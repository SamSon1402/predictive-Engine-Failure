import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved Random Forest model
@st.cache_resource
def load_model():
    with open('/content/random_forest.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Print model features for debugging
print("Model features:", model.feature_names_in_)

# Generate sample data
def generate_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'UID': range(1, n_samples + 1),
        'Type': np.random.choice(['L', 'M', 'H'], n_samples, p=[0.6, 0.3, 0.1]),
        'Air temperature [°C]': np.random.normal(27, 2, n_samples),
        'Process temperature [°C]': np.random.normal(37, 1, n_samples),
        'Rotational speed [rpm]': np.random.normal(2860, 100, n_samples),
        'Torque [Nm]': np.abs(np.random.normal(40, 10, n_samples)),
        'Tool wear [min]': np.random.uniform(0, 240, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate temperature difference
    df['Temperature difference [°C]'] = df['Process temperature [°C]'] - df['Air temperature [°C]']
    
    # Calculate failure modes
    df['TWF'] = df['Tool wear [min]'] > np.random.uniform(200, 240, n_samples)
    df['HDF'] = (df['Temperature difference [°C]'] < 8.6) & (df['Rotational speed [rpm]'] < 1380)
    df['PWF'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60 < 3500) | (df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60 > 9000)
    df['OSF'] = (df['Type'] == 'L') & (df['Tool wear [min]'] * df['Torque [Nm]'] > 11000) | \
                (df['Type'] == 'M') & (df['Tool wear [min]'] * df['Torque [Nm]'] > 12000) | \
                (df['Type'] == 'H') & (df['Tool wear [min]'] * df['Torque [Nm]'] > 13000)
    df['RNF'] = np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
    
    df['Target'] = (df['TWF'] | df['HDF'] | df['PWF'] | df['OSF'] | df['RNF']).astype(int)
    
    return df

# Load or generate data
@st.cache_data
def load_data():
    df = generate_data()
    
    # Create LabelEncoder for 'Type'
    le_type = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    
    return df, le_type

df, le_type = load_data()

# Streamlit app
st.title('Machine Failure Analysis Dashboard')

# Machine Learning Prediction Section
st.header('Machine Learning Failure Prediction')

# Input fields for prediction
st.write("Enter the following parameters to predict machine failure:")
type_input = st.selectbox('Type', ['L', 'M', 'H'])
air_temp = st.number_input('Air Temperature [°C]', value=27.0)
process_temp = st.number_input('Process Temperature [°C]', value=37.0)
rotational_speed = st.number_input('Rotational Speed [rpm]', value=2860)
torque = st.number_input('Torque [Nm]', value=40.0)
tool_wear = st.number_input('Tool Wear [min]', value=0)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Type': [type_input],
    'Air temperature [°C]': [air_temp],
    'Process temperature [°C]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear]
})

# Calculate temperature difference
input_data['Temperature difference [°C]'] = input_data['Process temperature [°C]'] - input_data['Air temperature [°C]']

# Add placeholder for 'Target' column (it's not used for prediction, but the model might expect it)
input_data['Target'] = 0

# Encode 'Type' column
input_data['Type'] = le_type.transform(input_data['Type'])

# Ensure the order of columns matches the model's expected features
input_data = input_data[model.feature_names_in_]

# Debug print
print("Input data for prediction:", input_data)

# Make prediction
if st.button('Predict Failure'):
    prediction = model.predict(input_data)
    failure_type = "Failure" if prediction[0] == 1 else "No Failure"
    st.write(f"Predicted Outcome: {failure_type}")

# User input for product quality variant selection
selected_variants = st.multiselect('Select Product Quality Variants', ['L', 'M', 'H'], default=['L', 'M', 'H'])

# Filter data based on user selection
filtered_df = df[df['Type'].isin(le_type.transform(selected_variants))]

# Overall failure rate distribution
st.subheader('Overall Failure Rate Distribution')
failure_rate = filtered_df.groupby('Type')['Target'].mean().reset_index()
failure_rate['Type'] = le_type.inverse_transform(failure_rate['Type'])
fig_overall = px.bar(failure_rate, x='Type', y='Target', 
                     title='Failure Rate by Product Quality',
                     labels={'Target': 'Failure Rate', 'Type': 'Product Quality'})
st.plotly_chart(fig_overall)

# Failure mode contribution
st.subheader('Failure Mode Contribution')
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_mode_data = filtered_df.groupby('Type')[failure_modes].mean().reset_index()
failure_mode_data['Type'] = le_type.inverse_transform(failure_mode_data['Type'])
failure_mode_data_melted = pd.melt(failure_mode_data, id_vars=['Type'], value_vars=failure_modes, 
                                   var_name='Failure_Mode', value_name='Rate')
fig_modes = px.bar(failure_mode_data_melted, x='Type', y='Rate', color='Failure_Mode', 
                   title='Failure Mode Contribution by Product Quality',
                   labels={'Rate': 'Failure Rate', 'Type': 'Product Quality'})
st.plotly_chart(fig_modes)

# Tool wear impact
st.subheader('Tool Wear Impact on Failure Rate')
filtered_df['Type'] = le_type.inverse_transform(filtered_df['Type'])
fig_tool_wear = px.scatter(filtered_df, x='Tool wear [min]', y='Target', color='Type', 
                           title='Tool Wear vs Failure Rate',
                           labels={'Tool wear [min]': 'Tool Wear (min)', 'Target': 'Failure (0/1)', 'Type': 'Product Quality'})
st.plotly_chart(fig_tool_wear)

# Interactive feature: Torque and Tool Wear adjustment
st.subheader('Failure Rate Simulation')
torque = st.slider('Select Torque (Nm)', min_value=0, max_value=100, value=40)
tool_wear = st.slider('Select Tool Wear (min)', min_value=0, max_value=240, value=120)

# Calculate failure probabilities based on user input
failure_probs = {
    'L': 1 if tool_wear * torque > 11000 else 0,
    'M': 1 if tool_wear * torque > 12000 else 0,
    'H': 1 if tool_wear * torque > 13000 else 0
}

fig_simulation = go.Figure(data=[go.Bar(x=list(failure_probs.keys()), y=list(failure_probs.values()))])
fig_simulation.update_layout(title='Simulated Failure Probabilities',
                             xaxis_title='Product Quality',
                             yaxis_title='Failure Probability')
st.plotly_chart(fig_simulation)

# Data breakdown
st.subheader('Data Breakdown')
st.write(f"Tool wear impact on failure likelihood:")
for variant in ['L', 'M', 'H']:
    threshold = 11000 if variant == 'L' else (12000 if variant == 'M' else 13000)
    st.write(f"- {variant}: Fails when Tool wear * Torque > {threshold} minNm")

st.write("\nOverstrain failure thresholds:")
st.write("- L: 11,000 minNm")
st.write("- M: 12,000 minNm")
st.write("- H: 13,000 minNm")

st.sidebar.info('This dashboard analyzes the impact of product quality variants (L, M, H) on machine failure rates. Use the interactive features to explore different scenarios and understand how various factors contribute to machine failures.')