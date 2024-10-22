# Machine Failure Analysis Dashboard

This Streamlit app provides a Machine Failure Analysis Dashboard that helps analyze the failure rates of machines based on various parameters like product quality, air temperature, process temperature, rotational speed, torque, and tool wear. It uses a trained Random Forest model to predict the likelihood of machine failures and allows for interactive simulations to explore different failure scenarios.

## Features

- **Failure Prediction**: Input machine parameters such as type, air temperature, process temperature, rotational speed, torque, and tool wear to predict whether the machine will fail or not.
- **Failure Rate Distribution**: Visualize the overall failure rate distribution based on product quality variants (L, M, H).
- **Failure Mode Contribution**: Explore the contribution of different failure modes (TWF, HDF, PWF, OSF, RNF) to the overall failure rate.
- **Tool Wear Impact**: Analyze how tool wear affects the machine's failure rate, with a scatter plot of tool wear vs. failure rate.
- **Failure Rate Simulation**: Adjust torque and tool wear values to simulate failure probabilities for different product quality types (L, M, H).
- **Interactive Insights**: Gain insights into the overstrain failure thresholds for each product quality type (L: 11,000 minNm, M: 12,000 minNm, H: 13,000 minNm).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/machine-failure-analysis-dashboard.git

cd machine-failure-analysis-dashboard

pip install -r requirements.txt

streamlit run app.py
