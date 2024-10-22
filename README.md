# Machine Failure Analysis Dashboard

This Streamlit app provides an interactive **Machine Failure Analysis Dashboard**. The app helps analyze machine failure rates based on various parameters such as product quality, air temperature, process temperature, rotational speed, torque, and tool wear. Using a pre-trained Random Forest model, it predicts the likelihood of machine failure and allows users to simulate different failure scenarios based on the input variables.

## Features

- **Failure Prediction**: Users can input machine parameters (type, air temperature, process temperature, rotational speed, torque, and tool wear) to predict machine failure using a pre-trained Random Forest model.
- **Failure Rate Distribution**: The dashboard visualizes the overall failure rate distribution across different product quality types (L, M, H).
- **Failure Mode Contribution**: Displays the contribution of various failure modes (TWF, HDF, PWF, OSF, RNF) to the overall failure rate.
- **Tool Wear Impact**: Users can visualize how tool wear impacts the machine's failure rate, with a scatter plot showing the relationship between tool wear and failure.
- **Failure Rate Simulation**: Simulate machine failures by adjusting torque and tool wear values for different product quality variants.
- **Interactive Insights**: Includes thresholds for overstrain failure based on product quality (L: 11,000 minNm, M: 12,000 minNm, H: 13,000 minNm).

## Prerequisites

Make sure you have the following installed before running the app:
- **Python 3.x**
- **Streamlit**
- **Plotly**
- **Pandas**
- **Scikit-learn**

## Installation

To install and run the app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-failure-analysis-dashboard.git

```bash
cd machine-failure-analysis-dashboard

```bash
pip install -r requirements.txt

```bash
streamlit run app.py


   
