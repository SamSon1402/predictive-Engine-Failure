# Machine Failure Analysis Dashboard

## Overview

This Streamlit app provides an interactive **Machine Failure Analysis Dashboard**. The app helps analyze machine failure rates based on various parameters such as product quality, air temperature, process temperature, rotational speed, torque, and tool wear. Using a pre-trained Random Forest model, it predicts the likelihood of machine failure and allows users to simulate different failure scenarios based on the input variables.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Application Components](#application-components)
- [Data Generation](#data-generation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features

- **Failure Prediction**: Input machine parameters to predict failure probability
- **Failure Rate Distribution**: Visualize failure rates across quality types
- **Failure Mode Contribution**: Analyze impact of different failure modes
- **Tool Wear Impact**: Understand relationship between tool wear and failures
- **Failure Rate Simulation**: Test different scenarios with interactive controls
- **Interactive Insights**: Quality-specific threshold monitoring

## Prerequisites

- Python 3.x
- Streamlit
- Plotly
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-failure-analysis-dashboard.git
   ```

2. Navigate to project directory:
   ```bash
   cd machine-failure-analysis-dashboard
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify model file:
   - Ensure `random_forest.pkl` is in the root directory
   - Or update model path in `app.py`

## Running the App

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

2. Access the dashboard:
   - Open web browser
   - Navigate to `http://localhost:8501`

## Application Components

### 1. Machine Learning Predictor
- Input parameters:
  - Type (L/M/H)
  - Air Temperature [°C]
  - Process Temperature [°C]
  - Rotational Speed [rpm]
  - Torque [Nm]
  - Tool Wear [min]
- Real-time failure prediction

### 2. Failure Analysis
- Distribution visualization
- Quality-specific analysis
- Historical trend analysis

### 3. Failure Modes
- TWF (Tool Wear Failure)
- HDF (Heat Dissipation Failure)
- PWF (Power Failure)
- OSF (Overstrain Failure)
- RNF (Random Failure)

### 4. Interactive Controls
- Parameter adjustment sliders
- Quality type selectors
- Time range filters

### 5. Threshold Monitoring
Quality-specific thresholds:
- L: 11,000 minNm
- M: 12,000 minNm
- H: 13,000 minNm

## Data Generation

The `generate_data()` function simulates:

### Input Features
- Air Temperature [°C]
- Process Temperature [°C]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

### Output Features
- TWF (Tool Wear Failure)
- HDF (Heat Dissipation Failure)
- PWF (Power Failure)
- OSF (Overstrain Failure)
- RNF (Random Failure)
- Target (0: No Failure, 1: Failure)

## Usage Examples

### Failure Prediction
1. Enter machine parameters
2. Click "Predict Failure"
3. View prediction results

### Simulation
1. Adjust torque/tool wear sliders
2. Select quality type
3. Observe failure probability changes

### Analysis
1. Choose time range
2. Select quality types
3. Examine visualizations
4. Export results

## Project Structure

```
machine-failure-analysis-dashboard/
├── app.py                 # Main application
├── random_forest.pkl      # Trained model
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
└── screenshots/          # UI screenshots
```

## Dependencies

```bash
streamlit>=1.2.0
pandas>=1.3.0
plotly>=5.3.0
scikit-learn>=0.24.2
numpy>=1.21.0
```

Install via:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Streamlit: Interactive dashboard framework
- Plotly: Data visualization library
- Scikit-learn: Machine learning tools
- Project contributors and maintainers

## Contact

- **Developer**: [Your Name]
- **Email**: your-email@example.com
- **GitHub**: [Your GitHub Profile]
- **LinkedIn**: [Your LinkedIn Profile]

---

*Last updated: [Current Date]*
