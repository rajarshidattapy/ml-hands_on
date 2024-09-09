
# Solar Power Forecasting using LSTM and CNN

This project forecasts solar power production based on historical cumulative power data using a combination of Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN). It employs deep learning techniques to predict daily solar power generation and cumulative power over one year, evaluating the model's performance using various metrics like R² and MAE.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Project Overview
This project uses an LSTM-CNN hybrid model to predict solar power generation based on past power consumption and gas meter readings. It helps forecast future power output, providing key insights for energy planning.

## Requirements
To run this project, you'll need the following libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- keras

Install the requirements using:

```bash
!pip install pandas numpy matplotlib scikit-learn tensorflow keras
```

## Dataset
The dataset consists of historical data on cumulative power generation, electricity consumption, and gas usage. It includes:
- `cum_power`: Cumulative solar power generated.
- `Elec_kW`: Electricity consumed.
- `Gas_mxm`: Gas usage data.

The CSV file `PV_Elec_Gas3.csv` should be placed in the project directory.

## Model Architecture
The project uses a hybrid deep learning model consisting of:
1. **LSTM Layer**: Captures temporal dependencies in the power data.
2. **Conv1D Layer**: Extracts local patterns and features from the data.
3. **MaxPooling**: Reduces the dimensionality of the data while preserving important features.
4. **BatchNormalization**: Normalizes activations to improve training stability.
5. **Fully Connected Layers**: Provides final predictions for the next 365 days of solar power generation.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/rajarshidattapy/ml-hands_on.git
cd ml-hands_on
```

2. Run the script to train the model and make predictions:

```bash
python ml-hands_on.ipynb
```

3. The model will generate a graph comparing the predicted and actual solar power over the validation set.

## Results
The model trains on one year of daily solar power data and forecasts the next year's output. After training for 20 epochs, the model's accuracy and loss metrics are displayed.

### Example Output:

- **True Cumulative Power (1 Year)**: `xxxx kW`
- **Predicted Cumulative Power (1 Year)**: `yyyy kW`
- **Accuracy**: `98.45%`

## Evaluation Metrics
The model performance is evaluated using:

- **R² Score**: Measures how well the model predicts actual power values.
- **Mean Absolute Error (MAE)**: Evaluates the average magnitude of errors in predictions.

For example:
- R² Score: `0.89`
- MAE: `1.15`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
