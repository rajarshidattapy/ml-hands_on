# Solar Power Generation Prediction with LSTM-CNN

This project deals with predicting solar power generation using an LSTM-CNN model. The model is built using TensorFlow and Keras, with data manipulation performed using pandas and NumPy, and visualizations created with Matplotlib.

## Overview

The code reads solar power generation data from a CSV file, preprocesses it to calculate daily power generation, and splits the data into training and validation sets. The project defines a function to split the time series data into input and output sequences and reshapes the data for the LSTM layer. An LSTM-CNN model is compiled and trained on the data. The model’s performance is evaluated using metrics like R-squared score and mean absolute error, and results are visualized using Matplotlib.

### Technologies Used:
- **Pandas**: For data manipulation and analysis, particularly for working with data frames.
- **NumPy**: For numerical operations and array manipulation.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For model evaluation metrics.
- **TensorFlow**: For building and training machine learning models.
- **Keras**: A high-level API running on top of TensorFlow for model development.


