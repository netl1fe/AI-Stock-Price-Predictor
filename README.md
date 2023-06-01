# GAINZ

This program uses a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) neural networks to predict stock prices. Leveraging technical analysis indicators and historical price data, the model forecasts the Open, High, Low, and Close prices for the next trading day.

## Usage

1. Clone the repository:
```
git clone https://github.com/user00011001/Gainz.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Open `V0.py` and modify the parameters as needed. You can set the ticker symbol, start and end date for the data, and adjust the neural network and training parameters.

4. Run the program:
```
python V0.py
```

5. The program will train the model and print the predicted Open, High, Low, and Close prices for the following day based on the trained model.

## Tuning Hyperparameters

The performance of the model heavily relies on the configuration of the hyperparameters. These include the sequence length, hidden dimensions, number of LSTM layers, and learning rate, among others. The model also uses early stopping, which you can configure by setting the patience level.

When tuning these hyperparameters, consider the following:

- **Sequence Length**: The sequence length represents the number of historical data points that the model will consider. A larger sequence length may capture more trends but could also lead to overfitting. 
- **Hidden Dimensions**: This parameter refers to the size of the LSTM's hidden state. More dimensions can capture more complex patterns but may also increase computational cost and overfitting risk.
- **Number of LSTM Layers**: Adding more layers can capture more complex relationships in the data but also raises the risk of overfitting and requires more computational resources.
- **Learning Rate**: The learning rate determines how much the model changes in response to the estimated error each time the model weights are updated. Choosing the right learning rate is crucial for good performance.
- **Early Stopping Patience**: This is the number of epochs with no improvement after which training will be stopped. The use of early stopping aims to prevent overfitting.

It's crucial to balance model complexity with the risk of overfitting. Using a validation set or cross-validation can provide a more robust way to tune these hyperparameters.

## Features

- Fetches historical stock price data using the yfinance library.
- Calculates several technical analysis indicators using the Technical Analysis Library (ta).
- Normalizes data using Scikit-learn's MinMaxScaler.
- Implements a Conv1D_LSTM model using PyTorch for predicting stock prices.
- Incorporates an early stopping mechanism during training to prevent overfitting.
