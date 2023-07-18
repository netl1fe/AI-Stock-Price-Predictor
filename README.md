# AI Stock & Crypto Price Predictor

This program is an AI-powered stock and cryptocurrency price predictor built using Python and Streamlit. It utilizes Long Short-Term Memory (LSTM) neural networks and technical analysis indicators to make predictions on the future prices of financial assets.

The program fetches historical price data from Yahoo Finance using the `yfinance` library. It then applies various technical analysis indicators from the `ta` library, including Bollinger Bands, Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and others, to extract relevant features from the data. The data is normalized using a Min-Max Scaler from `scikit-learn`.

The LSTM model is implemented using PyTorch and consists of multiple layers of LSTM cells followed by a fully connected layer for the output. The model is trained on the historical price data using mean squared error (MSE) loss and Adam optimizer.

## Setup and Usage

1. Run the training program using the following command:
   ```
   streamlit run trainer.py
   ```
4. Once model is saved, run the predictor with:
   ```
   streamlit run predictor.py
   ```
