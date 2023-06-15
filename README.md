# AI Stock & Crypto Price Predictor

This program is an AI-powered stock and cryptocurrency price predictor built using Python and Streamlit. It utilizes Long Short-Term Memory (LSTM) neural networks and technical analysis indicators to make predictions on the future prices of financial assets.

The program fetches historical price data from Yahoo Finance using the `yfinance` library. It then applies various technical analysis indicators from the `ta` library, including Bollinger Bands, Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and others, to extract relevant features from the data. The data is normalized using a Min-Max Scaler from `scikit-learn`.

The LSTM model is implemented using PyTorch and consists of multiple layers of LSTM cells followed by a fully connected layer for the output. The model is trained on the historical price data using mean squared error (MSE) loss and Adam optimizer.

## Setup and Usage

1. Clone the repository or download the program files.
2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Run the program using the following command:
   ```
   streamlit run app.py
   ```
4. Access the program interface in your web browser at `http://localhost:8501`.

## Usage Instructions

1. Upon running the program, you will see a sidebar with various customizable settings for the prediction model.
2. Adjust the settings according to your preferences, including the ticker symbol, number of layers, hidden dimension, number of epochs, learning rate, and sequence length.
3. Specify the start and end dates for the historical data.
4. Click the "Predict" button to start the prediction process.
5. The program will fetch the historical price data, apply technical analysis indicators, normalize the data, and train the LSTM model.
6. It will then make predictions for the next day's prices and display them on the interface.
7. If actual data is available for the predicted day, the program will also display the actual prices for comparison.
8. The program allows for multiple prediction cycles, where each cycle predicts the next day's prices based on the previous day's predictions.
9. The best predictions across all cycles are tracked, and the corresponding model is saved to a file named `best_model.pt`.

## Disclaimer

Please note that the predictions made by this program are based on historical data and technical analysis indicators. They should not be considered as financial advice or an accurate representation of future prices. Always conduct your own research.
