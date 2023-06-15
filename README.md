# AI Stock Price Predictor

This is a Streamlit-based application that uses an LSTM (Long Short-Term Memory) model to predict stock prices based on historical data and selected technical indicators.

## Installation

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the application with the command `streamlit run app.py`.

## Usage

1. Open the application in your browser.
2. Set the desired parameters in the sidebar:
   - Choose the ticker symbol for the stock you want to predict.
   - Specify the start and end dates for the historical data.
   - Adjust the model settings, such as the hidden dimension, number of layers, number of epochs, learning rate, and sequence length.
   - Determine the number of prediction cycles you want to run.
3. Click the "Predict" button to start the prediction process.
4. The application will display the predicted prices for the next day based on the selected parameters.
5. Additionally, the actual prices for the predicted day will be fetched and displayed if available.
6. The best predictions from all the cycles will be identified and highlighted.
7. The best model will be saved as "best_model.pt" in the current directory.

## Features

- Uses an LSTM model to predict stock prices.
- Includes various technical indicators for feature engineering, such as Bollinger Bands, MACD, RSI, VWAP, and more.
- Allows customization of model parameters and input settings.
- Provides visualization of the training loss during model training.
- Compares predictions across multiple cycles and identifies the best predictions.
- Fetches and displays actual prices for the predicted day.

## Limitations

- Stock market predictions are inherently uncertain and can be affected by various factors.
- The accuracy of the predictions may vary depending on the selected parameters, historical data, and market conditions.
- The program should not be used as the sole basis for financial decisions. It is intended for educational and informational purposes only.
