##AI Stock & Crypto Price Predictor

This program is an AI-powered stock and cryptocurrency price predictor built using Python and Streamlit. It utilizes Long Short-Term Memory (LSTM) neural networks and technical analysis indicators to make predictions on the future prices of financial assets.

The program fetches historical price data from Yahoo Finance using the `yfinance` library. It then applies various technical analysis indicators from the `ta` library, including Bollinger Bands, Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and others, to extract relevant features from the data. The data is normalized using a Min-Max Scaler from `scikit-learn`.

The LSTM model is implemented using PyTorch and consists of multiple layers of LSTM cells followed by a fully connected layer for the output. The model is trained on the historical price data using mean squared error (MSE) loss and Adam optimizer.

The program provides a user interface using Streamlit, allowing users to customize the prediction settings such as the ticker symbol, number of layers, hidden dimension, number of epochs, learning rate, and sequence length. Users can also specify the start and end dates for the historical data.

Upon clicking the "Predict" button, the program performs the following steps:

1. Fetches the historical price data for the specified ticker and date range.
2. Applies the selected technical analysis indicators to the data.
3. Normalizes the data using Min-Max Scaler.
4. Creates sequences of input-output pairs for training the LSTM model.
5. Trains the model on the data for the specified number of epochs.
6. Makes predictions for the next day's prices.
7. Displays the predicted prices and compares them to the actual prices (if available).

The program allows multiple prediction cycles, where each cycle predicts the next day's prices based on the previous day's predictions. It keeps track of the best predictions across all cycles and saves the corresponding model to a file named `best_model.pt`.

**Setup and Usage:**

1. Clone the repository or download the program files.
2. Install the required dependencies listed in the `requirements.txt` file using `pip install -r requirements.txt`.
3. Run the program using `streamlit run app.py`.
4. Access the program interface in your web browser at `http://localhost:8501`.

**Disclaimer:**

Please note that the predictions made by this program are based on historical data and technical analysis indicators. They should not be considered as financial advice or an accurate representation of future prices. Always conduct your own research.
