# AI Stock Price Predictor

AI Stock & Crypto Price Predictor is an intelligent application powered by Python, PyTorch, and Streamlit. It utilizes Long Short-Term Memory (LSTM) neural networks, as well as technical analysis indicators to generate future price predictions of stocks and cryptocurrencies.

The system is capable of fetching historical price data from Yahoo Finance with the help of the `yfinance` library. Post data extraction, several technical analysis indicators from the `ta` library. These indicators help derive meaningful features from the historical data.

Data normalization is achieved using the Min-Max Scaler from `scikit-learn`. The LSTM model, implemented with PyTorch, incorporates several LSTM cell layers followed by a fully connected layer to generate the output. This model is trained using the historical price data, with mean squared error (MSE) loss and the Adam optimizer enhancing the learning process.

## Running the Application

Follow the steps to run the application:

1. Begin the model training:
   ```
   streamlit run trainer.py
   ```

2. Launch the prediction application:
   ```
   streamlit run predictor.py
   ```

Please remember to train the model before running the predictor. The prediction application relies on the trained model for its operations. If the model isn't trained, the prediction application won't function as expected.
