# AI Stock Price Predictor

AI Stock & Crypto Price Predictor is an intelligent application powered by Python, PyTorch, and Streamlit. It utilizes advanced machine learning techniques with a special focus on Long Short-Term Memory (LSTM) neural networks, as well as technical analysis indicators to generate future price predictions of stocks and cryptocurrencies.

The system is capable of fetching historical price data from Yahoo Finance with the help of the `yfinance` library. Post data extraction, several technical analysis indicators from the `ta` library are applied, including but not limited to Bollinger Bands, Moving Average Convergence Divergence (MACD), and the Relative Strength Index (RSI). These indicators help derive meaningful features from the historical data.

Data normalization is achieved using the Min-Max Scaler from `scikit-learn`. The LSTM model, implemented with PyTorch, incorporates several LSTM cell layers followed by a fully connected layer to generate the output. This model is trained using the historical price data, with mean squared error (MSE) loss and the Adam optimizer enhancing the learning process.

## Setup and Usage

Here is a simple step-by-step guide to help you get started:

1. To initiate the training process, enter the following command:
   ```
   streamlit run trainer.py
   ```

2. Once the model has been successfully trained and saved, you can proceed to run the predictor using:
   ```
   streamlit run predictor.py
   ```

## Files Overview

The application comprises three significant Python files:

1. **data_fetcher.py**: Responsible for retrieving historical price data from Yahoo Finance through the `yfinance` library and applying technical analysis indicators using the `ta` library.

2. **trainer.py**: Manages the training process of the LSTM model, including loading and processing data, splitting it into training and validation datasets, and evaluating model performance.

3. **predictor.py**: Handles the task of loading the trained LSTM model and making future price predictions. An interactive web interface is created using `Streamlit`, where users can select a financial asset, and the program will provide the projected future price of the chosen asset.

## Installation Guide

Follow the steps below to install and run the project:

1. Clone the repository:
   ```
   git clone https://github.com/<username>/ai-stock-crypto-predictor.git
   cd ai-stock-crypto-predictor
   ```

2. Set up and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the necessary dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Follow the steps to run the application:

1. Fetch the data:
   ```
   python data_fetcher.py
   ```

2. Begin the model training:
   ```
   streamlit run trainer.py
   ```

3. Launch the prediction application:
   ```
   streamlit run predictor.py
   ```

Please remember to train the model before running the predictor. The prediction application relies on the trained model for its operations. If the model isn't trained, the prediction application won't function as expected.
