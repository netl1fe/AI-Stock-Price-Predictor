# GAINZ

This program utilizes LSTM (Long Short-Term Memory) neural networks to predict stock prices. By incorporating technical analysis indicators and historical price data, the model can forecast the Open, High, Low, and Close prices for the next day.

## Usage

1. Clone the repository:
```
git clone https://github.com/yourusername/stock-price-prediction.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Open `v0.py` and modify the program parameters to suit your needs.

4. Run the program:
```
python v0.py
```

5. The program will print the predicted Open, High, Low, and Close prices for the next day based on the trained model.

## Acknowledgments

- The program uses the yfinance library to fetch historical stock price data.
- Technical analysis indicators are provided by the Technical Analysis Library (ta).
- PyTorch is utilized for implementing and training the LSTM model.

## Limitations and Future Improvements

- Stock price prediction is a challenging task influenced by various factors such as model architecture, features, hyperparameters, data quality, and market unpredictability.
- Further experimentation and hyperparameter tuning may be required to improve prediction accuracy.
- Additional features, like volume-based indicators or macroeconomic data, could enhance the model's performance.
- Remember that past performance is not indicative of future results in the stock market, and predictions should not be the sole basis for financial decisions.
