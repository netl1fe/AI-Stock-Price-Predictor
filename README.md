## Gainz - Stock Price Prediction with LSTM

Gainz is a Python program that utilizes Long Short-Term Memory (LSTM) neural networks to predict the future price movement of stocks. It uses historical stock data, including opening price, high, low, close, and volume, to train an LSTM model and make predictions.

## Prerequisites

Before running the program, make sure you have the following dependencies installed:

- Python 3
- NumPy
- pandas
- yfinance
- scikit-learn
- PyTorch

You can install the dependencies by running the following command:

```
pip install numpy pandas yfinance scikit-learn torch
```

## Usage

1. Clone the repository or download the source code files.

2. Open the `gainz.py` file in a text editor.

3. Adjust the `tickers`, `start_date`, `end_date`, `seq_length`, `input_dim`, `hidden_dim`, `num_layers`, `output_dim`, `num_epochs`, and `learning_rate` variables according to your requirements.

4. Save the changes and close the file.

5. Open a terminal or command prompt and navigate to the directory where the `gainz.py` file is located.

6. Run the program using the following command:

```
python3 gainz.py
```

7. The program will start processing each ticker and display the training progress and the predicted next day values for each stock.

## Important Note

- The Gainz program uses historical stock data to train the LSTM model and make predictions. However, stock price prediction is a complex and uncertain task, and the program's predictions should be used for informational purposes only. It is recommended to perform thorough analysis and consult financial experts before making any investment decisions.

- The accuracy of the predictions depends on various factors, including the quality and relevance of the input data, the model architecture, and the nature of the stock market. Interpret the predictions with caution and do not solely rely on them for making investment decisions.

- Gainz assumes that you have access to a reliable data source for historical stock prices. It uses the `yfinance` library to fetch the data from Yahoo Finance. If you encounter any issues with data retrieval, make sure your internet connection is stable and the data source is accessible.

