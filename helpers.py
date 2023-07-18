import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange
from ta.trend import CCIIndicator
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator
from ta.utils import dropna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_selected_ta_features(data):
    # Bollinger Bands
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()

    # Moving Average Convergence Divergence (MACD)
    indicator_macd = MACD(close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = indicator_macd.macd()
    data['macd_signal'] = indicator_macd.macd_signal()
    data['macd_diff'] = indicator_macd.macd_diff()

    # Relative Strength Index (RSI)
    indicator_rsi = RSIIndicator(close=data["Close"], window=14)
    data['rsi'] = indicator_rsi.rsi()

    # Volume Weighted Average Price (VWAP)
    indicator_vwap = VolumeWeightedAveragePrice(
        high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
    data['vwap'] = indicator_vwap.volume_weighted_average_price()

    # Daily Return
    indicator_daily_return = DailyReturnIndicator(close=data['Close'])
    data['daily_return'] = indicator_daily_return.daily_return()

    # Simple Moving Average (SMA)
    indicator_sma = SMAIndicator(close=data["Close"], window=14)
    data['sma'] = indicator_sma.sma_indicator()

    # Exponential Moving Average (EMA)
    indicator_ema = EMAIndicator(close=data["Close"], window=14)
    data['ema'] = indicator_ema.ema_indicator()

    # Stochastic Oscillator
    indicator_so = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['so_k'] = indicator_so.stoch()
    data['so_d'] = indicator_so.stoch_signal()

    # Average True Range (ATR)
    indicator_atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['atr'] = indicator_atr.average_true_range()

    # Commodity Channel Index (CCI)
    indicator_cci = CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=20)
    data['cci'] = indicator_cci.cci()

    # On Balance Volume (OBV)
    indicator_obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
    data['obv'] = indicator_obv.on_balance_volume()

    data = data.dropna()
    return data


def reorder_data(data):
    # This function reorders the columns in a DataFrame.
    # Adjust as necessary to suit your specific needs.
    data = data[[ 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def normalize_data(data):
    original_data = data.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler, original_data

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def train_model(model, X_train, y_train, num_epochs, criterion, optimizer):
    for epoch in range(num_epochs):
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model, loss.item()

def model_loss(model, X, y, criterion):
    outputs = model(X)
    loss = criterion(outputs, y)
    return loss.item()
 
