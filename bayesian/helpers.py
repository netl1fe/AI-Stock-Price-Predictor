import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def add_selected_ta_features(data):
    data['SMA'] = ta.trend.sma_indicator(data['Close'])
    data['RSI'] = ta.momentum.rsi(data['Close'])
    data = data.dropna()
    return data

def reorder_data(data):
    data = data[['SMA', 'RSI', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

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
