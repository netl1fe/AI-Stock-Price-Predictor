import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = dropna(data)
    return data


def normalize_data(data):
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    scaler_open = MinMaxScaler()
    scaler_high = MinMaxScaler()
    scaler_low = MinMaxScaler()
    scaler_close = MinMaxScaler()
    prices = data[:, -4:]
    scaler_open.fit(prices[:, 0].reshape(-1, 1))
    scaler_high.fit(prices[:, 1].reshape(-1, 1))
    scaler_low.fit(prices[:, 2].reshape(-1, 1))
    scaler_close.fit(prices[:, 3].reshape(-1, 1))
    return (
        data_norm,
        scaler,
        scaler_open,
        scaler_high,
        scaler_low,
        scaler_close,
        scaler_open.data_min_,
        scaler_open.data_max_,
    )


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, :])  # Adjusted to include all features
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, X, y, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()  # set model to training mode
    for t in range(num_epochs):
        optimizer.zero_grad()
        inputs = X.to(device)
        targets = y.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            print(f"Epoch {t} train loss: {loss.item()}")
    model.eval()  # set model to evaluation mode after training
    return model


ticker = "SPY"
start_date = "1993-01-30"
end_date = "2023-05-24"
seq_length = 7
hidden_dim = 64
num_layers = 2
num_epochs = 100
learning_rate = 0.001

data = fetch_historical_data(ticker, start_date, end_date)
data, scaler, scaler_open, scaler_high, scaler_low, scaler_close, data_min, data_max = normalize_data(
    data.values
)
X, y = create_sequences(data, seq_length)

X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

input_dim = X.shape[2]
output_dim = data.shape[1]

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
model = train_model(model, X, y, num_epochs, learning_rate)

X_latest = data[-seq_length:].reshape(1, seq_length, input_dim)
X_latest = torch.from_numpy(X_latest).float().to(device)
model.eval()  # set model to evaluation mode before making predictions
prediction = model(X_latest)
prediction = prediction.detach().cpu().numpy()
prediction = prediction * (data_max - data_min) + data_min

print(f"Predicted next day prices for {ticker}:")
print(f"Open: {prediction[0, 0]}")
print(f"High: {prediction[0, 1]}")
print(f"Low: {prediction[0, 2]}")
print(f"Close: {prediction[0, 3]}")
