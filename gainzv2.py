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
    data = add_all_ta_features(data, open="Open", high="High",
                               low="Low", close="Close", volume="Volume", fillna=True)
    return data


def normalize_data(data):
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    scaler_close = MinMaxScaler()  # additional scaler for Close price
    close = data[:, -1].reshape(-1, 1)  # reshape to 2D array
    close_norm = scaler_close.fit_transform(close)
    # return close_norm and scaler_close
    return data_norm, close_norm, scaler, scaler_close


def create_sequences(data, close, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(close[i + seq_length])  # use close_norm here
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
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(num_epochs):
        y_pred = model(X)
        loss = criterion(y_pred.float(), y.view(y.size(0), -1))  # changed here
        if t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
        elif t % 10 == 1:
            print(f'Epoch {t} train loss: {loss.item()}')
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model


ticker = 'SPY'
start_date = '2000-01-01'
end_date = datetime.date.today().strftime('%Y-%m-%d')
seq_length = 7
hidden_dim = 32
num_layers = 2
num_epochs = 500
learning_rate = 0.001

data = fetch_historical_data(ticker, start_date, end_date)
data, close_norm, scaler, scaler_close = normalize_data(data.values)
X, y = create_sequences(data, close_norm, seq_length)
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

input_dim = X.shape[2]
output_dim = 1  # changed here
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
model = train_model(model, X, y, num_epochs, learning_rate)

X_latest = data[-seq_length:].reshape(1, seq_length, input_dim)
X_latest = torch.from_numpy(X_latest).float().to(device)
prediction = model(X_latest)
prediction = model(X_latest)
prediction = scaler_close.inverse_transform(
    prediction.cpu().detach().numpy().reshape(1, -1))  # use scaler_close here


print(f'Predicted next day price for {ticker}: {prediction[0][0]}')
