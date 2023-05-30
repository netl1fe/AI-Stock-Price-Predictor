import numpy as np
import pandas as pd
import yfinance as yf
from ta import add_all_ta_features
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from datetime import datetime, timedelta

yf.pdr_override()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

def add_selected_ta_features(data):
    # Add all TA features
    data = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    return data_norm, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, :])
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected input type is torch.Tensor, but got {type(x)}")

        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for t in range(num_epochs):
        optimizer.zero_grad()
        inputs = X_train.to(device)
        targets = y_train.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_inputs = X_val.to(device)
        val_targets = y_val.to(device)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_targets)
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            print(f"Epoch {t} train loss: {loss.item()}, validation loss: {val_loss.item()}")
    model.eval()
    return model

ticker = "SPY"
start_date = "1993-01-30"
end_date = "2023-05-24"
seq_length = 120
hidden_dim = 128
num_layers = 3
num_epochs = 400
learning_rate = 0.001
validation_split = 0.2

data = fetch_historical_data(ticker, start_date, end_date)
data = add_selected_ta_features(data)
data = data.fillna(method='ffill').dropna()
data, scaler = normalize_data(data.values)
X, y = create_sequences(data, seq_length)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, shuffle=False)

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_val = torch.from_numpy(X_val).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)

input_dim = X.shape[2]
output_dim = data.shape[1]

model1 = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
model2 = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)

# Training two models
model1 = train_model(model1, X_train, y_train, X_val, y_val, num_epochs, learning_rate)
model2 = train_model(model2, X_train, y_train, X_val, y_val, num_epochs, learning_rate)

# Ensembling predictions
X_predict = X[-1:, :, :]
model1.eval()
model2.eval()
prediction1 = model1(X_predict)
prediction2 = model2(X_predict)
prediction = (prediction1 + prediction2) / 2
prediction = prediction.detach().cpu().numpy()

# Un-normalize the prediction
prediction = scaler.inverse_transform(prediction)

end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d")
next_day_datetime = end_date_datetime + timedelta(days=1)
next_day = next_day_datetime.strftime("%Y-%m-%d")

print(f"Predicted prices for {ticker} on {next_day}:")
print(f"Open: {prediction[0, 0]}")
print(f"High: {prediction[0, 1]}")
print(f"Low: {prediction[0, 2]}")
print(f"Close: {prediction[0, 3]}")
