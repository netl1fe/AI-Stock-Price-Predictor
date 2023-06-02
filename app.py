import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Conv1D_LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Conv1D_LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        x = self.conv1(x)
        x = self.relu(x)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate, patience):
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for t in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        inputs = X_train.to(device)
        targets = y_train.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        model.eval()
        val_outputs = model(X_val.to(device))
        val_loss = criterion(val_outputs, y_val.to(device))

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if t % 10 == 0:
            print(f"Epoch {t} train loss: {loss.item()}, validation loss: {val_loss.item()}")
    
    return model

def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = dropna(data)
    return data


def add_selected_ta_features(data):
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()

    indicator_macd = MACD(
        close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = indicator_macd.macd()
    data['macd_signal'] = indicator_macd.macd_signal()
    data['macd_diff'] = indicator_macd.macd_diff()

    indicator_rsi = RSIIndicator(close=data["Close"], window=14)
    data['rsi'] = indicator_rsi.rsi()

    return data


def normalize_data(data):
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    return (
        data_norm,
        scaler,
    )


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
    model.train()
    progress_bar = st.progress(0)

    loss_values = []
    for t in range(num_epochs):
        optimizer.zero_grad()
        inputs = X.to(device)
        targets = y.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            loss_values.append(loss.item())
            print(f"Epoch {t} train loss: {loss.item()}")
        
        progress_bar.progress((t+1)/num_epochs)
    
    model.eval()
    
    training_loss_chart = st.line_chart(pd.DataFrame(loss_values, columns=['loss']))
    return model



st.title("AI Stock Price Predictor")

st.subheader("Enter your preferences")
ticker = st.text_input("Ticker Symbol", "SPY")
start_date = st.date_input("Start Date", datetime(1993, 1, 30))
end_date = st.date_input("End Date", datetime(2023, 5, 25))
seq_length = st.slider("Sequence Length", min_value=1, max_value=100, value=60)
hidden_dim = st.slider("Hidden Dimension Size", min_value=1, max_value=128, value=64)
num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=3)
num_epochs = st.slider("Number of Epochs", min_value=1, max_value=500, value=200)
learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)

if st.button("Predict"):
    st.subheader("Fetching Data and Predicting Prices...")
    end_date_datetime = end_date
    next_day_datetime = end_date_datetime + timedelta(days=1)
    next_day = next_day_datetime.strftime("%Y-%m-%d")

    data = fetch_historical_data(ticker, start_date, end_date)
    data = add_selected_ta_features(data)
    data = data.dropna()  
    data, scaler = normalize_data(data.values)
    X, y = create_sequences(data, seq_length)

    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    input_dim = X.shape[2]
    output_dim = data.shape[1]

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    model = train_model(model, X, y, num_epochs, learning_rate)

    X_predict = X[-1:, :, :]
    model.eval()
    prediction = model(X_predict)
    prediction = prediction.detach().cpu().numpy()

    prediction = scaler.inverse_transform(prediction)

    st.subheader(f"Predicted prices for {ticker} on {next_day}:")
    st.write(f"Open: {prediction[0, 0]}")
    st.write(f"High: {prediction[0, 1]}")
    st.write(f"Low: {prediction[0, 2]}")
    st.write(f"Close: {prediction[0, 3]}")

    # Fetch actual data for the predicted day
    actual_data = fetch_historical_data(ticker, next_day, next_day_datetime + timedelta(days=1))

    if not actual_data.empty:
        st.subheader(f"Actual prices for {ticker} on {next_day}:")
        st.write(f"Open: {actual_data.iloc[0]['Open']}")
        st.write(f"High: {actual_data.iloc[0]['High']}")
        st.write(f"Low: {actual_data.iloc[0]['Low']}")
        st.write(f"Close: {actual_data.iloc[0]['Close']}")
    else:
        st.write("The actual data for the predicted day is not yet available.")

st.markdown("---")
st.markdown("Built by netl1fe", unsafe_allow_html=True)
