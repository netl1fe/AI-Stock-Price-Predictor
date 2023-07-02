import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
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
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
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

def train_model(model, X, y, num_epochs, criterion, optimizer):
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

        progress_bar.progress((t + 1) / num_epochs)  # Calculate progress percentage as a float

    model.eval()
    training_loss_chart = st.line_chart(pd.DataFrame(loss_values, columns=['loss']))
    return model, training_loss_chart



def fetch_historical_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {str(e)}")
        return None

def add_selected_ta_features(data):
    # Bollinger Bands
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()

    # Moving Average Convergence Divergence (MACD)
    indicator_macd = MACD(
        close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
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

    return data

def reorder_data(data):
    cols = data.columns.tolist()
    new_order = ['Open', 'High', 'Low', 'Close'] + [col for col in cols if col not in ['Open', 'High', 'Low', 'Close']]
    data = data[new_order]
    return data

def normalize_data(data):
    if data.shape[0] <= 1:
        return data, None

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    return data_normalized, scaler

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Sidebar input
hidden_dim = st.sidebar.slider("Hidden Dimension", min_value=1, max_value=100, value=64)
num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=5, value=3)
num_epochs = int(st.sidebar.text_input("Number of Epochs", value='100')) 
learning_rate = float(st.sidebar.text_input("Learning Rate", value='0.001'))  
seq_length = st.sidebar.slider("Sequence Length", min_value=1, max_value=200, value=60)
ticker = st.sidebar.text_input("Ticker", "SPY")
num_cycles = int(st.sidebar.text_input("Number of Prediction Cycles", value='1'))  

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))

best_model = None
best_loss = np.inf

for cycle in range(num_cycles):
    st.subheader(f"Cycle {cycle + 1}/{num_cycles}")
    data = fetch_historical_data(ticker, start_date, end_date)
    if data is None:
        continue
    data = add_selected_ta_features(data)
    data = reorder_data(data.dropna())

    # Split data into training and validation sets before normalizing
    train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Normalize data
    train_data, train_scaler = normalize_data(train_data.values)
    valid_data, valid_scaler = normalize_data(valid_data.values)

    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_valid, y_valid = create_sequences(valid_data, seq_length)

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)

    input_dim = train_data.shape[1]

    model = LSTMModel(input_dim, hidden_dim, num_layers, input_dim).to(device)
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, training_loss_chart = train_model(model, X_train, y_train, num_epochs, criterion, optimizer)

    # Make prediction on validation set and calculate loss
    X_valid = torch.from_numpy(X_valid).float().to(device)
    y_valid = torch.from_numpy(y_valid).float().to(device)

    with torch.no_grad():
        prediction = model(X_valid)
        loss = criterion(prediction, y_valid)

    current_date = data.index[-seq_length] + pd.Timedelta(days=1)  # Update current_date

    st.write(f"Prediction for date: {current_date}")

    if loss < best_loss:
        best_loss = loss
        best_model = model

    st.write(f"Validation loss: {loss.item()}")

    # Inverse transform the predicted and actual prices
    predicted_prices = train_scaler.inverse_transform(prediction.cpu().numpy())
    true_prices = train_scaler.inverse_transform(y_train.cpu().numpy())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("**Predicted Prices**")
        st.write(f"**Open**: {predicted_prices[0][0]}")
        st.write(f"**High**: {predicted_prices[0][1]}")
        st.write(f"**Low**: {predicted_prices[0][2]}")
        st.write(f"**Close**: {predicted_prices[0][3]}")

    with col2:
        st.subheader("**Actual Prices**")
        st.write(f"**Open**: {true_prices[0][0]}")
        st.write(f"**High**: {true_prices[0][1]}")
        st.write(f"**Low**: {true_prices[0][2]}")
        st.write(f"**Close**: {true_prices[0][3]}")


if best_model is not None:
    st.balloons()
    st.subheader("Best Model")
    st.write(f"Best validation loss: {best_loss.item()}")

    # Evaluate the best model on the validation set
    with torch.no_grad():
        final_predictions = best_model(X_valid)
        final_loss = criterion(final_predictions, y_valid)

    st.write(f"Final validation loss: {final_loss.item()}")

    # Inverse transform the final predictions
    final_predicted_prices = train_scaler.inverse_transform(final_predictions.cpu().numpy())
    final_true_prices = train_scaler.inverse_transform(y_valid.cpu().numpy())

    # Plot the final predictions vs true prices
    st.subheader("Predictions vs True Prices")
    df = pd.DataFrame({'Predicted': final_predicted_prices.flatten(), 'True': final_true_prices.flatten()})
    st.line_chart(df)
else:
    st.write("No model was trained.")
