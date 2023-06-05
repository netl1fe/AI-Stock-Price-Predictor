#Transformer
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):
    def __init__(self, feature_size, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1)  # changed nhead from 4 to 1
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_dim)
    
    def forward(self, x):
        x = x.permute(1, 0, 2) 
        x = self.transformer_encoder(x)
        x = self.decoder(x[-1])
        return x

def train_model(model, X_train, y_train, num_epochs, learning_rate):
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for t in range(num_epochs):
        optimizer.zero_grad()
        inputs = X_train.to(device)
        targets = y_train.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if t % 10 == 0:
            print(f"Epoch {t} train loss: {loss.item()}")
    model.eval()

    return model

def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data = data.dropna()
    return data

def create_sequences(data, seq_length, dates):
    X, y, seq_dates = [], [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, :])
        seq_dates.append(dates[i + seq_length - 1])
    return np.array(X), np.array(y), seq_dates

def prepare_data(ticker, start_date, end_date, seq_length):
    df = fetch_historical_data(ticker, start_date, end_date)
    dates = df.index.tolist()  # store dates before converting df to numpy array
    data = df[['Open', 'High', 'Low', 'Close']].values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    X, y, seq_dates = create_sequences(data, seq_length, dates)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y, scaler, seq_dates

def predict_and_print(model, X, dates, scaler):
    model.eval()
    predicted = model(X.to(device)).cpu().detach().numpy()
    predicted_prices = scaler.inverse_transform(predicted)
    for i in range(len(predicted_prices)):
        print(f"Date: {dates[i]}, Predicted: {predicted_prices[i]}")


ticker = 'SPY'
start_date = '1993-01-30'
end_date = '2023-05-25'
num_layers = 2
learning_rate = 0.01
num_epochs = 100

X, y, scaler, dates = prepare_data(ticker, start_date, end_date, seq_length=60)
model = TransformerModel(X.shape[2], num_layers, X.shape[2]).to(device)
model = train_model(model, X, y, num_epochs, learning_rate)
predict_and_print(model, X, dates, scaler)