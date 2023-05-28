import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fetch_historical_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].values


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), :])
        y.append(data[(i + seq_length), :])
    return np.array(X), np.array(y)


def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler


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


def train_model(model, X_train, y_train, num_epochs, learning_rate):
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    print('Finished training')
    return model


def predict_price_movement(model, X):
    model.eval()
    with torch.no_grad():
        inputs = torch.Tensor(X).to(device)
        prediction = model(inputs)
        return prediction.cpu().numpy()


def print_prediction(stock, prediction):
    labels = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"\nPredicted next day values for {stock}:")
    for label, pred in zip(labels, prediction):
        print(f"{label}: {pred:.2f}")


tickers = ['SPY']
start_date = "1993-01-30"
end_date = "2023-05-24"
seq_length = 7
input_dim = 5
hidden_dim = 32
num_layers = 2
output_dim = 5
num_epochs = 100
learning_rate = 0.001

for ticker in tickers:
    print(f'Processing {ticker}:')

    data = fetch_historical_data(ticker, start_date, end_date)
    data, scaler = normalize_data(data)
    X, y = create_sequences(data, seq_length)

    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    model = train_model(model, X, y, num_epochs, learning_rate)

    # Prepare the most recent sequence from the data for prediction
    X_latest = data[-seq_length:].reshape(1, seq_length, input_dim)
    X_latest = torch.from_numpy(X_latest).float().to(device)

    # Make a prediction
    prediction = predict_price_movement(model, X_latest)

    # Invert the normalization step
    prediction = scaler.inverse_transform(np.array(prediction).reshape(1, -1))

    # Print out the prediction
    print_prediction(ticker, prediction[0])

print('Done with all predictions.')
