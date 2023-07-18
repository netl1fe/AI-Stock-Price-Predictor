import streamlit as st
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from helpers import fetch_historical_data, add_selected_ta_features, reorder_data, normalize_data, create_sequences, LSTMModel, train_model, model_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
seq_length = st.sidebar.slider("Sequence Length", min_value=1, max_value=200, value=60)

# Fetching and processing the data
@st.cache_data
def get_processed_data(ticker, start_date, end_date):
    data = fetch_historical_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None
    data = add_selected_ta_features(data)
    data = reorder_data(data.dropna())
    # Split data into training and validation sets before normalizing
    train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=False)
    # Normalize data
    train_data, train_scaler = normalize_data(train_data.values)
    valid_data, valid_scaler = normalize_data(valid_data.values)
    return train_data, valid_data, train_scaler


train_data, valid_data, train_scaler = get_processed_data(ticker, start_date, end_date)

# Create sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_valid, y_valid = create_sequences(valid_data, seq_length)

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_valid = torch.from_numpy(X_valid).float().to(device)
y_valid = torch.from_numpy(y_valid).float().to(device)

input_dim = train_data.shape[1]

# Define the function we want to optimize
def optimize_model(hidden_dim, num_layers, learning_rate):
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    model = LSTMModel(input_dim, hidden_dim, num_layers, input_dim).to(device)
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, _ = train_model(model, X_train, y_train, 100, criterion, optimizer)
    valid_loss = model_loss(model, X_valid, y_valid, criterion)
    return -valid_loss  # We want to maximize the negative loss (i.e., minimize the loss)

# Define the bounds of the parameters we want to optimize
pbounds = {
    'hidden_dim': (1, 100),
    'num_layers': (1, 5),
    'learning_rate': (0.0001, 0.01),
}

optimizer = BayesianOptimization(
    f=optimize_model,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

# Run the optimization
optimizer.maximize(init_points=2, n_iter=3)

best_params = optimizer.max['params']
best_params['hidden_dim'] = int(best_params['hidden_dim'])
best_params['num_layers'] = int(best_params['num_layers'])

# Show the best parameters
st.sidebar.markdown("### Best Parameters")
st.sidebar.markdown(f"Hidden Dimension: {best_params['hidden_dim']}")
st.sidebar.markdown(f"Number of Layers: {best_params['num_layers']}")
st.sidebar.markdown(f"Learning Rate: {best_params['learning_rate']}")

# Train the model with the best parameters
best_model = LSTMModel(input_dim, best_params['hidden_dim'], best_params['num_layers'], input_dim).to(device)
criterion = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
best_model, _ = train_model(best_model, X_train, y_train, 100, criterion, optimizer)
torch.save(best_model.state_dict(), 'best_model_weights.pth')  # Save the best model

# Show the loss on the validation set
st.sidebar.markdown("### Validation Loss")
valid_loss = model_loss(best_model, X_valid, y_valid, criterion)
st.sidebar.markdown(f"{valid_loss:.4f}")
