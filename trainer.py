import streamlit as st
import time
import json
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from helpers import fetch_historical_data, add_selected_ta_features, reorder_data, normalize_data, create_sequences, LSTMModel, train_model, model_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ticker = st.sidebar.text_input("Ticker", "SPY")
start_date = datetime(1993, 1, 30)
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
seq_length = st.sidebar.slider("Sequence Length", min_value=1, max_value=200, value=60)

@st.cache_data
def get_processed_data(ticker, start_date, end_date):
    data = fetch_historical_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None
    data = add_selected_ta_features(data)
    data = reorder_data(data.dropna())
    train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, train_scaler, original_train_data = normalize_data(train_data.values)
    valid_data, _, original_valid_data = normalize_data(valid_data.values)
    return train_data, valid_data, original_train_data, original_valid_data, train_scaler

train_data, valid_data, original_train_data, original_valid_data, train_scaler = get_processed_data(ticker, start_date, end_date)

X_train, y_train = create_sequences(train_data, seq_length)
X_valid, y_valid = create_sequences(valid_data, seq_length)

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_valid = torch.from_numpy(X_valid).float().to(device)
y_valid = torch.from_numpy(y_valid).float().to(device)

input_dim = train_data.shape[1]

best_valid_loss = float('inf')

def optimize_model(hidden_dim, num_layers, learning_rate):
    global best_valid_loss
    hidden_dim = int(hidden_dim)
    num_layers = int(num_layers)
    model = LSTMModel(input_dim, hidden_dim, num_layers, input_dim).to(device)
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, _ = train_model(model, X_train, y_train, 100, criterion, optimizer)
    valid_loss = model_loss(model, X_valid, y_valid, criterion)
    # Save the model if it's better than the current best
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
        # Save the best parameters
        best_params = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
        }
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)
    return -valid_loss

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

optimizer.maximize(init_points=200, n_iter=1000)

with open('best_params.json', 'r') as f:
    best_params = json.load(f)


best_params = optimizer.max['params']
best_params['hidden_dim'] = int(best_params['hidden_dim'])
best_params['num_layers'] = int(best_params['num_layers'])

# Show the best parameters with typing effect
st.markdown("### Best Parameters")

output_text = f"""
Hidden Dimension: {best_params['hidden_dim']}
Number of Layers: {best_params['num_layers']}
Learning Rate: {best_params['learning_rate']}
"""

typing_speed = 0.03


lines = output_text.strip().split("\n")


for line in lines:
    placeholder = st.empty()
    for char in line:
        placeholder.text(char)
        time.sleep(typing_speed)
    placeholder.text(line)
    st.write("")

best_model = LSTMModel(input_dim, best_params['hidden_dim'], best_params['num_layers'], input_dim).to(device)
criterion = torch.nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
best_model, _ = train_model(best_model, X_train, y_train, 100, criterion, optimizer)
torch.save(best_model.state_dict(), 'best_model_weights.pth')


st.markdown("### Validation Loss")
valid_loss = model_loss(best_model, X_valid, y_valid, criterion)

output_text = f"{valid_loss:.4f}"

lines = output_text.strip().split("\n")

for line in lines:
    placeholder = st.empty()
    for char in line:
        placeholder.text(char)
        time.sleep(typing_speed)
    placeholder.text(line)
    st.write("")

best_model = LSTMModel(input_dim, best_params['hidden_dim'], best_params['num_layers'], input_dim).to(device)
best_model.load_state_dict(torch.load('best_model_weights.pth'))
best_model.eval()

last_sequence = torch.from_numpy(valid_data[-seq_length:].reshape(1, -1, input_dim)).float().to(device)

predicted_value = best_model(last_sequence)
predicted_value = predicted_value.cpu().detach().numpy()

predicted_value = train_scaler.inverse_transform(predicted_value)

actual_values = original_valid_data[-1]

st.markdown("### Actual Values")

output_text = f"""
Open: {actual_values[0]:.2f}
High: {actual_values[1]:.2f}
Low: {actual_values[2]:.2f}
Close: {actual_values[3]:.2f}
"""

lines = output_text.strip().split("\n")

for line in lines:
    placeholder = st.empty()
    for char in line:
        placeholder.text(char)
        time.sleep(typing_speed)
    placeholder.text(line)
    st.write("")

st.markdown("### Predicted Next Values")

output_text = f"""
Open: {predicted_value[0][0]:.2f}
High: {predicted_value[0][1]:.2f}
Low: {predicted_value[0][2]:.2f}
Close: {predicted_value[0][3]:.2f}
"""

lines = output_text.strip().split("\n")

for line in lines:
    placeholder = st.empty()
    for char in line:
        placeholder.text(char)
        time.sleep(typing_speed)
    placeholder.text(line)
    st.write("")
