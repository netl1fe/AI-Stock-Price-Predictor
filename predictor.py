import streamlit as st
import time
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from helpers import fetch_historical_data, add_selected_ta_features, reorder_data, LSTMModel, create_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to simulate typing effect
def typing_effect(text, speed=0.03):
    lines = text.strip().split("\n")
    for line in lines:
        placeholder = st.empty()
        for char in line:
            placeholder.text(char)
            time.sleep(speed)
        placeholder.text(line) 
        st.write("")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365 * 5))
end_date = st.sidebar.date_input("End Date", datetime.now() - timedelta(days=1))
seq_length = 60  # Sequence length is now fixed to 60

# Fetching and processing the data
data = fetch_historical_data(ticker, start_date, end_date)
if data is None:
    st.write("No data available.")
    exit()

data = add_selected_ta_features(data)
data = reorder_data(data.dropna())
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.values)

# Load the best model
best_model = LSTMModel(data.shape[1], 42, 1, data.shape[1]).to(device)
best_model.load_state_dict(torch.load('bmw.pth'))
best_model.eval()

# Get the last sequence of data
last_sequence = torch.from_numpy(normalized_data[-seq_length:-1].reshape(1, -1, data.shape[1])).float().to(device)  # Exclude today's data from the sequence

# Make a prediction
predicted_value = best_model(last_sequence)
predicted_value = predicted_value.cpu().detach().numpy()

# Rescale the predicted value
predicted_value = scaler.inverse_transform(predicted_value)

# Fetch the actual values for the day of prediction
actual_values = data.values[-1]  # Get today's actual values

# Display the predicted values with typing effect
st.markdown("### Predicted Values")
output_text = f"""
Open: {predicted_value[0][0]:.2f}
High: {predicted_value[0][1]:.2f}
Low: {predicted_value[0][2]:.2f}
Close: {predicted_value[0][3]:.2f}
"""
typing_effect(output_text, speed=0.03)

# Display the actual values for the day of prediction with typing effect
st.markdown("### Actual Values")
output_text = f"""
Open: {actual_values[0]:.2f}
High: {actual_values[1]:.2f}
Low: {actual_values[2]:.2f}
Close: {actual_values[3]:.2f}
"""
typing_effect(output_text, speed=0.03)
