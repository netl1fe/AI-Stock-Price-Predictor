import streamlit as st
import time
import torch
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from helpers import fetch_historical_data, add_selected_ta_features, reorder_data, LSTMModel, create_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def typing_effect(text, speed=0.03):
    lines = text.strip().split("\n")
    for line in lines:
        placeholder = st.empty()
        for char in line:
            placeholder.text(char)
            time.sleep(speed)
        placeholder.text(line) 
        st.write("")

def fetch_single_day_data(ticker, date):
    data = yf.download(ticker, start=date, end=date + timedelta(days=1))
    return data

def calculate_accuracy(actual, predicted):
    return abs(actual - predicted)


ticker = st.sidebar.text_input("Ticker", "SPY")
start_date = datetime(1993, 1, 30)
end_date = st.sidebar.date_input("End Date", datetime.now())  
seq_length = 60 

data = fetch_historical_data(ticker, start_date, end_date)
if data is None:
    st.write("No data available.")
    exit()

data = add_selected_ta_features(data)
data = reorder_data(data.dropna())
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.values)

best_model = LSTMModel(data.shape[1], 94, 1, data.shape[1]).to(device)
best_model.load_state_dict(torch.load('bmw2.pth'))
best_model.eval()

last_sequence = torch.from_numpy(normalized_data[-seq_length-1:-1].reshape(1, -1, data.shape[1])).float().to(device)  # Exclude the last day's data from the sequence

predicted_value = best_model(last_sequence)
predicted_value = predicted_value.cpu().detach().numpy()

predicted_value = scaler.inverse_transform(predicted_value)

end_date_data = fetch_single_day_data(ticker, end_date)

if end_date_data.empty:
    st.write(f"No data available for {end_date.strftime('%Y-%m-%d')}.")
else:
    end_date_values = end_date_data.values[0]

accuracy_open = calculate_accuracy(end_date_values[0], predicted_value[0][0])
accuracy_high = calculate_accuracy(end_date_values[1], predicted_value[0][1])
accuracy_low = calculate_accuracy(end_date_values[2], predicted_value[0][2])
accuracy_close = calculate_accuracy(end_date_values[3], predicted_value[0][3])

st.markdown("### Predicted Values")
output_text = f"""
Open: {predicted_value[0][0]:.2f}
High: {predicted_value[0][1]:.2f}
Low: {predicted_value[0][2]:.2f}
Close: {predicted_value[0][3]:.2f}
"""
typing_effect(output_text, speed=0.03)

st.markdown("### Actual Values")
output_text = f"""
Open: {end_date_values[0]:.2f}
High: {end_date_values[1]:.2f}
Low: {end_date_values[2]:.2f}
Close: {end_date_values[3]:.2f}
"""
typing_effect(output_text, speed=0.03)

st.sidebar.markdown("### Accuracy Scores")
st.sidebar.write(f"Open: ${accuracy_open:.2f}")
st.sidebar.write(f"High: ${accuracy_high:.2f}")
st.sidebar.write(f"Low: ${accuracy_low:.2f}")
st.sidebar.write(f"Close: ${accuracy_close:.2f}")
