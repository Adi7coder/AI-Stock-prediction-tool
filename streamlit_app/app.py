import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Streamlit page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title
st.title("AI Stock Prediction Tool")

st.write("""
This app predicts the **next-day closing price** of selected stocks using an LSTM baseline model.
""")

# list of supported tickers
tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA"]

# user selects
selected_ticker = st.selectbox("Select a stock to predict:", tickers)

try:
    # load from local csv
    csv_path = f"./data/{selected_ticker}.csv"
    st.write(f"Loading data from `{csv_path}` ...")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if df.empty:
        st.error("The data is empty. Please check the CSV file.")
    else:
        st.subheader(f"{selected_ticker} - Raw Data Preview")
        st.dataframe(df, height=300)
        st.subheader(f"{selected_ticker} - Closing Price Plot")
         

        # draw a more compact chart
        col1, col2, col3 = st.columns([1, 2, 1])  # middle column is wider
        with col2:
            n_days = st.slider("Select number of days to plot", 30, 500, 100)
            st.line_chart(df['Close'].tail(n_days), use_container_width=True)


        if st.button("Run Prediction"):
            st.write(f"Running prediction for **{selected_ticker}**...")

            # force numeric close column
            close_prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            last_60_days = scaled_data[-60:]
            X_test = np.reshape(last_60_days, (1, 60, 1))

            model_path = f"./models/{selected_ticker}_lstm_model.h5"

            if os.path.exists(model_path):
                model = load_model(model_path)
                pred_price = model.predict(X_test)
                pred_price = scaler.inverse_transform(pred_price)
                st.success(f"Predicted next closing price for {selected_ticker}: **${pred_price[0][0]:.2f}**")
            else:
                st.warning(f"No trained LSTM model found for {selected_ticker}. Please train and save it as `{selected_ticker}_lstm_model.h5` in the models folder.")

except Exception as e:
    st.error(f"Error: {e}")
