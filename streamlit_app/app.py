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

# supported tickers
tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA"]

selected_ticker = st.selectbox("Select a stock to predict:", tickers)

try:
    csv_path = f"./data/{selected_ticker}.csv"
    st.write(f"Loading data from `{csv_path}` ...")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if df.empty:
        st.error("The data is empty. Please check the CSV file.")
    else:
        st.subheader(f"{selected_ticker} - Raw Data Preview")
        st.dataframe(df, height=300)
        st.subheader(f"{selected_ticker} - Closing Price Plot")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            n_days = st.slider("Select number of days to plot", 30, 500, 100)
            st.line_chart(df['Close'].tail(n_days), use_container_width=True)

        if st.button("Run Prediction"):
            st.write(f"Running prediction for **{selected_ticker}**...")

            # by default, use close prices only
            close_prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values.reshape(-1, 1)
            features = close_prices
            feature_count = 1

            # if sentiment-enhanced CSV exists, switch to it
            sentiment_csv_path = f"./data/{selected_ticker}_merged.csv"
            if os.path.exists(sentiment_csv_path):
                st.info("Sentiment data found — using sentiment-enhanced model.")
                df_sent = pd.read_csv(sentiment_csv_path, parse_dates=["Date"])
                df_sent["sentiment_score"] = df_sent["sentiment_score"].fillna(0)
                features = df_sent[["Close", "sentiment_score"]].values
                feature_count = 2

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(features)

            last_60_days = scaled_data[-60:]
            X_test = np.reshape(last_60_days, (1, 60, feature_count))

            # choose correct model
            if feature_count == 2:
                model_path = f"./models/{selected_ticker}_lstm_model_sentiment.h5"
            else:
                model_path = f"./models/{selected_ticker}_lstm_model.h5"

            if os.path.exists(model_path):
                model = load_model(model_path)
                pred_scaled = model.predict(X_test)

                # fill missing feature dims with zeros for inverse transform
                extended = np.concatenate(
                    [pred_scaled, np.zeros((pred_scaled.shape[0], feature_count - 1))],
                    axis=1
                )
                pred_price = scaler.inverse_transform(extended)

                st.success(f"Predicted next closing price for {selected_ticker}: **${pred_price[0][0]:.2f}**")
            else:
                st.warning(f"No trained LSTM model found for {selected_ticker}. Please train and save it as `{selected_ticker}_lstm_model.h5` or `{selected_ticker}_lstm_model_sentiment.h5` in the models folder.")

except Exception as e:
    st.error(f"Error: {e}")
