import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("🇮🇳 AI Stock Prediction Tool")
st.write("""
This app predicts the **next-day closing price** of selected Indian stocks using an LSTM model optionally enhanced with news sentiment.
""")

tickers = ["RELIANCE", "TCS", "INFY", "SBIN", "HDFCBANK", "ICICIBANK", "HINDUNILVR"]
selected_ticker = st.selectbox("Select a stock to predict:", tickers)

try:
    csv_path = f"./data/{selected_ticker}.csv"
    st.write(f"Loading data from `{csv_path}` ...")

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    if df.empty:
        st.error("The data is empty. Please check the CSV file.")
    else:
        st.subheader(f"{selected_ticker} - Raw Data Preview")
        st.dataframe(df, height=300)
        st.subheader(f"{selected_ticker} - Closing Price Plot")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            n_days = st.slider("Select number of days to plot", 30, 500, 100)
            st.line_chart(df.set_index("Date")["Close"].tail(n_days), use_container_width=True)

        if st.button("Run Prediction"):
            st.write(f"Running prediction for **{selected_ticker}**...")

            close_prices = df['Close'].values.reshape(-1, 1)
            features = close_prices
            feature_count = 1

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

            if feature_count == 2:
                model_path = f"./models/{selected_ticker}_lstm_model_sentiment.h5"
            else:
                model_path = f"./models/{selected_ticker}_lstm_model.h5"

            if os.path.exists(model_path):
                model = load_model(model_path)
                pred_scaled = model.predict(X_test)

                extended = np.concatenate(
                    [pred_scaled, np.zeros((pred_scaled.shape[0], feature_count - 1))],
                    axis=1
                )
                pred_price = scaler.inverse_transform(extended)

                st.success(f"Predicted next closing price for {selected_ticker}: **₹{pred_price[0][0]:.2f}**")
            else:
                st.warning(f"No trained LSTM model found for {selected_ticker}. Please train and save it as `{selected_ticker}_lstm_model.h5` or `{selected_ticker}_lstm_model_sentiment.h5` in the models folder.")

except Exception as e:
    st.error(f"Error: {e}")
