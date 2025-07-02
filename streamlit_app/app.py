import streamlit as st
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

#our page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title(" AI Stock Prediction Tool")

st.write("""
This app predicts the next-day closing price of AAPL using an LSTM baseline model.
""")

# the data from yfinance will be loaded here
ticker = 'AAPL'
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")[['Close']]

st.subheader("Raw Data")
st.line_chart(df['Close'])

#our prediction button named 'predict'
if st.button("Run Prediction"):
    st.write("Predicting...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])

    last_60_days = scaled_data[-60:]
    X_test = np.reshape(last_60_days, (1, 60, 1))

    model = load_model('./models/lstm_model.h5')

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    st.success(f"Predicted next closing price: ${pred_price[0][0]:.2f}")
