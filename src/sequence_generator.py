import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_and_sequence(data, seq_length=60, scaler_save_path="models/scalers/"):
    os.makedirs(scaler_save_path, exist_ok=True)
    sequences = {}
    scalers = {}

    for stock in data.columns:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[[stock]])
        scalers[stock] = scaler
        joblib.dump(scaler, f"{scaler_save_path}{stock}_scaler.pkl")

        X, y = [], []
        for i in range(len(scaled) - seq_length):
            X.append(scaled[i:i+seq_length])
            y.append(scaled[i+seq_length])
        sequences[stock] = (np.array(X), np.array(y))

    return sequences
