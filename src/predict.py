import numpy as np
from keras.models import load_model
import joblib

def predict_next(stock, latest_data, seq_length=60, model_path="models/", scaler_path="models/scalers/"):
    model = load_model(f"{model_path}{stock}_lstm_model.h5")
    scaler = joblib.load(f"{scaler_path}{stock}_scaler.pkl")

    latest_scaled = scaler.transform(latest_data[-seq_length:])
    latest_scaled = np.reshape(latest_scaled, (1, seq_length, 1))

    pred_scaled = model.predict(latest_scaled)
    pred = scaler.inverse_transform(pred_scaled)
    return pred[0][0]
