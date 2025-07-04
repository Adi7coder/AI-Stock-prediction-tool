from keras.models import Sequential
from keras.layers import LSTM, Dense
import os

def train_and_save_models(sequences, model_save_path="models/"):
    os.makedirs(model_save_path, exist_ok=True)
    for stock, (X, y) in sequences.items():
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        model.save(f"{model_save_path}{stock}_lstm_model.h5")
