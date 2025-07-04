import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start_date, end_date, save_path="data/raw/"):
    os.makedirs(save_path, exist_ok=True)
    data = yf.download(tickers, start=start_date, end=end_date)
    close_prices = data['Close']
    close_prices = close_prices.ffill()


    for stock in close_prices.columns:
        close_prices[[stock]].to_csv(f"{save_path}{stock}_raw.csv")

    return close_prices
