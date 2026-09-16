"""Feature engineering: moving averages, RSI, MACD, lag/rolling features - all
computed per-Symbol, so a coin's features never pull in another coin's rows."""
import numpy as np

from src import config
from src.db import load_df, save_df


def calculate_rsi(x, periods=14):
    delta = x.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def engineer_features(df):
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    g = df.groupby("Symbol")

    df["SMA_30"] = g["Close"].transform(lambda x: x.rolling(30).mean())
    df["SMA_60"] = g["Close"].transform(lambda x: x.rolling(60).mean())
    df["SMA_90"] = g["Close"].transform(lambda x: x.rolling(90).mean())

    df["EMA_30"] = g["Close"].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df["EMA_60"] = g["Close"].transform(lambda x: x.ewm(span=60, adjust=False).mean())
    df["EMA_90"] = g["Close"].transform(lambda x: x.ewm(span=90, adjust=False).mean())

    df["RSI"] = g["Close"].transform(calculate_rsi)

    df["EMA_12"] = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df["EMA_26"] = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = g["MACD"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    df["Midpoint"] = (df["High"] + df["Low"]) / 2
    df["Log_Return"] = g["Midpoint"].transform(lambda x: np.log(x / x.shift(1)))

    df["30_day_MA"] = g["Close"].transform(lambda x: x.rolling(30).mean())
    df["60_day_MA"] = g["Close"].transform(lambda x: x.rolling(60).mean())
    df["90_day_MA"] = g["Close"].transform(lambda x: x.rolling(90).mean())

    for lag in config.LAG_PERIODS:
        df[f"Close_lag_{lag}"] = g["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = g["Volume($)"].shift(lag)

    for window in config.ROLLING_WINDOWS:
        df[f"Close_rolling_mean_{window}"] = g["Close"].transform(lambda x: x.rolling(window).mean())
        df[f"Close_rolling_std_{window}"] = g["Close"].transform(lambda x: x.rolling(window).std())
        df[f"Volume_rolling_mean_{window}"] = g["Volume($)"].transform(lambda x: x.rolling(window).mean())
        df[f"Volume_rolling_std_{window}"] = g["Volume($)"].transform(lambda x: x.rolling(window).std())

    df = df.dropna()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    return df


def run():
    raw = load_df("raw_prices")
    raw["Date"] = raw["Date"].astype("datetime64[ns]")
    df = engineer_features(raw)
    save_df(df, "engineered_features")
    print(f"Saved {len(df)} rows to engineered_features ({df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    run()
