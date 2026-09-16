"""LSTM (Keras): predict next-day closing price for a single coin.

Time-based train/test split (no shuffling) and scaling fit on the training
period only - see the module docstring in the notebook this was lifted from
(5. Predictive Modeling(ML).ipynb) for why both of those matter here.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.db import load_df

MIN_TRAIN_ROWS = config.LSTM_WINDOW + 60


def make_sequences(arr, start, end, window, target_idx):
    xs, ys = [], []
    for i in range(start, end):
        xs.append(arr[i - window:i])
        ys.append(arr[i, target_idx])
    return np.array(xs), np.array(ys)


def build_model(window, n_features):
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _load_coin(coin):
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df[df["Symbol"] == coin].sort_values("Date").reset_index(drop=True)


def _train_predict(coin_df, features, window, split):
    """Shared core: scale on train-only data, build sequences, fit, predict.

    `split` is the row index where the test period begins - everything at or
    after it is held out, everything before it (down to `window` rows earlier,
    so the first test prediction has a full lookback) is used for training.
    Returns (actual_close, predicted_close, test_dates, final_train_loss).
    """
    target_idx = features.index("Close")
    data = coin_df[features].values

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(data[:split])
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[:split, [target_idx]])
    scaled = feature_scaler.transform(data)

    X_train, y_train = make_sequences(scaled, window, split, window, target_idx)
    X_test, y_test = make_sequences(scaled, split, len(scaled), window, target_idx)

    model = build_model(window, len(features))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    final_loss = history.history["loss"][-1]
    if not np.isfinite(final_loss):
        raise RuntimeError(f"Training loss is not finite ({final_loss}) - check scaling/inputs.")

    pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    predicted = close_scaler.inverse_transform(pred_scaled).flatten()
    actual = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_dates = coin_df["Date"].values[split:]
    return actual, predicted, test_dates, final_loss


def train_and_evaluate(coin=config.COIN, features=config.LSTM_FEATURES, window=config.LSTM_WINDOW):
    coin_df = _load_coin(coin)
    n_test = int(len(coin_df) * config.TEST_FRACTION)
    split = len(coin_df) - n_test

    actual, predicted, _, _ = _train_predict(coin_df, features, window, split)

    metrics = {
        "coin": coin,
        "n_train": split,
        "n_test": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }
    return metrics, (actual, predicted)


def predict_for_backtest(coin, cutoff_date, features=config.LSTM_FEATURES, window=config.LSTM_WINDOW):
    """Train on everything before `cutoff_date`, predict day-by-day through
    the rest of the coin's history. Used to build a shared, date-aligned set
    of predictions across coins for the portfolio backtest.

    Returns a DataFrame with columns [Date, Symbol, actual_close,
    predicted_close], or None if there isn't enough pre-cutoff history.
    """
    coin_df = _load_coin(coin)
    split = int((coin_df["Date"] < cutoff_date).sum())
    if split < MIN_TRAIN_ROWS or split >= len(coin_df):
        print(f"  skipping {coin}: only {split} rows before {cutoff_date.date()} (need >= {MIN_TRAIN_ROWS})")
        return None

    actual, predicted, test_dates, _ = _train_predict(coin_df, features, window, split)
    return pd.DataFrame({
        "Date": test_dates,
        "Symbol": coin,
        "actual_close": actual,
        "predicted_close": predicted,
    })


def run():
    metrics, _ = train_and_evaluate()
    print(f"{metrics['coin']} test set ({metrics['n_test']} days):")
    print(f"  MAE:  {metrics['mae']:,.2f}")
    print(f"  RMSE: {metrics['rmse']:,.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    return metrics


if __name__ == "__main__":
    run()
