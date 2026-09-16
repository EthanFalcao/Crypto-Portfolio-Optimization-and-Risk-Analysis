"""LSTM (Keras): predict a coin's next-day RETURN (not its price).

Earlier version of this predicted next-day price, which looked accurate
(2-10% MAPE) but turned out to be a useless signal - see
src/baseline_comparison.py. The problem: prices barely move day to day, so a
model can get a low price error just by roughly repeating today's price. That
doesn't require it to have learned anything about which direction things are
headed. Predicting the RETURN directly forces the model to actually commit to
a direction and size, which is what the portfolio optimizer needs anyway.

Time-based train/test split (no shuffling) and scaling fit on the training
period only - both matter because this is a time series (see the module
docstring in the notebook this was lifted from,
5. Predictive Modeling(ML).ipynb, for more on why).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from src import config
from src.db import load_df

MIN_TRAIN_ROWS = config.LSTM_WINDOW + 60


def add_return_features(coin_df):
    """Turn raw price/volume columns into RETURNS - how much each one moved
    compared to the previous day's close, instead of its raw price level.

    Why: a raw price series barely changes day to day (today's BTC price is
    almost always close to yesterday's), so a model fed raw prices can get
    away with just repeating the last value - it looks accurate but isn't
    predicting anything. Returns are small, close to zero, and don't have
    that shortcut available.
    """
    coin_df = coin_df.copy()
    yesterdays_close = coin_df["Close"].shift(1)

    coin_df["close_return"] = np.log(coin_df["Close"] / yesterdays_close)
    coin_df["open_return"] = np.log(coin_df["Open"] / yesterdays_close)
    coin_df["high_return"] = np.log(coin_df["High"] / yesterdays_close)
    coin_df["low_return"] = np.log(coin_df["Low"] / yesterdays_close)

    # +1 on both sides so a day with 0 volume doesn't take log(0).
    yesterdays_volume = coin_df["Volume($)"].shift(1)
    coin_df["volume_return"] = np.log((coin_df["Volume($)"] + 1) / (yesterdays_volume + 1))

    coin_df = coin_df.dropna(subset=["close_return"])  # first row has no "yesterday"
    coin_df = coin_df.reset_index(drop=True)
    return coin_df


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
    coin_df = df[df["Symbol"] == coin].sort_values("Date").reset_index(drop=True)
    coin_df = add_return_features(coin_df)
    return coin_df


def _train_predict(coin_df, features, target, window, split):
    """Shared core: scale on train-only data, build sequences, fit, predict.

    `split` is the row index where the test period begins - everything at or
    after it is held out, everything before it (down to `window` rows earlier,
    so the first test prediction has a full lookback) is used for training.
    Returns (actual_return, predicted_return, test_dates, final_train_loss).
    """
    target_idx = features.index(target)
    data = coin_df[features].values

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(data[:split])
    target_scaler = MinMaxScaler()
    target_scaler.fit(data[:split, [target_idx]])
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
    predicted_return = target_scaler.inverse_transform(pred_scaled).flatten()
    actual_return = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_dates = coin_df["Date"].values[split:]
    return actual_return, predicted_return, test_dates, final_loss


def train_and_evaluate(coin=config.COIN, features=config.LSTM_FEATURES, window=config.LSTM_WINDOW):
    coin_df = _load_coin(coin)
    n_test = int(len(coin_df) * config.TEST_FRACTION)
    split = len(coin_df) - n_test

    actual_return, predicted_return, _, _ = _train_predict(coin_df, features, config.LSTM_TARGET, window, split)

    # Correlation is the metric that actually matters for a return signal -
    # low error alone doesn't tell you if the direction is right. See
    # src/baseline_comparison.py for the full explanation of why.
    correlation = np.corrcoef(actual_return, predicted_return)[0, 1]
    guessed_direction_correctly = np.sign(actual_return) == np.sign(predicted_return)

    metrics = {
        "coin": coin,
        "n_train": split,
        "n_test": len(actual_return),
        "correlation": correlation,
        "directional_accuracy": guessed_direction_correctly.mean(),
        "mae": np.mean(np.abs(actual_return - predicted_return)),
    }
    return metrics, (actual_return, predicted_return)


def predict_for_backtest(coin, cutoff_date, features=config.LSTM_FEATURES, window=config.LSTM_WINDOW):
    """Train on everything before `cutoff_date`, predict day-by-day through
    the rest of the coin's history. Used to build a shared, date-aligned set
    of predictions across coins for the portfolio backtest.

    Returns a DataFrame with columns [Date, Symbol, actual_close,
    predicted_close], or None if there isn't enough pre-cutoff history. Prices
    (not raw returns) are handed back so every other script in this project
    (backtest.py, multi_model.py, ...) can keep working unchanged - the
    predicted return gets turned back into a price using the last known
    actual close, the same way you'd turn "up 3%" back into a dollar amount.
    """
    coin_df = _load_coin(coin)
    split = int((coin_df["Date"] < cutoff_date).sum())
    if split < MIN_TRAIN_ROWS or split >= len(coin_df):
        print(f"  skipping {coin}: only {split} rows before {cutoff_date.date()} (need >= {MIN_TRAIN_ROWS})")
        return None

    actual_return, predicted_return, test_dates, _ = _train_predict(coin_df, features, config.LSTM_TARGET, window, split)

    # coin_df["Close"] at position (split - 1) is the last known price before
    # the test period starts; position (split - 1 + k) is "yesterday's price"
    # for the k-th test day - exactly what's needed to turn a return back
    # into a price level.
    yesterdays_close = coin_df["Close"].values[split - 1: len(coin_df) - 1]
    actual_close = yesterdays_close * np.exp(actual_return)
    predicted_close = yesterdays_close * np.exp(predicted_return)

    return pd.DataFrame({
        "Date": test_dates,
        "Symbol": coin,
        "actual_close": actual_close,
        "predicted_close": predicted_close,
    })


def run():
    metrics, _ = train_and_evaluate()
    print(f"{metrics['coin']} test set ({metrics['n_test']} days):")
    print(f"  correlation (predicted vs. actual return): {metrics['correlation']:.4f}")
    print(f"  directional accuracy: {metrics['directional_accuracy']:.1%}")
    print(f"  MAE (return): {metrics['mae']:.4f}")
    return metrics


if __name__ == "__main__":
    run()
