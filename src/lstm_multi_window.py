"""Test the LSTM strategy across several historical windows, not just the
most recent one - the same question src/momentum_multi_window.py answers for
momentum, applied to the LSTM.

Important difference from momentum_multi_window.py: momentum needs no
training, so it can be freely re-evaluated on any stretch of history.
Retraining a fresh LSTM for every window here would mean 14 coins x 7
windows = 98 separate trainings, which is too slow to be practical. Instead,
each coin's LSTM is trained ONCE on its earliest available data, and its
predictions for everything after that are sliced into the same 7 windows.

That's a weaker, cheaper test than "retrain periodically and see how each
retrained model does" - the model answering window 7 (the most recent) was
never updated with anything from windows 2-6. Read the results as "how did
one early-trained model do across different later stretches of time," not
as a simulation of how this would run in production.
"""
import pandas as pd

from src import model
from src.backtest import compute_returns, run_strategy, performance_metrics
from src.multi_model import get_coins_to_model
from src.db import load_df

WINDOW_DAYS = 180


def find_shared_date_range(df, coins):
    close_prices = df.pivot(index="Date", columns="Symbol", values="Close")
    close_prices = close_prices[coins].dropna()
    return close_prices.index.min(), close_prices.index.max()


def build_full_prediction_history(coins, earliest_cutoff_date):
    """Train each coin's LSTM once, on data before earliest_cutoff_date, then
    predict day-by-day through the rest of its history."""
    prediction_frames = []
    for coin in coins:
        print(f"training {coin}...")
        coin_predictions = model.predict_for_backtest(coin, earliest_cutoff_date)
        if coin_predictions is not None:
            prediction_frames.append(coin_predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    actual_close = all_predictions.pivot(index="Date", columns="Symbol", values="actual_close")
    predicted_close = all_predictions.pivot(index="Date", columns="Symbol", values="predicted_close")

    # Only keep days every coin has a prediction for.
    shared_dates = actual_close.dropna().index
    actual_close = actual_close.loc[shared_dates].sort_index()
    predicted_close = predicted_close.loc[shared_dates].sort_index()
    return actual_close, predicted_close


def split_into_windows(actual_close, predicted_close, window_days):
    total_days = len(actual_close)
    num_windows = total_days // window_days
    leftover_days = total_days - (num_windows * window_days)

    windows = []
    for window_number in range(num_windows):
        window_start = leftover_days + window_number * window_days
        window_end = window_start + window_days
        windows.append((
            actual_close.iloc[window_start:window_end],
            predicted_close.iloc[window_start:window_end],
        ))
    return windows


def test_one_window(actual_close_window, predicted_close_window):
    actual_return, predicted_return = compute_returns(actual_close_window, predicted_close_window)

    lstm_daily_returns, _ = run_strategy(actual_close_window, actual_return, predicted_return, use_predictions=True)
    equal_weight_daily_returns, _ = run_strategy(actual_close_window, actual_return, predicted_return, use_predictions=False)

    lstm_metrics, _ = performance_metrics(lstm_daily_returns)
    equal_weight_metrics, _ = performance_metrics(equal_weight_daily_returns)
    return lstm_metrics, equal_weight_metrics


def run():
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    coins = get_coins_to_model(df)

    shared_start, _ = find_shared_date_range(df, coins)
    # Leave enough calendar days before the test period for the
    # shortest-history coin to have MIN_TRAIN_ROWS of training data.
    earliest_cutoff_date = shared_start + pd.Timedelta(days=model.MIN_TRAIN_ROWS + 10)
    print(f"Training each coin once, on data before {earliest_cutoff_date.date()}...")

    actual_close, predicted_close = build_full_prediction_history(coins, earliest_cutoff_date)
    print(f"Coins with enough history: {list(actual_close.columns)}")
    print(f"Testable days: {len(actual_close)} ({actual_close.index.min().date()} to {actual_close.index.max().date()})")

    windows = split_into_windows(actual_close, predicted_close, WINDOW_DAYS)
    print(f"Testing {len(windows)} non-overlapping {WINDOW_DAYS}-day windows\n")

    lstm_wins = 0
    lstm_total_returns = []
    equal_weight_total_returns = []

    for window_number, (actual_close_window, predicted_close_window) in enumerate(windows, start=1):
        lstm_metrics, equal_weight_metrics = test_one_window(actual_close_window, predicted_close_window)

        window_start_date = actual_close_window.index.min().date()
        window_end_date = actual_close_window.index.max().date()
        print(f"Window {window_number} ({window_start_date} to {window_end_date}):")
        print(f"  LSTM:         total_return = {lstm_metrics['total_return']:.4f}, "
              f"sharpe = {lstm_metrics['sharpe_ratio']:.4f}")
        print(f"  equal-weight: total_return = {equal_weight_metrics['total_return']:.4f}, "
              f"sharpe = {equal_weight_metrics['sharpe_ratio']:.4f}")

        if lstm_metrics["total_return"] > equal_weight_metrics["total_return"]:
            lstm_wins += 1

        lstm_total_returns.append(lstm_metrics["total_return"])
        equal_weight_total_returns.append(equal_weight_metrics["total_return"])

    average_lstm_return = sum(lstm_total_returns) / len(lstm_total_returns)
    average_equal_weight_return = sum(equal_weight_total_returns) / len(equal_weight_total_returns)

    print(f"\nLSTM beat equal-weight in {lstm_wins} of {len(windows)} windows")
    print(f"Average total return per window: LSTM = {average_lstm_return:.4f}, "
          f"equal-weight = {average_equal_weight_return:.4f}")

    return lstm_total_returns, equal_weight_total_returns


if __name__ == "__main__":
    run()
