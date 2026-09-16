"""Test the momentum strategy (src/baseline_comparison.py's best signal) over
several separate historical windows, not just the most recent one - answers
whether momentum's edge was a persistent effect or specific to one lucky
stretch of time.

No LSTM training needed here - momentum only uses yesterday's actual return,
so this whole script is quick to run.
"""
from src.backtest import run_strategy, performance_metrics
from src.multi_model import get_coins_to_model
from src.db import load_df

WINDOW_DAYS = 180


def load_price_history():
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    coins = get_coins_to_model(df)

    close_prices = df.pivot(index="Date", columns="Symbol", values="Close")
    close_prices = close_prices[coins]
    close_prices = close_prices.dropna()  # only keep days every coin has a price
    return close_prices.sort_index()


def split_into_windows(close_prices, window_days):
    """Break the price history into consecutive, non-overlapping windows.
    Any leftover days at the very start (too few to fill a full window) are
    dropped, so every window is the same length."""
    total_days = len(close_prices)
    num_windows = total_days // window_days
    leftover_days = total_days - (num_windows * window_days)

    windows = []
    for window_number in range(num_windows):
        window_start = leftover_days + window_number * window_days
        window_end = window_start + window_days
        window = close_prices.iloc[window_start:window_end]
        windows.append(window)

    return windows


def test_one_window(window_prices):
    actual_return = window_prices.pct_change()
    momentum_predicted_return = actual_return.shift(1)

    momentum_daily_returns, _ = run_strategy(window_prices, actual_return, momentum_predicted_return, use_predictions=True)
    equal_weight_daily_returns, _ = run_strategy(window_prices, actual_return, momentum_predicted_return, use_predictions=False)

    momentum_metrics, _ = performance_metrics(momentum_daily_returns)
    equal_weight_metrics, _ = performance_metrics(equal_weight_daily_returns)
    return momentum_metrics, equal_weight_metrics


def run():
    close_prices = load_price_history()
    coins = list(close_prices.columns)
    print(f"Coins: {coins}")
    print(f"Full shared history: {close_prices.index.min().date()} to {close_prices.index.max().date()} "
          f"({len(close_prices)} days)")

    windows = split_into_windows(close_prices, WINDOW_DAYS)
    print(f"Testing {len(windows)} non-overlapping {WINDOW_DAYS}-day windows\n")

    momentum_wins = 0
    momentum_total_returns = []
    equal_weight_total_returns = []

    for window_number, window_prices in enumerate(windows, start=1):
        momentum_metrics, equal_weight_metrics = test_one_window(window_prices)

        window_start_date = window_prices.index.min().date()
        window_end_date = window_prices.index.max().date()
        print(f"Window {window_number} ({window_start_date} to {window_end_date}):")
        print(f"  momentum:     total_return = {momentum_metrics['total_return']:.4f}, "
              f"sharpe = {momentum_metrics['sharpe_ratio']:.4f}")
        print(f"  equal-weight: total_return = {equal_weight_metrics['total_return']:.4f}, "
              f"sharpe = {equal_weight_metrics['sharpe_ratio']:.4f}")

        if momentum_metrics["total_return"] > equal_weight_metrics["total_return"]:
            momentum_wins += 1

        momentum_total_returns.append(momentum_metrics["total_return"])
        equal_weight_total_returns.append(equal_weight_metrics["total_return"])

    average_momentum_return = sum(momentum_total_returns) / len(momentum_total_returns)
    average_equal_weight_return = sum(equal_weight_total_returns) / len(equal_weight_total_returns)

    print(f"\nMomentum beat equal-weight in {momentum_wins} of {len(windows)} windows")
    print(f"Average total return per window: momentum = {average_momentum_return:.4f}, "
          f"equal-weight = {average_equal_weight_return:.4f}")

    return momentum_total_returns, equal_weight_total_returns


if __name__ == "__main__":
    run()
