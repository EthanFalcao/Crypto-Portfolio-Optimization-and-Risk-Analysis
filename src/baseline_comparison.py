"""Sanity check for the LSTM's predicted-return signal: is it actually better
than doing something trivial?

Compares three ways of guessing tomorrow's return for each coin:
1. The LSTM's prediction (src/model.py)
2. "Predict no change" - always guess 0% return
3. "Predict momentum" - guess today repeats yesterday's actual return

Each signal is fed into the same Mean-Variance optimizer, same trading costs,
same stop-loss rule as src/backtest.py, so the four strategies below are
compared on equal footing. If the LSTM doesn't beat these trivial guesses,
that's a sign the problem is the signal, not the optimizer or the backtest.
"""
import pandas as pd

from src import config
from src.backtest import build_aligned_predictions, compute_returns, run_strategy, performance_metrics
from src.db import load_df


def print_signal_accuracy(signal_name, predicted_return, actual_return):
    """How well does this signal line up with what actually happened?
    Correlation near 0 means the signal carries no real information."""
    all_predicted_values = []
    all_actual_values = []

    for coin in actual_return.columns:
        # Put predicted and actual in one small DataFrame first, then drop
        # NaN rows together - otherwise a signal with an extra leading NaN
        # (like the momentum one below, which is shifted by a day) ends up
        # with a different number of values than actual_return, and the two
        # lists we're building can't be lined up anymore.
        coin_data = pd.DataFrame({
            "predicted": predicted_return[coin],
            "actual": actual_return[coin],
        }).dropna()

        correlation = coin_data["predicted"].corr(coin_data["actual"])
        print(f"    {coin}: correlation = {correlation:.4f}")

        all_predicted_values.extend(coin_data["predicted"].tolist())
        all_actual_values.extend(coin_data["actual"].tolist())

    combined = pd.DataFrame({"predicted": all_predicted_values, "actual": all_actual_values})
    overall_correlation = combined["predicted"].corr(combined["actual"])
    mean_absolute_error = (combined["predicted"] - combined["actual"]).abs().mean()
    print(f"  {signal_name}: overall correlation = {overall_correlation:.4f}, "
          f"mean absolute error = {mean_absolute_error:.4f}")


def run():
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    most_recent_date = df["Date"].max()
    cutoff_date = most_recent_date - pd.Timedelta(days=config.BACKTEST_WINDOW_DAYS)

    actual_close, predicted_close = build_aligned_predictions(cutoff_date)
    actual_return, lstm_predicted_return = compute_returns(actual_close, predicted_close)

    # Signal 1: always guess "no change" (0% return).
    zero_predicted_return = actual_return * 0

    # Signal 2: guess today repeats yesterday's actual return (momentum).
    momentum_predicted_return = actual_return.shift(1)

    signals = {
        "LSTM": lstm_predicted_return,
        "zero (predict no change)": zero_predicted_return,
        "momentum (predict yesterday's return)": momentum_predicted_return,
    }

    print("Step 1: how well does each signal correlate with what actually happened?\n")
    for signal_name, predicted_return in signals.items():
        print(f"{signal_name}:")
        print_signal_accuracy(signal_name, predicted_return, actual_return)
        print()

    print("Step 2: run each signal through the same Mean-Variance optimizer, same costs, same stop-loss.\n")
    results = {}
    for signal_name, predicted_return in signals.items():
        strategy_returns = run_strategy(actual_close, actual_return, predicted_return, use_predictions=True)
        metrics, _ = performance_metrics(strategy_returns)
        results[signal_name] = metrics

    baseline_returns = run_strategy(actual_close, actual_return, lstm_predicted_return, use_predictions=False)
    results["equal-weight (no predictions at all)"] = performance_metrics(baseline_returns)[0]

    for signal_name, metrics in results.items():
        print(f"  {signal_name}: total_return = {metrics['total_return']:.4f}, "
              f"sharpe_ratio = {metrics['sharpe_ratio']:.4f}")

    return results


if __name__ == "__main__":
    run()
