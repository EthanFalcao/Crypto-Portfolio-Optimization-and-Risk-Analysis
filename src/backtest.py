"""Backtest: simulate two portfolio strategies over the same recent stretch
of days and coins, so we can honestly compare them:

1. "Mean-Variance strategy" - every day, use the LSTM's predicted returns to
   pick new portfolio weights (see src/optimizer.py).
2. "Equal-weight baseline" - every day, just split the portfolio evenly across
   the same coins. No predictions involved.

We report both. The point isn't to make strategy 1 look good - it's to see
honestly whether prediction + optimization actually beats simple, naive
diversification, once trading costs are included.
"""
import pandas as pd
import matplotlib.pyplot as plt

from src import config, model
from src.db import load_df, save_df
from src.multi_model import get_coins_to_model
from src.optimizer import mean_variance_weights

# Need at least this many days of recent returns before we trust a
# covariance estimate. With only 1-2 days, variance isn't even defined.
MIN_COV_OBSERVATIONS = 10


def build_aligned_predictions(cutoff_date):
    """Train + predict every coin over the same shared test window, then line
    them all up by date so every coin has a value on every kept day."""
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    coins = get_coins_to_model(df)

    prediction_frames = []
    for coin in coins:
        print(f"predicting {coin}...")
        coin_predictions = model.predict_for_backtest(coin, cutoff_date)
        if coin_predictions is not None:
            prediction_frames.append(coin_predictions)

    if len(prediction_frames) < 2:
        raise RuntimeError("Not enough coins had history for this backtest window.")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)

    # Turn the long format (one row per coin per day) into a wide format
    # (one column per coin), so every coin lines up by date.
    actual_close = all_predictions.pivot(index="Date", columns="Symbol", values="actual_close")
    predicted_close = all_predictions.pivot(index="Date", columns="Symbol", values="predicted_close")
    actual_close = actual_close.sort_index()
    predicted_close = predicted_close.sort_index()

    # Only keep days where every coin has a value (some coins may have a
    # slightly shorter test period than others).
    days_with_all_coins = actual_close.dropna().index
    actual_close = actual_close.loc[days_with_all_coins]
    predicted_close = predicted_close.loc[days_with_all_coins]

    print(f"Backtest universe: {list(actual_close.columns)}")
    print(f"Shared test days: {len(days_with_all_coins)}")
    return actual_close, predicted_close


def compute_returns(actual_close, predicted_close):
    actual_return = actual_close.pct_change()

    # The model predicts tomorrow's price. To turn that into a predicted
    # RETURN, compare it to today's actual price (the last price we really
    # know) - not tomorrow's actual price, which we don't know yet.
    yesterdays_actual_close = actual_close.shift(1)
    predicted_return = (predicted_close - yesterdays_actual_close) / yesterdays_actual_close

    return actual_return, predicted_return


def find_stopped_out_coins(price_history, today_index):
    """Coins down more than STOP_LOSS_DRAWDOWN from their recent peak, using
    only prices from before today (no lookahead)."""
    lookback_start = today_index - config.STOP_LOSS_LOOKBACK_DAYS
    if lookback_start < 0:
        lookback_start = 0
    recent_prices = price_history.iloc[lookback_start:today_index]

    if len(recent_prices) == 0:
        return []

    stopped_out = []
    for coin in recent_prices.columns:
        peak_price = recent_prices[coin].max()
        latest_price = recent_prices[coin].iloc[-1]
        drop_from_peak = (peak_price - latest_price) / peak_price
        if drop_from_peak > config.STOP_LOSS_DRAWDOWN:
            stopped_out.append(coin)
    return stopped_out


def run_strategy(actual_close, actual_return, predicted_return, use_predictions):
    """Simulate rebalancing once a day. If use_predictions is False, this is
    the equal-weight baseline (still subject to the same stop-loss + costs).

    Returns (daily_portfolio_returns, daily_weights) - daily_weights is a
    DataFrame (one row per day, one column per coin) so a dashboard can show
    what the portfolio actually held on any given day, not just the P&L.
    """
    all_coins = list(actual_return.columns)
    all_dates = actual_return.index

    previous_weights = pd.Series(0.0, index=all_coins)
    daily_portfolio_returns = []
    daily_weights = []

    for day_index, today in enumerate(all_dates):
        if day_index == 0:
            # No history yet to base a decision on - sit out day 1.
            daily_portfolio_returns.append(0.0)
            daily_weights.append(previous_weights.copy())
            continue

        stopped_out = find_stopped_out_coins(actual_close, day_index)
        coins_available_today = []
        for coin in all_coins:
            if coin not in stopped_out:
                coins_available_today.append(coin)
        if len(coins_available_today) < 3:
            # Stop-loss would leave too few coins to satisfy the max-weight
            # rule (e.g. 2 coins can't add up to 1 if each is capped at 0.35).
            coins_available_today = all_coins

        days_of_cov_history = min(day_index, config.COV_LOOKBACK_DAYS)
        have_enough_history_for_cov = days_of_cov_history >= MIN_COV_OBSERVATIONS

        if use_predictions and have_enough_history_for_cov:
            todays_predicted_returns = predicted_return.loc[today, coins_available_today]

            # Drop any coin whose prediction is missing for some reason.
            coins_with_predictions = []
            for coin in coins_available_today:
                if pd.notna(todays_predicted_returns[coin]):
                    coins_with_predictions.append(coin)
            todays_predicted_returns = todays_predicted_returns[coins_with_predictions]

            cov_window_start = day_index - days_of_cov_history
            recent_returns = actual_return[coins_with_predictions].iloc[cov_window_start:day_index]
            covariance_matrix = recent_returns.cov()

            todays_weights = mean_variance_weights(todays_predicted_returns, covariance_matrix)
        else:
            # Either this is the baseline strategy, or it's too early in the
            # backtest to trust a covariance estimate yet - split evenly.
            equal_share = 1 / len(coins_available_today)
            todays_weights = pd.Series(equal_share, index=coins_available_today)

        # Put today's weights into a full-length Series (0 for any coin we
        # didn't invest in today) so it lines up with previous_weights.
        full_weights = pd.Series(0.0, index=all_coins)
        for coin in todays_weights.index:
            full_weights[coin] = todays_weights[coin]

        # Turnover = how much we had to buy/sell to get from yesterday's
        # weights to today's. Trading isn't free, so this costs us a bit.
        weight_changes = (full_weights - previous_weights).abs()
        turnover = weight_changes.sum() / 2
        trading_cost = turnover * (config.TRANSACTION_COST_BPS / 10_000)

        todays_market_returns = actual_return.loc[today]
        portfolio_return_today = (full_weights * todays_market_returns).sum() - trading_cost

        daily_portfolio_returns.append(portfolio_return_today)
        daily_weights.append(full_weights.copy())
        previous_weights = full_weights

    returns_series = pd.Series(daily_portfolio_returns, index=all_dates)
    weights_df = pd.DataFrame(daily_weights, index=all_dates)
    return returns_series, weights_df


def performance_metrics(daily_returns):
    portfolio_value = (1 + daily_returns).cumprod()
    total_return = portfolio_value.iloc[-1] - 1

    num_days = len(daily_returns)
    num_years = num_days / 365
    annualized_return = (1 + total_return) ** (1 / num_years) - 1

    # Sharpe ratio: average daily return divided by how much it bounces
    # around, scaled up to a yearly number. Higher = better risk-adjusted return.
    if daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (365 ** 0.5)
    else:
        sharpe_ratio = float("nan")

    running_peak = portfolio_value.cummax()
    drawdown = (portfolio_value - running_peak) / running_peak
    max_drawdown = drawdown.min()

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }
    return metrics, portfolio_value


def run():
    """Run all three strategies (LSTM-driven, momentum-driven, equal-weight)
    over the shared backtest window, print an honest comparison, and save
    everything a dashboard would need (summary metrics, day-by-day portfolio
    value, and day-by-day weights - what the portfolio held, and how that
    changed over time) to Turso."""
    df = load_df("engineered_features")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    most_recent_date = df["Date"].max()
    cutoff_date = most_recent_date - pd.Timedelta(days=config.BACKTEST_WINDOW_DAYS)

    actual_close, predicted_close = build_aligned_predictions(cutoff_date)
    actual_return, lstm_predicted_return = compute_returns(actual_close, predicted_close)
    momentum_predicted_return = actual_return.shift(1)

    print("\nRunning the LSTM-driven Mean-Variance strategy...")
    lstm_returns, lstm_weights = run_strategy(actual_close, actual_return, lstm_predicted_return, use_predictions=True)

    print("Running the momentum-driven Mean-Variance strategy...")
    momentum_returns, momentum_weights = run_strategy(actual_close, actual_return, momentum_predicted_return, use_predictions=True)

    print("Running the equal-weight baseline...")
    equal_weight_returns, equal_weight_weights = run_strategy(actual_close, actual_return, lstm_predicted_return, use_predictions=False)

    strategies = {
        "lstm": (lstm_returns, lstm_weights),
        "momentum": (momentum_returns, momentum_weights),
        "equal_weight": (equal_weight_returns, equal_weight_weights),
    }

    print(f"\nBacktest window: {cutoff_date.date()} to {most_recent_date.date()} ({len(actual_return)} days)")
    print(f"Coins used: {list(actual_return.columns)}")
    print()

    results_rows = []
    equity_curve_rows = []
    daily_weight_rows = []
    portfolio_values_by_strategy = {}
    run_time = pd.Timestamp.now()

    for strategy_name, (daily_returns, daily_weights) in strategies.items():
        metrics, portfolio_value = performance_metrics(daily_returns)
        print(f"{strategy_name}: total_return = {metrics['total_return']:.4f}, "
              f"sharpe_ratio = {metrics['sharpe_ratio']:.4f}")

        portfolio_values_by_strategy[strategy_name] = portfolio_value
        results_rows.append({"strategy": strategy_name, "n_days": len(daily_returns), "run_at": run_time, **metrics})

        for date, value in portfolio_value.items():
            equity_curve_rows.append({"strategy": strategy_name, "date": date, "portfolio_value": value})

        # Every day's weights, not just the latest - so a dashboard can show
        # how the portfolio's composition changed over time, not just a
        # single snapshot.
        for date, weights_that_day in daily_weights.iterrows():
            for coin, weight in weights_that_day.items():
                daily_weight_rows.append({"strategy": strategy_name, "date": date, "coin": coin, "weight": weight})

    save_df(pd.DataFrame(results_rows), "backtest_results")
    save_df(pd.DataFrame(equity_curve_rows), "backtest_equity_curve")
    save_df(pd.DataFrame(daily_weight_rows), "backtest_daily_weights")
    print("\nSaved backtest_results, backtest_equity_curve, and backtest_daily_weights to Turso")

    plt.figure(figsize=(10, 6))
    for strategy_name, portfolio_value in portfolio_values_by_strategy.items():
        plt.plot(portfolio_value.index, portfolio_value, label=strategy_name)
    plt.title("Portfolio value over the backtest window (starting at 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png")
    print("Saved chart to backtest_equity_curve.png")

    return results_rows


if __name__ == "__main__":
    run()
