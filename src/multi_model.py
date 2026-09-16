"""Run the single-coin LSTM (src/model.py) on every coin, not just BTC, so we
can compare how well it predicts each one. Standalone script - not part of
src/main.py, since training ~15 LSTMs one after another is slow and shouldn't
silently become part of every pipeline run.
"""
import pandas as pd

from src import config, model
from src.db import load_df, save_df


def get_coins_to_model(df):
    """Every coin in the data except stablecoins (see config.STABLECOINS for why)."""
    coins = []
    for coin in df["Symbol"].unique():
        if coin not in config.STABLECOINS:
            coins.append(coin)
    return sorted(coins)


def has_enough_history(df, coin):
    coin_rows = df[df["Symbol"] == coin]
    num_train_rows = len(coin_rows) - int(len(coin_rows) * config.TEST_FRACTION)
    return num_train_rows >= model.MIN_TRAIN_ROWS


def run():
    df = load_df("engineered_features")
    coins = get_coins_to_model(df)

    results = []
    for coin in coins:
        if not has_enough_history(df, coin):
            print(f"skipping {coin}: not enough history to train on")
            continue

        print(f"training {coin}...")
        try:
            metrics, _ = model.train_and_evaluate(coin=coin)
        except RuntimeError as error:
            print(f"  {coin} failed: {error}")
            continue

        results.append(metrics)

    if len(results) == 0:
        print("No coin had enough history to train on.")
        return results

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("correlation", ascending=False)
    save_df(results_df, "model_metrics")

    print("\nPer-coin results, best (highest correlation between predicted and actual return) first:")
    print(results_df.to_string(index=False))
    return results


if __name__ == "__main__":
    run()
