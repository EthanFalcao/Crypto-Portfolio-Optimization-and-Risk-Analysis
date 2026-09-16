"""Fetch top coins (CoinGecko) and their historical OHLCV (Yahoo Finance), save to Turso."""
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from src import config
from src.db import save_df


def fetch_top_coins_tickers(vs_currency="usd", limit=30):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h",
        "x_cg_demo_api_key": config.coingecko_api_key(),
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching data: Status code {response.status_code}")
        return []

    data = response.json()
    return [coin["symbol"].upper() for coin in data]


def fetch_historical_prices(top_coins):
    symbols = [f"{coin}-USD" for coin in top_coins]
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = pd.DataFrame()
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        data = yf.download(symbol, start=config.START_DATE, end=end_date, auto_adjust=False)
        if not data.empty:
            # yfinance returns MultiIndex columns (Price, Ticker) even for a
            # single ticker - flatten to just the price field name.
            data.columns = data.columns.get_level_values(0)
            data["Symbol"] = symbol.replace("-USD", "")
            df = pd.concat([df, data], axis=0)

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Volume": "Volume($)"})
    df["Daily Return"] = df.groupby("Symbol")["Close"].pct_change()
    df = df[~df["Symbol"].isin(config.EXCLUDE_SYMBOLS)]
    return df


def run():
    top_coins = fetch_top_coins_tickers(limit=config.TOP_N_COINS)
    print(f"Top coins: {top_coins}")
    df = fetch_historical_prices(top_coins)
    save_df(df, "raw_prices")
    print(f"Saved {len(df)} rows to raw_prices ({df['Symbol'].nunique()} symbols)")
    return df


if __name__ == "__main__":
    run()
