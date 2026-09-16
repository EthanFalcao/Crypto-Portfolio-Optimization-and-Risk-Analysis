# Crypto-Portfolio-Optimization-and-Risk-Analysis

A small pipeline that pulls historical price data for the top cryptocurrencies, engineers
technical-indicator features, and trains an LSTM (Keras) to predict next-day closing price.

## Layout

`src/` is the production pipeline:
- `src/config.py` — env vars (Turso, CoinGecko) and pipeline constants
- `src/db.py` — Turso (hosted SQLite-compatible) read/write helpers
- `src/data_fetch.py` — top coins from CoinGecko, historical OHLCV from Yahoo Finance
- `src/features.py` — moving averages, RSI, MACD, lag/rolling features, computed per-coin
- `src/model.py` — the LSTM: build, train, evaluate on a single coin (BTC by default)
- `src/main.py` — runs the three stages above in order: `python -m src.main`

`1. Data Preprocessing.ipynb`, `3. Feature Engineering.ipynb`, and
`5. Predictive Modeling(ML).ipynb` were the original, notebook-based versions of the same
three stages. They're kept on disk (not deleted) but are superseded by `src/` — treat them
as a reference/history of how this was built, not something to run going forward.

`2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb` are still the live,
exploratory notebooks — correlation analysis, Random Forest feature importance, RFE, PCA —
used to inform the pipeline, not part of it.

## What's actually implemented

- Data collection, feature engineering, and a single-coin LSTM (see above), reading/writing
  a shared Turso database.

**Not implemented yet** (despite being a natural next step, don't assume they exist):
portfolio optimization (Kelly Criterion / Mean-Variance), risk management / stop-loss rules,
a backtesting framework, ARIMA/XGBoost models, or a real-time data pipeline.

## Setup

1. Clone this repository and create a virtual environment:
   ```bash
   git clone https://github.com/EthanFalcao/Crypto-Portfolio-Optimization-and-Risk-Analysis.git
   cd Crypto-Portfolio-Optimization-and-Risk-Analysis
   python -m venv .venv
   .venv/Scripts/activate   # or `source .venv/bin/activate` on macOS/Linux
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in:
   - `COINGECKO_API_KEY` — free key from [coingecko.com](https://www.coingecko.com/en/api).
   - `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` — create a free database at
     [turso.tech](https://turso.tech).
3. Run the pipeline: `python -m src.main` (fetches live data, engineers features, trains
   and evaluates the LSTM, all against your Turso database).
4. `2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb` can be run any time
   after step 3 has populated the database at least once.
