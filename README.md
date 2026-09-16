# Crypto-Portfolio-Optimization-and-Risk-Analysis

A pipeline that pulls historical price data for the top cryptocurrencies, engineers
technical-indicator features, trains an LSTM (Keras) per coin to predict next-day closing
price, and runs those predictions through a Mean-Variance portfolio optimizer with a simple
stop-loss rule - backtested against a naive equal-weight baseline.

## Layout

`src/` is the production pipeline:
- `src/config.py` — env vars (Turso, CoinGecko) and pipeline constants
- `src/db.py` — Turso (hosted SQLite-compatible) read/write helpers
- `src/data_fetch.py` — top coins from CoinGecko, historical OHLCV from Yahoo Finance
- `src/features.py` — moving averages, RSI, MACD, lag/rolling features, computed per-coin
- `src/model.py` — the LSTM: build, train, evaluate on one coin (BTC by default)
- `src/multi_model.py` — runs `model.py` across every coin (except stablecoins) so their
  accuracy can be compared: `python -m src.multi_model`
- `src/optimizer.py` — Mean-Variance portfolio weights (long-only, capped position size)
- `src/backtest.py` — simulates the Mean-Variance strategy vs. an equal-weight baseline
  over a shared recent window, with a trailing stop-loss: `python -m src.backtest`
- `src/main.py` — runs fetch → features → single-coin LSTM in order: `python -m src.main`

`1. Data Preprocessing.ipynb`, `3. Feature Engineering.ipynb`, and
`5. Predictive Modeling(ML).ipynb` were the original, notebook-based versions of the same
three stages. They're kept on disk (not deleted) but are superseded by `src/` — treat them
as a reference/history of how this was built, not something to run going forward.

`2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb` are still the live,
exploratory notebooks — correlation analysis, Random Forest feature importance, RFE, PCA —
used to inform the pipeline, not part of it.

## What's actually implemented

- Data collection, feature engineering, and an LSTM trained per-coin (see above)
- A Mean-Variance optimizer that turns predicted returns into portfolio weights, with a
  per-asset weight cap (max exposure) and a trailing stop-loss (both risk-management rules)
- A backtest comparing that strategy to naive equal-weight diversification, including
  trading costs

**Honest result, not a cherry-picked one:** over the last 180 days (14 coins with enough
history - `python -m src.backtest`'s own output), the Mean-Variance strategy driven by the
LSTM's predictions **lost money** (-9.1% total return, Sharpe -0.18) while the equal-weight
baseline **made money** (+20.4%, Sharpe 1.11). This isn't a bug - checked that the optimizer's
weights are valid (sum to 1, respect the cap) - it's a real finding: the LSTM predicts
next-day *price* well (2-10% MAPE per coin), but the day-to-day *return* implied by those
predictions barely correlates with what actually happens (correlation near zero, negative
for several coins). A Mean-Variance optimizer takes that noisy signal at face value and
confidently concentrates the portfolio into whichever coin looks best that day, which
amplifies noise into losses rather than skill into gains. This is a well-documented failure
mode of plugging raw return forecasts into Markowitz without accounting for estimation error
(see: Michaud, "The Markowitz Optimization Enigma"). Fixing it - if there's something worth
fixing rather than concluding the signal isn't useful yet - would mean either predicting
returns directly instead of price levels, or shrinking/blending the predicted returns with a
market prior (e.g. Black-Litterman) before handing them to the optimizer.

**Not implemented yet:** ARIMA/XGBoost model comparisons, a real-time data pipeline.

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
4. Optional, once step 3 has populated the database:
   - `python -m src.multi_model` — per-coin model accuracy comparison
   - `python -m src.backtest` — Mean-Variance vs. equal-weight backtest (saves a chart to
     `backtest_equity_curve.png`)
   - `2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb`
