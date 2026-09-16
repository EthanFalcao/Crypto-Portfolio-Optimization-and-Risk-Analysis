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
- `src/baseline_comparison.py` — sanity check: is the LSTM's predicted-return signal
  actually better than trivial guesses (no change, or "today repeats yesterday")? See
  below - `python -m src.baseline_comparison`
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
history), the Mean-Variance strategy driven by the LSTM's predictions **lost money**
(total return roughly -0.5% to -9% across two separate runs - see below on why it isn't
exactly reproducible) while a naive equal-weight baseline **made money** (+20.4%, Sharpe
1.11). This isn't a bug - checked that the optimizer's weights are valid (sum to 1, respect
the cap) - it's a real finding: the LSTM predicts next-day *price* well (2-10% MAPE per
coin), but the day-to-day *return* implied by those predictions doesn't correlate with what
actually happens (overall correlation **-0.06**, i.e. slightly worse than a coin flip).

**`src/baseline_comparison.py` digs into this further**, by feeding three different signals
into the exact same optimizer/costs/stop-loss: the LSTM, always guessing "no change" (0%),
and a naive momentum guess ("today repeats yesterday"). The result was clarifying:

| signal | correlation with actual return | total return |
|---|---|---|
| LSTM | -0.067 | -0.5% |
| predict no change | undefined (constant) | +8.6% |
| **momentum (yesterday's return)** | **+0.083** | **+49.2%** |
| equal-weight (no signal at all) | n/a | +20.4% |

The LSTM isn't just unhelpful, it's the *worst* signal of the four - worse than guessing
zero. Simple momentum, which took an afternoon to add as a sanity check, beat everything
else by a wide margin. **The likely actual source of edge in this window was momentum, not
the LSTM** - a good reminder to sanity-check a model against trivial baselines before trusting
it. Caveat: this is one 180-day window: momentum's strength here could be specific to this
stretch rather than a persistent effect, and the LSTM's exact number varies run to run since
its training is stochastic (different random weight init each time) - the qualitative result
(LSTM signal is uninformative to slightly harmful) held across both runs, though.

**Not implemented yet:** testing the above across multiple time windows (walk-forward, to
see if momentum's edge holds up or was specific to this stretch), predicting returns
directly instead of price levels, ARIMA/XGBoost model comparisons, a real-time data pipeline.

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
   - `python -m src.baseline_comparison` — checks the LSTM's signal against trivial baselines
   - `2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb`
