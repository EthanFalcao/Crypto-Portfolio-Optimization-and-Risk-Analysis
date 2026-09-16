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
- `src/momentum_multi_window.py` — re-tests the momentum signal across several historical
  windows instead of just the most recent one: `python -m src.momentum_multi_window`
- `src/dashboard.py` — Streamlit dashboard: current allocation + equity curve for each
  strategy, click-through in the sidebar: `streamlit run src/dashboard.py`. Reads whatever
  `src/backtest.py` last saved - it does not retrain the LSTM live (that takes 10+ minutes),
  so run `python -m src.backtest` first to refresh the numbers it shows.
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
zero. Momentum, which took an afternoon to add as a sanity check, beat everything else by a
wide margin **in that one window**. The LSTM's exact number also varies run to run since its
training is stochastic (different random weight init each time) - the qualitative result
(LSTM signal is uninformative to slightly harmful) held across repeated runs, though.

**But that single window overclaimed momentum's edge - `src/momentum_multi_window.py`
re-tested momentum vs. equal-weight across 7 separate, non-overlapping 180-day windows
(the full ~3.8 years of shared history this 14-coin universe has) instead of just the most
recent one:**

| window | momentum | equal-weight | winner |
|---|---|---|---|
| 2023-04 to 2023-09 | -28.8% | -5.8% | equal-weight |
| 2023-10 to 2024-03 | +109.0% | +86.4% | momentum |
| 2024-03 to 2024-09 | -19.5% | -13.3% | equal-weight |
| 2024-09 to 2025-03 | +45.1% | +66.6% | equal-weight |
| 2025-03 to 2025-09 | +16.9% | +34.6% | equal-weight |
| 2025-09 to 2026-03 | +51.3% | -14.5% | momentum |
| 2026-03 to 2026-09 | +50.5% | +22.8% | momentum |

**Momentum only beat equal-weight in 3 of 7 windows.** The single-window result above was
one of momentum's best two windows, not a representative one - a clean example of why
"picked the most recent backtest" is a form of cherry-picking even when nothing was
deliberately cherry-picked. Averaged across all 7 windows momentum's mean return is somewhat
higher (32.1% vs. 25.3%), but it's also swingier in both directions (its best window +109%,
its worst -28.8%, vs. equal-weight's +86.4%/-13.3%) - consistent with "more aggressive,
higher variance," not "reliably better." **Conclusion: neither the LSTM nor simple momentum
has demonstrated a durable edge over naive diversification in this universe** - the honest
state of this project right now is that nothing built so far reliably beats equal-weight.

**Not implemented yet:** predicting returns directly instead of price levels, ARIMA/XGBoost
model comparisons, a real-time data pipeline.

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
   - `python -m src.momentum_multi_window` — re-tests momentum across multiple time windows
   - `streamlit run src/dashboard.py` — dashboard (run `python -m src.backtest` first so it
     has something current to show)
   - `2. Exploratory Data Analysis.ipynb` and `4. Feature Selection.ipynb`
