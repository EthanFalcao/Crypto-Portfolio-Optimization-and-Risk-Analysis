# Crypto-Portfolio-Optimization-and-Risk-Analysis

A pipeline that pulls historical price data for the top cryptocurrencies, engineers
technical-indicator features, trains an LSTM (Keras) per coin to predict next-day *return*,
and runs those predictions through a Mean-Variance portfolio optimizer with a simple
stop-loss rule - backtested against a naive equal-weight baseline, with a Streamlit
dashboard to browse the results.

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
- `src/dashboard.py` — Streamlit dashboard: equity curve, current portfolio allocation, and
  how that allocation changed over time, for each strategy, click-through in the sidebar:
  `streamlit run src/dashboard.py`. Reads whatever `src/backtest.py` last saved - it does
  not retrain the LSTM live (that takes 10+ minutes), so run `python -m src.backtest` first
  to refresh the numbers it shows.
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

**The LSTM originally predicted next-day *price*, which looked accurate (2-10% MAPE per
coin) but wasn't a real signal** - a price series barely moves day to day, so a model can
get a low price error just by roughly repeating today's value, without predicting anything.
It's been rebuilt (`src/model.py`) to predict next-day *return* directly instead, which is
harder to fake and is what the portfolio optimizer actually needs.

**Per-coin correlation is still close to zero even with the corrected model**
(`python -m src.multi_model`, all 14 coins): best is WBT at **+0.06**, several are negative
(DOGE -0.02, ETH -0.009, ADA -0.003), and directional accuracy sits at 45-55% across the
board - a coin flip. The LSTM does not have demonstrated real predictive skill for any of
these coins, honestly measured this time (not via the misleading price metric above).

**The backtest result is unstable in a way that itself confirms the above.** Three separate
runs of the exact same corrected model, on the exact same data, gave the LSTM strategy
**+14.0% (Sharpe 0.78)**, **+50.8% (Sharpe 1.92)**, and **+66.0% (Sharpe 1.91)** - nothing
changed between runs except the random weight initialization Keras starts training from.
When a model's backtest P&L swings that widely between runs with literally the same code
and data, that's a sign the result is dominated by which random pattern the optimizer
happened to concentrate into, not by learned skill - consistent with the ~0 correlation
number above. A real edge would be expected to reproduce far more consistently run to run.
(For reference, momentum and equal-weight - which don't involve any random training - are
exactly reproducible: every run gives the same number.)

**`src/baseline_comparison.py`**, re-run with the corrected model, confirms the same
picture from a different angle - it compares the LSTM against two trivial signals ("predict
no change" and momentum) on the same optimizer/costs/stop-loss:

| signal | correlation with actual return | MAE | total return |
|---|---|---|---|
| LSTM | +0.006 | 0.0211 | +50.8% (see instability note above) |
| predict no change | undefined (constant) | **0.0207** | +8.6% |
| **momentum (yesterday's return)** | **+0.083** | 0.0294 | +49.2% |
| equal-weight (no signal at all) | n/a | n/a | +20.4% |

The LSTM's correlation is closer to zero than useful in either direction now (+0.006, up
from -0.067 with the old price-based model, but still not meaningfully different from no
signal at all) - and its error (MAE) is actually slightly *worse* than just guessing "no
change." Its backtest return looks competitive with momentum here, but per the instability
note above, that number isn't trustworthy on its own - a different training run gave +14.0%
using the identical setup. Momentum remains the one signal with a real (if modest) positive
correlation to what actually happens, though see the multi-window test below for why even
that shouldn't be taken as a proven edge.

**`src/momentum_multi_window.py`** doesn't involve the LSTM at all, so its result still
stands: it re-tested the momentum signal (which looked strong in a single recent backtest)
across 7 separate, non-overlapping 180-day windows - the full ~3.8 years of shared history
this 14-coin universe has - instead of just the most recent one:

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

**Not implemented yet:** re-testing the LSTM itself across multiple historical windows the
way momentum was (to see whether its instability is a property of this one window or
inherent to training it at all), ARIMA/XGBoost model comparisons, a real-time data pipeline.

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

## Deploying the dashboard

The dashboard only reads from Turso - it doesn't need to run the pipeline itself - so it can
be hosted for free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repo to GitHub (already done if you're reading this there).
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click
   "New app". Point it at this repo, the branch you want to deploy, and set the main file
   path to `src/dashboard.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   TURSO_DATABASE_URL = "libsql://..."
   TURSO_AUTH_TOKEN = "..."
   ```
   (`COINGECKO_API_KEY` isn't needed for the dashboard - it only reads from Turso.)
4. Deploy. Refresh the numbers it shows by running `python -m src.backtest` locally
   (or on a schedule - see below) whenever you want the dashboard to reflect new data; it
   doesn't retrain anything itself.

**Keeping it current:** since the dashboard only shows what `src/backtest.py` last saved,
consider a scheduled GitHub Action that runs `python -m src.main` and `python -m
src.backtest` daily or weekly (the sibling Baleen project referenced in this project's
history does exactly this) - not set up yet, see the roadmap below.
