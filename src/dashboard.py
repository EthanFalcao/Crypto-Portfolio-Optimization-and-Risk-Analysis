"""Streamlit dashboard: shows how the portfolio is doing and what it's
currently holding, for each strategy this project has built.

Reads results that src/backtest.py already computed and saved to Turso - it
does NOT retrain the LSTM live (that takes 10+ minutes for all 14 coins).
Run `python -m src.backtest` first to refresh the numbers, then:

    streamlit run src/dashboard.py
"""
import os
import sys

# "streamlit run src/dashboard.py" runs this file directly, which does NOT
# put the project's root folder on Python's import path - so `from src.db
# import ...` below would fail without this. Adding the parent-of-src folder
# (the repo root) fixes it, no matter which folder you run the command from.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st

# On Streamlit Community Cloud, TURSO_DATABASE_URL/TURSO_AUTH_TOKEN are set in
# the app's "Secrets" box, which Streamlit exposes as st.secrets. Copy them
# into the environment so src/config.py's plain os.environ.get() calls (used
# both here and by every other script in this project) can find them too.
# Locally, where there's no secrets.toml file, this just does nothing.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass

from src.db import load_df

STRATEGY_DISPLAY_NAMES = {
    "lstm": "LSTM (Mean-Variance)",
    "momentum": "Momentum (Mean-Variance)",
    "equal_weight": "Equal-weight (baseline)",
}


@st.cache_data(ttl=300)
def load_backtest_data():
    results = load_df("backtest_results")
    equity_curve = load_df("backtest_equity_curve")
    latest_weights = load_df("backtest_latest_weights")

    equity_curve["date"] = pd.to_datetime(equity_curve["date"])
    latest_weights["as_of_date"] = pd.to_datetime(latest_weights["as_of_date"])
    return results, equity_curve, latest_weights


def show_metrics_row(strategy_results):
    row = strategy_results.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total return", f"{row['total_return']:.1%}")
    col2.metric("Annualized return", f"{row['annualized_return']:.1%}")
    col3.metric("Sharpe ratio", f"{row['sharpe_ratio']:.2f}")
    col4.metric("Max drawdown", f"{row['max_drawdown']:.1%}")


def show_equity_curve(equity_curve, selected_strategy):
    st.subheader("Portfolio value over time")
    chart_data = equity_curve.pivot(index="date", columns="strategy", values="portfolio_value")
    chart_data = chart_data.rename(columns=STRATEGY_DISPLAY_NAMES)
    st.line_chart(chart_data)
    st.caption(f"Highlighted strategy: {STRATEGY_DISPLAY_NAMES[selected_strategy]}. "
               "All three are shown together on purpose - the point of this project "
               "was to check whether the fancier strategies actually beat the simple one.")


def show_current_weights(latest_weights, selected_strategy):
    st.subheader("Current portfolio allocation")
    strategy_weights = latest_weights[latest_weights["strategy"] == selected_strategy]
    strategy_weights = strategy_weights[strategy_weights["weight"] > 0.001]
    strategy_weights = strategy_weights.sort_values("weight", ascending=False)

    as_of_date = strategy_weights["as_of_date"].iloc[0].date()
    st.caption(f"As of {as_of_date}")

    chart_data = strategy_weights.set_index("coin")["weight"]
    st.bar_chart(chart_data)


def show_honest_caveats():
    with st.expander("Read this before trusting any of the above"):
        st.markdown("""
- **The LSTM strategy has lost money in every backtest run so far.** Its predicted
  returns have close to zero (sometimes negative) correlation with what actually
  happens - see `src/baseline_comparison.py`.
- **The momentum strategy only beat equal-weight in 3 of 7 independent 6-month
  windows** when tested across ~3.8 years of history (`src/momentum_multi_window.py`).
  The strong number in this single backtest window is one of momentum's best two
  results, not a typical one.
- **Neither strategy has a demonstrated, durable edge over naive equal-weight
  diversification.** That's the honest state of this project - see the README.
        """)


def run():
    st.set_page_config(page_title="Crypto Portfolio Dashboard", layout="wide")
    st.title("Crypto Portfolio Dashboard")

    results, equity_curve, latest_weights = load_backtest_data()

    available_strategies = list(results["strategy"].unique())
    selected_strategy = st.sidebar.radio(
        "Strategy",
        options=available_strategies,
        format_func=lambda name: STRATEGY_DISPLAY_NAMES.get(name, name),
    )

    st.header(STRATEGY_DISPLAY_NAMES.get(selected_strategy, selected_strategy))

    strategy_results = results[results["strategy"] == selected_strategy]
    show_metrics_row(strategy_results)
    show_equity_curve(equity_curve, selected_strategy)
    show_current_weights(latest_weights, selected_strategy)
    show_honest_caveats()


run()
