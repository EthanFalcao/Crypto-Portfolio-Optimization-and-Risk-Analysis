"""Environment-based config: secrets and pipeline constants in one place."""
import os

from dotenv import load_dotenv

load_dotenv()


def _get_secret(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def coingecko_api_key():
    return _get_secret("COINGECKO_API_KEY")


def turso_database_url():
    return _get_secret("TURSO_DATABASE_URL")


def turso_auth_token():
    return _get_secret("TURSO_AUTH_TOKEN")


# --- Data collection ---
TOP_N_COINS = 20
START_DATE = "2011-9-1"
EXCLUDE_SYMBOLS = ["SHIB", "ICP", "STETH", "WBTC", "USDC", "UNI"]

# --- Feature engineering ---
LAG_PERIODS = [1, 3, 7]
ROLLING_WINDOWS = [7, 30]

# --- LSTM ---
# Features and target are RETURNS (how much each one moved vs. the previous
# close), not raw price levels - see src/model.py's module docstring for why.
COIN = "BTC"
LSTM_FEATURES = ["close_return", "open_return", "high_return", "low_return", "volume_return", "RSI", "MACD"]
LSTM_TARGET = "close_return"
LSTM_WINDOW = 60
TEST_FRACTION = 0.15

# --- Portfolio: modeling universe ---
# Always ~$1 - "predicting" it is meaningless, and a near-zero-variance asset
# breaks Mean-Variance's covariance math. Excluded from modeling/optimization
# only; still present in raw_prices/engineered_features.
STABLECOINS = ["USDT", "USDC", "USDS"]

# --- Backtest ---
BACKTEST_WINDOW_DAYS = 180
COV_LOOKBACK_DAYS = 30

# --- Optimizer (Mean-Variance) ---
RISK_AVERSION = 3.0
MAX_ASSET_WEIGHT = 0.35  # also the "max exposure" risk limit
TRANSACTION_COST_BPS = 10

# --- Risk management: trailing stop-loss ---
STOP_LOSS_DRAWDOWN = 0.15
STOP_LOSS_LOOKBACK_DAYS = 30
