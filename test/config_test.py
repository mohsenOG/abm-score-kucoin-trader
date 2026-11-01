"""Settings for the hourly history, signal, and SL/TP test workflow."""

from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
HISTORY_DIR = TEST_DIR / "history"
SIGNALS_DIR = TEST_DIR / "signals"
RESULTS_DIR = TEST_DIR / "results"
OHLC_COLUMNS = ["open", "high", "low", "close", "volume", "turnover"]
TIMEFRAME_SECONDS = 3600
DOWNLOAD_BATCH_SIZE = 1500
USE_HEIKIN_ASHI = True
# EMA200 seeds on candle 200; the first fully warmed crossover is on candle 201.
WARMUP_CANDLES = 200

# Percentage units: 0.1 means 0.1%, not 10%.
SL_MIN_PCT = 0.1
SL_MAX_PCT = 5.0
SL_STEP_PCT = 0.1
TP_MIN_PCT = 0.1
TP_MAX_PCT = 5.0
TP_STEP_PCT = 0.1
FEE_PER_SIDE_PCT = 0.06
RETURN_DECIMALS = 12  # Remove floating-point noise at exactly break-even returns.
TRAIN_FRACTION = 0.70
TOP_N = 10
