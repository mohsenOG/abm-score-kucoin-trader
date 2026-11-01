from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from data_engine import DataEngine
from indicators import compute_bias_and_votes
from kucoin_rest import KuCoinREST
from signals import compute_signals


HISTORY_DIR = Path(__file__).resolve().parent / "history"
SIGNALS_DIR = Path(__file__).resolve().parent / "signals"
OHLC_COLUMNS = ["open", "high", "low", "close", "volume", "turnover"]
WARMUP_CANDLES = 200  # EMA200 seeds on candle 200; the first valid crossover is on candle 201.


def download_history(symbol: str, count: int) -> Path:
    rest = KuCoinREST()
    engine = DataEngine(rest)
    try:
        engine.sync_server_time()
        end_at = int((engine.get_latest_closed_candle_ts() + pd.Timedelta(hours=1)).timestamp()) - 1
        candles = []
        while len(candles) < count:
            # KuCoin's per-request limit is independent of the 200-candle warmup.
            batch_size = min(1500, count - len(candles))
            batch = rest.get_klines(
                start_at=end_at - batch_size * 3600 + 1,
                end_at=end_at,
                symbol=symbol,
            )
            if not batch:
                # ponytail: Empty ranges stop the download; use listing bounds to scan long trading gaps.
                raise ValueError(f"No candles returned for the next range; downloaded {len(candles)} of {count}.")
            candles.extend(batch)
            end_at = min(candle.ts for candle in batch) - 1
            print(f"Downloaded {len(candles)}/{count} candles.")
        history = engine._candles_to_df(candles)[OHLC_COLUMNS]
    finally:
        rest.session.close()

    first = history.index.min().strftime("%Y-%m-%d_%Hh")
    last = history.index.max().strftime("%Y-%m-%d_%Hh")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{symbol}_1h_{first}_to_{last}_{len(history)}candles.csv"
    history.to_csv(path, index_label="candle_open_utc")
    print(f"Original OHLC history saved: {path}")
    return path


def read_history(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path, index_col="candle_open_utc")
    history.index = pd.to_datetime(history.index, utc=True)
    return history[OHLC_COLUMNS].sort_index()


def write_signals(history: pd.DataFrame, history_path: Path) -> Path:
    if len(history) <= WARMUP_CANDLES:
        raise ValueError(f"History must contain more than {WARMUP_CANDLES} candles for warmup and testing.")

    # All reused indicators use trailing windows or past values, never future candles.
    frame = compute_signals(compute_bias_and_votes(history, use_heikin_ashi=True))
    frame["candle_number"] = range(1, len(frame) + 1)
    evaluated = frame.iloc[WARMUP_CANDLES:]
    signals = evaluated[evaluated["buy_signal"] | evaluated["sell_signal"]].copy()
    signals["side"] = signals["buy_signal"].map({True: "BUY", False: "SELL"})
    signals["symbol"] = history_path.name.split("_", 1)[0]
    signals["candle_close_utc"] = signals.index + pd.Timedelta(hours=1)
    signals = signals[
        ["symbol", "candle_number", "candle_close_utc", "side", "open", "high", "low", "close", "bias", "total_votes"]
    ]

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    run_time = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = SIGNALS_DIR / f"{history_path.stem}_signals_HA_{run_time}.csv"
    signals.to_csv(path, index_label="candle_open_utc", mode="x")
    print(f"History: {len(history)} candles; warmup: {WARMUP_CANDLES}; tested: {len(evaluated)}; signals: {len(signals)}.")
    print(f"Heikin-Ashi signals saved: {path}")
    return path


def main() -> int:
    histories = sorted(HISTORY_DIR.glob("*_1h_*.csv"))
    print("1. Download history")
    if histories:
        print("2. Use existing history")
    choice = input("Choose an option: ").strip()

    if choice == "1":
        symbol = input("Ticker (for example BTC-USDT): ").strip().upper().replace("/", "-")
        if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
            raise ValueError("Enter a ticker such as BTC-USDT or BTC/USDT.")
        count = int(input(f"Total candles to download (includes {WARMUP_CANDLES} warmup candles): "))
        if count <= WARMUP_CANDLES:
            raise ValueError(f"Download at least {WARMUP_CANDLES + 1} candles.")
        history_path = download_history(symbol, count)
    elif choice == "2" and histories:
        for number, path in enumerate(histories, 1):
            print(f"{number}. {path.name}")
        number = int(input("Choose a history file: "))
        if not 1 <= number <= len(histories):
            raise ValueError("Choose a history file from the list.")
        history_path = histories[number - 1]
    else:
        raise ValueError("Choose an available option.")

    history = read_history(history_path)
    write_signals(history, history_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n[shutdown] Stopped by user.")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
