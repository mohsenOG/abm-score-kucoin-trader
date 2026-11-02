from __future__ import annotations

import time
import traceback

import pandas as pd

import config
from data_engine import DataEngine
from indicators import compute_bias_and_votes
from kucoin_rest import KuCoinREST
from signals import compute_signals


def _print_signal(ts: pd.Timestamp, row: pd.Series) -> None:
    side = "BUY" if bool(row["buy_signal"]) else "SELL"
    print(
        f"[signal] symbol={config.SYMBOL} timeframe={config.KLINE_TYPE} ts={ts} side={side} "
        f"close={float(row['close']):.4f} bias={float(row['bias']):.5f} votes={int(row['total_votes'])}/21",
        flush=True,
    )


def _print_warmup_signals(df: pd.DataFrame) -> None:
    signals = df[df["buy_signal"] | df["sell_signal"]].tail(10)
    print("=== last historical signals (max 10) ===")
    if signals.empty:
        print("(none)")
    for ts, row in signals.iterrows():
        _print_signal(ts, row)


def main() -> int:
    rest = KuCoinREST()
    engine = DataEngine(rest)
    try:
        engine.sync_server_time()
        engine.warmup()
        df = compute_signals(compute_bias_and_votes(engine.df))
        _print_warmup_signals(df)
        engine.mark_processed(df.index.max())
    except Exception:
        print("[fatal] startup failed")
        traceback.print_exc()
        return 1

    while True:
        try:
            engine.maybe_resync_time()
            result = engine.fetch_and_update()
            if result.new_candle_ts:
                df = compute_signals(compute_bias_and_votes(result.df))
                for ts in sorted(result.new_candle_ts):
                    row = df.loc[ts]
                    if bool(row["buy_signal"]) or bool(row["sell_signal"]):
                        _print_signal(ts, row)
                    engine.mark_processed(ts)
            time.sleep(config.POLL_INTERVAL_SECONDS)
        except Exception:
            print("[loop] unhandled exception")
            traceback.print_exc()
            time.sleep(config.HTTP_RETRY_PERIOD_SECONDS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[shutdown] Stopped by user.", flush=True)
