from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config
from kucoin_rest import Candle, KuCoinREST


@dataclass
class FetchResult:
    df: pd.DataFrame
    new_candle_ts: List[pd.Timestamp]


class DataEngine:
    def __init__(self, rest: KuCoinREST) -> None:
        self.rest = rest
        self.server_offset_ms: int = 0
        self._last_server_sync_ms: int = 0
        self.df: pd.DataFrame = pd.DataFrame()
        self.last_processed_ts: Optional[pd.Timestamp] = None

    def _timeframe_delta(self) -> pd.Timedelta:
        return pd.Timedelta(seconds=config.TIMEFRAME_SECONDS)

    def _timeframe_freq(self) -> str:
        return f"{config.TIMEFRAME_SECONDS}s"

    def sync_server_time(self) -> None:
        server_ms = self.rest.get_server_time_ms()
        local_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        self.server_offset_ms = server_ms - local_ms
        self._last_server_sync_ms = local_ms
        print(f"[time] offset_ms={self.server_offset_ms}")

    def server_now_utc(self) -> pd.Timestamp:
        local_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        return pd.to_datetime(local_ms + self.server_offset_ms, unit="ms", utc=True)

    def maybe_resync_time(self) -> None:
        local_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        if local_ms - self._last_server_sync_ms >= config.SERVER_TIME_RESYNC_SECONDS * 1000:
            self.sync_server_time()

    def _candles_to_df(self, candles: List[Candle]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"]).astype(float)
        rows = [
            {
                "ts": pd.to_datetime(candle.ts, unit="s", utc=True),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "turnover": candle.turnover,
            }
            for candle in candles
        ]
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.loc[:self.get_latest_closed_candle_ts()]

    def _merge_df(self, new_df: pd.DataFrame) -> None:
        if new_df.empty:
            return
        if self.df.empty:
            self.df = new_df.copy()
            return
        combined = pd.concat([self.df, new_df], axis=0)
        self.df = combined[~combined.index.duplicated(keep="last")].sort_index()

    def _detect_missing_candles(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[pd.Timestamp]:
        if self.df.empty or start_ts > end_ts:
            return []
        expected = pd.date_range(start=start_ts, end=end_ts, freq=self._timeframe_freq(), tz="UTC")
        present = set(self.df.index)
        return [ts for ts in expected if ts not in present]

    def _backfill_missing(self, missing: List[pd.Timestamp]) -> None:
        if not missing:
            return
        delta = self._timeframe_delta()
        missing_sorted = sorted(missing)
        seg_start = seg_end = missing_sorted[0]

        def flush(start: pd.Timestamp, end: pd.Timestamp) -> None:
            candles = self.rest.get_klines(
                start_at=int(start.timestamp()),
                end_at=int((end + delta).timestamp()) - 1,
            )
            self._merge_df(self._candles_to_df(candles))

        for ts in missing_sorted[1:]:
            if ts == seg_end + delta and ts - seg_start < 1500 * delta:
                seg_end = ts
            else:
                flush(seg_start, seg_end)
                seg_start = seg_end = ts
        flush(seg_start, seg_end)

    def warmup(self) -> None:
        last_closed = self.get_latest_closed_candle_ts()
        candles = self.rest.get_klines(
            start_at=int((last_closed - (config.WARMUP_CANDLES - 1) * self._timeframe_delta()).timestamp()),
            end_at=int((last_closed + self._timeframe_delta()).timestamp()) - 1,
        )
        self.df = self._candles_to_df(candles)
        if self.df.empty:
            raise RuntimeError("Warmup returned no closed hourly candles.")
        print(f"[warmup] symbol={config.SYMBOL} candles={len(self.df)} first={self.df.index.min()} last={self.df.index.max()}")
        self.last_processed_ts = None

    def get_latest_closed_candle_ts(self) -> pd.Timestamp:
        candidate = self.server_now_utc() - pd.Timedelta(seconds=config.SAFE_CLOSE_DELAY_SECONDS)
        return candidate.floor(self._timeframe_freq()) - self._timeframe_delta()

    def fetch_and_update(self) -> FetchResult:
        last_closed = self.get_latest_closed_candle_ts()
        if self.last_processed_ts is not None and last_closed <= self.last_processed_ts:
            return FetchResult(df=self.df.copy(), new_candle_ts=[])

        candles = self.rest.get_klines(
            start_at=int((last_closed - (config.FETCH_LAST_N_CANDLES - 1) * self._timeframe_delta()).timestamp()),
            end_at=int((last_closed + self._timeframe_delta()).timestamp()) - 1,
        )
        self._merge_df(self._candles_to_df(candles))
        if self.df.empty:
            return FetchResult(df=self.df.copy(), new_candle_ts=[])

        missing = self._detect_missing_candles(self.df.index.min(), last_closed)
        self._backfill_missing(missing)
        eligible = self.df.index[self.df.index <= last_closed]
        if self.last_processed_ts is None:
            new_ts = [last_closed] if last_closed in eligible else []
        else:
            new_ts = [ts for ts in eligible if ts > self.last_processed_ts]
        return FetchResult(df=self.df.copy(), new_candle_ts=new_ts)

    def mark_processed(self, ts: pd.Timestamp) -> None:
        if self.last_processed_ts is None or ts > self.last_processed_ts:
            self.last_processed_ts = ts
