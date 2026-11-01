from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

import config


@dataclass(frozen=True)
class Candle:
    ts: int  # Candle start time in UTC seconds.
    open: float
    close: float
    high: float
    low: float
    volume: float
    turnover: float


class KuCoinREST:
    """KuCoin V2 public REST client for hourly spot candles."""

    BASE_URL = "https://api.kucoin.com"

    def __init__(self) -> None:
        self.session = requests.Session()

    def _reset_session(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()

    def _request(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
            try:
                response = self.session.get(
                    self.BASE_URL + path,
                    params=params,
                    headers={"kc-enable-ns": "true"},
                    timeout=(config.HTTP_CONNECT_TIMEOUT_SECONDS, config.HTTP_READ_TIMEOUT_SECONDS),
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") != "200000":
                    raise RuntimeError(
                        f"KuCoin API error: code={payload.get('code')} msg={payload.get('msg')}"
                    )
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_exc = exc
                self._reset_session()
                print(
                    f"[kucoin_rest] request failed attempt={attempt}/{config.HTTP_MAX_RETRIES} "
                    f"path={path} params={params} err={repr(exc)}"
                )
                if attempt >= config.HTTP_MAX_RETRIES:
                    break
                time.sleep(config.HTTP_RETRY_PERIOD_SECONDS)

        raise RuntimeError(f"KuCoin REST request failed after retries: {path} last={repr(last_exc)}")

    def get_server_time_ms(self) -> int:
        response = self._request("/api/ua/v2/server/status", params={"tradeType": "SPOT"})
        # KuCoin's gateway prefix is "5-"; kc-enable-ns requests nanoseconds.
        server_ns = response.headers["x-out-time"].removeprefix("5-")
        return int(server_ns) // 1_000_000

    def get_klines(self, *, start_at: int, end_at: int, symbol: Optional[str] = None) -> List[Candle]:
        response = self._request(
            "/api/ua/v2/market/kline",
            params={
                "tradeType": "SPOT",
                "symbol": config.SYMBOL if symbol is None else symbol,
                "klineType": "TRADE",
                "interval": config.KLINE_TYPE,
                "startAt": int(start_at),
                "endAt": int(end_at),
            },
        )
        rows = response.json()["data"]["list"]
        candles: List[Candle] = []
        for row in rows:
            # V2 order: timestamp, open, high, low, close, volume, turnover.
            candle = Candle(
                ts=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                turnover=float(row[6]),
            )
            if not (0 < candle.low <= candle.open <= candle.high and candle.low <= candle.close <= candle.high):
                raise ValueError(f"Invalid KuCoin candle at {candle.ts}")
            if start_at <= candle.ts <= end_at:
                candles.append(candle)
        return candles
