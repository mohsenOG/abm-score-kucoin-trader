from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

import config

AGENT_COUNT = 21


def pine_ema(src: pd.Series, length: int) -> pd.Series:
    """TradingView/Pine ta.ema parity (SMA seed, then recursive EMA)."""
    length = int(length)
    alpha = 2.0 / (length + 1.0)

    s = src.astype(float)
    sma = s.rolling(length).mean()

    out = pd.Series(np.nan, index=s.index, dtype=float)
    prev = float("nan")
    for i in range(len(s)):
        if prev != prev:  # NaN
            v = float(sma.iloc[i])
            out.iloc[i] = v
            prev = v
        else:
            x = float(s.iloc[i])
            v = alpha * x + (1.0 - alpha) * prev
            out.iloc[i] = v
            prev = v
    return out


def compute_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    ha_open = np.empty(len(df), dtype=float)
    ha_open[:] = np.nan
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    hc = ha_close.to_numpy(dtype=float)

    for i in range(len(df)):
        if i == 0 or np.isnan(ha_open[i - 1]):
            ha_open[i] = (o[i] + c[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + hc[i - 1]) / 2.0

    ha_open_s = pd.Series(ha_open, index=df.index, name="ha_open")
    ha_high = pd.concat([df["high"], ha_open_s, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open_s, ha_close], axis=1).min(axis=1)

    out = df.copy()
    out["ha_open"] = ha_open_s
    out["ha_high"] = ha_high
    out["ha_low"] = ha_low
    out["ha_close"] = ha_close
    return out


def _ema(s: pd.Series, n: int) -> pd.Series:
    return pine_ema(s, int(n))


def _sma(s: pd.Series, n: int) -> pd.Series:
    return SMAIndicator(close=s.astype(float), window=int(n), fillna=False).sma_indicator()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    return RSIIndicator(close=s.astype(float), window=int(n), fillna=False).rsi()


def _macd(s: pd.Series, fast: int, slow: int, signal: int):
    fast_ema = pine_ema(s, int(fast))
    slow_ema = pine_ema(s, int(slow))
    macd_line = fast_ema - slow_ema
    signal_line = pine_ema(macd_line, int(signal))
    return macd_line, signal_line


def compute_bias_and_votes(df: pd.DataFrame, *, use_heikin_ashi: bool | None = None) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if use_heikin_ashi is None:
        use_heikin_ashi = config.USE_HEIKIN_ASHI
    if use_heikin_ashi:
        if not {"ha_open", "ha_high", "ha_low", "ha_close"}.issubset(out.columns):
            out = compute_heikin_ashi(out)
        out["src_open"] = out["ha_open"]
        out["src_high"] = out["ha_high"]
        out["src_low"] = out["ha_low"]
        out["src_close"] = out["ha_close"]
    else:
        out["src_open"] = out["open"]
        out["src_high"] = out["high"]
        out["src_low"] = out["low"]
        out["src_close"] = out["close"]

    src_close = out["src_close"]

    # EMA trend
    out["ema_short_1"] = _ema(src_close, 5)
    out["ema_long_1"] = _ema(src_close, 10)
    out["ema_short_2"] = _ema(src_close, 5)
    out["ema_long_2"] = _ema(src_close, 20)
    out["ema_short_3"] = _ema(src_close, 10)
    out["ema_long_3"] = _ema(src_close, 20)
    out["ema_short_4"] = _ema(src_close, 10)
    out["ema_long_4"] = _ema(src_close, 50)

    out["ema_trend_vote"] = (
        (out["ema_short_1"] > out["ema_long_1"]).astype(int) - (out["ema_short_1"] < out["ema_long_1"]).astype(int)
        + (out["ema_short_2"] > out["ema_long_2"]).astype(int) - (out["ema_short_2"] < out["ema_long_2"]).astype(int)
        + (out["ema_short_3"] > out["ema_long_3"]).astype(int) - (out["ema_short_3"] < out["ema_long_3"]).astype(int)
        + (out["ema_short_4"] > out["ema_long_4"]).astype(int) - (out["ema_short_4"] < out["ema_long_4"]).astype(int)
    ).astype(int)

    # SMA trend
    out["sma_short_1"] = _sma(src_close, 5)
    out["sma_long_1"] = _sma(src_close, 10)
    out["sma_short_2"] = _sma(src_close, 5)
    out["sma_long_2"] = _sma(src_close, 20)
    out["sma_short_3"] = _sma(src_close, 10)
    out["sma_long_3"] = _sma(src_close, 50)

    out["sma_trend_vote"] = (
        (out["sma_short_1"] > out["sma_long_1"]).astype(int) - (out["sma_short_1"] < out["sma_long_1"]).astype(int)
        + (out["sma_short_2"] > out["sma_long_2"]).astype(int) - (out["sma_short_2"] < out["sma_long_2"]).astype(int)
        + (out["sma_short_3"] > out["sma_long_3"]).astype(int) - (out["sma_short_3"] < out["sma_long_3"]).astype(int)
    ).astype(int)

    # RSI contrarian
    out["rsi"] = _rsi(src_close, 14)
    out["rsi_vote"] = np.where(out["rsi"] < 30, 1, np.where(out["rsi"] > 70, -1, 0)).astype(int)

    # EMA fundamentalists
    out["ema20"] = _ema(src_close, 20)
    out["ema50"] = _ema(src_close, 50)
    out["ema100"] = _ema(src_close, 100)
    out["ema200"] = _ema(src_close, 200)

    out["ema_fund_vote"] = (
        np.where(src_close < out["ema20"] * (1 - 0.01), 1, np.where(src_close > out["ema20"] * (1 + 0.015), -1, 0))
        + np.where(src_close < out["ema50"] * (1 - 0.01), 1, np.where(src_close > out["ema50"] * (1 + 0.02), -1, 0))
        + np.where(src_close < out["ema100"] * (1 - 0.015), 1, np.where(src_close > out["ema100"] * (1 + 0.025), -1, 0))
        + np.where(src_close < out["ema200"] * (1 - 0.015), 1, np.where(src_close > out["ema200"] * (1 + 0.05), -1, 0))
    ).astype(int)

    # MACD
    out["macd1"], out["signal1"] = _macd(src_close, 12, 26, 9)
    out["macd2"], out["signal2"] = _macd(src_close, 20, 50, 10)
    out["macd3"], out["signal3"] = _macd(src_close, 8, 21, 6)

    out["macd_vote"] = (
        (out["macd1"] > out["signal1"]).astype(int) - (out["macd1"] < out["signal1"]).astype(int)
        + (out["macd2"] > out["signal2"]).astype(int) - (out["macd2"] < out["signal2"]).astype(int)
        + (out["macd3"] > out["signal3"]).astype(int) - (out["macd3"] < out["signal3"]).astype(int)
    ).astype(int)

    # Bollinger (population stddev ddof=0)
    out["bb_basis"] = src_close.rolling(20).mean()
    out["bb_stddev"] = src_close.rolling(20).std(ddof=0)
    out["bb_upper"] = out["bb_basis"] + 2.0 * out["bb_stddev"]
    out["bb_lower"] = out["bb_basis"] - 2.0 * out["bb_stddev"]
    out["bb_vote"] = np.where(src_close < out["bb_lower"], 1, np.where(src_close > out["bb_upper"], -1, 0)).astype(int)

    # Breakout ranges exclude the current candle.
    out["high5"] = out["src_high"].rolling(5).max().shift(1)
    out["low5"] = out["src_low"].rolling(5).min().shift(1)
    out["high17"] = out["src_high"].rolling(17).max().shift(1)
    out["low17"] = out["src_low"].rolling(17).min().shift(1)
    out["high21"] = out["src_high"].rolling(21).max().shift(1)
    out["low21"] = out["src_low"].rolling(21).min().shift(1)

    out["breakout_vote"] = (
        np.where(src_close > out["high5"], 1, np.where(src_close < out["low5"], -1, 0))
        + np.where(src_close > out["high17"], 1, np.where(src_close < out["low17"], -1, 0))
        + np.where(src_close > out["high21"], 1, np.where(src_close < out["low21"], -1, 0))
    ).astype(int)

    # ROC
    out["roc13"] = (src_close - src_close.shift(13)) / src_close.shift(13) * 100.0
    out["roc27"] = (src_close - src_close.shift(27)) / src_close.shift(27) * 100.0
    out["bar_change"] = src_close / src_close.shift(1) - 1.0

    out["roc_vote"] = (
        np.where((out["roc13"] > 3.5) & (out["bar_change"] > 0.01), 1, np.where((out["roc13"] < -3.5) & (out["bar_change"] < -0.01), -1, 0))
        + np.where((out["roc27"] > 2.0) & (out["bar_change"] > 0.01), 1, np.where((out["roc27"] < -2.0) & (out["bar_change"] < -0.01), -1, 0))
    ).astype(int)

    out["total_votes"] = (
        out["ema_trend_vote"]
        + out["sma_trend_vote"]
        + out["rsi_vote"]
        + out["ema_fund_vote"]
        + out["macd_vote"]
        + out["bb_vote"]
        + out["breakout_vote"]
        + out["roc_vote"]
    ).astype(int)

    out["bias"] = out["total_votes"].astype(float) / float(AGENT_COUNT)

    return out
