from __future__ import annotations

import pandas as pd

import config


def crossover(x: pd.Series, y: pd.Series) -> pd.Series:
    # Pine: x[1] <= y[1] and x > y
    return (x.shift(1) <= y.shift(1)) & (x > y)


def crossunder(x: pd.Series, y: pd.Series) -> pd.Series:
    # Pine: x[1] >= y[1] and x < y
    return (x.shift(1) >= y.shift(1)) & (x < y)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    thr = float(config.BIAS_THRESHOLD)
    buy_y = pd.Series(thr, index=out.index, dtype=float)
    sell_y = pd.Series(-thr, index=out.index, dtype=float)

    out["buy_signal"] = crossover(out["bias"], buy_y).fillna(False)
    out["sell_signal"] = crossunder(out["bias"], sell_y).fillna(False)
    return out
