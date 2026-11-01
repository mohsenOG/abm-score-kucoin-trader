"""Independent 1x trades on original OHLC, with no capital allocation."""

from __future__ import annotations

from decimal import Decimal
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

import config_test as cfg
from report import write_report


def percentage_range(start: float, stop: float, step: float) -> np.ndarray:
    start, stop, step = (Decimal(str(value)) for value in (start, stop, step))
    if not all(value.is_finite() for value in (start, stop, step)):
        raise ValueError("SL/TP ranges must be finite.")
    if not 0 < start <= stop < 100 or step <= 0 or (stop - start) % step:
        raise ValueError("SL/TP ranges need 0 < minimum <= maximum < 100 and an exact positive step.")
    return np.array([float(start + i * step) for i in range(int((stop - start) / step) + 1)])


def optimize(
    history: pd.DataFrame,
    signals: pd.DataFrame,
    sl_values: np.ndarray | None = None,
    tp_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sum completed trade returns in percentage points; leave open trades unrealized."""
    sl = percentage_range(cfg.SL_MIN_PCT, cfg.SL_MAX_PCT, cfg.SL_STEP_PCT) if sl_values is None else np.asarray(sl_values, dtype=float)
    tp = percentage_range(cfg.TP_MIN_PCT, cfg.TP_MAX_PCT, cfg.TP_STEP_PCT) if tp_values is None else np.asarray(tp_values, dtype=float)
    for values in (sl, tp):
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not ((values > 0) & (values < 100)).all():
            raise ValueError("SL/TP values must be finite percentages between 0 and 100.")
    if not np.isfinite(cfg.FEE_PER_SIDE_PCT) or not 0 <= cfg.FEE_PER_SIDE_PCT < 100:
        raise ValueError("The fee must be a finite percentage between 0 and 100.")

    delta = pd.Timedelta(seconds=cfg.TIMEFRAME_SECONDS)
    if history.empty or not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("History needs timezone-aware candle_open_utc timestamps.")
    if history.index.hasnans or not history.index.is_unique or not history.index.is_monotonic_increasing:
        raise ValueError("History timestamps must be unique and chronological.")
    if not (history.index.to_series().diff().dropna() == delta).all():
        raise ValueError("History has missing or irregular candles; exits cannot be determined safely.")
    prices = history[["open", "high", "low", "close"]].to_numpy(dtype=float)
    opens, highs, lows, closes = prices.T
    if not np.isfinite(prices).all() or not (prices > 0).all() or not (
        (lows <= opens) & (lows <= closes) & (highs >= opens) & (highs >= closes)
    ).all():
        raise ValueError("History contains invalid OHLC prices.")
    if not isinstance(signals.index, pd.DatetimeIndex) or signals.index.tz is None or signals.index.hasnans:
        raise ValueError("Signals need timezone-aware candle_open_utc timestamps.")
    if not signals.index.is_monotonic_increasing or not signals.index.is_unique:
        raise ValueError("Signals must be unique and chronological.")
    if not signals["side"].isin(["BUY", "SELL"]).all():
        raise ValueError("Signal sides must be BUY or SELL.")
    signal_positions = history.index.get_indexer(signals.index)
    if (signal_positions < 0).any():
        raise ValueError("Every signal candle must exist in the OHLC history.")
    close_times = pd.to_datetime(signals["candle_close_utc"], utc=True)
    if not (close_times.array == signals.index + delta).all():
        raise ValueError("Signal candle_close_utc must equal candle_open_utc plus the timeframe.")

    shape = (len(sl), len(tp))
    completed = np.zeros(shape, dtype=int)
    wins = np.zeros(shape, dtype=int)
    tp_exits = np.zeros(shape, dtype=int)
    total = np.zeros(shape)
    # ponytail: Exact percentiles store O(signals * combinations) durations; use duration histograms for much larger runs.
    holding_hours = np.full((len(signals), *shape), np.nan)
    candle_hours = ((history.index - history.index[0]) / pd.Timedelta(hours=1)).to_numpy()
    timeframe_hours = delta / pd.Timedelta(hours=1)
    pending = 0
    for signal_number, (position, side) in enumerate(zip(signal_positions, signals["side"])):
        entry_index = position + 1
        if entry_index == len(history):
            pending += 1
            continue
        entry = opens[entry_index]
        direction = 1 if side == "BUY" else -1
        stops = entry * (1 - direction * sl / 100)
        targets = entry * (1 + direction * tp / 100)
        # First passage times are shared by all pairs, avoiding 2,500 candle scans.
        # ponytail: O(signals * remaining candles); use a range-search tree for much larger histories.
        adverse = -lows[entry_index:] if direction == 1 else highs[entry_index:]
        favorable = highs[entry_index:] if direction == 1 else -lows[entry_index:]
        stop_at = np.searchsorted(np.maximum.accumulate(adverse), -direction * stops)
        target_at = np.searchsorted(np.maximum.accumulate(favorable), direction * targets)
        stopped = stop_at[:, None] <= target_at[None, :]
        exit_at = np.minimum(stop_at[:, None], target_at[None, :])
        closed = exit_at < len(adverse)
        # Clamp only for array access; unfinished trades never contribute returns.
        exit_open = direction * opens[entry_index + np.minimum(exit_at, len(adverse) - 1)]
        # SL wins ties, including a candle that gaps through TP and later reaches SL.
        exit_price = direction * np.where(
            stopped,
            np.minimum(exit_open, direction * stops[:, None]),
            np.maximum(exit_open, direction * targets[None, :]),
        )
        net = np.round(direction * (exit_price / entry - 1) * 100 - 2 * cfg.FEE_PER_SIDE_PCT, cfg.RETURN_DECIMALS)
        completed += closed
        wins += closed & (net > 0)
        tp_exits += closed & ~stopped
        total += np.where(closed, net, 0)
        # Use the selected exit only: SL still wins a TP-gap/SL intrabar tie.
        at_open = np.where(stopped, exit_open <= direction * stops[:, None], exit_open >= direction * targets[None, :])
        exit_hours = candle_hours[entry_index + np.minimum(exit_at, len(adverse) - 1)]
        duration = exit_hours - candle_hours[entry_index] + np.where(at_open, 0, timeframe_hours)
        holding_hours[signal_number] = np.where(closed, duration, np.nan)

    win_rate = np.full(shape, np.nan)
    np.divide(100 * wins, completed, out=win_rate, where=completed > 0)
    tp_hit_rate = np.full(shape, np.nan)
    np.divide(100 * tp_exits, completed, out=tp_hit_rate, where=completed > 0)
    avg_holding, p95_holding, p99_holding, max_holding = (np.full(shape, np.nan) for _ in range(4))
    for row, column in zip(*np.nonzero(completed)):
        durations = holding_hours[:, row, column]
        durations = durations[~np.isnan(durations)]
        avg_holding[row, column] = durations.mean()
        p95_holding[row, column], p99_holding[row, column] = np.percentile(
            durations, cfg.HOLDING_PERCENTILES, method=cfg.HOLDING_PERCENTILE_METHOD
        )
        max_holding[row, column] = durations.max()
    results = pd.DataFrame({
        "SL": np.repeat(sl, len(tp)),
        "TP": np.tile(tp, len(sl)),
        "Win Rate": win_rate.ravel(),
        "Total Net Profit": total.ravel(),
        "Completed Trades": completed.ravel(),
        "Open Trades": (len(signals) - pending - completed).ravel(),
        "Pending Entries": pending,
        "avg_holding_hours": avg_holding.ravel(),
        "p95_holding_hours": p95_holding.ravel(),
        "p99_holding_hours": p99_holding.ravel(),
        "max_holding_hours": max_holding.ravel(),
        "tp_hit_rate_pct": tp_hit_rate.ravel(),
    })
    return results.sort_values("Total Net Profit", ascending=False, kind="stable").reset_index(drop=True)


def chronological_validation(history: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if not 0 < cfg.TRAIN_FRACTION < 1:
        raise ValueError("TRAIN_FRACTION must be between 0 and 1.")
    split = int(len(signals) * cfg.TRAIN_FRACTION)
    if split == 0 or split == len(signals):
        print("Validation skipped: both signal sets need at least one signal.")
        return pd.DataFrame()
    boundary = pd.Timestamp(signals.iloc[split]["candle_close_utc"])
    training = optimize(history.loc[history.index < boundary], signals.iloc[:split])
    candidates = training[training["Completed Trades"] > 0]
    if candidates.empty:
        print("Validation skipped: no training trades completed before the boundary.")
        return pd.DataFrame()
    best = candidates.iloc[0]
    validation = optimize(history, signals.iloc[split:], np.array([best["SL"]]), np.array([best["TP"]]))
    report = pd.concat([best.to_frame().T, validation], ignore_index=True)
    report.insert(0, "Set", ["Training", "Validation"])
    report["Signals"] = [split, len(signals) - split]
    report["Boundary UTC"] = boundary.isoformat()
    for column in ("Completed Trades", "Open Trades", "Pending Entries"):
        report[column] = report[column].astype(int)
    return report


def write_heatmap(results: pd.DataFrame, path: Path) -> None:
    grid = results.pivot(index="SL", columns="TP", values="Total Net Profit").sort_index(ascending=False)
    cell = 12
    left, top = 90, 65
    width, height = max(720, left + len(grid.columns) * cell + 50), top + len(grid) * cell + 145
    scale = max(abs(results["Total Net Profit"]).max(), np.finfo(float).eps)

    def color(value: float) -> str:
        fraction = min(abs(value) / scale, 1)
        endpoint = (22, 135, 105) if value >= 0 else (190, 40, 50)
        return "#" + "".join(f"{round(255 + (channel - 255) * fraction):02x}" for channel in endpoint)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<title>SL/TP total net profit heatmap</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#222">',
        '<text x="20" y="25" font-size="18">Total net profit after fees (percentage points)</text>',
        f'<text x="20" y="45">{len(results)} combinations; completed trades only; independent 1x returns</text>',
    ]
    for row, (sl, values) in enumerate(grid.iterrows()):
        y = top + row * cell
        if row % max(1, len(grid) // 10) == 0 or row == len(grid) - 1:
            parts.append(f'<text x="{left - 10}" y="{y + 10}" text-anchor="end">{sl:g}</text>')
        for column, (tp, value) in enumerate(values.items()):
            title = escape(f"SL {sl:g}%; TP {tp:g}%; net {value:.4f} pp")
            parts.append(f'<rect x="{left + column * cell}" y="{y}" width="{cell}" height="{cell}" fill="{color(value)}"><title>{title}</title></rect>')
    bottom = top + len(grid) * cell
    for column, tp in enumerate(grid.columns):
        if column % max(1, len(grid.columns) // 10) == 0 or column == len(grid.columns) - 1:
            parts.append(f'<text x="{left + column * cell + cell / 2}" y="{bottom + 20}" text-anchor="middle">{tp:g}</text>')
    parts.extend([
        f'<text x="{left + len(grid.columns) * cell / 2}" y="{bottom + 42}" text-anchor="middle">TP (%)</text>',
        f'<text x="20" y="{top + len(grid) * cell / 2}">SL (%)</text>',
    ])
    for i in range(100):
        parts.append(f'<rect x="{left + i * 3}" y="{bottom + 65}" width="3" height="12" fill="{color(scale * (2 * i / 99 - 1))}"/>')
    for x, value in ((left, -scale), (left + 150, 0), (left + 300, scale)):
        parts.append(f'<text x="{x}" y="{bottom + 95}" text-anchor="middle">{value:.2f}</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def run_backtest(history: pd.DataFrame, signals_path: Path) -> None:
    signals = pd.read_csv(signals_path, index_col="candle_open_utc")
    signals.index = pd.to_datetime(signals.index, utc=True)
    signals["candle_close_utc"] = pd.to_datetime(signals["candle_close_utc"], utc=True)
    results = optimize(history, signals)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = cfg.RESULTS_DIR / signals_path.stem
    results_path = prefix.with_name(prefix.name + "_optimization.csv")
    heatmap_path = prefix.with_name(prefix.name + "_heatmap.svg")
    results.to_csv(results_path, index=False, mode="x")
    write_heatmap(results, heatmap_path)
    print(f"\nTop {min(cfg.TOP_N, len(results))} of {len(results)} SL/TP combinations, ranked by total net profit:")
    print("SL, TP and win rate are percentages; profit is the sum of independent 1x net returns in percentage points.")
    print(results.head(cfg.TOP_N).to_string(index=False, float_format=lambda value: f"{value:.4f}", na_rep="N/A"))
    print("Open trades are unrealized. Pending entries have no next candle. Both are excluded from profit and win rate.")
    print(f"Full ranked table: {results_path}")
    print(f"Profit heatmap: {heatmap_path}")
    validation = chronological_validation(history, signals)
    if not validation.empty:
        validation_path = prefix.with_name(prefix.name + "_validation.csv")
        validation.to_csv(validation_path, index=False, mode="x")
        print(f"\nChronological validation ({cfg.TRAIN_FRACTION:.0%}/{1 - cfg.TRAIN_FRACTION:.0%} of signals):")
        print(validation.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"Validation saved: {validation_path}")
    report_path = prefix.with_name(prefix.name + "_report.html")
    write_report(history, signals, results, validation, signals_path, heatmap_path, report_path)
    print(f"HTML dashboard: {report_path}")
