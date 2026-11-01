"""Run with: python test/check_backtest.py"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config_test as cfg
from backtest import chronological_validation, optimize, percentage_range, write_heatmap
from indicators import compute_bias_and_votes
from main_test import write_signals
from signals import compute_signals


def candles(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float, index=pd.date_range("2025-01-01", periods=len(rows), freq="h", tz="UTC"))
    frame["volume"] = frame["turnover"] = 1.0
    return frame


def signals(history: pd.DataFrame, positions: list[int], sides: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"side": sides}, index=history.index[positions])
    frame["candle_close_utc"] = frame.index + pd.Timedelta(seconds=cfg.TIMEFRAME_SECONDS)
    return frame


def check() -> None:
    assert len(percentage_range(0.1, 5.0, 0.1)) == 50
    assert np.array_equal(percentage_range(0.1, 0.3, 0.1), [0.1, 0.2, 0.3])
    for args in ((0, 5, 0.1), (0.1, 5, 0), (0.1, 0.25, 0.1)):
        try:
            percentage_range(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid range accepted")

    with patch.object(cfg, "FEE_PER_SIDE_PCT", 0.06):
        # Signal candle extremes must never close a trade. Entry uses the next Open.
        for side, exit_row in (("BUY", (100, 101.1, 99.9, 101)), ("SELL", (100, 100.1, 98.9, 99))):
            history = candles([(200, 300, 1, 200), exit_row])
            result = optimize(history, signals(history, [0], [side]), [1], [1]).iloc[0]
            assert result["Completed Trades"] == 1 and result["Win Rate"] == 100
            assert np.isclose(result["Total Net Profit"], 0.88)

        # Both levels in one candle: SL wins for BUY and SELL.
        for side in ("BUY", "SELL"):
            history = candles([(100, 100, 100, 100), (100, 102, 98, 100)])
            result = optimize(history, signals(history, [0], [side]), [1], [1]).iloc[0]
            assert np.isclose(result["Total Net Profit"], -1.12)
            assert result["Win Rate"] == 0

        # Gap fills in both directions, with TP-gap/SL ties also losing.
        for side, gap, expected in (
            ("BUY", (98, 98.5, 97, 98), -2.12),
            ("BUY", (102, 103, 101.5, 102), 1.88),
            ("SELL", (102, 103, 101.5, 102), -2.12),
            ("SELL", (98, 98.5, 97, 98), 1.88),
            ("BUY", (102, 103, 98, 100), -1.12),
            ("SELL", (98, 102, 97, 100), -1.12),
        ):
            history = candles([(100, 100, 100, 100), (100, 100.2, 99.8, 100), gap])
            result = optimize(history, signals(history, [0], [side]), [1], [1]).iloc[0]
            assert np.isclose(result["Total Net Profit"], expected), (side, gap, result)

        # A TP below fees is a net loss, not a win.
        history = candles([(100, 100, 100, 100), (100, 100.2, 100, 100.1)])
        result = optimize(history, signals(history, [0], ["BUY"]), [1], [0.1]).iloc[0]
        assert np.isclose(result["Total Net Profit"], -0.02) and result["Win Rate"] == 0
        result = optimize(history, signals(history, [0], ["BUY"]), [1], [0.12]).iloc[0]
        assert result["Total Net Profit"] == 0 and result["Win Rate"] == 0

        history = candles([(100, 100, 100, 100)] * 3 + [(100, 102, 100, 101), (100, 100, 100, 100)])
        trades = signals(history, [0, 1, 3, 4], ["BUY", "BUY", "SELL", "BUY"])
        result = optimize(history, trades, [1], [1]).iloc[0]
        assert result["Completed Trades"] == 2 and result["Open Trades"] == 1 and result["Pending Entries"] == 1
        assert np.isclose(result["Total Net Profit"], 1.76)
        empty = optimize(history, trades.iloc[:0], [1], [1]).iloc[0]
        assert empty["Total Net Profit"] == 0 and empty["Completed Trades"] == 0 and pd.isna(empty["Win Rate"])

        # Compare the optimized grid against a direct chronological candle scan.
        rng = np.random.default_rng(7)
        opens = 100 + rng.normal(0, 0.5, 40).cumsum()
        history = candles([(o, o + up, o - down, o) for o, up, down in zip(opens, rng.uniform(0, 2, 40), rng.uniform(0, 2, 40))])
        trades = signals(history, list(range(40)), ["BUY", "SELL"] * 20)
        grid = optimize(history, trades, [0.1, 1, 5], [0.1, 1, 5])
        for _, result in grid.iterrows():
            returns = []
            for position, side in enumerate(trades.side):
                if position + 1 == len(history):
                    continue
                entry = history.iloc[position + 1].open
                direction = 1 if side == "BUY" else -1
                stop = entry * (1 - direction * result.SL / 100)
                target = entry * (1 + direction * result.TP / 100)
                for row in history.iloc[position + 1:].itertuples():
                    stop_hit = row.low <= stop if direction == 1 else row.high >= stop
                    target_hit = row.high >= target if direction == 1 else row.low <= target
                    if stop_hit or target_hit:
                        exit_price = (min(row.open, stop) if direction == 1 else max(row.open, stop)) if stop_hit else (max(row.open, target) if direction == 1 else min(row.open, target))
                        returns.append(direction * (exit_price / entry - 1) * 100 - 0.12)
                        break
            assert result["Completed Trades"] == len(returns)
            assert np.isclose(result["Total Net Profit"], sum(returns))
            assert np.isclose(result["Win Rate"], 100 * sum(value > 0 for value in returns) / len(returns))

        # Training trades may not see validation candles.
        history = candles([(100, 100, 100, 100)] * 10 + [(100, 110, 90, 100)] * 2)
        trades = signals(history, list(range(10)), ["BUY"] * 10)
        with contextlib.redirect_stdout(io.StringIO()):
            assert chronological_validation(history, trades).empty
        history.iloc[2, history.columns.get_loc("high")] = 100.3
        validation = chronological_validation(history, trades)
        altered = history.copy()
        altered.iloc[8:, altered.columns.get_loc("high")] = 150
        altered.iloc[8:, altered.columns.get_loc("low")] = 50
        other = chronological_validation(altered, trades)
        pd.testing.assert_series_equal(validation.iloc[0], other.iloc[0])
        assert validation["Signals"].tolist() == [7, 3]

    # Indicators and signals at a timestamp do not change when later candles arrive.
    prices = 100 + np.sin(np.arange(260) / 6) * 3 + np.arange(260) / 50
    history = candles([(p, p + 1, p - 1, p) for p in prices])
    frame = compute_bias_and_votes(history, use_heikin_ashi=True)
    assert frame.ema200.first_valid_index() == history.index[199]
    pd.testing.assert_frame_equal(compute_signals(frame).iloc[:230], compute_signals(compute_bias_and_votes(history.iloc[:230], use_heikin_ashi=True)))
    with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        with patch.object(cfg, "SIGNALS_DIR", Path(directory)):
            path = write_signals(history, Path("CHECK-USDT_1h.csv"))
            saved = pd.read_csv(path)
            assert len(saved) > 0 and (saved.candle_number > cfg.WARMUP_CANDLES).all()
        path = Path(directory) / "heatmap.svg"
        write_heatmap(grid, path)
        assert ElementTree.parse(path).getroot().tag.endswith("svg")
    print("Backtest checks passed.")


if __name__ == "__main__":
    check()
