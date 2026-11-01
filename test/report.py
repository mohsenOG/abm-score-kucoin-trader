"""Write a standalone dashboard from the existing backtest outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config_test as cfg


def report_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    display = frame.copy()
    display["Average Net Profit per Trade"] = display["Total Net Profit"] / display["Completed Trades"].replace(0, float("nan"))
    return json.loads(display.to_json(orient="records", double_precision=15))


def write_report(
    history: pd.DataFrame,
    signals: pd.DataFrame,
    results: pd.DataFrame,
    validation: pd.DataFrame,
    signals_path: Path,
    heatmap_path: Path,
    report_path: Path,
) -> None:
    seconds = cfg.TIMEFRAME_SECONDS
    timeframe = f"{seconds / 3600:g}h" if seconds % 3600 == 0 else f"{seconds / 60:g}m"
    data = {
        "symbol": signals_path.name.split("_", 1)[0],
        "timeframe": timeframe,
        "start": history.index[0].isoformat(),
        "end": (history.index[-1] + pd.Timedelta(seconds=seconds)).isoformat(),
        "signals": len(signals),
        "warmup": cfg.WARMUP_CANDLES,
        "fee": cfg.FEE_PER_SIDE_PCT,
        "trainFraction": cfg.TRAIN_FRACTION,
        "topN": cfg.TOP_N,
        "pageSize": cfg.REPORT_PAGE_SIZE,
        "csvName": report_path.name.removesuffix("_report.html") + "_results.csv",
        "results": report_records(results),
        "validation": report_records(validation),
    }
    # JSON is inside a script element; escape '<' so metadata cannot close it.
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    html = template.replace("<!-- HEATMAP -->", heatmap_path.read_text(encoding="utf-8"))
    html = html.replace("/* REPORT_DATA */", payload)
    with report_path.open("x", encoding="utf-8") as report:
        report.write(html)
