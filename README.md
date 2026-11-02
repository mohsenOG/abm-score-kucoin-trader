# abm-score-kucoin-trader

A Python project.

Project layout:

- `src/`: application code and configuration.
- `backtest/`: backtest scripts, report template, and saved history, signals, and results.
- `requirements.txt`: Python dependencies.

Run these commands from the project root with the project virtual environment activated.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run live signals:

```powershell
python src/main.py
```

Run the backtest workflow:

```powershell
python backtest/main_test.py
```

Run the offline checks:

```powershell
python backtest/check_backtest.py
```
