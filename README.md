# 📈 Enhanced SMA Backtester

A powerful, beginner-friendly backtesting engine for stock trading strategies using Python.

---

## 🚀 What It Does

This tool lets you simulate different trading strategies on historical stock data to evaluate performance. It includes:

- Simple/Exponential Moving Averages (SMA/EMA)
- Bollinger Bands (trend and mean-reversion)
- Momentum-based signals (z-scores)
- Hybrid and Ensemble approaches
- Risk management tools (e.g., stop loss, volatility scaling)
- Visual and statistical performance analysis

---

## 🛠️ Features

### 📉 Strategy Types

- `basic`: SMA/EMA crossover
- `confirmed`: SMA crossover with multi-day confirmation
- `filtered`: Adds trend/volatility filters
- `bollinger`: Mean reversion via Bollinger Bands
- `hybrid_bollinger`: Bollinger + long-term trend filter
- `momentum`: Z-score based price movement
- `ensemble`: Weighted signal combination

### 🔒 Risk Management

- Stop loss / take profit exits
- Dynamic volatility-based position sizing
- Signal suppression during high-volatility markets

### 📊 Performance Metrics

- Sharpe Ratio
- Max Drawdown
- Win Rate
- Cumulative return curves
- Rolling Sharpe plots
- Monthly return heatmaps

---

## 🧪 Example Usage

```python
from backtester import EnhancedSMABacktester  # or use your actual file name

# Create a backtest instance
bt = EnhancedSMABacktester('SPY', '2020-01-01', '2024-01-01')

# Run basic strategy
results, diagnostics = bt.run_backtest(strategy_type='basic')

# Run advanced hybrid Bollinger strategy
results, diagnostics = bt.run_backtest(
    strategy_type='hybrid_bollinger',
    window=20,
    num_std=2,
    trend_window=200,
    use_vol_sizing=True,
    target_vol=0.12,
    add_exits=True,
    profit_target=0.03,
    stop_loss=0.02
)

bt.print_results()
bt.plot_results()
