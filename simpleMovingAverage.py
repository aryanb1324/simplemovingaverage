import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedSMABacktester:
    """
    A comprehensive backtesting system for Simple Moving Average trading strategies
    with diagnostic tools and strategy improvements.
    """
    
    def __init__(self, symbol='SPY', start_date='2020-01-01', end_date='2024-01-01'):
        """
        Initialize the backtester with symbol and date range.
        
        Parameters:
        symbol (str): Stock symbol to backtest
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results = {}
        self.diagnostics = {}
        
    def fetch_data(self):
        """Download historical price data using yfinance."""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            self.data = self.data.dropna()
            print(f"Successfully downloaded {len(self.data)} days of data for {self.symbol}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        return True
    
    def calculate_sma(self, short_window=20, long_window=50, ma_type='SMA'):
        """
        Calculate Simple or Exponential Moving Averages.
        
        Parameters:
        short_window (int): Short MA period
        long_window (int): Long MA period
        ma_type (str): 'SMA' or 'EMA'
        """
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
            
        if ma_type == 'SMA':
            self.data['MA_short'] = self.data['Close'].rolling(window=short_window).mean()
            self.data['MA_long'] = self.data['Close'].rolling(window=long_window).mean()
        elif ma_type == 'EMA':
            self.data['MA_short'] = self.data['Close'].ewm(span=short_window).mean()
            self.data['MA_long'] = self.data['Close'].ewm(span=long_window).mean()
        
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type
        
    def generate_signals(self, strategy='basic', confirmation_days=1, vol_threshold=0.0):
        """
        Generate trading signals with various enhancements.
        
        Parameters:
        strategy (str): 'basic', 'confirmed', 'filtered'
        confirmation_days (int): Days of confirmation required
        vol_threshold (float): Minimum volatility threshold
        """
        self.data['signal'] = 0
        self.data['signal_raw'] = 0
        
        self.data.loc[self.data['MA_short'] > self.data['MA_long'], 'signal_raw'] = 1
        self.data.loc[self.data['MA_short'] < self.data['MA_long'], 'signal_raw'] = -1
        
        if strategy == 'basic':
            self.data['signal'] = self.data['signal_raw']
            
        elif strategy == 'confirmed':
            self.data['signal'] = self.data['signal_raw'].copy()
            for i in range(confirmation_days, len(self.data)):
                if not all(self.data['signal_raw'].iloc[i-confirmation_days:i+1] == 
                          self.data['signal_raw'].iloc[i]):
                    self.data.iloc[i, self.data.columns.get_loc('signal')] = 0
                    
        elif strategy == 'filtered':
            self.add_volatility_filter_scaling(vol_threshold)
            self.add_trend_filter()
            
        self.data['position'] = self.data['signal'].shift(1)
        self.data['position'] = self.data['position'].fillna(0)
        
    def add_volatility_filter_scaling(self, vol_threshold=0.01):
        """Scale signals based on volatility instead of eliminating them"""
        if 'returns' not in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        
        scaling = (self.data['volatility'] / vol_threshold).clip(0, 1)
        self.data['signal'] = self.data['signal_raw'] * scaling
        
    def add_volatility_filter(self, vol_threshold=0.01):
        """Only trade when volatility is above threshold (original method)"""
        if 'returns' not in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        
        low_vol_mask = self.data['volatility'] < vol_threshold
        self.data.loc[low_vol_mask, 'signal'] = 0
        
    def add_trend_filter(self, trend_threshold=0.02):
        """Use trend strength to filter signals"""
        self.data['MA_200'] = self.data['Close'].rolling(200).mean()
        self.data['trend_strength'] = abs(self.data['Close'] - self.data['MA_200']) / self.data['MA_200']
        
        weak_trend = self.data['trend_strength'] < trend_threshold
        self.data.loc[weak_trend, 'signal'] = 0
        
    def triple_ma_system(self, fast=10, medium=20, slow=50):
        """More robust signals with three MAs"""
        self.data['MA_fast'] = self.data['Close'].rolling(fast).mean()
        self.data['MA_medium'] = self.data['Close'].rolling(medium).mean()
        self.data['MA_slow'] = self.data['Close'].rolling(slow).mean()
        
        bullish = ((self.data['MA_fast'] > self.data['MA_medium']) & 
                   (self.data['MA_medium'] > self.data['MA_slow']))
        bearish = ((self.data['MA_fast'] < self.data['MA_medium']) & 
                   (self.data['MA_medium'] < self.data['MA_slow']))
        
        self.data['signal'] = np.where(bullish, 1, np.where(bearish, -1, 0))
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        
    def bollinger_bands_strategy(self, window=20, num_std=2):
        """Mean reversion strategy using Bollinger Bands"""
        self.data['BB_middle'] = self.data['Close'].rolling(window).mean()
        self.data['BB_std'] = self.data['Close'].rolling(window).std()
        self.data['BB_upper'] = self.data['BB_middle'] + num_std * self.data['BB_std']
        self.data['BB_lower'] = self.data['BB_middle'] - num_std * self.data['BB_std']
        
        self.data['signal'] = np.where(
            self.data['Close'] < self.data['BB_lower'], 1,  
            np.where(self.data['Close'] > self.data['BB_upper'], -1, 0)  
        )
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        
    def hybrid_bollinger_strategy(self, window=20, num_std=2, trend_window=200):
        """
        NEW: Hybrid Bollinger Bands with trend filter
        Combines mean reversion with long-term trend filtering
        """
        self.data['BB_middle'] = self.data['Close'].rolling(window).mean()
        self.data['BB_std'] = self.data['Close'].rolling(window).std()
        self.data['BB_upper'] = self.data['BB_middle'] + num_std * self.data['BB_std']
        self.data['BB_lower'] = self.data['BB_middle'] - num_std * self.data['BB_std']
        self.data['MA_trend'] = self.data['Close'].rolling(trend_window).mean()

        long_condition = (self.data['Close'] < self.data['BB_lower']) & (self.data['Close'] > self.data['MA_trend'])
        short_condition = (self.data['Close'] > self.data['BB_upper']) & (self.data['Close'] < self.data['MA_trend'])
        
        self.data['signal'] = np.where(long_condition, 1, np.where(short_condition, -1, 0))
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        
    def enhanced_momentum_strategy(self, lookback=12, z_score_window=252):
        """
        IMPROVED: Enhanced momentum strategy using z-scores and longer lookback
        """
        self.data['momentum'] = self.data['Close'].pct_change(lookback)
        
        self.data['momentum_z'] = (
            (self.data['momentum'] - self.data['momentum'].rolling(z_score_window).mean()) /
            self.data['momentum'].rolling(z_score_window).std()
        )
        
        self.data['signal'] = np.where(
            self.data['momentum_z'] > 1.0, 1,
            np.where(self.data['momentum_z'] < -1.0, -1, 0)
        )
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        
    def momentum_strategy(self, lookback=12, quantile_threshold=0.75):
        """Original momentum strategy (kept for compatibility)"""
        self.data['momentum'] = self.data['Close'].pct_change(lookback)
        
        momentum_high = self.data['momentum'].rolling(252).quantile(quantile_threshold)
        momentum_low = self.data['momentum'].rolling(252).quantile(1 - quantile_threshold)
        
        self.data['signal'] = np.where(
            self.data['momentum'] > momentum_high, 1,
            np.where(self.data['momentum'] < momentum_low, -1, 0)
        )
        self.data['position'] = self.data['signal'].shift(1).fillna(0)
        
    def ensemble_strategy(self, weights=None):
        """
        NEW: Ensemble strategy combining multiple signals
        """
        if weights is None:
            weights = {'sma': 0.4, 'bollinger': 0.3, 'momentum': 0.3}
            
        original_data = self.data.copy()
        
        self.calculate_sma(20, 50)
        self.generate_signals('basic')
        sma_signal = self.data['signal'].copy()
        
        self.data = original_data.copy()
        self.bollinger_bands_strategy()
        bollinger_signal = self.data['signal'].copy()
        
        self.data = original_data.copy()
        self.enhanced_momentum_strategy()
        momentum_signal = self.data['signal'].copy()
        
        self.data = original_data.copy()
        self.data['signal'] = (
            sma_signal * weights['sma'] +
            bollinger_signal * weights['bollinger'] +
            momentum_signal * weights['momentum']
        )
        
        self.data['position'] = self.data['signal'].shift(1).fillna(0).clip(-1, 1)
        
    def add_exit_rules(self, profit_target=0.03, stop_loss=0.02, time_exit=5):
        """
        NEW: Add profit target, stop loss, and time-based exits
        """
        if 'returns' not in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            
        rolling_returns = self.data['returns'].rolling(time_exit).sum()
        
        profit_exit = rolling_returns > profit_target
        loss_exit = rolling_returns < -stop_loss
        
        exit_mask = profit_exit | loss_exit
        self.data.loc[exit_mask, 'position'] = 0
        
    def add_macro_filter(self, vol_panic_threshold=0.02):
        """
        NEW: Add macro/risk-off filter to avoid trading during high volatility periods
        """
        if 'volatility' not in self.data.columns:
            if 'returns' not in self.data.columns:
                self.data['returns'] = self.data['Close'].pct_change()
            self.data['volatility'] = self.data['returns'].rolling(20).std()

        panic_periods = self.data['volatility'].rolling(5).mean() > vol_panic_threshold
        self.data.loc[panic_periods, 'position'] = 0
        
    def dynamic_position_sizing(self, target_vol=0.15):
        """Adjust position size based on volatility"""
        if 'returns' not in self.data.columns:
            self.data['returns'] = self.data['Close'].pct_change()
            
        self.data['rolling_vol'] = self.data['returns'].rolling(20).std() * np.sqrt(252)
        
        vol_adjustment = target_vol / self.data['rolling_vol']
        vol_adjustment = vol_adjustment.fillna(1).clip(0.1, 2.0) 
        
        self.data['vol_adjusted_position'] = (self.data['position'] * vol_adjustment).clip(-1, 1)
        
    def calculate_returns(self, transaction_cost=0.001, use_vol_sizing=False):
        """
        Calculate strategy returns and cumulative performance.
        
        Parameters:
        transaction_cost (float): Transaction cost as a fraction
        use_vol_sizing (bool): Use volatility-adjusted position sizing
        """
        self.data['returns'] = self.data['Close'].pct_change()
        
        position_col = 'vol_adjusted_position' if use_vol_sizing and 'vol_adjusted_position' in self.data.columns else 'position'
        
        self.data['position_change'] = self.data[position_col].diff().abs()
        
        self.data['strategy_returns_gross'] = self.data[position_col] * self.data['returns']
        
        transaction_costs = self.data['position_change'] * transaction_cost
        self.data['strategy_returns'] = self.data['strategy_returns_gross'] - transaction_costs
        
        self.data['cum_returns'] = (1 + self.data['returns']).cumprod()
        self.data['cum_strategy'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cum_benchmark'] = self.data['cum_returns']
        
    def analyze_market_conditions(self):
        """Analyze market conditions and strategy suitability"""
        if self.data is None or 'returns' not in self.data.columns:
            return {}
            
        trend_up_days = (self.data['Close'] > self.data['Close'].shift(20)).sum()
        sideways_ratio = 1 - (trend_up_days / len(self.data))
        
        if 'position' in self.data.columns:
            signal_changes = self.data['position'].diff().abs().sum()
            avg_holding_period = len(self.data) / max(signal_changes, 1)
        else:
            signal_changes = 0
            avg_holding_period = 0
            
        if 'volatility' in self.data.columns:
            avg_volatility = self.data['volatility'].mean() * np.sqrt(252)
            vol_stability = self.data['volatility'].std() / self.data['volatility'].mean()
        else:
            avg_volatility = self.data['returns'].std() * np.sqrt(252)
            vol_stability = 0
            
        if sideways_ratio > 0.6:
            market_regime = "Sideways/Choppy"
        elif avg_volatility > 0.25:
            market_regime = "High Volatility"
        else:
            market_regime = "Trending"
            
        self.diagnostics = {
            'market_regime': market_regime,
            'sideways_market_ratio': round(sideways_ratio, 3),
            'signal_frequency': int(signal_changes),
            'avg_holding_period': round(avg_holding_period, 1),
            'annualized_volatility': round(avg_volatility, 3),
            'volatility_stability': round(vol_stability, 3)
        }
        
        return self.diagnostics
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if 'strategy_returns' not in self.data.columns:
            raise ValueError("Returns not calculated. Please run calculate_returns first.")
            
        strategy_returns = self.data['strategy_returns'].dropna()
        benchmark_returns = self.data['returns'].dropna()

        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[-min_length:]
        benchmark_returns = benchmark_returns[-min_length:]
        
        total_return_strategy = (self.data['cum_strategy'].iloc[-1] - 1) * 100
        total_return_benchmark = (self.data['cum_benchmark'].iloc[-1] - 1) * 100
        
        trading_days = 252
        years = len(strategy_returns) / trading_days
        
        annualized_return_strategy = ((1 + strategy_returns.mean()) ** trading_days - 1) * 100
        annualized_return_benchmark = ((1 + benchmark_returns.mean()) ** trading_days - 1) * 100
        
        volatility_strategy = strategy_returns.std() * np.sqrt(trading_days) * 100
        volatility_benchmark = benchmark_returns.std() * np.sqrt(trading_days) * 100
        
        risk_free_rate = 0.02
        sharpe_strategy = (annualized_return_strategy/100 - risk_free_rate) / (volatility_strategy/100) if volatility_strategy > 0 else 0
        sharpe_benchmark = (annualized_return_benchmark/100 - risk_free_rate) / (volatility_benchmark/100) if volatility_benchmark > 0 else 0
        
        def calculate_max_drawdown(cum_returns):
            peak = cum_returns.expanding().max()
            drawdown = (cum_returns - peak) / peak
            return drawdown.min() * 100
        
        max_dd_strategy = calculate_max_drawdown(self.data['cum_strategy'])
        max_dd_benchmark = calculate_max_drawdown(self.data['cum_benchmark'])
        
        winning_trades = len(strategy_returns[strategy_returns > 0])
        losing_trades = len(strategy_returns[strategy_returns < 0])
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = strategy_returns[strategy_returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = strategy_returns[strategy_returns < 0].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        
        self.results = {
            'Total Return Strategy (%)': round(total_return_strategy, 2),
            'Total Return Benchmark (%)': round(total_return_benchmark, 2),
            'Annualized Return Strategy (%)': round(annualized_return_strategy, 2),
            'Annualized Return Benchmark (%)': round(annualized_return_benchmark, 2),
            'Volatility Strategy (%)': round(volatility_strategy, 2),
            'Volatility Benchmark (%)': round(volatility_benchmark, 2),
            'Sharpe Ratio Strategy': round(sharpe_strategy, 3),
            'Sharpe Ratio Benchmark': round(sharpe_benchmark, 3),
            'Max Drawdown Strategy (%)': round(max_dd_strategy, 2),
            'Max Drawdown Benchmark (%)': round(max_dd_benchmark, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 3),
            'Total Trades': total_trades
        }
        
        return self.results
    
    def robustness_testing(self):
        """Test strategy across different market conditions"""
        if len(self.data) < 500:
            print("Insufficient data for robustness testing")
            return
            
        periods = []
        data_length = len(self.data)
        
        period_length = data_length // 4
        for i in range(4):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < 3 else data_length
            periods.append((start_idx, end_idx, f"Period {i+1}"))
            
        print("\nRobustness Testing Results:")
        print("=" * 50)
        
        current_data = self.data.copy()
        
        for start_idx, end_idx, period_name in periods:
            self.data = current_data.iloc[start_idx:end_idx].copy()
            
            try:
                self.calculate_returns()
                metrics = self.calculate_performance_metrics()
                print(f"{period_name}: Sharpe = {metrics['Sharpe Ratio Strategy']:.2f}, "
                      f"Return = {metrics['Annualized Return Strategy (%)']:.1f}%")
            except:
                print(f"{period_name}: Insufficient data for testing")
        
        self.data = current_data 
        print("=" * 50)
        
    def run_backtest(self, strategy_type='basic', short_window=20, long_window=50, 
                    ma_type='SMA', transaction_cost=0.001, **kwargs):
        """
        Run the complete backtest process with various strategy options.
        
        Parameters:
        strategy_type (str): 'basic', 'confirmed', 'filtered', 'triple_ma', 'bollinger', 
                           'hybrid_bollinger', 'momentum', 'enhanced_momentum', 'ensemble'
        short_window (int): Short MA period
        long_window (int): Long MA period
        ma_type (str): 'SMA' or 'EMA'
        transaction_cost (float): Transaction cost as a fraction
        **kwargs: Additional parameters for specific strategies
        """
        print(f"Running {strategy_type} backtest for {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Transaction cost: {transaction_cost*100:.2f}%")
        print("-" * 50)
        
        if not self.fetch_data():
            return None
            
        if strategy_type == 'triple_ma':
            fast = kwargs.get('fast', 10)
            medium = kwargs.get('medium', 20)
            slow = kwargs.get('slow', 50)
            print(f"Triple MA windows: {fast} / {medium} / {slow}")
            self.triple_ma_system(fast, medium, slow)
            
        elif strategy_type == 'bollinger':
            window = kwargs.get('window', 20)
            num_std = kwargs.get('num_std', 2)
            print(f"Bollinger Bands: {window} period, {num_std} std dev")
            self.bollinger_bands_strategy(window, num_std)
            
        elif strategy_type == 'hybrid_bollinger':
            window = kwargs.get('window', 20)
            num_std = kwargs.get('num_std', 2)
            trend_window = kwargs.get('trend_window', 200)
            print(f"Hybrid Bollinger: {window} period, {num_std} std dev, {trend_window} trend filter")
            self.hybrid_bollinger_strategy(window, num_std, trend_window)
            
        elif strategy_type == 'momentum':
            lookback = kwargs.get('lookback', 12)
            quantile = kwargs.get('quantile_threshold', 0.75)
            print(f"Momentum: {lookback} day lookback, {quantile} quantile")
            self.momentum_strategy(lookback, quantile)
            
        elif strategy_type == 'enhanced_momentum':
            lookback = kwargs.get('lookback', 12)
            z_score_window = kwargs.get('z_score_window', 252)
            print(f"Enhanced Momentum: {lookback} day lookback, {z_score_window} z-score window")
            self.enhanced_momentum_strategy(lookback, z_score_window)
            
        elif strategy_type == 'ensemble':
            weights = kwargs.get('weights', {'sma': 0.4, 'bollinger': 0.3, 'momentum': 0.3})
            print(f"Ensemble weights: {weights}")
            self.ensemble_strategy(weights)
            
        else:
            print(f"{ma_type} windows: {short_window} / {long_window}")
            self.calculate_sma(short_window, long_window, ma_type)
            
            confirmation_days = kwargs.get('confirmation_days', 1)
            vol_threshold = kwargs.get('vol_threshold', 0.0)
            
            if strategy_type == 'confirmed':
                print(f"Confirmation days: {confirmation_days}")
            elif strategy_type == 'filtered':
                print(f"Volatility threshold: {vol_threshold:.3f}")
                
            self.generate_signals(strategy_type, confirmation_days, vol_threshold)
        
        if kwargs.get('add_exits', False):
            profit_target = kwargs.get('profit_target', 0.03)
            stop_loss = kwargs.get('stop_loss', 0.02)
            time_exit = kwargs.get('time_exit', 5)
            print(f"Adding exits: profit={profit_target:.1%}, stop={stop_loss:.1%}, time={time_exit}d")
            self.add_exit_rules(profit_target, stop_loss, time_exit)
            
        if kwargs.get('add_macro_filter', False):
            vol_panic_threshold = kwargs.get('vol_panic_threshold', 0.02)
            print(f"Adding macro filter: volatility threshold={vol_panic_threshold:.1%}")
            self.add_macro_filter(vol_panic_threshold)
        
        use_vol_sizing = kwargs.get('use_vol_sizing', False)
        if use_vol_sizing:
            target_vol = kwargs.get('target_vol', 0.15)
            print(f"Using volatility sizing, target vol: {target_vol:.1%}")
            self.dynamic_position_sizing(target_vol)
            
        self.calculate_returns(transaction_cost, use_vol_sizing)
        results = self.calculate_performance_metrics()
        
        diagnostics = self.analyze_market_conditions()
        
        print("Backtest completed successfully!")
        return results, diagnostics
    
    def plot_results(self, figsize=(16, 12)):
        """Create comprehensive visualization of backtest results."""
        if self.data is None:
            raise ValueError("No data to plot. Please run backtest first.")
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'{self.symbol} - Enhanced Strategy Backtest Results', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(self.data.index, self.data['Close'], label='Price', alpha=0.7, linewidth=1)
        
        if 'MA_short' in self.data.columns:
            ax1.plot(self.data.index, self.data['MA_short'], label=f'MA Short', color='orange', alpha=0.8)
            ax1.plot(self.data.index, self.data['MA_long'], label=f'MA Long', color='red', alpha=0.8)
        
        if 'BB_upper' in self.data.columns:
            ax1.plot(self.data.index, self.data['BB_upper'], label='BB Upper', color='gray', alpha=0.5)
            ax1.plot(self.data.index, self.data['BB_lower'], label='BB Lower', color='gray', alpha=0.5)
            ax1.fill_between(self.data.index, self.data['BB_upper'], self.data['BB_lower'], alpha=0.1, color='gray')
        
        if 'position' in self.data.columns:
            buy_signals = self.data[self.data['position'] == 1].index
            sell_signals = self.data[self.data['position'] == -1].index
            
            ax1.scatter(buy_signals, self.data.loc[buy_signals, 'Close'], 
                       marker='^', color='green', s=30, alpha=0.6, label='Buy')
            ax1.scatter(sell_signals, self.data.loc[sell_signals, 'Close'], 
                       marker='v', color='red', s=30, alpha=0.6, label='Sell')
        
        ax1.set_title('Price and Signals')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(self.data.index, (self.data['cum_strategy'] - 1) * 100, 
                label='Strategy', linewidth=2)
        ax2.plot(self.data.index, (self.data['cum_benchmark'] - 1) * 100, 
                label='Benchmark', linewidth=2, alpha=0.8)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        strategy_peak = self.data['cum_strategy'].expanding().max()
        strategy_dd = (self.data['cum_strategy'] - strategy_peak) / strategy_peak * 100
        
        benchmark_peak = self.data['cum_benchmark'].expanding().max()
        benchmark_dd = (self.data['cum_benchmark'] - benchmark_peak) / benchmark_peak * 100
        
        ax3.fill_between(self.data.index, strategy_dd, 0, alpha=0.3, label='Strategy DD', color='red')
        ax3.fill_between(self.data.index, benchmark_dd, 0, alpha=0.3, label='Benchmark DD', color='blue')
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 0]
        if len(self.data) > 252:
            rolling_window = 252
            strategy_rolling_returns = self.data['strategy_returns'].rolling(rolling_window)
            benchmark_rolling_returns = self.data['returns'].rolling(rolling_window)
            
            strategy_sharpe = (strategy_rolling_returns.mean() * 252) / (strategy_rolling_returns.std() * np.sqrt(252))
            benchmark_sharpe = (benchmark_rolling_returns.mean() * 252) / (benchmark_rolling_returns.std() * np.sqrt(252))
            
            ax4.plot(self.data.index, strategy_sharpe, label='Strategy', alpha=0.8)
            ax4.plot(self.data.index, benchmark_sharpe, label='Benchmark', alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Rolling 1Y Sharpe Ratio')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        ax5 = axes[1, 1]
        if 'position' in self.data.columns:
            ax5_twin = ax5.twinx()
            
            ax5.plot(self.data.index, self.data['position'], label='Position', 
                    color='green', alpha=0.7, linewidth=1)
            ax5.set_ylabel('Position', color='green')
            ax5.set_ylim(-1.2, 1.2)
            
            if 'volatility' in self.data.columns:
                vol_annualized = self.data['volatility'] * np.sqrt(252) * 100
                ax5_twin.plot(self.data.index, vol_annualized, label='Volatility', 
                            color='orange', alpha=0.6, linewidth=1)
                ax5_twin.set_ylabel('Volatility (%)', color='orange')
            
            ax5.set_title('Position & Volatility')
            ax5.grid(True, alpha=0.3)
        
        ax6 = axes[1, 2]
        if 'strategy_returns' in self.data.columns:
            monthly_returns = self.data['strategy_returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            
            if len(monthly_returns) > 12:
                monthly_data = monthly_returns.to_frame()
                monthly_data.index = pd.to_datetime(monthly_data.index)
                monthly_data['Year'] = monthly_data.index.year
                monthly_data['Month'] = monthly_data.index.month
                
                pivot_data = monthly_data.pivot(index='Year', columns='Month', values='strategy_returns')
                
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                           center=0, ax=ax6, cbar_kws={'label': 'Return (%)'})
                ax6.set_title('Monthly Returns Heatmap')
            else:
                monthly_returns.plot(kind='bar', ax=ax6, alpha=0.7)
                ax6.set_title('Monthly Returns')
                ax6.set_ylabel('Return (%)')
                plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def print_results(self):
        """Print formatted results and diagnostics."""
        if not self.results:
            print("No results to display. Please run backtest first.")
            return
            
        print("\n" + "="*70)
        print(f"BACKTEST RESULTS - {self.symbol}")
        if hasattr(self, 'ma_type'):
            print(f"{self.ma_type} Windows: {getattr(self, 'short_window', 'N/A')} / {getattr(self, 'long_window', 'N/A')}")
        print("="*70)
        
        for metric, value in self.results.items():
            print(f"{metric:<40}: {value}")

        if self.diagnostics:
            print("\nMARKET DIAGNOSTICS:")
            print("-" * 40)
            for metric, value in self.diagnostics.items():
                print(f"{metric.replace('_', ' ').title():<30}: {value}")
        
        print("\nSTRATEGY RECOMMENDATIONS:")
        print("-" * 40)
        self._generate_recommendations()
        
        print("="*70)
        
    def _generate_recommendations(self):
        """Generate strategy improvement recommendations based on diagnostics"""
        if not self.diagnostics or not self.results:
            return
            
        sharpe = self.results.get('Sharpe Ratio Strategy', 0)
        market_regime = self.diagnostics.get('market_regime', '')
        sideways_ratio = self.diagnostics.get('sideways_market_ratio', 0)
        signal_freq = self.diagnostics.get('signal_frequency', 0)
        win_rate = self.results.get('Win Rate (%)', 0)
        
        recommendations = []
        
        if sharpe < 0.5:
            if market_regime == "Sideways/Choppy":
                recommendations.append("• Consider hybrid_bollinger strategy instead of trend following")
                recommendations.append("• Add volatility filters to reduce whipsaws")
            else:
                recommendations.append("• Try enhanced_momentum strategy or longer MA periods")
                recommendations.append("• Add confirmation days to reduce false signals")
        
        if signal_freq > len(self.data) * 0.1: 
            recommendations.append("• Reduce whipsaws with confirmation periods")
            recommendations.append("• Consider EMA instead of SMA for smoother signals")
            recommendations.append("• Add trend strength filters")
        
        if win_rate < 45:
            recommendations.append("• Improve entry timing with additional indicators")
            recommendations.append("• Consider ensemble strategy for better signal quality")
        
        if sideways_ratio > 0.6:
            recommendations.append("• Market is choppy - try hybrid_bollinger strategy")
        elif sideways_ratio < 0.3:
            recommendations.append("• Strong trending market - optimize MA periods")
        
        if sharpe > 0 and sharpe < 1.0:
            recommendations.append("• Add volatility-based position sizing")
            recommendations.append("• Consider exit rules for better risk management")
        
        if not recommendations:
            recommendations.append("• Strategy performing well - test robustness across periods")
            recommendations.append("• Consider portfolio diversification")
        
        for rec in recommendations[:5]: 
            print(rec)
    
    def parameter_optimization(self, param_ranges=None, optimization_metric='sharpe'):
        """
        Perform parameter optimization over specified ranges.
        
        Parameters:
        param_ranges (dict): Dictionary of parameter ranges
        optimization_metric (str): Metric to optimize ('sharpe', 'return', 'profit_factor')
        """
        if param_ranges is None:
            param_ranges = {
                'short_window': range(5, 50, 5),
                'long_window': range(20, 200, 20)
            }
        
        print("Starting parameter optimization...")
        print(f"Optimization metric: {optimization_metric}")
        
        results_matrix = []
        short_windows = param_ranges['short_window']
        long_windows = param_ranges['long_window']
        
        original_data = self.data.copy()
        
        for short_win in short_windows:
            row = []
            for long_win in long_windows:
                if short_win >= long_win: 
                    row.append(np.nan)
                    continue
                    
                try:
                    self.data = original_data.copy()
                    
                    self.calculate_sma(short_win, long_win)
                    self.generate_signals()
                    self.calculate_returns()
                    metrics = self.calculate_performance_metrics()
                    
                    if optimization_metric == 'sharpe':
                        score = metrics.get('Sharpe Ratio Strategy', 0)
                    elif optimization_metric == 'return':
                        score = metrics.get('Annualized Return Strategy (%)', 0)
                    elif optimization_metric == 'profit_factor':
                        score = metrics.get('Profit Factor', 0)
                    else:
                        score = metrics.get('Sharpe Ratio Strategy', 0)
                    
                    row.append(score)
                    
                except Exception as e:
                    row.append(np.nan)
            
            results_matrix.append(row)
        
        self.data = original_data
        
        results_df = pd.DataFrame(results_matrix, 
                                index=short_windows, 
                                columns=long_windows)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(results_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': optimization_metric.title()})
        plt.title(f'{self.symbol} - SMA Parameter Optimization ({optimization_metric.title()})')
        plt.xlabel('Long Window')
        plt.ylabel('Short Window')
        plt.show()
        
        best_score = results_df.max().max()
        best_params = results_df.stack().idxmax()
        
        print(f"Best parameters: Short={best_params[0]}, Long={best_params[1]}")
        print(f"Best {optimization_metric}: {best_score:.3f}")
        
        return results_df, best_params

def run_strategy_comparison(symbol='SPY', start_date='2020-01-01', end_date='2024-01-01'):
    """
    Compare different strategy variations on the same dataset.
    """
    print(f"Strategy Comparison for {symbol}")
    print("=" * 60)
    
    strategies = [
        ('Basic SMA', {'strategy_type': 'basic', 'short_window': 20, 'long_window': 50}),
        ('Confirmed SMA', {'strategy_type': 'confirmed', 'short_window': 20, 'long_window': 50, 'confirmation_days': 3}),
        ('Filtered SMA (Fixed)', {'strategy_type': 'filtered', 'short_window': 20, 'long_window': 50, 'vol_threshold': 0.01}),
        ('EMA Cross', {'strategy_type': 'basic', 'short_window': 12, 'long_window': 26, 'ma_type': 'EMA'}),
        ('Triple MA', {'strategy_type': 'triple_ma', 'fast': 10, 'medium': 20, 'slow': 50}),
        ('Bollinger Bands', {'strategy_type': 'bollinger', 'window': 20, 'num_std': 2}),
        ('Hybrid Bollinger', {'strategy_type': 'hybrid_bollinger', 'window': 20, 'num_std': 2, 'trend_window': 200}),
        ('Enhanced Momentum', {'strategy_type': 'enhanced_momentum', 'lookback': 12, 'z_score_window': 252}),
        ('Ensemble Strategy', {'strategy_type': 'ensemble', 'weights': {'sma': 0.4, 'bollinger': 0.3, 'momentum': 0.3}})
    ]
    
    results_summary = []
    
    for strategy_name, params in strategies:
        print(f"\nTesting {strategy_name}...")
        
        backtester = EnhancedSMABacktester(symbol, start_date, end_date)
        
        try:
            results, diagnostics = backtester.run_backtest(**params)
            
            if results:
                results_summary.append({
                    'Strategy': strategy_name,
                    'Annual Return (%)': results['Annualized Return Strategy (%)'],
                    'Volatility (%)': results['Volatility Strategy (%)'],
                    'Sharpe Ratio': results['Sharpe Ratio Strategy'],
                    'Max DD (%)': results['Max Drawdown Strategy (%)'],
                    'Win Rate (%)': results['Win Rate (%)'],
                    'Total Trades': results['Total Trades'],
                    'Market Regime': diagnostics.get('market_regime', 'Unknown')
                })
        
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")
            continue
    
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        comparison_df = comparison_df.set_index('Strategy')
        
        print("\n" + "=" * 90)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 90)
        print(comparison_df.round(2))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        comparison_df['Sharpe Ratio'].plot(kind='bar', ax=axes[0, 0], title='Sharpe Ratio Comparison')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].scatter(comparison_df['Volatility (%)'], comparison_df['Annual Return (%)'])
        for i, strategy in enumerate(comparison_df.index):
            axes[0, 1].annotate(strategy, 
                               (comparison_df['Volatility (%)'].iloc[i], 
                                comparison_df['Annual Return (%)'].iloc[i]),
                               fontsize=8, ha='left')
        axes[0, 1].set_xlabel('Volatility (%)')
        axes[0, 1].set_ylabel('Annual Return (%)')
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        comparison_df['Max DD (%)'].plot(kind='bar', ax=axes[1, 0], 
                                        title='Maximum Drawdown Comparison', color='red', alpha=0.7)
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        comparison_df['Win Rate (%)'].plot(kind='bar', ax=axes[1, 1], 
                                          title='Win Rate Comparison', color='green', alpha=0.7)
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\nTOP PERFORMING STRATEGIES:")
        print("-" * 40)
        top_sharpe = comparison_df.nlargest(3, 'Sharpe Ratio')
        for i, (strategy, data) in enumerate(top_sharpe.iterrows(), 1):
            print(f"{i}. {strategy}: Sharpe = {data['Sharpe Ratio']:.2f}, "
                  f"Return = {data['Annual Return (%)']:.1f}%, Trades = {data['Total Trades']}")
    
    return comparison_df if results_summary else None

def main():
    """Enhanced demonstration of the backtesting system with new strategies."""
    
    print("Enhanced SMA Backtesting System - Updated with Improvements")
    print("==========================================================")
    
    backtester = EnhancedSMABacktester('SPY', '2020-01-01', '2024-01-01')
    
    print("\n1. Testing Hybrid Bollinger Strategy...")
    results, diagnostics = backtester.run_backtest(
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
    
    if results:
        backtester.print_results()
        
    print("\n2. Testing Ensemble Strategy...")
    backtester2 = EnhancedSMABacktester('SPY', '2020-01-01', '2024-01-01')
    results2, diagnostics2 = backtester2.run_backtest(
        strategy_type='ensemble',
        weights={'sma': 0.4, 'bollinger': 0.3, 'momentum': 0.3},
        use_vol_sizing=True
    )
    
    if results2:
        backtester2.print_results()
    
    print("\n3. Running Comprehensive Strategy Comparison...")
    comparison_results = run_strategy_comparison('SPY', '2020-01-01', '2024-01-01')
    
    print("\nDemo completed! Key improvements implemented:")
    print("• ✅ Hybrid Bollinger strategy with trend filter")
    print("• ✅ Fixed filtered SMA strategy (scaling vs elimination)")
    print("• ✅ Enhanced momentum with z-scores")
    print("• ✅ Ensemble strategy combining multiple signals")
    print("• ✅ Exit rules (profit targets, stop losses, time-based)")
    print("• ✅ Macro volatility filters")
    print("• ✅ Updated recommendations system")

if __name__ == "__main__":
    main()