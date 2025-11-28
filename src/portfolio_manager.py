import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_collection import get_data
from data_collection_fred import get_fred_data
from allocation_strategy import AllocationStrategy


class PortfolioManager:
    def __init__(self, symbols, allocation_strategy: AllocationStrategy, start_date, end_date, fred_series_to_fetch=None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.asset_data_with_signals = {}
        self.portfolio_returns = None
        self.aligned_data = {}
        self.raw_data = {}
        self.macro_data = None
        self.allocation_strategy = allocation_strategy
        self.fred_series_to_fetch = fred_series_to_fetch if fred_series_to_fetch is not None else {}

    def _fetch_all_data(self):
        """Fetches historical data for all symbols."""
        print("Fetching market data for all assets...")
        for symbol in self.symbols[:]:
            try:
                self.raw_data[symbol] = get_data(symbol, self.start_date, self.end_date)
            except Exception as e:
                print(f"Could not fetch market data for {symbol}: {e}")
                self.symbols.remove(symbol)
        if not self.raw_data:
            raise ValueError("No market data fetched for any symbol. Exiting.")
        return self.raw_data

    def _fetch_macro_data(self):
        """Fetches macroeconomic data from FRED if series are specified."""
        if not self.fred_series_to_fetch:
            print("No FRED series specified. Skipping macro data fetch.")
            self.macro_data = None
        else:
            print("Fetching macroeconomic data...")
            self.macro_data = get_fred_data(series_ids=self.fred_series_to_fetch, start_date=self.start_date, end_date=self.end_date)
        return self.macro_data

    def _align_data(self):
        """Aligns all asset data to a common date range."""
        if not self.asset_data_with_signals:
            raise ValueError("No asset data with signals to align.")
        
        close_df = pd.DataFrame({sym: df['Close'] for sym, df in self.asset_data_with_signals.items()})
        close_df.dropna(inplace=True)
        aligned_index = close_df.index
        
        aligned_dfs = {}
        for symbol, df in self.asset_data_with_signals.items():
            aligned_dfs[symbol] = df.reindex(aligned_index).ffill()

        self.aligned_data = aligned_dfs
        self.symbols = list(self.aligned_data.keys())

    def _calculate_portfolio_performance(self, weights_df):
        """Calculates daily and cumulative returns for the portfolio and a benchmark."""
        print("Calculating portfolio and benchmark returns...")
        
        asset_returns = pd.DataFrame({s: self.aligned_data[s]['Close'].pct_change() for s in self.symbols})
        asset_returns = asset_returns.loc[weights_df.index]

        num_assets = len(self.symbols)

        # --- Strategy Performance ---
        portfolio_daily_returns = (weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # --- BENCHMARK: Static Risk Parity (Monthly Rebalanced) ---
        print("  - Calculating static risk parity benchmark...")
        rp_weights_df = pd.DataFrame(index=asset_returns.index, columns=self.symbols).fillna(0.0)
        rebalance_dates = asset_returns.resample('BMS').first().index
        
        last_rp_weights = np.array([1/num_assets] * num_assets)
        if not asset_returns.empty:
            if asset_returns.index[0] not in rebalance_dates:
                rp_weights_df.loc[asset_returns.index[0]] = last_rp_weights

        for date in asset_returns.index:
            if date in rebalance_dates:
                hist_returns = asset_returns.loc[:date].tail(63)
                if len(hist_returns) > 1:
                    inv_vol = 1 / hist_returns.std()
                    if inv_vol.sum() > 0:
                        last_rp_weights = inv_vol / inv_vol.sum()
            rp_weights_df.loc[date] = last_rp_weights
        
        rp_weights_df.ffill(inplace=True)
        risk_parity_daily_returns = (rp_weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        risk_parity_cumulative_returns = (1 + risk_parity_daily_returns).cumprod()

        # --- BENCHMARK: Monthly Rebalanced Equal Weight ---
        print("  - Calculating monthly rebalanced equal-weight benchmark...")
        ew_rebalanced_weights_df = pd.DataFrame(index=asset_returns.index, columns=self.symbols)
        equal_weight = np.array([1/num_assets] * num_assets)
        
        for date in rebalance_dates:
            if date in ew_rebalanced_weights_df.index:
                ew_rebalanced_weights_df.loc[date] = equal_weight
        
        ew_rebalanced_weights_df.iloc[0] = equal_weight
        ew_rebalanced_weights_df.ffill(inplace=True)
        
        ew_rebalanced_daily_returns = (ew_rebalanced_weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        ew_rebalanced_cumulative_returns = (1 + ew_rebalanced_daily_returns).cumprod()

        # --- Combine Results ---
        self.portfolio_returns = pd.DataFrame({
            'Strategy Daily': portfolio_daily_returns,
            'Strategy Cumulative': cumulative_returns,
            'Risk Parity Daily': risk_parity_daily_returns,
            'Risk Parity Cumulative': risk_parity_cumulative_returns,
            'Equal Weight Rebalanced Daily': ew_rebalanced_daily_returns,
            'Equal Weight Rebalanced Cumulative': ew_rebalanced_cumulative_returns
        }).dropna()

    def _get_allocation_weights(self):
        """Delegates weight determination to the allocation strategy object."""
        return self.allocation_strategy.get_weights(self.aligned_data, macro_data=self.macro_data)

    def run_portfolio_backtest(self):
        """Main method to run the portfolio backtest."""
        print(f"\n--- Running Portfolio Backtest ({', '.join(self.symbols)}) ---")
        print(f"Allocation Strategy: {self.allocation_strategy.__class__.__name__}")
        
        self._fetch_all_data()
        self._fetch_macro_data()
        self.asset_data_with_signals = self.raw_data
        self._align_data()

        if not self.symbols:
            print("No symbols remaining after data alignment. Backtest aborted.")
            return

        weights_df = self._get_allocation_weights()
        self._calculate_portfolio_performance(weights_df)
        self.plot_performance()
            
        return self.portfolio_returns, weights_df

    # --- Performance and Plotting ---

    def sharpe_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sharpe Ratio."""
        excess_returns = return_series - risk_free_rate / annualization_factor
        return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

    def sortino_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sortino Ratio."""
        excess_returns = return_series - risk_free_rate / annualization_factor
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf
        return np.sqrt(annualization_factor) * excess_returns.mean() / downside_std

    def max_drawdown(self, cumulative_returns_series):
        """Calculates the Maximum Drawdown."""
        running_max = cumulative_returns_series.cummax()
        drawdown = (cumulative_returns_series - running_max) / running_max
        return drawdown.min()

    def plot_performance(self):
        """Plots the cumulative performance of the strategy vs. the benchmark."""
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            print("Portfolio backtest did not generate returns data to plot.")
            return
            
        strat_final = self.portfolio_returns['Strategy Cumulative'].iloc[-1]
        risk_parity_final = self.portfolio_returns['Risk Parity Cumulative'].iloc[-1]
        ew_rebalanced_final = self.portfolio_returns['Equal Weight Rebalanced Cumulative'].iloc[-1]
        
        strat_sharpe = self.sharpe_ratio(self.portfolio_returns['Strategy Daily'])
        risk_parity_sharpe = self.sharpe_ratio(self.portfolio_returns['Risk Parity Daily'])
        ew_rebalanced_sharpe = self.sharpe_ratio(self.portfolio_returns['Equal Weight Rebalanced Daily'])

        strat_sortino = self.sortino_ratio(self.portfolio_returns['Strategy Daily'])
        risk_parity_sortino = self.sortino_ratio(self.portfolio_returns['Risk Parity Daily'])
        ew_rebalanced_sortino = self.sortino_ratio(self.portfolio_returns['Equal Weight Rebalanced Daily'])

        strat_mdd = self.max_drawdown(self.portfolio_returns['Strategy Cumulative'])
        risk_parity_mdd = self.max_drawdown(self.portfolio_returns['Risk Parity Cumulative'])
        ew_rebalanced_mdd = self.max_drawdown(self.portfolio_returns['Equal Weight Rebalanced Cumulative'])

        print("\n--- Performance Summary ---")
        print(f"Strategy Final Cumulative Return: {strat_final:.2%}")
        print(f"Strategy Annualized Sharpe Ratio: {strat_sharpe:.2f}")
        print(f"Strategy Annualized Sortino Ratio: {strat_sortino:.2f}")
        print(f"Strategy Maximum Drawdown: {strat_mdd:.2%}")
        print("-" * 20)
        print(f"Risk Parity Benchmark Final Cumulative Return: {risk_parity_final:.2%}")
        print(f"Risk Parity Benchmark Annualized Sharpe Ratio: {risk_parity_sharpe:.2f}")
        print(f"Risk Parity Benchmark Annualized Sortino Ratio: {risk_parity_sortino:.2f}")
        print(f"Risk Parity Benchmark Maximum Drawdown: {risk_parity_mdd:.2%}")
        print("-" * 20)
        print(f"Equal Weight Rebalanced Benchmark Final Cumulative Return: {ew_rebalanced_final:.2%}")
        print(f"Equal Weight Rebalanced Benchmark Annualized Sharpe Ratio: {ew_rebalanced_sharpe:.2f}")
        print(f"Equal Weight Rebalanced Benchmark Annualized Sortino Ratio: {ew_rebalanced_sortino:.2f}")
        print(f"Equal Weight Rebalanced Benchmark Maximum Drawdown: {ew_rebalanced_mdd:.2%}")

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))
        self.portfolio_returns['Strategy Cumulative'].plot(label='Strategy', lw=2.5, color='blue')
        self.portfolio_returns['Risk Parity Cumulative'].plot(label='Benchmark (Static Risk Parity)', lw=1.5, linestyle=':', color='orange')
        self.portfolio_returns['Equal Weight Rebalanced Cumulative'].plot(label='Benchmark (Monthly Rebalanced Equal Weight)', lw=1.5, linestyle='-.', color='green')
        plt.title(f"Performance: '{self.allocation_strategy.__class__.__name__}' vs. Benchmarks", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
