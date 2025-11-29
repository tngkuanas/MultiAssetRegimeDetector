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
        """Calculates daily and cumulative returns for the portfolio and benchmarks."""
        print("Calculating portfolio and benchmark returns...")
        
        asset_returns = pd.DataFrame({s: self.aligned_data[s]['Close'].pct_change() for s in self.symbols})
        asset_returns = asset_returns.loc[weights_df.index]

        num_assets = len(self.symbols)

        # --- Strategy Performance ---
        portfolio_daily_returns = (weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # --- BENCHMARK 1: Static Risk Parity (Monthly Rebalanced) ---
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

        # --- BENCHMARK 2: Monthly Rebalanced Equal Weight ---
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

    def run_portfolio_backtest(self, plot=False):
        """Main method to run the portfolio backtest."""
        print(f"\n--- Running Portfolio Backtest ({', '.join(self.symbols)}) ---")
        print(f"Allocation Strategy: {self.allocation_strategy.__class__.__name__}")
        
        self._fetch_all_data()
        self._fetch_macro_data()
        self.asset_data_with_signals = self.raw_data
        self._align_data()

        if not self.symbols:
            print("No symbols remaining after data alignment. Backtest aborted.")
            # Return empty results to prevent crash in optimizer
            return pd.DataFrame(), pd.DataFrame(), pd.Series()

        weights_df, regime_labels = self.allocation_strategy.get_weights(self.aligned_data, macro_data=self.macro_data)
        
        if weights_df.empty:
            print("Allocation strategy returned empty weights. Backtest aborted.")
            return pd.DataFrame(), pd.DataFrame(), pd.Series()
            
        self._calculate_portfolio_performance(weights_df)
        
        if plot:
            self.plot_performance()
            
        return self.portfolio_returns, weights_df, regime_labels

    # --- Performance and Plotting ---

    def sharpe_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sharpe Ratio."""
        if return_series.std() == 0: return 0.0
        excess_returns = return_series - risk_free_rate / annualization_factor
        return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

    def sortino_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sortino Ratio."""
        excess_returns = return_series - risk_free_rate / annualization_factor
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.empty: return np.inf
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
            
        # Dynamically get all cumulative columns to plot
        cumulative_cols = [col for col in self.portfolio_returns.columns if 'Cumulative' in col]
        
        # --- Performance Summary ---
        print("\n--- Performance Summary ---")
        for col in cumulative_cols:
            name = col.replace(' Cumulative', '')
            daily_col = name + ' Daily'
            
            final_return = self.portfolio_returns[col].iloc[-1]
            sharpe = self.sharpe_ratio(self.portfolio_returns[daily_col])
            sortino = self.sortino_ratio(self.portfolio_returns[daily_col])
            mdd = self.max_drawdown(self.portfolio_returns[col])
            
            print(f"--- {name} ---")
            print(f"Final Cumulative Return: {final_return:.2%}")
            print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
            print(f"Annualized Sortino Ratio: {sortino:.2f}")
            print(f"Maximum Drawdown: {mdd:.2%}")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))
        for col in cumulative_cols:
            name = col.replace(' Cumulative', '')
            lw = 2.5 if name == 'Strategy' else 1.5
            ls = '-' if name == 'Strategy' else ':'
            self.portfolio_returns[col].plot(label=name, lw=lw, linestyle=ls)
            
        plt.title(f"Performance: '{self.allocation_strategy.__class__.__name__}' vs. Benchmarks", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
