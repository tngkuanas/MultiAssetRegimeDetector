import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from data_collection import get_data
from data_collection_fred import get_fred_data
from strategy_manager import StrategyManager

class PortfolioManager:
    def __init__(self, symbols, strategy_manager, start_date, end_date, allocation_strategy="equal_weight", fred_series_to_fetch=None):
        self.symbols = symbols
        self.strategy_manager = strategy_manager
        self.start_date = start_date
        self.end_date = end_date
        self.asset_data_with_signals = {}
        self.portfolio_returns = None
        self.aligned_data = None
        self.allocation_strategy = allocation_strategy
        self.fred_series_to_fetch = fred_series_to_fetch if fred_series_to_fetch is not None else {}

    def _fetch_all_data(self):
        """Fetches historical data for all symbols."""
        print("Fetching market data for all assets...")
        all_raw_data = {}
        for symbol in self.symbols:
            try:
                all_raw_data[symbol] = get_data(symbol, self.start_date, self.end_date)
            except Exception as e:
                print(f"Could not fetch market data for {symbol}: {e}")
                self.symbols.remove(symbol)
        if not all_raw_data:
            raise ValueError("No market data fetched for any symbol. Exiting.")
        return all_raw_data

    def _fetch_macro_data(self):
        """Fetches macroeconomic data from FRED if series are specified."""
        if not self.fred_series_to_fetch:
            print("No FRED series specified. Skipping macro data fetch.")
            return None
        
        print("Fetching macroeconomic data...")
        macro_data = get_fred_data(series_ids=self.fred_series_to_fetch, start_date=self.start_date, end_date=self.end_date)
        return macro_data

    def _generate_all_signals(self, all_raw_data, macro_data):
        """Generates trading signals for each asset using the StrategyManager and macro_data."""
        print("Generating signals for all assets...")
        for symbol, data in all_raw_data.items():
            print(f"  - Generating signals for {symbol}")
            self.asset_data_with_signals[symbol] = self.strategy_manager.process(data.copy(), macro_data)

    def _align_data(self):
        """Aligns all asset data with signals to a common date range."""
        if not self.asset_data_with_signals:
            raise ValueError("No asset data with signals to align.")

        aligned_index = next(iter(self.asset_data_with_signals.values())).index.copy()

        for df in self.asset_data_with_signals.values():
            aligned_index = aligned_index.intersection(df.index)
        
        aligned_index = aligned_index.sort_values()

        aligned_dfs = {}
        for symbol, df in self.asset_data_with_signals.items():
            aligned_dfs[symbol] = df.reindex(aligned_index)
            aligned_dfs[symbol].dropna(inplace=True)
            if aligned_dfs[symbol].empty:
                print(f"Warning: {symbol} became empty after alignment. Removing from portfolio.")
                del self.asset_data_with_signals[symbol]
                self.symbols.remove(symbol)
        
        if not aligned_dfs:
            raise ValueError("All asset data became empty after alignment. Cannot proceed.")

        self.aligned_data = aligned_dfs

    def _determine_equal_weight_weights(self):
        """Determines daily portfolio weights based on equal weighting for active signals."""
        print("Determining portfolio weights using 'equal_weight' strategy...")
        weights_df = pd.DataFrame(index=next(iter(self.aligned_data.values())).index)

        for date in weights_df.index:
            active_assets = []
            for symbol, df in self.aligned_data.items():
                if date in df.index and df.loc[date, 'Signal'] > 0 and df.loc[date, 'PSignal'] == 1:
                    active_assets.append(symbol)
            
            if active_assets:
                weight_per_asset = 1 / len(active_assets)
                for symbol in self.symbols:
                    weights_df.loc[date, symbol] = weight_per_asset if symbol in active_assets else 0
            else:
                for symbol in self.symbols:
                    weights_df.loc[date, symbol] = 0
        
        return weights_df.fillna(0)

    def _determine_relative_momentum_weights(self):
        """Determines weights by picking the single asset with the highest relative momentum."""
        print("Determining portfolio weights using 'relative_momentum' strategy...")
        weights_df = pd.DataFrame(0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        for date in weights_df.index:
            active_assets_momentum = {}
            for symbol, df in self.aligned_data.items():
                if date in df.index and 'Momentum' in df.columns and df.loc[date, 'Signal'] > 0:
                    active_assets_momentum[symbol] = df.loc[date, 'Momentum']

            if active_assets_momentum:
                winner = max(active_assets_momentum, key=active_assets_momentum.get)
                weights_df.loc[date, winner] = 1 # Allocate 100% to the winner
        
        return weights_df.fillna(0)

    def _determine_crash_aware_weights(self):
        """
        Determines weights based on predicted regimes (Calm, Volatile, Crash).
        Allocates across asset classes (Equities, Bonds, Gold) based on the current regime.
        Assumes 'SPY', 'QQQ', 'VOO', 'MSFT', 'AAPL' are equities, 'BND', 'TLT' are bonds, 'GLD' is gold for this example.
        """
        print("Determining portfolio weights using 'crash_aware' strategy...")
        weights_df = pd.DataFrame(0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        # Define asset classes based on common interpretations of symbols
        # This can be made more dynamic by passing asset types to the constructor
        equities = [s for s in self.symbols if s in ['SPY', 'QQQ', 'VOO', 'MSFT', 'AAPL']]
        bonds = [s for s in self.symbols if s in ['BND', 'TLT']]
        gold = [s for s in self.symbols if s in ['GLD']]
        
        # Ensure that asset classes are not empty for allocation
        if not equities and (len(equities) + len(bonds) + len(gold)) > 0:
            print("Warning: No equity symbols found among the provided. Equity allocation will be 0.")
        if not bonds and (len(equities) + len(bonds) + len(gold)) > 0:
            print("Warning: No bond symbols found among the provided. Bond allocation will be 0.")
        if not gold and (len(equities) + len(bonds) + len(gold)) > 0:
            print("Warning: No gold symbols found among the provided. Gold allocation will be 0.")

        for date in weights_df.index:
            current_regime = None
            # The PSignal is a global regime, so it should be consistent across assets for a given date
            # Just pick from the first asset that has a PSignal value for this date
            for symbol, df in self.aligned_data.items():
                if date in df.index and 'PSignal' in df.columns and pd.notna(df.loc[date, 'PSignal']):
                    current_regime = df.loc[date, 'PSignal']
                    break
            
            if current_regime is None:
                # If no regime detected (e.g., due to NaNs), assume cash
                continue # Weights already 0
            
            # --- Allocation Rules based on Regime ---
            target_alloc = {}
            if current_regime == "Calm":
                target_alloc = {'equity': 1.0, 'bond': 0.0, 'gold': 0.0}
            elif current_regime == "Volatile":
                target_alloc = {'equity': 0.50, 'bond': 0.30, 'gold': 0.20}
            elif current_regime == "Crash":
                target_alloc = {'equity': 0.20, 'bond': 0.50, 'gold': 0.30}
            else:
                # Unknown regime, default to cash
                continue

            # Apply target allocations
            if equities:
                eq_weight_per_asset = target_alloc['equity'] / len(equities)
                for s in equities: weights_df.loc[date, s] = eq_weight_per_asset
            if bonds:
                bond_weight_per_asset = target_alloc['bond'] / len(bonds)
                for s in bonds: weights_df.loc[date, s] = bond_weight_per_asset
            if gold:
                gold_weight_per_asset = target_alloc['gold'] / len(gold)
                for s in gold: weights_df.loc[date, s] = gold_weight_per_asset
        
        return weights_df.fillna(0) # Fill any remaining NaNs with 0 (e.g., if an asset class was empty)


    def _calculate_returns(self, weights_df):
        """Calculates daily and cumulative returns for both the portfolio and a benchmark."""
        print("Calculating portfolio and benchmark returns...")
        asset_daily_returns = pd.DataFrame(index=weights_df.index)
        
        for symbol, df in self.aligned_data.items():
            if 'Close' in df.columns:
                asset_daily_returns[symbol] = df['Close'].pct_change()
            else:
                asset_daily_returns[symbol] = 0

        asset_daily_returns.dropna(inplace=True)
        weights_df = weights_df.reindex(asset_daily_returns.index).fillna(0)

        portfolio_daily_returns = (weights_df.shift(1) * asset_daily_returns).sum(axis=1)
        portfolio_daily_returns.dropna(inplace=True)
        cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1

        benchmark_daily_returns = asset_daily_returns.mean(axis=1)
        benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod() - 1
        
        self.portfolio_returns = pd.DataFrame({
            'Strategy Daily Returns': portfolio_daily_returns,
            'Strategy Cumulative Returns': cumulative_returns,
            'Benchmark Daily Returns': benchmark_daily_returns,
            'Benchmark Cumulative Returns': benchmark_cumulative_returns
        }).dropna()

    def sharpe_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sharpe Ratio."""
        excess_returns = return_series - risk_free_rate / annualization_factor
        return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

    def run_portfolio_backtest(self):
        """Main method to run the portfolio backtest."""
        print(f"\n--- Running Portfolio Backtest ({', '.join(self.symbols)}) ---")
        print(f"Allocation Strategy: {self.allocation_strategy}")
        
        # Fetch both market and macro data
        all_raw_data = self._fetch_all_data()
        macro_data = self._fetch_macro_data()
        
        # Generate signals using both data sources
        self._generate_all_signals(all_raw_data, macro_data)
        
        self._align_data()
        
        self.symbols = list(self.aligned_data.keys()) 
        if not self.symbols:
            print("No symbols remaining after data alignment. Backtest aborted.")
            return

        if self.allocation_strategy == "equal_weight":
            weights_df = self._determine_equal_weight_weights()
        elif self.allocation_strategy == "relative_momentum":
            weights_df = self._determine_relative_momentum_weights()
        elif self.allocation_strategy == "crash_aware":
            weights_df = self._determine_crash_aware_weights()
        else:
            raise ValueError(f"Unknown allocation_strategy: {self.allocation_strategy}")

        self._calculate_returns(weights_df)

        if self.portfolio_returns is not None and not self.portfolio_returns.empty:
            strat_final_return = self.portfolio_returns['Strategy Cumulative Returns'].iloc[-1]
            strat_sharpe = self.sharpe_ratio(self.portfolio_returns['Strategy Daily Returns'])

            bench_final_return = self.portfolio_returns['Benchmark Cumulative Returns'].iloc[-1]
            bench_sharpe = self.sharpe_ratio(self.portfolio_returns['Benchmark Daily Returns'])

            print("\n--- Performance Summary ---")
            print(f"Strategy Final Cumulative Return: {strat_final_return:.2%}")
            print(f"Strategy Annualized Sharpe Ratio: {strat_sharpe:.2f}")
            print("-" * 20)
            print(f"Benchmark Final Cumulative Return: {bench_final_return:.2%}")
            print(f"Benchmark Annualized Sharpe Ratio: {bench_sharpe:.2f}")

            plt.figure(figsize=(12, 6))
            self.portfolio_returns['Strategy Cumulative Returns'].plot(label='Strategy', legend=True)
            self.portfolio_returns['Benchmark Cumulative Returns'].plot(label='Benchmark (Buy and Hold)', legend=True)
            plt.title(f"Portfolio Performance vs. Benchmark ({', '.join(self.symbols)})")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Returns")
            plt.grid(True)
            plt.show()
        else:
            print("Portfolio backtest did not generate returns data.")
            
        return self.portfolio_returns, weights_df # Optionally return the results for further analysis
