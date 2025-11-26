import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from data_collection import get_data
from data_collection_fred import get_fred_data
from strategy_manager import StrategyManager
from regime_detection.shu_mulvey_model import ShuMulveyModel
from regime_detection.statistical_jump_model import StatisticalJumpModel
from feature_engineering import calculate_jump_model_features


class PortfolioManager:
    def __init__(self, symbols, strategy_manager, start_date, end_date, allocation_strategy="equal_weight", fred_series_to_fetch=None):
        self.symbols = symbols
        self.strategy_manager = strategy_manager
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
        for symbol in self.symbols:
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

    def _generate_all_signals(self):
        """Generates trading signals for each asset using the StrategyManager and macro_data."""
        # For pure allocation strategies, signals are generated inside the allocation logic, so we bypass this.
        if self.allocation_strategy in ['shu_mulvey', 'jump_model_risk_parity']:
            print(f"Bypassing signal generation for pure allocation strategy: '{self.allocation_strategy}'.")
            self.asset_data_with_signals = self.raw_data
            return

        print("Generating signals for all assets...")
        for symbol, data in self.raw_data.items():
            print(f"  - Generating signals for {symbol}")
            self.asset_data_with_signals[symbol] = self.strategy_manager.process(data.copy(), self.macro_data)

    def _align_data(self):
        """Aligns all asset data to a common date range."""
        if not self.asset_data_with_signals:
            raise ValueError("No asset data with signals to align.")
        
        # Create a combined DataFrame to find the common date index
        close_df = pd.DataFrame({sym: df['Close'] for sym, df in self.asset_data_with_signals.items()})
        close_df.dropna(inplace=True)
        aligned_index = close_df.index
        
        aligned_dfs = {}
        for symbol, df in self.asset_data_with_signals.items():
            aligned_dfs[symbol] = df.reindex(aligned_index).ffill() # Use ffill to prevent data loss

        self.aligned_data = aligned_dfs
        self.symbols = list(self.aligned_data.keys())

    def _calculate_portfolio_performance(self, weights_df):
        """Calculates daily and cumulative returns for the portfolio and a benchmark."""
        print("Calculating portfolio and benchmark returns...")
        
        asset_returns = pd.DataFrame({s: self.aligned_data[s]['Close'].pct_change() for s in self.symbols})
        asset_returns = asset_returns.loc[weights_df.index] # Align returns with weights index

        num_assets = len(self.symbols) # Define num_assets here

        # --- Strategy Performance ---
        # Shift weights by 1 to trade on the next day's open
        portfolio_daily_returns = (weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # --- BENCHMARK: Static Risk Parity (Monthly Rebalanced) ---
        print("  - Calculating static risk parity benchmark...")
        rp_weights_df = pd.DataFrame(index=asset_returns.index, columns=self.symbols).fillna(0.0)
        rebalance_dates = asset_returns.resample('BMS').first().index
        
        last_rp_weights = np.array([1/num_assets] * num_assets) # Initialize with equal weights
        if not asset_returns.empty:
            if asset_returns.index[0] not in rebalance_dates: # If first day isn't a rebalance day, init weights
                rp_weights_df.loc[asset_returns.index[0]] = last_rp_weights


        for date in asset_returns.index:
            if date in rebalance_dates:
                # Look back 63 trading days for volatility calculation
                hist_returns = asset_returns.loc[:date].tail(63)
                if len(hist_returns) > 1:
                    # Inverse volatility weighting
                    inv_vol = 1 / hist_returns.std()
                    if inv_vol.sum() > 0: # Avoid division by zero
                        last_rp_weights = inv_vol / inv_vol.sum()
                    # else, keep last month's weights
            rp_weights_df.loc[date] = last_rp_weights # Assign weights for each day
        
        rp_weights_df.ffill(inplace=True)
        risk_parity_daily_returns = (rp_weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        risk_parity_cumulative_returns = (1 + risk_parity_daily_returns).cumprod()

        # --- BENCHMARK: Monthly Rebalanced Equal Weight ---
        print("  - Calculating monthly rebalanced equal-weight benchmark...")
        ew_rebalanced_weights_df = pd.DataFrame(index=asset_returns.index, columns=self.symbols)
        equal_weight = np.array([1/num_assets] * num_assets)
        
        # Set weights on rebalance days
        for date in rebalance_dates:
            if date in ew_rebalanced_weights_df.index:
                ew_rebalanced_weights_df.loc[date] = equal_weight
        
        # Ensure the portfolio is invested from the very first day
        ew_rebalanced_weights_df.iloc[0] = equal_weight
        
        # Forward-fill the weights from each rebalance day
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
        """Dispatcher for allocation strategies."""
        if self.allocation_strategy == "equal_weight":
            return self._determine_equal_weight_weights()
        elif self.allocation_strategy == "shu_mulvey":
            return self._determine_shu_mulvey_weights()
        elif self.allocation_strategy == "jump_model_risk_parity":
            return self._determine_jump_model_weights()
        else:
            raise ValueError(f"Unknown allocation_strategy: {self.allocation_strategy}")

    def run_portfolio_backtest(self):
        """Main method to run the portfolio backtest."""
        print(f"\n--- Running Portfolio Backtest ({', '.join(self.symbols)}) ---")
        print(f"Allocation Strategy: {self.allocation_strategy}")
        
        self._fetch_all_data()
        self._fetch_macro_data()
        self._generate_all_signals()
        self._align_data()

        if not self.symbols:
            print("No symbols remaining after data alignment. Backtest aborted.")
            return

        weights_df = self._get_allocation_weights()
        self._calculate_portfolio_performance(weights_df)
        self.plot_performance()
            
        return self.portfolio_returns, weights_df

    # --- Allocation Strategies ---

    def _determine_jump_model_weights(self, hysteresis_period=5, max_turnover=0.5):
        """
        Determines weights using the Statistical Jump Model and regime-switched risk parity.
        Rebalances and retrains monthly, targeting a specific portfolio volatility based on the active regime.
        """
        print("Determining portfolio weights using 'jump_model_risk_parity' walk-forward strategy...")
        
        weights_df = pd.DataFrame(index=next(iter(self.aligned_data.values())).index, columns=self.symbols).fillna(0.0)
        
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS')

        last_weights = np.array([1/len(self.symbols)] * len(self.symbols))
        
        # Define volatility targets for each regime (0=Low, 1=Mid, 2=High)
        vol_targets = {0: 0.15, 1: 0.10, 2: 0.05} # Annualized volatility

        for date in weights_df.index:
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                
                # 1. Get all historical data up to the current rebalance date
                historical_data = {s: df.loc[:date] for s, df in self.aligned_data.items()}
                
                # 2. Calculate features for the jump model
                sjm_features = calculate_jump_model_features(historical_data)
                
                if sjm_features.empty:
                    print(f"  - Not enough data for feature calculation on {date}. Holding previous weights.")
                    weights_df.loc[date] = last_weights
                    continue

                # 3. Fit the Statistical Jump Model
                sjm = StatisticalJumpModel(n_states=3, jump_penalty=100)
                regime_labels = sjm.fit(sjm_features)

                # 4. Apply Hysteresis to stabilize the regime signal
                stable_regimes = regime_labels.rolling(window=hysteresis_period).apply(lambda x: x.iloc[-1] if x.nunique() == 1 else np.nan, raw=False).ffill()
                
                if pd.isna(stable_regimes.iloc[-1]):
                    print(f"  - Regime is unstable on {date}. Holding previous weights.")
                    weights_df.loc[date] = last_weights
                    continue
                
                current_regime = int(stable_regimes.iloc[-1])
                target_vol = vol_targets[current_regime]
                print(f"  - Current Stable Regime: {current_regime} | Target Volatility: {target_vol:.0%}")

                # 5. Get returns and regime-specific covariance
                historical_returns = pd.DataFrame({
                    symbol: np.log(df['Close']).diff()
                    for symbol, df in historical_data.items()
                })
                
                # Align returns with regimes for filtering
                aligned_returns, aligned_regimes = historical_returns.align(regime_labels, join='inner', axis=0)
                
                regime_specific_returns = aligned_returns[aligned_regimes == current_regime]
                
                if len(regime_specific_returns) < 30: # Need enough data for stable covariance
                    print(f"  - Not enough data in regime {current_regime} for covariance. Using full history.")
                    cov_matrix = aligned_returns.iloc[-90:].cov() * 252 # Fallback
                else:
                    cov_matrix = regime_specific_returns.cov() * 252 # Annualize
                
                # 6. Run Optimizer
                num_assets = len(self.symbols)
                initial_weights = last_weights
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Sum of weights is 1
                    {'type': 'ineq', 'fun': lambda w: max_turnover - np.sum(np.abs(w - last_weights))} # L1 Turnover
                ]
                
                bounds = tuple((0.0, 1.0) for _ in range(num_assets))
                
                result = minimize(self._volatility_objective, initial_weights,
                                  args=(cov_matrix, target_vol),
                                  method='SLSQP',
                                  bounds=bounds,
                                  constraints=constraints)
                
                if result.success:
                    last_weights = result.x
                else:
                    print(f"  - Optimizer failed on {date}, holding previous weights.")

            weights_df.loc[date] = last_weights
            
        weights_df.ffill(inplace=True)
        return weights_df

    def _volatility_objective(self, weights, cov_matrix, target_vol):
        """Objective function for the optimizer: minimize difference to target volatility."""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return (portfolio_vol - target_vol)**2

    def _determine_shu_mulvey_weights(self):
        """
        Determines weights using a walk-forward version of the Shu-Mulvey GBDT model
        and mean-variance optimization. Rebalances and retrains monthly.
        """
        print("Determining portfolio weights using 'shu_mulvey' walk-forward strategy...")
        
        weights_df = pd.DataFrame(index=next(iter(self.aligned_data.values())).index, columns=self.symbols).fillna(0.0)
        
        # Get monthly rebalance dates, starting at least 1 year in for initial training data.
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS') # Business Month Start

        last_weights = np.array([1/len(self.symbols)] * len(self.symbols)) # Start with equal weight

        for date in weights_df.index:
            # Check if today is a rebalance date
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                
                # --- Walk-Forward Training ---
                # 1. Get all historical data up to the current rebalance date
                historical_data = {s: df.loc[:date] for s, df in self.aligned_data.items()}
                historical_macro_data = self.macro_data.loc[:date] if self.macro_data is not None else None
                
                # 2. Initialize and fit the Shu-Mulvey model on this historical slice
                model = ShuMulveyModel(assets_data=historical_data, macro_data=historical_macro_data)
                model.fit() # This now fits only on past data

                # 3. Predict regime probabilities for the *next* period based on the model just trained
                regime_probs = model.predict_proba()

                # 4. Calculate historical regime-conditional returns *using only past data*
                regime_returns = {}
                for asset in self.symbols:
                    returns = np.log(historical_data[asset]['Close']).diff()
                    
                    # Align returns with regimes from the just-trained model
                    # This is the critical fix for the logic error. The regimes were trained on returns
                    # that were dropped of NaNs, so we must align them before using them.
                    aligned_returns, aligned_regimes = returns.align(model.regime_labels[asset], join='inner')
                    
                    regime_data = pd.DataFrame({'returns': aligned_returns, 'regime': aligned_regimes})
                    regime_returns[asset] = regime_data.groupby('regime')['returns'].mean()
                # --- Allocation Logic ---
                # 5. Calculate forward-looking expected returns
                expected_returns = []
                for asset in self.symbols:
                    if not regime_probs.get(asset):
                        exp_ret = 0 # Handle cases with no prediction
                    else:
                        probs = pd.Series(regime_probs[asset])
                        cond_returns = regime_returns.get(asset, pd.Series())
                        # Align and compute dot product, handle missing regimes
                        exp_ret = probs.dot(cond_returns.reindex(probs.index).fillna(0.0))
                    expected_returns.append(exp_ret)

                # 6. Estimate covariance (using recent history)
                returns_subset = pd.DataFrame({s: np.log(df['Close']).diff() for s, df in historical_data.items()}).iloc[-90:]
                cov_matrix = returns_subset.cov() * 252 # Annualize

                # 7. Run Optimizer
                num_assets = len(self.symbols)
                args = (np.array(expected_returns), cov_matrix)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                
                result = minimize(self._neg_sharpe, num_assets*[1./num_assets], args=args,
                                  method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    last_weights = result.x
                else:
                    print(f"  - Optimizer failed on {date}, holding previous weights.")

            weights_df.loc[date] = last_weights

        # Forward-fill weights from rebalance dates to the rest of the month
        weights_df.replace(0, np.nan, inplace=True)
        weights_df.ffill(inplace=True)
        weights_df.fillna(0, inplace=True)

        return weights_df

    def _neg_sharpe(self, weights, expected_returns, cov_matrix, risk_free_rate=0.02):
        """Helper for the optimizer: calculates the negative Sharpe ratio."""
        portfolio_return = np.sum(expected_returns * weights) * 252 # Annualize
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_volatility == 0:
            return 0
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    def _determine_equal_weight_weights(self):
        """Determines daily portfolio weights based on equal weighting for active signals."""
        print("Determining portfolio weights using 'equal_weight' strategy...")
        weights_df = pd.DataFrame(index=next(iter(self.aligned_data.values())).index)
        for date in weights_df.index:
            active_assets = [s for s, df in self.aligned_data.items() if df.loc[date, 'Signal'] > 0 and df.loc[date, 'PSignal'] == 1]
            if active_assets:
                weight = 1 / len(active_assets)
                for s in self.symbols: weights_df.loc[date, s] = weight if s in active_assets else 0
            else:
                 for s in self.symbols: weights_df.loc[date, s] = 0
        return weights_df.fillna(0)


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
        plt.title(f"Performance: '{self.allocation_strategy}' vs. Benchmarks", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
