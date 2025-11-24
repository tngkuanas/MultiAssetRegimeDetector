import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from data_collection import get_data
from data_collection_fred import get_fred_data
from strategy_manager import StrategyManager
from regime_detection.shu_mulvey_model import ShuMulveyModel
from regime_detection.sma_crossover_regime import SMACrossoverRegime
from regime_detection.correlation_regime_model import CorrelationRegimeModel


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
        if self.allocation_strategy in ['shu_mulvey', 'risk_parity', 'ml_dynamic_tilt']:
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

        # Shift weights by 1 to trade on the next day's open
        portfolio_daily_returns = (weights_df.shift(1) * asset_returns).sum(axis=1).dropna()
        
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # Benchmark: Equal-weighted portfolio of the same assets
        benchmark_daily_returns = asset_returns.mean(axis=1).dropna()
        benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod()
        
        self.portfolio_returns = pd.DataFrame({
            'Strategy Daily': portfolio_daily_returns,
            'Strategy Cumulative': cumulative_returns,
            'Benchmark Daily': benchmark_daily_returns,
            'Benchmark Cumulative': benchmark_cumulative_returns
        }).dropna()

    def _get_allocation_weights(self):
        """Dispatcher for allocation strategies."""
        if self.allocation_strategy == "equal_weight":
            return self._determine_equal_weight_weights()
        elif self.allocation_strategy == "protective_momentum":
            return self._determine_protective_momentum_weights()
        elif self.allocation_strategy == "aggressive_momentum":
            return self._determine_aggressive_momentum_weights()
        elif self.allocation_strategy == "correlation_aware":
            return self._determine_correlation_aware_weights()
        elif self.allocation_strategy == "risk_parity":
            return self._determine_risk_parity_weights()
        elif self.allocation_strategy == "ml_dynamic_tilt":
            return self._determine_ml_dynamic_tilt_weights()
        elif self.allocation_strategy == "shu_mulvey":
            return self._determine_shu_mulvey_weights()
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

    def _determine_protective_momentum_weights(self):
        """
        Allocates to top momentum assets in a 'risk-on' regime, and to
        defensive assets in a 'risk-off' regime. The regime itself is
        calculated internally using an SMA crossover model on a market index.
        """
        print("Determining portfolio weights using 'protective_momentum' strategy...")
        weights_df = pd.DataFrame(0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        # --- 1. Generate Market Regime Signal ---
        print("  - Generating market regime signal from VOO...")
        market_index_data = get_data("VOO", self.start_date, self.end_date)
        regime_model = SMACrossoverRegime(window=200)
        market_regime_signal = regime_model.process(market_index_data) # This is now a DataFrame with PSignal

        # --- 2. Define Asset Classes ---
        risk_assets = [s for s in self.symbols if s in ["VOO", "VXUS", "GOOG", "SPY", "QQQ", "EFA", "EEM"]]
        defensive_assets = [s for s in self.symbols if s in ["BND", "TLT", "GLD"]]

        # --- 3. Rebalance Monthly ---
        rebalance_dates = weights_df.resample('M').last().index
        last_weights = pd.Series(0.0, index=self.symbols)

        for date in weights_df.index:
            if date in rebalance_dates:
                current_weights = pd.Series(0.0, index=self.symbols)
                
                # Get the regime for the current date
                try:
                    regime = market_regime_signal.loc[date, 'PSignal']
                except KeyError:
                    # If date is not in the regime signal, hold previous weights
                    last_weights.loc[:] = last_weights
                    continue

                if regime == 1: # Risk-On
                    momentums = {s: self.aligned_data[s].loc[date, 'MomentumScore'] for s in risk_assets if s in self.aligned_data and 'MomentumScore' in self.aligned_data[s].columns}
                    momentums = {k: v for k, v in momentums.items() if pd.notna(v)}

                    if momentums:
                        winner = max(momentums, key=momentums.get)
                        current_weights[winner] = 1.0
                    elif defensive_assets: # Fallback if no momentum
                        for s in defensive_assets: current_weights[s] = 1.0 / len(defensive_assets)
                
                else: # Risk-Off
                    if defensive_assets:
                        for s in defensive_assets: current_weights[s] = 1.0 / len(defensive_assets)
                
                last_weights = current_weights

            weights_df.loc[date] = last_weights

        return weights_df.fillna(0)

    def _determine_aggressive_momentum_weights(self):
        """
        Allocates 100% to the single asset with the highest momentum in a 
        'risk-on' regime, and allocates to cash in a 'risk-off' regime.
        This strategy does not use leveraged products directly but achieves
        high returns through concentration.
        """
        print("Determining portfolio weights using 'aggressive_momentum' strategy...")
        weights_df = pd.DataFrame(0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        # --- 1. Generate Market Regime Signal ---
        print("  - Generating market regime signal from VOO...")
        market_index_data = get_data("VOO", self.start_date, self.end_date)
        regime_model = SMACrossoverRegime(window=200)
        market_regime_signal = regime_model.process(market_index_data)

        # --- 2. Rebalance Monthly ---
        rebalance_dates = weights_df.resample('M').last().index
        last_weights = pd.Series(0.0, index=self.symbols)

        for date in weights_df.index:
            if date in rebalance_dates:
                current_weights = pd.Series(0.0, index=self.symbols)
                
                try:
                    regime = market_regime_signal.loc[date, 'PSignal']
                except KeyError:
                    last_weights.loc[:] = last_weights
                    continue

                if regime == 1: # Risk-On
                    momentums = {s: self.aligned_data[s].loc[date, 'MomentumScore'] for s in self.symbols if 'MomentumScore' in self.aligned_data[s].columns}
                    momentums = {k: v for k, v in momentums.items() if pd.notna(v)}

                    if momentums:
                        winner = max(momentums, key=momentums.get)
                        current_weights[winner] = 1.0
                
                # In a Risk-Off regime, weights remain 0 (i.e., cash)
                
                last_weights = current_weights

            weights_df.loc[date] = last_weights

        return weights_df.fillna(0)

    def _determine_correlation_aware_weights(self):
        """
        Allocates to assets by dynamically tilting the portfolio based on
        market trend and stock-bond correlation signals.
        """
        print("Determining portfolio weights using 'correlation_aware' (dynamic tilt) strategy...")
        weights_df = pd.DataFrame(0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        # --- 1. Generate Signals ---
        print("  - Generating market trend signal from VOO...")
        stock_index_data = get_data("VOO", self.start_date, self.end_date)
        trend_model = SMACrossoverRegime(window=200)
        trend_signal = trend_model.process(stock_index_data)

        print("  - Generating stock-bond correlation signal from VOO/BND...")
        bond_index_data = get_data("BND", self.start_date, self.end_date)
        corr_model = CorrelationRegimeModel(window=60, threshold=0.1)
        corr_signal = corr_model.process(stock_index_data, bond_index_data)

        # --- 2. Define Asset Classes & Allocation Profiles ---
        risk_assets = [s for s in self.symbols if s in ["VOO", "VXUS", "GOOG"]]
        safe_haven_assets = [s for s in self.symbols if s in ["BND"]]
        crisis_assets = [s for s in self.symbols if s in ["GLD"]]

        # Define allocation dictionaries {asset_class: percentage}
        allocations = {
            "risk_on":  {"risk": 0.7, "safe": 0.2, "crisis": 0.1},
            "risk_off": {"risk": 0.2, "safe": 0.6, "crisis": 0.2},
            "crisis":   {"risk": 0.1, "safe": 0.1, "crisis": 0.8}
        }

        # --- 3. Rebalance Monthly ---
        rebalance_dates = weights_df.resample('ME').last().index
        last_weights = pd.Series(0.0, index=self.symbols)

        for date in weights_df.index:
            if date in rebalance_dates:
                current_weights = pd.Series(0.0, index=self.symbols)
                
                try:
                    trend = trend_signal.loc[date, 'PSignal']
                    correlation = corr_signal.loc[date, 'CorrelationSignal']
                except KeyError:
                    last_weights.loc[:] = last_weights
                    continue
                
                # Choose allocation profile based on signals
                if trend == 1:
                    active_allocation = allocations["risk_on"]
                else: # Trend is "risk-off"
                    if correlation == 1: # Bonds are diversifying
                        active_allocation = allocations["risk_off"]
                    else: # Bonds are not diversifying
                        active_allocation = allocations["crisis"]
                
                # Apply the chosen allocation profile
                if risk_assets:
                    for s in risk_assets:
                        current_weights[s] = active_allocation["risk"] / len(risk_assets)
                if safe_haven_assets:
                    for s in safe_haven_assets:
                        current_weights[s] = active_allocation["safe"] / len(safe_haven_assets)
                if crisis_assets:
                    for s in crisis_assets:
                        current_weights[s] = active_allocation["crisis"] / len(crisis_assets)

                last_weights = current_weights

            weights_df.loc[date] = last_weights
            
        return weights_df.fillna(0)

    def _determine_risk_parity_weights(self):
        """
        Allocates to assets based on the inverse of their volatility, aiming
        to equalize the risk contribution of each asset.
        """
        print("Determining portfolio weights using 'risk_parity' strategy...")
        weights_df = pd.DataFrame(0.0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)
        
        # --- Rebalance Monthly ---
        rebalance_dates = weights_df.resample('ME').last().index
        last_weights = pd.Series(1.0 / len(self.symbols), index=self.symbols) # Start with equal weight
        
        volatility_lookback = 60 # days

        for date in weights_df.index:
            if date in rebalance_dates:
                # Get recent data slice to calculate volatility
                historical_data = {s: df.loc[:date] for s, df in self.aligned_data.items()}
                
                # Calculate historical volatility for each asset
                returns_subset = pd.DataFrame({
                    s: np.log(df['Close']).diff() for s, df in historical_data.items()
                }).iloc[-volatility_lookback:]
                
                # Calculate volatility (standard deviation of returns)
                volatilities = returns_subset.std()
                
                # Inverse volatility
                inverse_volatilities = 1 / volatilities
                
                # Check for infinite values that can result from zero volatility
                inverse_volatilities.replace([np.inf, -np.inf], 0, inplace=True)

                # Sum of inverse volatilities
                sum_inv_vol = np.sum(inverse_volatilities)
                
                if sum_inv_vol > 0:
                    # Weights are proportional to the inverse of their volatility
                    current_weights = inverse_volatilities / sum_inv_vol
                    last_weights = current_weights
                else:
                    # Fallback to previous weights if something went wrong
                    last_weights.loc[:] = last_weights

            weights_df.loc[date] = last_weights

        return weights_df.fillna(0)

    def _determine_ml_dynamic_tilt_weights(self):
        """
        Uses the ShuMulveyModel (ML) to predict a regime and then applies a
        dynamic tilt allocation based on that prediction.
        """
        print("Determining portfolio weights using 'ml_dynamic_tilt' strategy...")
        weights_df = pd.DataFrame(0.0, index=next(iter(self.aligned_data.values())).index, columns=self.symbols)

        # --- 1. Define Asset Classes & Allocation Profiles ---
        risk_assets = [s for s in self.symbols if s in ["VOO", "VXUS", "GOOG"]]
        safe_haven_assets = [s for s in self.symbols if s in ["BND"]]
        crisis_assets = [s for s in self.symbols if s in ["GLD"]]

        # DATA-DRIVEN MAPPING based on previous analysis:
        # Regime 1: High Return, High Vol -> "risk_on"
        # Regime 0: Low Return, Medium Vol -> "neutral"
        # Regime 2: Negative Return, High Vol -> "crisis"
        allocations = {
            1: {"risk": 0.7, "safe": 0.2, "crisis": 0.1},  # "risk_on"
            0: {"risk": 0.4, "safe": 0.4, "crisis": 0.2},  # "neutral"
            2: {"risk": 0.1, "safe": 0.3, "crisis": 0.6}   # "crisis"
        }

        # --- 2. Rebalance Monthly ---
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS')
        
        last_weights = pd.Series(0.0, index=self.symbols)

        for date in weights_df.index:
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                current_weights = pd.Series(0.0, index=self.symbols)

                # --- Walk-Forward ML Prediction ---
                historical_data = {s: df.loc[:date] for s, df in self.aligned_data.items()}
                historical_macro_data = self.macro_data.loc[:date] if self.macro_data is not None else None
                
                model = ShuMulveyModel(assets_data=historical_data, macro_data=historical_macro_data)
                model.fit()
                
                # Predict regime for all risk assets and find the consensus
                regime_probs = model.predict_proba()
                
                risk_asset_predictions = []
                for asset in risk_assets:
                    if asset in regime_probs and regime_probs[asset]:
                        top_regime = max(regime_probs[asset], key=regime_probs[asset].get)
                        risk_asset_predictions.append(top_regime)
                
                # Determine consensus by mode (most frequent prediction)
                if risk_asset_predictions:
                    consensus_regime = pd.Series(risk_asset_predictions).mode()[0]
                else:
                    consensus_regime = 0 # Default to neutral if no predictions

                # Choose allocation profile based on the consensus regime
                active_allocation = allocations.get(consensus_regime, allocations[0])

                # Apply the chosen allocation profile
                if risk_assets:
                    for s in risk_assets: current_weights[s] = active_allocation["risk"] / len(risk_assets)
                if safe_haven_assets:
                    for s in safe_haven_assets: current_weights[s] = active_allocation["safe"] / len(safe_haven_assets)
                if crisis_assets:
                    for s in crisis_assets: current_weights[s] = active_allocation["crisis"] / len(crisis_assets)
                
                last_weights = current_weights

            weights_df.loc[date] = last_weights
            
        return weights_df.fillna(0)
        
    # --- Performance and Plotting ---

    def sharpe_ratio(self, return_series, annualization_factor=252, risk_free_rate=0.0):
        """Calculates the annualized Sharpe Ratio."""
        excess_returns = return_series - risk_free_rate / annualization_factor
        return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()

    def plot_performance(self):
        """Plots the cumulative performance of the strategy vs. the benchmark."""
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            print("Portfolio backtest did not generate returns data to plot.")
            return
            
        strat_final = self.portfolio_returns['Strategy Cumulative'].iloc[-1]
        bench_final = self.portfolio_returns['Benchmark Cumulative'].iloc[-1]
        strat_sharpe = self.sharpe_ratio(self.portfolio_returns['Strategy Daily'])
        bench_sharpe = self.sharpe_ratio(self.portfolio_returns['Benchmark Daily'])

        print("\n--- Performance Summary ---")
        print(f"Strategy Final Cumulative Return: {strat_final:.2%}")
        print(f"Strategy Annualized Sharpe Ratio: {strat_sharpe:.2f}")
        print("-" * 20)
        print(f"Benchmark Final Cumulative Return: {bench_final:.2%}")
        print(f"Benchmark Annualized Sharpe Ratio: {bench_sharpe:.2f}")

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 7))
        self.portfolio_returns['Strategy Cumulative'].plot(label='Strategy', lw=2)
        self.portfolio_returns['Benchmark Cumulative'].plot(label='Benchmark (Equal Weight)', lw=2, linestyle='--')
        plt.title(f"Performance: '{self.allocation_strategy}' vs. Benchmark", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
