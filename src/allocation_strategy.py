from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from regime_detection.shu_mulvey_model import ShuMulveyModel
from regime_detection.statistical_jump_model import StatisticalJumpModel
from feature_engineering import calculate_jump_model_features

class AllocationStrategy(ABC):
    requires_signals = True

    @abstractmethod
    def get_weights(self, aligned_data, **kwargs):
        """
        Determines daily portfolio weights based on the allocation strategy.

        :param aligned_data: A dictionary where keys are asset symbols and values are
                             DataFrames containing aligned historical data and signals.
        :return: A DataFrame with dates as the index, asset symbols as columns,
                 and portfolio weights as values.
        """
        pass

class EqualWeightAllocationStrategy(AllocationStrategy):
    def get_weights(self, aligned_data, **kwargs):
        """
        Determines daily portfolio weights based on equal weighting for active signals.
        """
        print("Determining portfolio weights using 'equal_weight' strategy...")
        symbols = list(aligned_data.keys())
        weights_df = pd.DataFrame(index=next(iter(aligned_data.values())).index)
        for date in weights_df.index:
            active_assets = [s for s, df in aligned_data.items() if df.loc[date, 'Signal'] > 0 and df.loc[date, 'PSignal'] == 1]
            if active_assets:
                weight = 1 / len(active_assets)
                for s in symbols:
                    weights_df.loc[date, s] = weight if s in active_assets else 0
            else:
                for s in symbols:
                    weights_df.loc[date, s] = 0
        return weights_df.fillna(0)

class ShuMulveyAllocationStrategy(AllocationStrategy):
    requires_signals = False

    def get_weights(self, aligned_data, **kwargs):
        """
        Determines weights using a walk-forward version of the Shu-Mulvey GBDT model
        and mean-variance optimization. Rebalances and retrains monthly.
        """
        print("Determining portfolio weights using 'shu_mulvey' walk-forward strategy...")
        
        symbols = list(aligned_data.keys())
        macro_data = kwargs.get('macro_data')
        
        weights_df = pd.DataFrame(index=next(iter(aligned_data.values())).index, columns=symbols).fillna(0.0)
        
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS')

        last_weights = np.array([1/len(symbols)] * len(symbols))

        for date in weights_df.index:
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                
                historical_data = {s: df.loc[:date] for s, df in aligned_data.items()}
                historical_macro_data = macro_data.loc[:date] if macro_data is not None else None
                
                model = ShuMulveyModel(assets_data=historical_data, macro_data=historical_macro_data)
                model.fit()

                regime_probs = model.predict_proba()

                regime_returns = {}
                for asset in symbols:
                    returns = np.log(historical_data[asset]['Close']).diff()
                    aligned_returns, aligned_regimes = returns.align(model.regime_labels[asset], join='inner')
                    regime_data = pd.DataFrame({'returns': aligned_returns, 'regime': aligned_regimes})
                    regime_returns[asset] = regime_data.groupby('regime')['returns'].mean()

                expected_returns = []
                for asset in symbols:
                    if not regime_probs.get(asset):
                        exp_ret = 0
                    else:
                        probs = pd.Series(regime_probs[asset])
                        cond_returns = regime_returns.get(asset, pd.Series())
                        exp_ret = probs.dot(cond_returns.reindex(probs.index).fillna(0.0))
                    expected_returns.append(exp_ret)

                returns_subset = pd.DataFrame({s: np.log(df['Close']).diff() for s, df in historical_data.items()}).iloc[-90:]
                cov_matrix = returns_subset.cov() * 252

                num_assets = len(symbols)
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

        weights_df.replace(0, np.nan, inplace=True)
        weights_df.ffill(inplace=True)
        weights_df.fillna(0, inplace=True)

        return weights_df

    def _neg_sharpe(self, weights, expected_returns, cov_matrix, risk_free_rate=0.02):
        """Helper for the optimizer: calculates the negative Sharpe ratio."""
        portfolio_return = np.sum(expected_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_volatility == 0:
            return 0
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

class JumpModelRiskParityAllocationStrategy(AllocationStrategy):
    requires_signals = False
    
    def __init__(self, hysteresis_period=5, max_turnover=0.5):
        self.hysteresis_period = hysteresis_period
        self.max_turnover = max_turnover

    def get_weights(self, aligned_data, **kwargs):
        """
        Determines weights using the Statistical Jump Model and regime-switched risk parity.
        Rebalances and retrains monthly, targeting a specific portfolio volatility based on the active regime.
        """
        print("Determining portfolio weights using 'jump_model_risk_parity' walk-forward strategy...")
        
        symbols = list(aligned_data.keys())
        
        weights_df = pd.DataFrame(index=next(iter(aligned_data.values())).index, columns=symbols).fillna(0.0)
        
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS')

        last_weights = np.array([1/len(symbols)] * len(symbols))
        
        vol_targets = {0: 0.15, 1: 0.10, 2: 0.05}

        for date in weights_df.index:
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                
                historical_data = {s: df.loc[:date] for s, df in aligned_data.items()}
                
                sjm_features = calculate_jump_model_features(historical_data)
                
                if sjm_features.empty:
                    print(f"  - Not enough data for feature calculation on {date}. Holding previous weights.")
                    weights_df.loc[date] = last_weights
                    continue

                sjm = StatisticalJumpModel(n_states=3, jump_penalty=100)
                regime_labels = sjm.fit(sjm_features)

                stable_regimes = regime_labels.rolling(window=self.hysteresis_period).apply(lambda x: x.iloc[-1] if x.nunique() == 1 else np.nan, raw=False).ffill()
                
                if pd.isna(stable_regimes.iloc[-1]):
                    print(f"  - Regime is unstable on {date}. Holding previous weights.")
                    weights_df.loc[date] = last_weights
                    continue
                
                current_regime = int(stable_regimes.iloc[-1])
                target_vol = vol_targets[current_regime]
                print(f"  - Current Stable Regime: {current_regime} | Target Volatility: {target_vol:.0%}")

                historical_returns = pd.DataFrame({
                    symbol: np.log(df['Close']).diff()
                    for symbol, df in historical_data.items()
                })
                
                aligned_returns, aligned_regimes = historical_returns.align(regime_labels, join='inner', axis=0)
                
                regime_specific_returns = aligned_returns[aligned_regimes == current_regime]
                
                if len(regime_specific_returns) < 30:
                    print(f"  - Not enough data in regime {current_regime} for covariance. Using full history.")
                    cov_matrix = aligned_returns.iloc[-90:].cov() * 252
                else:
                    cov_matrix = regime_specific_returns.cov() * 252
                
                num_assets = len(symbols)
                initial_weights = last_weights
                
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                    {'type': 'ineq', 'fun': lambda w: self.max_turnover - np.sum(np.abs(w - last_weights))}
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
