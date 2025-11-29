from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
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

class JumpModelRiskParityAllocationStrategy(AllocationStrategy):
    requires_signals = False
    
    def __init__(self, hysteresis_period=5, max_turnover=0.5, vol_targets=None, n_states=3, jump_penalty=100):
        self.hysteresis_period = hysteresis_period
        self.max_turnover = max_turnover
        self.vol_targets = vol_targets if vol_targets is not None else {0: 0.15, 1: 0.10, 2: 0.05}
        self.n_states = n_states
        self.jump_penalty = jump_penalty

    def get_weights(self, aligned_data, **kwargs):
        """
        Determines weights using the Statistical Jump Model and regime-switched risk parity.
        Rebalances and retrains monthly, targeting a specific portfolio volatility based on the active regime.
        Returns both the final weights and the regime labels for the full period.
        """
        print("Determining portfolio weights using 'jump_model_risk_parity' walk-forward strategy...")
        
        symbols = list(aligned_data.keys())
        full_index = next(iter(aligned_data.values())).index
        weights_df = pd.DataFrame(index=full_index, columns=symbols).fillna(0.0)
        
        min_training_period = pd.DateOffset(years=1)
        first_rebalance_date = weights_df.index[0] + min_training_period
        rebalance_dates = pd.date_range(start=first_rebalance_date, end=weights_df.index[-1], freq='BMS')

        last_weights = np.array([1/len(symbols)] * len(symbols))
        
        # This loop calculates the weights on a walk-forward basis
        for date in weights_df.index:
            if date in rebalance_dates:
                print(f"Rebalancing for month of {date.strftime('%Y-%m')}...")
                
                historical_data = {s: df.loc[:date] for s, df in aligned_data.items()}
                
                sjm_features = calculate_jump_model_features(historical_data)
                
                if sjm_features.empty:
                    weights_df.loc[date] = last_weights
                    continue

                sjm = StatisticalJumpModel(n_states=self.n_states, jump_penalty=self.jump_penalty)
                regime_labels = sjm.fit(sjm_features)

                stable_regimes = regime_labels.rolling(window=self.hysteresis_period).apply(lambda x: x.iloc[-1] if x.nunique() == 1 else np.nan, raw=False).ffill()
                
                if pd.isna(stable_regimes.iloc[-1]):
                    weights_df.loc[date] = last_weights
                    continue
                
                current_regime = int(stable_regimes.iloc[-1])
                target_vol = self.vol_targets.get(current_regime)
                
                # Fallback if regime code is unexpected
                if target_vol is None:
                    print(f"  - Warning: No vol target for regime {current_regime}. Using last weights.")
                    weights_df.loc[date] = last_weights
                    continue

                historical_returns = pd.DataFrame({
                    symbol: np.log(df['Close']).diff()
                    for symbol, df in historical_data.items()
                })
                
                aligned_returns, aligned_regimes = historical_returns.align(regime_labels, join='inner', axis=0)
                regime_specific_returns = aligned_returns[aligned_regimes == current_regime]
                
                cov_matrix = aligned_returns.iloc[-90:].cov() * 252 if len(regime_specific_returns) < 30 else regime_specific_returns.cov() * 252
                
                result = minimize(self._volatility_objective, last_weights,
                                  args=(cov_matrix, target_vol),
                                  method='SLSQP',
                                  bounds=tuple((0.0, 1.0) for _ in symbols),
                                  constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                                               {'type': 'ineq', 'fun': lambda w: self.max_turnover - np.sum(np.abs(w - last_weights))}])
                
                if result.success:
                    last_weights = result.x
                
            weights_df.loc[date] = last_weights
            
        weights_df.ffill(inplace=True)
        
        # --- Final Regime Calculation for UI ---
        # To provide a consistent regime map for the UI, we run the SJM once on the full history.
        print("Calculating final regime map for UI...")
        final_sjm_features = calculate_jump_model_features(aligned_data)
        final_sjm = StatisticalJumpModel(n_states=self.n_states, jump_penalty=self.jump_penalty)
        final_regime_labels = final_sjm.fit(final_sjm_features)
        
        return weights_df, final_regime_labels

    def _volatility_objective(self, weights, cov_matrix, target_vol):
        """Objective function for the optimizer: minimize difference to target volatility."""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return (portfolio_vol - target_vol)**2
