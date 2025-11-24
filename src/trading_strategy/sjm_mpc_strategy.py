import pandas as pd
import numpy as np
from hmmlearn.hmm import GMMHMM
from scipy.optimize import minimize
from feature_engineering import FeatureEngineering

class SjmAndMpcStrategy:
    def __init__(self, n_regimes=3, sjm_lookback=1000, turnover_penalty=0.001, risk_aversion=0.5):
        """
        Initializes the SJM+MPC Holistic Strategy.
        
        :param n_regimes: The number of market regimes to model (e.g., Calm, Volatile, Crash).
        :param sjm_lookback: The rolling window size for training the SJM-proxy model.
        :param turnover_penalty: The cost factor for changing portfolio weights (trade cost).
        :param risk_aversion: The factor for the mean-variance risk term in the optimizer.
        """
        self.n_regimes = n_regimes
        self.sjm_lookback = sjm_lookback
        self.turnover_penalty = turnover_penalty
        self.risk_aversion = risk_aversion
        self.feature_engineer = FeatureEngineering()
        # Add min_covar to prevent "not positive definite" errors
        self.sjm_proxy = GMMHMM(n_components=self.n_regimes, n_mix=2, covariance_type="full", n_iter=10, min_covar=1e-3)
        self.feature_columns = [
            'ewma_returns', 'downside_deviation', 'macd', 'macd_signal', 
            'rsi', 'bb_width', 'stoch_k'
        ]

    def _run_sjm_proxy(self, features_df):
        """
        Fits the SJM-proxy (GMMHMM) on the provided features and extracts model parameters.
        """
        self.sjm_proxy.fit(features_df)
        
        # Extract transition matrix, means, and covariances for each regime
        trans_matrix = self.sjm_proxy.transmat_
        
        # Determine the most recent regime
        latest_regime = self.sjm_proxy.predict(features_df)[-1]
        
        return trans_matrix, latest_regime

    def _get_mpc_objective(self, weights, expected_returns, expected_cov, previous_weights):
        """
        The objective function for the MPC optimizer to MINIMIZE.
        It calculates: -Expected_Return + Risk_Aversion*Variance + Turnover_Penalty
        """
        # Portfolio's expected return (we want to maximize this, so we take the negative)
        portfolio_return = -np.sum(weights * expected_returns)
        
        # Portfolio's variance (risk)
        portfolio_variance = self.risk_aversion * np.dot(weights.T, np.dot(expected_cov, weights))
        
        # Turnover penalty (cost of trading)
        turnover = self.turnover_penalty * np.sum(np.abs(weights - previous_weights))
        
        return portfolio_return + portfolio_variance + turnover

    def _run_mpc_optimizer(self, forecasted_returns, forecasted_cov, previous_weights):
        """
        Runs the MPC-inspired portfolio optimization.
        """
        n_assets = len(forecasted_returns)
        
        # Initial guess for the weights (equal weight)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Constraints
        constraints = (
            # Weights must sum to 1 (fully invested)
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        )
        
        # Bounds for each weight (0 <= w <= 0.4)
        bounds = tuple((0, 0.4) for _ in range(n_assets))
        
        # Run the optimization
        result = minimize(
            fun=self._get_mpc_objective,
            x0=initial_weights,
            args=(forecasted_returns, forecasted_cov, previous_weights),
            method='SLSQP', # Sequential Least Squares Programming, good for constrained optimization
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            # If optimization fails, fall back to previous weights or equal weight
            return previous_weights if previous_weights is not None else initial_weights

    def process(self, historical_data_all_assets, macro_data, previous_weights):
        """
        The main method to be called by the PortfolioManager at each step of the backtest.
        
        :param historical_data_all_assets: A dict where keys are symbols and values are their historical data dfs.
        :param macro_data: A df of macroeconomic features.
        :param previous_weights: A numpy array of the portfolio weights from the previous period.
        :return: A numpy array of the new, optimal portfolio weights.
        """
        symbols = list(historical_data_all_assets.keys())
        asset_features_list = []
        asset_returns_list = []
        
        # 1. Engineer features and get returns for each asset
        for symbol in symbols:
            df = historical_data_all_assets[symbol]
            if df.empty or len(df) < 50: # Basic check for enough data
                continue
            features = self.feature_engineer.calculate_features(df)
            asset_log_returns = np.log(features['Close']).diff().dropna()
            
            common_index = features.index.intersection(asset_log_returns.index)
            if not common_index.empty:
                asset_features_list.append(features.loc[common_index][self.feature_columns])
                asset_returns_list.append(asset_log_returns.loc[common_index])

        if not asset_features_list:
            return previous_weights # Not enough data to proceed

        # Create feature DF with a multi-level index, then flatten it
        asset_features_df = pd.concat(asset_features_list, axis=1, keys=symbols)
        asset_features_df.columns = [f'{symbol}_{feature}' for symbol, feature in asset_features_df.columns]

        # Create returns DF with a single-level index (symbols)
        asset_returns_df = pd.concat(asset_returns_list, axis=1)
        asset_returns_df.columns = symbols
        
        # 2. Combine with macro data
        # Now that both have single-level columns, they can be joined
        if macro_data is not None:
            features_for_sjm = asset_features_df.join(macro_data)
        else:
            features_for_sjm = asset_features_df
        
        # Align all data to a common index and take the lookback window
        common_index = features_for_sjm.index.intersection(asset_returns_df.index)
        features_for_sjm = features_for_sjm.loc[common_index].tail(self.sjm_lookback)
        returns_for_sjm = asset_returns_df.loc[common_index].tail(self.sjm_lookback)

        # --- FIX for NaN values ---
        # Fill any remaining NaNs that might result from joining/alignment
        features_for_sjm.ffill(inplace=True)
        features_for_sjm.bfill(inplace=True)
        returns_for_sjm.ffill(inplace=True)
        returns_for_sjm.bfill(inplace=True)

        if features_for_sjm.empty or len(features_for_sjm) < self.n_regimes:
            return previous_weights # Not enough data for HMM

        # 3. Run the SJM-proxy to get model parameters
        trans_matrix, latest_regime = self._run_sjm_proxy(features_for_sjm)
        
        # 4. Generate forecasts based on regime analysis
        regime_assignments = self.sjm_proxy.predict(features_for_sjm)
        n_assets = returns_for_sjm.shape[1]

        # Calculate historical mean returns and covariances for each regime
        regime_mean_returns = []
        regime_covariances = []
        for i in range(self.n_regimes):
            regime_asset_returns = returns_for_sjm[regime_assignments == i]
            if len(regime_asset_returns) > n_assets: # Need more samples than assets for stable covariance
                regime_mean_returns.append(regime_asset_returns.mean().values)
                regime_covariances.append(regime_asset_returns.cov().values)
            else: # Handle regimes with too few samples
                regime_mean_returns.append(np.zeros(n_assets))
                regime_covariances.append(np.full((n_assets, n_assets), 1e-6)) # Small non-zero matrix

        # Get transition probabilities from the most recent regime
        next_regime_prob = trans_matrix[latest_regime]
        
        # Forecast expected returns as a probability-weighted average of regime means
        forecasted_returns = np.dot(next_regime_prob, np.array(regime_mean_returns))
        
        # Forecast covariance as a probability-weighted average of regime covariances
        forecasted_cov = sum(prob * cov for prob, cov in zip(next_regime_prob, regime_covariances))
        # Add regularization to forecasted covariance to ensure it's positive definite
        forecasted_cov += np.eye(n_assets) * 1e-6
        
        # 5. Run the MPC optimizer
        new_weights = self._run_mpc_optimizer(forecasted_returns, forecasted_cov, previous_weights)
        
        return new_weights

