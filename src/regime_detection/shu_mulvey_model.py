import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import FeatureEngineering

class ShuMulveyModel:
    """
    Implements the two-stage regime detection and forecasting model based on the paper
    by Shu & Mulvey.
    
    Stage 1: Non-parametric jump detection to identify historical regimes.
    Stage 2: Gradient-Boosted Decision Tree (GBDT) to forecast next-period regime probabilities.
    """
    def __init__(self, assets_data, macro_data, n_regimes=3, jump_threshold=3.0):
        self.assets_data = assets_data
        self.macro_data = macro_data
        self.n_regimes = n_regimes
        self.jump_threshold = jump_threshold
        self.models = {}
        self.regime_labels = {}
        self.feature_generator = FeatureEngineering()

    def _detect_regimes(self, returns):
        """
        Detects regimes based on return jumps and volatility clustering.
        This is a practical approximation of the non-parametric method.
        Regime 0: Low volatility
        Regime 1: High volatility
        Regime 2: Jump (positive or negative)
        """
        rolling_vol = returns.rolling(window=22).std()
        jumps = returns.abs() > (rolling_vol.shift(1) * self.jump_threshold)
        
        # Use a rolling quantile to avoid look-ahead bias
        high_vol_threshold = rolling_vol.rolling(window=252, min_periods=30).quantile(0.75)
        is_high_vol = rolling_vol > high_vol_threshold
        
        regime = pd.Series(0, index=returns.index, name="regime") # Default to low vol
        regime[is_high_vol] = 1 # High vol regime
        regime[jumps] = 2 # Jump regime overrides others
        
        return regime

    def fit(self):
        """
        Fit the two-stage model for each asset.
        """
        for asset_name, asset_df in self.assets_data.items():
            # Stage 1: Detect regimes
            returns = np.log(asset_df['Close']).diff().dropna()
            self.regime_labels[asset_name] = self._detect_regimes(returns)
            
            # Prepare features for this asset
            features = self.feature_generator.calculate_features(asset_df, self.macro_data)
            
            # Align features with next-period regime labels for training
            # We want to predict t+1's regime using t's features
            data = features.join(self.regime_labels[asset_name].shift(-1)).dropna()
            
            # Ensure we have enough data and all regimes are present
            if len(data) < 50 or len(data['regime'].unique()) < self.n_regimes:
                print(f"Skipping GBDT for {asset_name} due to insufficient data or regimes.")
                continue

            X = data.drop(columns=['regime'])
            y = data['regime']
            
            # Stage 2: Train GBDT model
            gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            gbdt.fit(X, y)
            self.models[asset_name] = gbdt
            print(f"Successfully trained GBDT model for {asset_name}.")

    def predict_proba(self):
        """
        Predict regime probabilities for the next period for all assets.
        """
        if not self.models:
            print("Models not trained yet. Fitting first.")
            self.fit()
            
        predictions = {}
        for asset_name, model in self.models.items():
            # Get the latest features for prediction
            features = self.feature_generator.calculate_features(self.assets_data[asset_name], self.macro_data)
            
            if features.empty:
                print(f"Could not generate features for {asset_name} to predict.")
                continue

            latest_features = features.iloc[[-1]]
            
            # Ensure columns match what the model was trained on
            model_cols = self.models[asset_name].feature_names_in_
            latest_features = latest_features.reindex(columns=model_cols, fill_value=0)

            probabilities = model.predict_proba(latest_features)
            # Ensure we return probabilities for all possible regimes
            class_probs = dict(zip(self.models[asset_name].classes_, probabilities[0]))
            full_probs = {i: class_probs.get(i, 0.0) for i in range(self.n_regimes)}
            predictions[asset_name] = full_probs
            
        return predictions

    def process(self):
        """
        A generic process method that fits and predicts.
        For compatibility with StrategyManager if ever needed, though it's used directly by PortfolioManager.
        """
        return self.predict_proba()
