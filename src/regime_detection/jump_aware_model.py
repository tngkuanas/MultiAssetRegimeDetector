import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class JumpAwareModel:
    def __init__(self, n_components=3, volatility_window=21, kurtosis_window=63, extreme_threshold_std=2):
        """
        Initializes the JumpAwareModel for regime detection.
        This model identifies regimes (Calm, Volatile, Crash) by clustering
        features related to volatility, kurtosis, and extreme negative returns.

        :param n_components: Number of regimes to detect. (Should be 3: Calm, Volatile, Crash).
        :param volatility_window: Rolling window for volatility calculation (e.g., 21 days for 1 month).
        :param kurtosis_window: Rolling window for kurtosis calculation (e.g., 63 days for 3 months).
        :param extreme_threshold_std: Number of standard deviations below which a return is considered extreme.
        """
        if n_components != 3:
            raise ValueError("JumpAwareModel currently supports only 3 components (regimes).")
        self.n_components = n_components
        self.volatility_window = volatility_window
        self.kurtosis_window = kurtosis_window
        self.extreme_threshold_std = extreme_threshold_std
        
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.regime_labels = {} # To map GMM cluster labels to meaningful names

    def _feature_engineer(self, data):
        """
        Engineers features for regime detection: volatility, kurtosis, and extreme negative returns.
        """
        df = data.copy()
        
        # Ensure 'Returns' column exists
        if 'Returns' not in df.columns:
            # If not present, calculate log returns.
            # This handles cases where the model is called directly on raw price data.
            df['Log_Close'] = np.log(df['Close'])
            df['Returns'] = df['Log_Close'].diff().fillna(0)
        
        # Volatility: Rolling standard deviation of returns
        df['Volatility'] = df['Returns'].rolling(window=self.volatility_window).std()

        # Kurtosis: Rolling kurtosis of returns
        df['Kurtosis'] = df['Returns'].rolling(window=self.kurtosis_window).kurt()
        
        # Extreme Negative Returns: Flag days with returns below a threshold
        # Calculate a rolling mean and std for the threshold
        rolling_mean = df['Returns'].rolling(window=self.volatility_window).mean()
        rolling_std = df['Returns'].rolling(window=self.volatility_window).std()
        
        df['Extreme_Negative'] = (df['Returns'] < (rolling_mean - self.extreme_threshold_std * rolling_std)).astype(int)
        
        # Fill NaNs created by rolling windows
        df.dropna(inplace=True)
        
        # Select features for clustering
        # Include the engineered features and any other non-price columns (i.e., macro data)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Log_Close', 'Returns', 'Momentum', 'Signal', 'PSignal']
        macro_cols = [col for col in df.columns if col not in price_cols and col not in ['Volatility', 'Kurtosis', 'Extreme_Negative']]
        
        feature_cols = ['Volatility', 'Kurtosis', 'Extreme_Negative'] + macro_cols
        
        # Ensure all feature columns exist before selecting
        features = df[[col for col in feature_cols if col in df.columns]]
        
        return features, df.index

    def fit(self, data):
        """
        Fits the GMM model using engineered features.
        :param data: pandas DataFrame with 'Close' prices and optionally 'Returns'.
        """
        features, _ = self._feature_engineer(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.gmm.fit(scaled_features)
        
        # Predict labels on the training data to assign meaningful names to regimes
        predicted_labels = self.gmm.predict(scaled_features)
        
        # Map GMM labels to human-readable regimes (Calm, Volatile, Crash)
        # This mapping is heuristic based on the mean of features for each cluster
        regime_means = pd.DataFrame(scaled_features, columns=features.columns, index=features.index).groupby(predicted_labels).mean()
        
        # Heuristic mapping:
        # Crash: Highest 'Extreme_Negative', highest 'Volatility' and 'Kurtosis'
        # Calm: Lowest 'Volatility', lowest 'Kurtosis', lowest 'Extreme_Negative'
        # Volatile: In between
        
        # Sort by Extreme_Negative, then Volatility to find potential crash/volatile regimes
        sorted_regimes = regime_means.sort_values(by=['Extreme_Negative', 'Volatility', 'Kurtosis'], ascending=[False, False, False])
        
        # Assign labels heuristically
        gmm_labels = sorted_regimes.index.tolist()
        
        # Default mapping, adjust if needed
        self.regime_labels[gmm_labels[0]] = "Crash" # Most extreme
        self.regime_labels[gmm_labels[1]] = "Volatile" # Middle
        self.regime_labels[gmm_labels[2]] = "Calm" # Least extreme
        
        # Double check if the most extreme is truly the 'Crash'
        # If the 'Crash' assigned regime has highest mean 'Extreme_Negative' and 'Volatility', then it's fine.
        # Otherwise, adjust logic or warn. For now, this is a simple heuristic.
        print("JumpAwareModel: GMM labels mapped to regimes:", self.regime_labels)

    def predict(self, data):
        """
        Predicts the regime for new data.
        :param data: pandas DataFrame with 'Close' prices and optionally 'Returns'.
        :return: A list of predicted regime labels (e.g., "Calm", "Volatile", "Crash").
        """
        features, index = self._feature_engineer(data)
        
        # Scale features using the scaler fitted during training
        scaled_features = self.scaler.transform(features)
        
        # Predict GMM cluster labels
        gmm_predictions = self.gmm.predict(scaled_features)
        
        # Map GMM labels to human-readable regime names
        predicted_regimes = [self.regime_labels[label] for label in gmm_predictions]
        
        return pd.Series(predicted_regimes, index=index, name="Regime")
