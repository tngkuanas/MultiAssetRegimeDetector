import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, lookback_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20):
        """
        Initializes the FeatureEngineering class with parameters for the technical indicators.
        :param lookback_period: General lookback period for RSI, Stochastic Oscillator, etc.
        :param macd_fast: Fast period for MACD.
        :param macd_slow: Slow period for MACD.
        :param macd_signal: Signal period for MACD.
        :param bb_window: Window for Bollinger Bands.
        """
        self.lookback_period = lookback_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_window = bb_window

    def calculate_features(self, df, macro_df=None):
        """
        Calculates all specified technical indicators for the given DataFrame.
        The DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.
        :param df: The asset's price DataFrame.
        :param macro_df: Optional DataFrame with macroeconomic data, indexed by date.
        """
        data = df.copy()

        # --- Basic Returns and Volatility ---
        data['log_returns'] = np.log(data['Close']).diff()
        data['rolling_volatility'] = data['log_returns'].rolling(window=self.lookback_period).std() * np.sqrt(self.lookback_period)

        # --- Lagged Returns ---
        for lag in [1, 5, 10, 21]:
            data[f'lagged_return_{lag}d'] = data['log_returns'].shift(lag)

        # --- Momentum ---
        # Using RSI as a momentum indicator
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.lookback_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.lookback_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi'] = data['rsi'].fillna(50)

        # --- Drawdown ---
        rolling_max = data['Close'].cummax()
        data['drawdown'] = (data['Close'] - rolling_max) / rolling_max

        # --- MACD ---
        ewma_fast = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ewma_slow = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        data['macd'] = ewma_fast - ewma_slow
        data['macd_signal'] = data['macd'].ewm(span=self.macd_signal, adjust=False).mean()

        # --- Bollinger Bands ---
        sma = data['Close'].rolling(window=self.bb_window).mean()
        std = data['Close'].rolling(window=self.bb_window).std()
        data['bb_width'] = ((sma + (std * 2)) - (sma - (std * 2))) / sma

        # --- Stochastic Oscillator ---
        low_n = data['Low'].rolling(window=self.lookback_period).min()
        high_n = data['High'].rolling(window=self.lookback_period).max()
        data['stoch_k'] = 100 * ((data['Close'] - low_n) / (high_n - low_n))
        data['stoch_k'] = data['stoch_k'].fillna(50)

        # --- Merge Macro Data ---
        if macro_df is not None:
            data = data.join(macro_df, how='left')
            data.ffill(inplace=True) # Forward-fill macro data for days with missing values

        # Drop intermediate columns and any rows with NaN values
        data.drop(columns=['log_returns'], inplace=True)
        data.dropna(inplace=True)

        return data

def calculate_jump_model_features(aligned_data):
    """
    Calculates the 'Holy Trinity' of features for the Statistical Jump Model.
    The features are standardized using a rolling 252-day window to avoid lookahead bias.

    :param aligned_data: A dictionary of DataFrames, where keys are asset symbols
                         and values are the corresponding aligned price data.
    :return: A DataFrame containing the standardized features.
    """
    # 1. Create a unified DataFrame of log returns for all assets
    returns_df = pd.DataFrame({
        symbol: np.log(df['Close']).diff()
        for symbol, df in aligned_data.items()
    })

    # 2. Calculate the four core features
    # Feature 1: 20-day Realized Volatility (averaged across assets)
    volatility = returns_df.rolling(window=20).std().mean(axis=1) * np.sqrt(252) # Annualized

    # Feature 2: 21-day Momentum (averaged across assets)
    momentum_21d = returns_df.rolling(window=21).sum().mean(axis=1)

    # Feature 3: 63-day Momentum (averaged across assets)
    momentum_63d = returns_df.rolling(window=63).sum().mean(axis=1)

    # Feature 4: 63-day Average Pairwise Correlation
    # Use the built-in rolling.corr() for efficiency and correctness
    rolling_corr = returns_df.rolling(window=63).corr()
    
    # The result is a multi-index DF. Group by date and calculate the mean of the lower triangle.
    avg_correlation = rolling_corr.groupby(level=0).apply(
        lambda x: x.where(np.tril(np.ones(x.shape), k=-1).astype(bool)).stack().mean()
    )
    avg_correlation.name = 'avg_correlation'

    # 3. Combine features into a single DataFrame
    features = pd.DataFrame({
        'volatility': volatility,
        'momentum_21d': momentum_21d,
        'momentum_63d': momentum_63d,
        'avg_correlation': avg_correlation
    }).dropna()

    # 4. Standardize the features using a rolling window to prevent lookahead bias
    rolling_mean = features.rolling(window=252, min_periods=63).mean()
    rolling_std = features.rolling(window=252, min_periods=63).std()
    
    standardized_features = (features - rolling_mean) / rolling_std
    
    return standardized_features.dropna()
