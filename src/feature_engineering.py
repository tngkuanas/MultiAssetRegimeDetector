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
