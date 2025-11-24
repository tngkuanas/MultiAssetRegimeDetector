import pandas as pd
import numpy as np

class SMACrossoverRegime:
    """
    A regime detection model that determines the market regime based on a
    Simple Moving Average (SMA) crossover of a given market index.
    """
    def __init__(self, window=200):
        """
        Initializes the SMACrossoverRegime model.

        Args:
            window (int): The lookback window for the Simple Moving Average.
        """
        if window <= 0:
            raise ValueError("SMA window must be positive.")
        self.window = window

    def process(self, data):
        """
        Determines the market regime based on the SMA crossover.

        Args:
            data (pd.DataFrame): DataFrame with at least a 'Close' column,
                                 representing the market index data.

        Returns:
            pd.DataFrame: A DataFrame with a single 'PSignal' column.
                          -  1: "Risk-On" (Close is at or above the SMA)
                          - -1: "Risk-Off" (Close is below the SMA)
        """
        if 'Close' not in data.columns:
            raise ValueError("Input data must contain a 'Close' column.")
            
        data_copy = data.copy()
        
        # Calculate the Simple Moving Average
        sma = data_copy['Close'].rolling(window=self.window, min_periods=1).mean()
        
        # Determine the regime
        data_copy['PSignal'] = np.where(data_copy['Close'] >= sma, 1, -1)
        
        return data_copy[['PSignal']]
