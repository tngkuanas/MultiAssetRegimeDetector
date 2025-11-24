import pandas as pd

class MomentumScorer:
    """
    A strategy object that doesn't generate trade signals, but calculates
    the momentum of an asset over a specified lookback period and adds it
    to the DataFrame.
    """
    def __init__(self, lookback_period=126):
        """
        Initializes the MomentumScorer.

        Args:
            lookback_period (int): The number of days to look back to calculate
                                   the momentum/return. Defaults to 126 (approx. 6 months).
        """
        if lookback_period <= 0:
            raise ValueError("Lookback period must be positive.")
        self.lookback_period = lookback_period

    def process(self, data, macro_data=None):
        """
        Calculates the momentum for the asset and appends it as a new column.

        Args:
            data (pd.DataFrame): DataFrame with at least a 'Close' column.
            macro_data (pd.DataFrame, optional): Not used by this strategy, but
                                                 included for compatibility with
                                                 the StrategyManager. Defaults to None.

        Returns:
            pd.DataFrame: The original DataFrame with a 'MomentumScore' column added.
        """
        if 'Close' not in data.columns:
            raise ValueError("Input data must contain a 'Close' column.")
        
        data_copy = data.copy()
        
        # Calculate momentum as the percentage change over the lookback period
        data_copy['MomentumScore'] = data_copy['Close'].pct_change(periods=self.lookback_period)
        
        return data_copy
