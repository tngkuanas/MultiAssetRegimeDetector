import pandas as pd

class CorrelationRegimeModel:
    """
    A regime detection model that determines the correlation regime between
    two assets, typically stocks and bonds, to determine the effectiveness
    of bonds as a hedge.
    """
    def __init__(self, window=60, threshold=0.1):
        """
        Initializes the CorrelationRegimeModel.

        Args:
            window (int): The lookback window for calculating rolling correlation.
            threshold (float): The correlation value above which bonds are
                               considered a poor hedge.
        """
        if window <= 0:
            raise ValueError("Correlation window must be positive.")
        self.window = window
        self.threshold = threshold

    def process(self, stock_data, bond_data):
        """
        Determines the correlation regime.

        Args:
            stock_data (pd.DataFrame): DataFrame for the stock index.
            bond_data (pd.DataFrame): DataFrame for the bond index.

        Returns:
            pd.DataFrame: A DataFrame with a single 'CorrelationSignal' column.
                          -  1: "Normal" (Correlation is below threshold)
                          - -1: "Crisis" (Correlation is at or above threshold)
        """
        if 'Close' not in stock_data.columns or 'Close' not in bond_data.columns:
            raise ValueError("Input data must contain a 'Close' column.")
            
        # Calculate daily returns
        stock_returns = stock_data['Close'].pct_change()
        bond_returns = bond_data['Close'].pct_change()
        
        # Calculate rolling correlation
        rolling_corr = stock_returns.rolling(window=self.window).corr(bond_returns)
        
        # Create the signal DataFrame
        signal_df = pd.DataFrame(index=rolling_corr.index)
        signal_df['CorrelationSignal'] = 1 # Default to Normal
        signal_df.loc[rolling_corr >= self.threshold, 'CorrelationSignal'] = -1
        
        return signal_df
