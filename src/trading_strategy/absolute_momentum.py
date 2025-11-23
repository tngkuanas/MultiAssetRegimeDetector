import pandas as pd
import numpy as np

class AbsoluteMomentum:
    def __init__(self, lookback_period=252):
        """
        Initializes the AbsoluteMomentum strategy.
        :param lookback_period: The number of trading days to look back for momentum calculation.
                                Defaults to 252, approximately one year.
        """
        if lookback_period < 1:
            raise ValueError("Lookback period must be a positive integer.")
        self.lookback_period = lookback_period

    def process(self, data):
        """
        Generates a trading signal based on absolute momentum.
        Signal is 1 if the return over the lookback period is positive, 0 otherwise.
        Adds 'Signal' and 'Momentum' columns to the DataFrame.
        """
        df = data.copy()
        
        # Calculate the momentum (return over the lookback period)
        # Using log returns for the calculation
        df['Momentum'] = np.log(df['Close']).diff(self.lookback_period)
        
        # Generate the signal
        df['Signal'] = 0
        df.loc[df['Momentum'] > 0, 'Signal'] = 1
        
        df.dropna(inplace=True)
        
        return df
