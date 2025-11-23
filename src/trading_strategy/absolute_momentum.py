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

    def process(self, data, macro_data=None):
        """
        Generates a trading signal based on absolute momentum, filtered by a macro indicator.
        Signal is 1 if momentum is positive AND the T10Y2Y spread is positive.
        """
        df = data.copy()

        # Calculate momentum
        df['Momentum'] = np.log(df['Close']).diff(self.lookback_period)
        
        # Merge macro data
        if macro_data is not None and '10y-2y_spread' in macro_data.columns:
            df = df.join(macro_data['10y-2y_spread'])
            df.ffill(inplace=True)
            df.dropna(inplace=True) # Drop rows where macro data might be missing at the start
            
            # Generate signal based on both momentum and macro indicator
            df['Signal'] = 0
            df.loc[(df['Momentum'] > 0) & (df['10y-2y_spread'] > 0), 'Signal'] = 1
        else:
            # If no macro data, fall back to original momentum-only logic
            df['Signal'] = 0
            df.loc[df['Momentum'] > 0, 'Signal'] = 1

        df.dropna(inplace=True)
        return df
