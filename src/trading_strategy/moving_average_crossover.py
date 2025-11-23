import numpy as np
import pandas as pd

class MovingAverageCrossover:
    def __init__(self):
        # No longer needs symbol, start_date, end_date as data is passed directly
        pass

    def _set_multiplier(self, direction):
        if direction == "long":
            pos_multiplier, neg_multiplier = 1, 0
        elif direction == "long_short":
            pos_multiplier, neg_multiplier = 1, -1
        else: # direction == "short"
            pos_multiplier, neg_multiplier = 0, -1
        return pos_multiplier, neg_multiplier

    def process(self, data, period_1=12, period_2=21, direction="long"):
        df = data.copy()
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)

        df[f"MA_{period_1}"] = df["Close"].rolling(window=period_1).mean()
        df[f"MA_{period_2}"] = df["Close"].rolling(window=period_2).mean()

        df["Signal"] = 0
        df.loc[df[f"MA_{period_1}"] > df[f"MA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{period_1}"] <= df[f"MA_{period_2}"], "Signal"] = neg_multiplier

        df.dropna(inplace=True)
        return df
