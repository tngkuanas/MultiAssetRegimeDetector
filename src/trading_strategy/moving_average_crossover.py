import numpy as np
import pandas as pd

class MovingAverageCrossover:
    def __init__(self, period_1=12, period_2=21, direction="long"):
        self.period_1 = period_1
        self.period_2 = period_2
        self.direction = direction

    def _set_multiplier(self):
        if self.direction == "long":
            pos_multiplier, neg_multiplier = 1, 0
        elif self.direction == "long_short":
            pos_multiplier, neg_multiplier = 1, -1
        else: # direction == "short"
            pos_multiplier, neg_multiplier = 0, -1
        return pos_multiplier, neg_multiplier

    def process(self, data, macro_data=None):
        df = data.copy()
        pos_multiplier, neg_multiplier = self._set_multiplier()

        df[f"MA_{self.period_1}"] = df["Close"].rolling(window=self.period_1).mean()
        df[f"MA_{self.period_2}"] = df["Close"].rolling(window=self.period_2).mean()

        df["Signal"] = 0
        df.loc[df[f"MA_{self.period_1}"] > df[f"MA_{self.period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{self.period_1}"] <= df[f"MA_{self.period_2}"], "Signal"] = neg_multiplier

        df.dropna(inplace=True)
        return df
