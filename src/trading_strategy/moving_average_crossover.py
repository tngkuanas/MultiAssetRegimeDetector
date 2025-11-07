import numpy as np
import yfinance as yf
import pandas as pd
from data_collection import get_data

class MovingAverageCrossover:
    def __init__(self, symbol, start_date, end_date):
        self.df = self._extract_data(symbol, start_date, end_date)
        self.sharpe = 0

    def _extract_data(self, symbol, start_date, end_date):
        df = get_data(symbol, start_date, end_date)
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = df["High"] / df["Low"] - 1
        df["Bench_C_Rets"], sharpe = self._calculate_returns(df, True)
        self.sharpe = sharpe
        df.dropna(inplace=True)
        return df

    def _set_multiplier(self, direction):
        if direction == "long":
            pos_multiplier, neg_multiplier = 1, 0
        elif direction == "long_short":
            pos_multiplier, neg_multiplier = 1, -1
        else:
            pos_multiplier, neg_multiplier = 0, -1
        return pos_multiplier, neg_multiplier

    def _calculate_returns(self, df, is_benchmark):
        if not is_benchmark:
            multiplier_1 = df["Signal"].shift(1)
            multiplier_2 = 1 if "PSignal" not in df.columns else df["PSignal"].shift(1)
            log_rets = np.log(df["Open"].shift(-1) / df["Open"]) * multiplier_1 * multiplier_2
        else:
            log_rets = np.log(df["Close"] / df["Close"].shift(1))
        sharpe_ratio = self.sharpe_ratio(log_rets)
        c_log_rets = log_rets.cumsum()
        c_log_rets_exp = np.exp(c_log_rets) - 1
        return c_log_rets_exp, sharpe_ratio

    def sharpe_ratio(self, return_series):
        N = 255
        rf = 0.005
        mean = return_series.mean() * N - rf
        sigma = return_series.std() * np.sqrt(N)
        sharpe = round(mean / sigma, 3)
        return sharpe

    def change_df(self, new_df, drop_cols=[]):
        new_df = new_df.drop(columns=drop_cols)
        self.df = new_df

    def backtest_ma_crossover(self, period_1, period_2, direction, drop_cols=[]):
        df = self.df
        pos_multiplier, neg_multiplier = self._set_multiplier(direction)
        if f"MA_{period_1}" or f"MA_{period_2}" not in df.columns:
            df[f"MA_{period_1}"] = df["Close"].rolling(window=period_1).mean()
            df[f"MA_{period_2}"] = df["Close"].rolling(window=period_2).mean()
            df.dropna(inplace=True)
        df["Bench_C_Rets"], sharpe_ratio_bench = self._calculate_returns(df, True)
        df.loc[df[f"MA_{period_1}"] > df[f"MA_{period_2}"], "Signal"] = pos_multiplier
        df.loc[df[f"MA_{period_1}"] <= df[f"MA_{period_2}"], "Signal"] = neg_multiplier
        df["Strat_C_Rets"], sharpe_ratio_strat = self._calculate_returns(df, False)
        bench_rets = df["Bench_C_Rets"].values.astype(float)
        strat_rets = df["Strat_C_Rets"].values.astype(float)
        print("Sense check:", round(df["Close"].values[-1] / df["Close"].values[0] - 1, 3), round(bench_rets[-1], 3))
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)
        df = df.dropna()
        self.df = df
        return df, sharpe_ratio_bench, sharpe_ratio_strat
