import yfinance as yf
import pandas as pd
from datetime import datetime

def get_data(symbol, start_date, end_date):
        data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False )
        if isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            known_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            for col in data.columns:
                chosen = None
                for part in col:
                    if isinstance(part, str) and part in known_fields:
                        chosen = part
                        break
                if chosen is None:
                    for part in reversed(col):
                        if isinstance(part, str) and part.strip() != "":
                            chosen = part
                            break
                if chosen is None:
                    chosen = "_".join([str(x) for x in col])
                new_cols.append(chosen)
            data.columns = new_cols
        if "Close" not in data.columns and "Adj Close" in data.columns:
            data["Close"] = data["Adj Close"]
        essentials = ["Open", "High", "Low", "Close"]
        if not all(col in data.columns for col in essentials):
            raise KeyError(f"Downloaded data missing essentials. Found: {list(data.columns)}")
        cols_to_keep = essentials + (["Volume"] if "Volume" in data.columns else [])
        data = data[cols_to_keep]
        return data