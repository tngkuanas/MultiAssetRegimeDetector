import numpy as np
import pandas as pd

def preprocess_data(data):
    data["Log"] = np.log(data["Close"])
    data["Returns"] = data["Log"].diff()
    data["Range"] = (data["High"] / data["Low"]) - 1
    data.dropna(inplace=True)
    return data
