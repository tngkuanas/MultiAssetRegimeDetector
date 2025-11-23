import numpy as np
import pandas as pd

def preprocess_data(data):
    data["Log"] = np.log(data["Close"])
    data["Returns"] = data["Log"].diff()
    data["Range"] = (data["High"] / data["Low"]) - 1
    data.dropna(inplace=True)
    return data

def get_X_train(data):
    X_train = data[["Returns", "Range"]].iloc[:500]
    return X_train

def get_X_test(data):
    X_test = data[["Returns", "Range"]].iloc[500:]
    return X_test
    