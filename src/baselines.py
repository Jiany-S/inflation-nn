# src/baselines.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def mae(y_true, y_pred): return float(np.mean(np.abs(np.array(y_true)-np.array(y_pred))))

def naive_last(y: pd.Series):
    return y.shift(1)

def seasonal_naive(y: pd.Series, season=12):
    return y.shift(season)

def arima_forecast(y_train: pd.Series, y_test: pd.Series, order=(1,0,0)):
    model = ARIMA(y_train, order=order)
    fitted = model.fit()
    pred = fitted.forecast(steps=len(y_test))
    return pd.Series(pred, index=y_test.index)
