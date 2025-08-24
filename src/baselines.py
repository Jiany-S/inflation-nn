# src/baselines.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge, Lasso

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

def _tabular_from_sequences(X3d, y):
    # average over time dimension (simple tabular projection)
    return X3d.mean(axis=1), y

def ridge_mae(X_test, y_test, X_train, y_train, alpha=1.0):
    Xtr, ytr = _tabular_from_sequences(X_train, y_train)
    Xte, yte = _tabular_from_sequences(X_test, y_test)
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = m.predict(Xte)
    return float(np.mean(np.abs(pred - yte)))

def lasso_mae(X_test, y_test, X_train, y_train, alpha=0.01):
    Xtr, ytr = _tabular_from_sequences(X_train, y_train)
    Xte, yte = _tabular_from_sequences(X_test, y_test)
    m = Lasso(alpha=alpha).fit(Xtr, ytr)
    pred = m.predict(Xte)
    return float(np.mean(np.abs(pred - yte)))
