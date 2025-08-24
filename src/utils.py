# src/utils.py
import numpy as np
import pandas as pd
from fredapi import Fred
from datetime import datetime
from pathlib import Path
from .config import RESULTS_DIR

def fetch_series_monthly(fred: Fred, code: str, how: str = "mean") -> pd.Series:
    s = fred.get_series(code).dropna()
    s.index = pd.to_datetime(s.index)
    df = s.to_frame("val")
    df.index = df.index.to_period("M").to_timestamp()
    df = df.groupby(level=0).mean() if how == "mean" else df.groupby(level=0).last()
    return df["val"]

def make_sequences(features_2d: np.ndarray, target_1d: np.ndarray, n_lags: int):
    X, y = [], []
    for i in range(n_lags, len(features_2d)):
        X.append(features_2d[i - n_lags:i, :])
        y.append(target_1d[i])
    return np.array(X), np.array(y)

def forecast_next(model, scaler, feat_df, n_lags):
    block = scaler.transform(feat_df.tail(n_lags))
    X_in = block.reshape(1, n_lags, feat_df.shape[1])
    return float(model.predict(X_in, verbose=0)[0, 0])

def make_run_dir(prefix="run"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out