# src/data.py
import numpy as np
import pandas as pd
from fredapi import Fred
from sklearn.preprocessing import StandardScaler

from .config import API_KEY, SERIES, SELECTED_FEATURES
from .utils import fetch_series_monthly, make_sequences

def _pct(x): return x.pct_change() * 100
def _yoy(x): return x.pct_change(12) * 100
def _diff(x): return x.diff()
def _logret(x): return np.log(x).diff() * 100

def load_raw_from_fred():
    fred = Fred(api_key=API_KEY)
    cols = {}
    for name, code in SERIES.items():
        how = "mean" if name in ["OilPrice"] else "last"
        cols[name] = fetch_series_monthly(fred, code, how=how)
    return pd.DataFrame(cols).sort_index()

def load_dataset():
    """Return (df, features) where df has engineered features + 'Inflation' target."""
    raw = load_raw_from_fred()

    # Target: MoM CPI % change
    infl_mom = _pct(raw["CPI"])

    # Engineered predictors
    feat = pd.DataFrame(index=raw.index)
    feat["Unemp_d"]      = _diff(raw["Unemployment"])
    feat["Rate_d"]       = _diff(raw["InterestRate"])
    feat["Oil_ret"]      = _logret(raw["OilPrice"])
    feat["PPI_yoy"]      = _yoy(raw["PPI"])
    feat["M2_yoy"]       = _yoy(raw["M2"])
    feat["Retail_yoy"]   = _yoy(raw["RetailSales"])
    feat["Employment_d"] = _diff(raw["Employment"])
    feat["Housing_d"]    = _diff(raw["HousingStarts"])
    feat["T10Y"]         = raw["Treasury10Y"]   # level
    feat["Sentiment"]    = raw["Sentiment"]     # level

    df = pd.concat([feat, infl_mom.rename("Inflation")], axis=1).dropna()
    return df, SELECTED_FEATURES

def load_and_prepare_data(n_lags=36):
    """
    Builds scaled sequences with chronological split.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr
    """
    df, features = load_dataset()

    # Chronological split
    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end   = int(n_total * 0.85)

    feat_df = df[features]
    y_sr    = df["Inflation"]

    feat_train = feat_df.iloc[:train_end].copy()
    feat_val   = feat_df.iloc[train_end:val_end].copy()
    feat_test  = feat_df.iloc[val_end:].copy()

    y_train_sr = y_sr.iloc[:train_end].copy()
    y_val_sr   = y_sr.iloc[train_end:val_end].copy()
    y_test_sr  = y_sr.iloc[val_end:].copy()

    # Scale on train only
    scaler = StandardScaler().fit(feat_train.values)
    X_train_sc = scaler.transform(feat_train.values)
    X_val_sc   = scaler.transform(feat_val.values)
    X_test_sc  = scaler.transform(feat_test.values)

    # Sequences for LSTM
    X_train, y_train = make_sequences(X_train_sc, y_train_sr.values, n_lags)
    X_val,   y_val   = make_sequences(X_val_sc,   y_val_sr.values,   n_lags)
    X_test,  y_test  = make_sequences(X_test_sc,  y_test_sr.values,  n_lags)

    return X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr
