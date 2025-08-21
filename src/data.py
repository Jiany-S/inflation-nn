# src/data.py
import numpy as np
import pandas as pd
from fredapi import Fred
from sklearn.preprocessing import StandardScaler

from .config import API_KEY, SERIES, SELECTED_FEATURES
from .utils import fetch_series_monthly, make_sequences

def _pct(x): return x.pct_change(fill_method=None) * 100
def _yoy(x): return x.pct_change(12, fill_method=None) * 100
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

    # Base engineered predictors
    feat = pd.DataFrame(index=raw.index)
    feat["Unemp_d"]       = _diff(raw["Unemployment"])
    feat["Rate_d"]        = _diff(raw["InterestRate"])
    feat["Oil_ret"]       = _logret(raw["OilPrice"])
    feat["PPI_yoy"]       = _yoy(raw["PPI"])
    feat["M2_yoy"]        = _yoy(raw["M2"])
    feat["Retail_yoy"]    = _yoy(raw["RetailSales"])
    feat["Employment_d"]  = _diff(raw["Employment"])
    feat["Housing_d"]     = _diff(raw["HousingStarts"])
    feat["T10Y"]          = raw["Treasury10Y"]     # level (expectations proxy)
    feat["Sentiment"]     = raw["Sentiment"]       # level

    # Simple polynomials
    feat["PPI_yoy_sq"]    = feat["PPI_yoy"].pow(2)
    feat["Oil_ret_sq"]    = feat["Oil_ret"].pow(2)
    feat["Rate_d_sq"]     = feat["Rate_d"].pow(2)
    feat["Unemp_d_sq"]    = feat["Unemp_d"].pow(2)

    # Interactions
    feat["Rate_Unemp"]    = feat["Rate_d"] * feat["Unemp_d"]
    feat["Oil_PPI"]       = feat["Oil_ret"] * feat["PPI_yoy"]

    # Rolling features on inflation itself (regime memory)
    infl_ma3              = infl_mom.rolling(3).mean()
    infl_ma6              = infl_mom.rolling(6).mean()
    infl_vol6             = infl_mom.rolling(6).std()
    feat["Infl_ma3"]      = infl_ma3
    feat["Infl_ma6"]      = infl_ma6
    feat["Infl_vol6"]     = infl_vol6
    feat["Inflation_prev"]= infl_mom  # sequences will lag it automatically

    df = pd.concat([feat, infl_mom.rename("Inflation")], axis=1).dropna()

    return df, SELECTED_FEATURES

from sklearn.preprocessing import StandardScaler
from .utils import make_sequences

def load_and_prepare_data(n_lags=60):
    """
    Build scaled sequences with chronological split.
    Uses carry-over context so val/test have n_lags lookback without leakage.
    Returns: X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr
    """
    df, features = load_dataset()

    # ------------- chronological split -------------
    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end   = int(n_total * 0.85)

    feat_df = df[features]
    y_sr    = df["Inflation"]

    # raw splits
    feat_train = feat_df.iloc[:train_end].copy()
    feat_val   = feat_df.iloc[train_end:val_end].copy()
    feat_test  = feat_df.iloc[val_end:].copy()

    y_train_sr = y_sr.iloc[:train_end].copy()
    y_val_sr   = y_sr.iloc[train_end:val_end].copy()
    y_test_sr  = y_sr.iloc[val_end:].copy()

    # ------------- scaler (fit on TRAIN only) -------------
    scaler = StandardScaler().fit(feat_train.values)

    # ------------- TRAIN sequences -------------
    X_train_sc = scaler.transform(feat_train.values)
    X_train, y_train = make_sequences(X_train_sc, y_train_sr.values, n_lags)

    # ------------- VAL sequences with context from TRAIN -------------
    # Block covers [train_end - n_lags, val_end)
    val_ctx_start = max(0, train_end - n_lags)
    feat_val_block = feat_df.iloc[val_ctx_start:val_end].values
    y_val_block    = y_sr.iloc[val_ctx_start:val_end].values

    X_val_sc = scaler.transform(feat_val_block)
    X_val_all, y_val_all = make_sequences(X_val_sc, y_val_block, n_lags)

    # Keep only sequences whose TARGET falls inside [train_end, val_end)
    first_target_idx_in_block = val_ctx_start + n_lags
    keep_from = max(0, train_end - first_target_idx_in_block)
    X_val = X_val_all[keep_from:]
    y_val = y_val_all[keep_from:]

    # ------------- TEST sequences with context from VAL -------------
    test_ctx_start = max(0, val_end - n_lags)
    feat_test_block = feat_df.iloc[test_ctx_start:].values
    y_test_block    = y_sr.iloc[test_ctx_start:].values

    X_test_sc = scaler.transform(feat_test_block)
    X_test_all, y_test_all = make_sequences(X_test_sc, y_test_block, n_lags)

    first_target_idx_in_test_block = test_ctx_start + n_lags
    keep_from_test = max(0, val_end - first_target_idx_in_test_block)
    X_test = X_test_all[keep_from_test:]
    y_test = y_test_all[keep_from_test:]

    # ------------- sanity checks -------------
    for name, X, y in [("train", X_train, y_train),
                       ("val",   X_val,   y_val),
                       ("test",  X_test,  y_test)]:
        if len(X) == 0:
            raise ValueError(
                f"{name} split still has 0 sequences after n_lags={n_lags}. "
                f"Lower n_lags or adjust split sizes."
            )

    return X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr

