import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fredapi import Fred
from .config import API_KEY, SERIES, SELECTED_FEATURES
from .utils import fetch_series_monthly, make_sequences

def load_and_prepare_data(n_lags=36):
    fred = Fred(api_key=API_KEY)

    cols = {}
    for name, code in SERIES.items():
        how = "mean" if name in ["OilPrice"] else "last"
        cols[name] = fetch_series_monthly(fred, code, how=how)

    df = pd.DataFrame(cols).sort_index()
    df["Inflation"] = df["CPI"].pct_change() * 100
    df = df[SELECTED_FEATURES + ["Inflation"]].dropna()

    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end   = int(n_total * 0.85)

    feat_df = df[SELECTED_FEATURES]
    targ_sr = df["Inflation"]

    feat_train = feat_df.iloc[:train_end].copy()
    feat_val   = feat_df.iloc[train_end:val_end].copy()
    feat_test  = feat_df.iloc[val_end:].copy()

    y_train_sr = targ_sr.iloc[:train_end].copy()
    y_val_sr   = targ_sr.iloc[train_end:val_end].copy()
    y_test_sr  = targ_sr.iloc[val_end:].copy()

    scaler = StandardScaler()
    feat_train_sc = scaler.fit_transform(feat_train)
    feat_val_sc   = scaler.transform(feat_val)
    feat_test_sc  = scaler.transform(feat_test)

    X_train, y_train = make_sequences(feat_train_sc, y_train_sr.values, n_lags)
    X_val,   y_val   = make_sequences(feat_val_sc,   y_val_sr.values,   n_lags)
    X_test,  y_test  = make_sequences(feat_test_sc,  y_test_sr.values,  n_lags)

    return (X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr)
