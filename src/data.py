# src/data.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from fredapi import Fred
from sklearn.preprocessing import StandardScaler

from .config import API_KEY, SERIES, SELECTED_FEATURES
from .utils import fetch_series_monthly, make_sequences


# ----------------------------
# Helpers for transformations
# ----------------------------
def _pct(x: pd.Series) -> pd.Series:
    # Month-over-month % change * 100
    return x.pct_change(fill_method=None) * 100.0


def _yoy(x: pd.Series) -> pd.Series:
    # Year-over-year % change * 100
    return x.pct_change(12, fill_method=None) * 100.0


def _diff(x: pd.Series) -> pd.Series:
    return x.diff()


def _logret(x: pd.Series) -> pd.Series:
    # Log return * 100 for approximate % return
    return np.log(x).diff() * 100.0


# ----------------------------
# FRED key loading (robust)
# ----------------------------
def _load_fred() -> Fred:
    """
    Load FRED with an API key:
    - Prefer FRED_API_KEY from .env / environment.
    - Fallback to config.API_KEY.
    Validate as 32 lower-case alphanumeric; raise helpful error otherwise.
    """
    # Try to find and load a .env (first via auto-discovery, then project root)
    _ = load_dotenv(find_dotenv())
    if not os.getenv("FRED_API_KEY"):
        project_root_env = Path(__file__).resolve().parents[1] / ".env"
        if project_root_env.exists():
            load_dotenv(project_root_env, override=False)

    env_key = (os.getenv("FRED_API_KEY") or "").strip()
    cfg_key = (API_KEY or "").strip()

    key = env_key or cfg_key

    # Normalize and validate (FRED complains if not 32 lower-case alnum)
    # If user provided uppercase, attempt to normalize to lower-case if format matches.
    if re.fullmatch(r"[A-Za-z0-9]{32}", key or ""):
        key = key.lower()

    if not re.fullmatch(r"[a-z0-9]{32}", key or ""):
        raise RuntimeError(
            "FRED API key missing/invalid.\n"
            "Set FRED_API_KEY in your .env (project root) as a 32-character lower-case alphanumeric string, e.g.:\n"
            "FRED_API_KEY=0123456789abcdef0123456789abcdef\n"
            "Alternatively, set API_KEY in src/config.py. If you just set it, restart your shell/IDE."
        )

    return Fred(api_key=key)


# ----------------------------
# Data loading / feature build
# ----------------------------
def load_raw_from_fred() -> pd.DataFrame:
    fred = _load_fred()
    cols: dict[str, pd.Series] = {}

    for name, code in SERIES.items():
        how = "mean" if name in ["OilPrice"] else "last"
        cols[name] = fetch_series_monthly(fred, code, how=how)

    df = pd.DataFrame(cols).sort_index()

    # Sanity: ensure the essential columns exist (better error messages)
    required = [
        "CPI",
        "Unemployment",
        "InterestRate",
        "OilPrice",
        "PPI",
        "M2",
        "RetailSales",
        "Employment",
        "HousingStarts",
        "Treasury10Y",
        "Sentiment",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required series from FRED fetch: {missing}. "
            f"Check your SERIES mapping in config and that the FRED codes are valid."
        )

    return df


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the modeling dataframe with engineered features and 'Inflation' target.
    Returns:
        df: DataFrame with predictors + 'Inflation'
        features: list of column names to use as features
    """
    raw = load_raw_from_fred()

    # Target: MoM CPI % change
    infl_mom = _pct(raw["CPI"])

    # Base engineered predictors
    feat = pd.DataFrame(index=raw.index)
    feat["Unemp_d"] = _diff(raw["Unemployment"])
    feat["Rate_d"] = _diff(raw["InterestRate"])
    feat["Oil_ret"] = _logret(raw["OilPrice"])
    feat["PPI_yoy"] = _yoy(raw["PPI"])
    feat["M2_yoy"] = _yoy(raw["M2"])
    feat["Retail_yoy"] = _yoy(raw["RetailSales"])
    feat["Employment_d"] = _diff(raw["Employment"])
    feat["Housing_d"] = _diff(raw["HousingStarts"])
    feat["T10Y"] = raw["Treasury10Y"]  # level (expectations proxy)
    feat["Sentiment"] = raw["Sentiment"]  # level

    # Simple polynomials
    feat["PPI_yoy_sq"] = feat["PPI_yoy"].pow(2)
    feat["Oil_ret_sq"] = feat["Oil_ret"].pow(2)
    feat["Rate_d_sq"] = feat["Rate_d"].pow(2)
    feat["Unemp_d_sq"] = feat["Unemp_d"].pow(2)

    # Interactions
    feat["Rate_Unemp"] = feat["Rate_d"] * feat["Unemp_d"]
    feat["Oil_PPI"] = feat["Oil_ret"] * feat["PPI_yoy"]

    # Rolling features on inflation itself (regime memory)
    infl_ma3 = infl_mom.rolling(window=3).mean()
    infl_ma6 = infl_mom.rolling(window=6).mean()
    infl_vol6 = infl_mom.rolling(window=6).std()

    feat["Infl_ma3"] = infl_ma3
    feat["Infl_ma6"] = infl_ma6
    feat["Infl_vol6"] = infl_vol6
    feat["Inflation_prev"] = infl_mom  # sequences will lag it automatically

    # Final DF (drop rows with any NaNs from lags/rolling)
    df = pd.concat([feat, infl_mom.rename("Inflation")], axis=1).dropna()

    return df, SELECTED_FEATURES


def load_and_prepare_data(
    n_lags: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, StandardScaler, pd.Series]:
    """
    Build scaled sequences with chronological split.
    Uses carry-over context so val/test have n_lags lookback without leakage.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr
    """
    df, features = load_dataset()

    # ------------- chronological split -------------
    n_total = len(df)
    if n_total < (n_lags * 3 + 30):  # heuristic minimum for splits
        raise ValueError(
            f"Not enough rows ({n_total}) for n_lags={n_lags}. "
            f"Either lower n_lags or fetch a longer history."
        )

    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    feat_df = df[features]
    y_sr = df["Inflation"]

    # raw splits
    feat_train = feat_df.iloc[:train_end].copy()
    feat_val = feat_df.iloc[train_end:val_end].copy()
    feat_test = feat_df.iloc[val_end:].copy()

    y_train_sr = y_sr.iloc[:train_end].copy()
    y_val_sr = y_sr.iloc[train_end:val_end].copy()
    y_test_sr = y_sr.iloc[val_end:].copy()

    # ------------- scaler (fit on TRAIN only) -------------
    scaler = StandardScaler().fit(feat_train.values)

    # ------------- TRAIN sequences -------------
    X_train_sc = scaler.transform(feat_train.values)
    X_train, y_train = make_sequences(X_train_sc, y_train_sr.values, n_lags)

    # ------------- VAL sequences with context from TRAIN -------------
    # Block covers [train_end - n_lags, val_end)
    val_ctx_start = max(0, train_end - n_lags)
    feat_val_block = feat_df.iloc[val_ctx_start:val_end].values
    y_val_block = y_sr.iloc[val_ctx_start:val_end].values

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
    y_test_block = y_sr.iloc[test_ctx_start:].values

    X_test_sc = scaler.transform(feat_test_block)
    X_test_all, y_test_all = make_sequences(X_test_sc, y_test_block, n_lags)

    first_target_idx_in_test_block = test_ctx_start + n_lags
    keep_from_test = max(0, val_end - first_target_idx_in_test_block)
    X_test = X_test_all[keep_from_test:]
    y_test = y_test_all[keep_from_test:]

    # ------------- sanity checks -------------
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        if len(X) == 0:
            raise ValueError(
                f"{name} split still has 0 sequences after n_lags={n_lags}. "
                f"Lower n_lags or adjust split sizes."
            )

    return X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr
