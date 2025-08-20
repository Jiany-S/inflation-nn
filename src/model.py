import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from fredapi import Fred
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Config
# -------------------------------
API_KEY = "9fc23489b39a61d02d244276cb0d000b"
N_LAGS = 36
TRAIN = True  # set False to just load and evaluate

# Small, focused feature set
SELECTED_FEATURES = ["CPI", "Unemployment", "InterestRate", "OilPrice", "PPI"]

# If larger set:
# SELECTED_FEATURES = ["CPI", "Unemployment", "InterestRate", "M2", "Treasury10Y",
#                      "OilPrice", "PPI", "RetailSales", "Sentiment", "Employment", "HousingStarts"]

SERIES = {
    "CPI": "CPIAUCNS",            # monthly
    "Unemployment": "UNRATE",     # monthly
    "InterestRate": "FEDFUNDS",   # monthly
    "M2": "M2SL",                 # monthly
    "Treasury10Y": "GS10",        # monthly
    "OilPrice": "DCOILWTICO",     # daily -> monthly avg
    "PPI": "PPIACO",              # monthly
    "RetailSales": "RSXFS",       # monthly
    "Sentiment": "UMCSENT",       # monthly
    "Employment": "PAYEMS",       # monthly
    "HousingStarts": "HOUST"      # monthly
}

# -------------------------------
# Helpers
# -------------------------------
def fetch_series_monthly(fred: Fred, code: str, how: str = "mean") -> pd.Series:
    """Fetch a FRED series and aggregate to monthly (handles daily -> monthly)."""
    s = fred.get_series(code).dropna()
    s.index = pd.to_datetime(s.index)
    df = s.to_frame("val")
    # Move to month buckets, then aggregate duplicates (daily -> monthly)
    df.index = df.index.to_period("M").to_timestamp()
    if how == "last":
        df = df.groupby(level=0).last()
    else:
        df = df.groupby(level=0).mean()
    return df["val"]

def make_sequences(features_2d: np.ndarray, target_1d: np.ndarray, n_lags: int):
    """Build 3D sequences for LSTM from scaled features and aligned target arrays."""
    X, y = [], []
    for i in range(n_lags, len(features_2d)):
        X.append(features_2d[i - n_lags:i, :])
        y.append(target_1d[i])
    return np.array(X), np.array(y)

# -------------------------------
# 1) Data
# -------------------------------
fred = Fred(api_key=API_KEY)

# Fetch, align monthly, and combine
cols = {}
for name, code in SERIES.items():
    how = "mean" if name in ["OilPrice"] else "last"
    cols[name] = fetch_series_monthly(fred, code, how=how)

df = pd.DataFrame(cols).sort_index()

# Target: monthly inflation (% change in CPI)
df["Inflation"] = df["CPI"].pct_change() * 100

# Keep only selected features + target
df = df[SELECTED_FEATURES + ["Inflation"]].dropna()

# -------------------------------
# 2) Split BEFORE scaling (chronological)
# -------------------------------
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

# -------------------------------
# 3) Scale using TRAIN only
# -------------------------------
scaler = StandardScaler()
feat_train_sc = scaler.fit_transform(feat_train)
feat_val_sc   = scaler.transform(feat_val)
feat_test_sc  = scaler.transform(feat_test)

# -------------------------------
# 4) Build sequences for LSTM
# -------------------------------
X_train, y_train = make_sequences(feat_train_sc, y_train_sr.values, N_LAGS)
X_val,   y_val   = make_sequences(feat_val_sc,   y_val_sr.values,   N_LAGS)
X_test,  y_test  = make_sequences(feat_test_sc,  y_test_sr.values,  N_LAGS)

print("Shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("  X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# Baseline for the test segment (last month = next month), aligned to sequences
# For a segment of length L, sequences start at index N_LAGS, so:
y_test_seg = y_test_sr.values
baseline_preds = y_test_seg[N_LAGS - 1:-1]  # previous step inside test segment
baseline_true  = y_test_seg[N_LAGS:]
baseline_mae   = np.mean(np.abs(baseline_preds - baseline_true)) if len(baseline_true) else np.nan

# -------------------------------
# 5) Train or Load
# -------------------------------
if TRAIN:
    model = Sequential([
        LSTM(32, dropout=0.2, input_shape=(N_LAGS, len(SELECTED_FEATURES))),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss='mse',
        metrics=['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test MAE: {test_mae:.4f}")
    print(f"Baseline MAE (last month): {baseline_mae:.4f}")

    # Save model and scaler
    model.save("inflation_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    # Curves
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.show()

    plt.plot(history.history['mae'], label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.show()

else:
    model = load_model("inflation_model.keras")
    scaler = joblib.load("scaler.pkl")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Reloaded Model Test MAE: {test_mae:.4f}")
    print(f"Baseline MAE (last month): {baseline_mae:.4f}")
