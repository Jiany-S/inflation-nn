import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import random
import os

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

SELECTED_FEATURES = ["CPI", "Unemployment", "InterestRate", "OilPrice", "PPI"]

SERIES = {
    "CPI": "CPIAUCNS",          # monthly CPI (NSA)
    "Unemployment": "UNRATE",   # monthly
    "InterestRate": "FEDFUNDS", # monthly
    "M2": "M2SL",               # monthly
    "Treasury10Y": "GS10",      # monthly
    "OilPrice": "DCOILWTICO",   # daily -> monthly avg
    "PPI": "PPIACO",            # monthly
    "RetailSales": "RSXFS",     # monthly
    "Sentiment": "UMCSENT",     # monthly
    "Employment": "PAYEMS",     # monthly
    "HousingStarts": "HOUST"    # monthly
}

# Reproducibility
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# -------------------------------
# Helpers
# -------------------------------
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

# -------------------------------
# 1) Data
# -------------------------------
fred = Fred(api_key=API_KEY)

cols = {}
for name, code in SERIES.items():
    how = "mean" if name in ["OilPrice"] else "last"
    cols[name] = fetch_series_monthly(fred, code, how=how)

df = pd.DataFrame(cols).sort_index()

# Target: monthly inflation (% change in CPI)
df["Inflation"] = df["CPI"].pct_change() * 100
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

print("Shapes:",
      "X_train", X_train.shape, "y_train", y_train.shape,
      "X_val",   X_val.shape,   "y_val",   y_val.shape,
      "X_test",  X_test.shape,  "y_test",  y_test.shape)

# Baseline (last-month) on test segment, aligned to sequences
y_test_seg = y_test_sr.values
baseline_preds = y_test_seg[N_LAGS - 1:-1]
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

    model.compile(optimizer=Adam(learning_rate=0.003), loss='mse', metrics=['mae'])

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

    # Save artifacts
    model.save("inflation_model.keras")
    joblib.dump(scaler, "scaler.pkl")

    # Curves (loss/MAE are in target scale since y is unscaled)
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

# -------------------------------
# 6) Forecast next month
# -------------------------------
next_forecast = forecast_next(model, scaler, feat_df, N_LAGS)
print(f"Next-month forecast (target scale): {next_forecast:.4f}")
