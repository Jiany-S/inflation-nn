import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fredapi import Fred
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Data
# -------------------------------
fred = Fred(api_key="9fc23489b39a61d02d244276cb0d000b")

series = {
    "CPI": "CPIAUCNS",
    "Unemployment": "UNRATE",
    "InterestRate": "FEDFUNDS",
    "M2": "M2SL",
    "Treasury10Y": "GS10"
}

df = pd.DataFrame({name: fred.get_series(code) for name, code in series.items()})
df["Inflation"] = df["CPI"].pct_change() * 100
df = df.dropna()

def make_supervised(data, target_col="Inflation", n_lags=12):
    df = data.copy()
    for lag in range(1, n_lags + 1):
        for col in data.columns:
            if col != target_col:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

X, y = make_supervised(df, target_col="Inflation", n_lags=24)
print("Supervised shape:", X.shape, y.shape)

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# chronological split: 70% train, 15% val, 15% test
n = len(X_scaled)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X_scaled[:train_end], y[:train_end]
X_val, y_val     = X_scaled[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X_scaled[val_end:], y[val_end:]

# -------------------------------
# 2. Train or Load
# -------------------------------
TRAIN = True   # flase just loads

if TRAIN:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),   # drop 30% of neurons
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=0.005)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop]
    )

    # final test evaluation
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print("Final Test MAE:", test_mae)

    # save trained model
    model.save("inflation_model.h5")

    # plot training curves
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.show()

    plt.plot(history.history['mae'], label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE')
    plt.legend(); plt.show()

else:
    # just load existing model
    model = load_model("inflation_model.h5")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print("Reloaded Model Test MAE:", test_mae)
