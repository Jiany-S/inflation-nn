# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from .config import LR, EPOCHS, BATCH_SIZE, PATIENCE, MIN_LR

def build_model(n_lags, n_features):
    model = Sequential([
        Input(shape=(n_lags, n_features)),
        GRU(256, dropout=0.1, return_sequences=True),
        GRU(128, dropout=0.1),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss="mse", metrics=["mae"])
    return model

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    early   = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(PATIENCE//3, 4),
                                min_lr=MIN_LR, verbose=1)


    def _shape(o):
        try:
            return o.shape
        except Exception:
            return type(o)

    print("SHAPES:")
    print("  X_train", _shape(X_train), "y_train", _shape(y_train))
    print("  X_val  ", _shape(X_val),   "y_val  ", _shape(y_val))
    print("  X_test ", _shape(X_test),  "y_test ", _shape(y_test))

    # hard assertions to catch issues early
    assert X_train.ndim == 3, f"X_train must be 3D, got {X_train.ndim}D"
    assert X_val.ndim   == 3, f"X_val must be 3D, got {X_val.ndim}D"
    assert X_test.ndim  == 3, f"X_test must be 3D, got {X_test.ndim}D"
    assert y_train.ndim == 1, f"y_train must be 1D, got {y_train.ndim}D"
    assert y_val.ndim   == 1, f"y_val must be 1D, got {y_val.ndim}D"
    assert y_test.ndim  == 1, f"y_test must be 1D, got {y_test.ndim}D"

    # ensure float32 tensors
    X_train = X_train.astype("float32")
    X_val   = X_val.astype("float32")
    X_test  = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_val   = y_val.astype("float32")
    y_test  = y_test.astype("float32")

    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early, plateau],
        verbose=1,
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test MAE: {mae:.4f}")
    return history, loss, mae
