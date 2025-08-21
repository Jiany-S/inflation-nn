import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from .data import load_and_prepare_data
from .model import build_model, train_model
from .config import N_LAGS, RESULTS_DIR

def plot_loss(history, fname="learning_curves.png"):
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()
    out = (RESULTS_DIR / fname)
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
    print(f"saved: {out}")

def plot_predictions(model, X_test, y_test, y_test_sr, fname="preds_vs_actuals.png"):
    preds = model.predict(X_test, verbose=0).ravel()

    # Align timestamps to the sequence-reduced arrays
    # (robust: compute offset instead of assuming N_LAGS)
    offset = len(y_test_sr) - len(y_test)
    idx = pd.to_datetime(y_test_sr.index)[offset:]  # same length as y_test/preds

    fig, ax = plt.subplots()
    ax.plot(idx, y_test, label="actual")
    ax.plot(idx, preds, label="predicted")
    ax.set_xlabel("Date"); ax.set_ylabel("Inflation (pp)")
    ax.legend()

    # Make the x-axis readable
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   # every 3 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out = (RESULTS_DIR / fname)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    # load data
    X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)
    # build + train (train_model should return (history, loss, mae))
    model = build_model(N_LAGS, len(df.columns) - 1)
    history, loss, mae = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler)

    # plots
    plot_loss(history, "learning_curves.png")
    plot_predictions(model, X_test, y_test, y_test_sr, "preds_vs_actuals.png")


