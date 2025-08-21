# src/visualize.py
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from .config import RESULTS_DIR

# Ensure output dir exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_learning_curves(history, fname: str = "learning_curves.png"):
    """
    Save training/validation loss curves to results/.
    """
    fig, ax = plt.subplots()
    ax.plot(history.history.get("loss", []), label="train")
    ax.plot(history.history.get("val_loss", []), label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

def plot_preds_vs_actuals(model, X_test, y_test, y_test_sr, fname: str = "preds_vs_actuals.png"):
    """
    Predict on X_test and plot predictions vs actuals, aligned to the test timestamps.
    y_test_sr should be the unsequenced test target Series (with a DateTimeIndex).
    """
    preds = model.predict(X_test, verbose=0).ravel()

    # Align timestamps (sequences shorten y by N_LAGS, so compute offset generically)
    offset = len(y_test_sr) - len(y_test)
    idx = pd.to_datetime(y_test_sr.index)[offset:]  # same length as y_test/preds

    fig, ax = plt.subplots()
    ax.plot(idx, y_test, label="actual")
    ax.plot(idx, preds, label="predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation (pp)")
    ax.legend()

    # Make x-axis readable (quarterly ticks)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

def save_metrics_json(metrics: dict, fname: str = "metrics.json"):
    """
    Save a metrics dictionary as JSON under results/.
    """
    out = RESULTS_DIR / fname
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {out}")
