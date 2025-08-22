# src/visualize.py
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model

from .config import RESULTS_DIR, N_LAGS
from .data import load_and_prepare_data  # provides splits + df + scaler + y_test_sr
from .uncertainty import (
    mc_dropout_predict,      # returns mean, std
    plot_uncertainty,        # plots mean + band
    permutation_importance,  # returns dict {feature -> MAE increase}
    plot_feature_importance  # bar chart
)

# Ensure output dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(history, fname: str = "learning_curves.png"):
    """Save training/validation loss curves to results/ (history comes from train.py)."""
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
    y_test_sr is the unsequenced test target Series (DateTimeIndex).
    """
    preds = model.predict(X_test, verbose=0).ravel()

    # Align timestamps (sequence construction shortens y by N_LAGS)
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
    """Save a metrics dictionary as JSON under results/."""
    out = RESULTS_DIR / fname
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved: {out}")


def run_all_plots():
    """
    One-shot: load data + model, then create:
      - preds_vs_actuals.png
      - uncertainty.png (MC-Dropout)
      - feature_importance.png (+ JSON)
    """
    # 1) Load data/splits (no duplication; we call your data loader)
    X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)

    # 2) Load trained model (prefer saved_models/, fall back to project root)
    candidates = [
        RESULTS_DIR.parent / "saved_models" / "inflation_model.keras",
        Path("inflation_model.keras"),
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "Trained model not found. Expected saved_models/inflation_model.keras or ./inflation_model.keras"
        )
    model = load_model(model_path)

    # 3) Preds vs Actuals
    plot_preds_vs_actuals(model, X_test, y_test, y_test_sr, fname="preds_vs_actuals.png")

    # 4) Uncertainty (MC-Dropout) — calls functions from src/uncertainty.py
    mean, std = mc_dropout_predict(model, X_test, n_samples=100)
    offset = len(y_test_sr) - len(y_test)
    idx = pd.to_datetime(y_test_sr.index)[offset:]
    plot_uncertainty(idx, y_test, mean, std, fname="uncertainty.png", z=1.28)

    # 5) Permutation Feature Importance — calls functions from src/uncertainty.py
    feature_names = [c for c in df.columns if c != "Inflation"]
    imps = permutation_importance(model, X_test, y_test, feature_names, n_repeats=5)
    (RESULTS_DIR / "feature_importance.json").write_text(json.dumps(imps, indent=2))
    plot_feature_importance(imps, fname="feature_importance.png")

    print("All plots done.")


if __name__ == "__main__":
    run_all_plots()
