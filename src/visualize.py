# src/visualize.py
import json
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
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
    plot_feature_importance,  # bar chart
    empirical_coverage
)

# Ensure output dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves(history, fname: str = "learning_curves.png") -> None:
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


def plot_preds_vs_actuals(
    model,
    X_test: np.ndarray,
    y_test: Sequence[float],
    y_test_sr: pd.Series,
    fname: str = "preds_vs_actuals.png",
) -> None:
    """
    Predict on X_test and plot predictions vs actuals, aligned to the test timestamps.
    y_test_sr is the unsequenced test target Series (DateTimeIndex).
    """
    preds = model.predict(X_test, verbose=0).ravel()

    # Align timestamps (sequence construction shortens y by N_LAGS)
    offset = len(y_test_sr) - len(y_test)
    idx = pd.to_datetime(y_test_sr.index)[offset:]  # same length as y_test/preds

    fig, ax = plt.subplots()
    ax.plot(idx, y_test, label="actual", linewidth=2)
    ax.plot(idx, preds, label="predicted", linestyle="--", linewidth=2)
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


def plot_preds_with_bands(
    dates: Sequence,
    y_true: Sequence[float],
    y_pred_mean: Sequence[float],
    y_pred_lower: Optional[Sequence[float]] = None,
    y_pred_upper: Optional[Sequence[float]] = None,
    title: str = "Predictions vs Actuals (with Uncertainty Bands)",
    fname: str = "preds_with_bands.png",
    show: bool = False,
) -> None:
    """
    Plot actual vs predicted values with optional uncertainty bands and save to results/.

    Args:
        dates: 1D array-like of datetime-like or indexable x-axis values.
        y_true: 1D array-like of actual values.
        y_pred_mean: 1D array-like of predicted mean values (same length as y_true).
        y_pred_lower: 1D array-like of lower bound (optional).
        y_pred_upper: 1D array-like of upper bound (optional).
        title: Plot title string.
        fname: Output filename under RESULTS_DIR.
        show: If True, also display the figure window (useful in notebooks).
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)

    if y_true.shape[0] != y_pred_mean.shape[0]:
        raise ValueError("y_true and y_pred_mean must have the same length")
    if dates.shape[0] != y_true.shape[0]:
        raise ValueError("dates and y_true must have the same length")

    has_bands = (y_pred_lower is not None) and (y_pred_upper is not None)
    if has_bands:
        y_pred_lower = np.asarray(y_pred_lower, dtype=float)
        y_pred_upper = np.asarray(y_pred_upper, dtype=float)
        if (y_pred_lower.shape[0] != y_true.shape[0]) or (y_pred_upper.shape[0] != y_true.shape[0]):
            raise ValueError("Uncertainty bands must have same length as y_true")
        # Ensure lower <= upper
        y_pred_lower, y_pred_upper = np.minimum(y_pred_lower, y_pred_upper), np.maximum(y_pred_lower, y_pred_upper)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, y_true, label="Actual", linewidth=2)
    ax.plot(dates, y_pred_mean, label="Predicted (mean)", linewidth=2, linestyle="--")

    if has_bands:
        ax.fill_between(dates, y_pred_lower, y_pred_upper, alpha=0.2, label="Uncertainty band")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    # Date formatting if dates look like datetimes
    try:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    except Exception:
        # If dates aren't datetime-like, just skip formatting
        pass

    fig.tight_layout()
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"saved: {out}")


def save_metrics_json(metrics: dict, fname: str = "metrics.json") -> None:
    """Save a metrics dictionary as JSON under results/."""
    out = RESULTS_DIR / fname
    out.write_text(json.dumps(metrics, indent=2))
    print(f"saved: {out}")


def run_all_plots() -> None:
    """
    One-shot: load data + model, then create:
      - preds_vs_actuals.png
      - uncertainty.png (MC-Dropout)
      - feature_importance.png (+ JSON)
    """
    # 1) Load data/splits
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

    # 4) Uncertainty (MC-Dropout)
    mean, std = mc_dropout_predict(model, X_test, n_samples=100)
    cov = empirical_coverage(y_test, mean, std, z=1.28)  # ~80% band
    (RESULTS_DIR / "coverage.txt").write_text(f"80% band empirical coverage: {cov:.3f}\n")
    print(f"80% band empirical coverage: {cov:.3f}")

    offset = len(y_test_sr) - len(y_test)
    idx = pd.to_datetime(y_test_sr.index)[offset:]
    plot_uncertainty(idx, y_test, mean, std, fname="uncertainty.png", z=1.28)

    # 5) Permutation Feature Importance
    feature_names = [c for c in df.columns if c != "Inflation"]
    imps = permutation_importance(model, X_test, y_test, feature_names, n_repeats=5)
    (RESULTS_DIR / "feature_importance.json").write_text(json.dumps(imps, indent=2))
    plot_feature_importance(imps, fname="feature_importance.png")

    print("All plots done.")


if __name__ == "__main__":
    run_all_plots()
