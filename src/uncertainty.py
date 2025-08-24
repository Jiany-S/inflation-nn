# src/uncertainty.py
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from .config import RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "mc_dropout_predict",
    "plot_uncertainty",
    "permutation_importance",
    "plot_feature_importance",
]

# ---------------------------
# MC Dropout
# ---------------------------
def mc_dropout_predict(model, X, n_samples: int = 100):
    """
    Run model multiple times with dropout active to get mean & std of predictions.
    Assumes model contains Dropout layers. Uses model(x, training=True).
    Returns (mean, std) arrays shaped like (T,).
    """
    preds = [model(X, training=True).numpy().ravel() for _ in range(n_samples)]
    P = np.stack(preds, axis=0)  # [n_samples, T]
    return P.mean(axis=0), P.std(axis=0)

def plot_uncertainty(idx, y_true, y_mean, y_std, fname: str = "uncertainty.png", z: float = 1.28):
    """
    Plot mean prediction with an ~80% band (z≈1.28). Use z=1.96 for ~95%.
    """
    lo = y_mean - z * y_std
    hi = y_mean + z * y_std
    fig, ax = plt.subplots()
    ax.plot(idx, y_true, label="actual")
    ax.plot(idx, y_mean, label="predicted")
    ax.fill_between(idx, lo, hi, alpha=0.2, label="uncertainty band")
    ax.set_xlabel("Date"); ax.set_ylabel("Inflation (pp)"); ax.legend()
    out = RESULTS_DIR / fname
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

# ---------------------------
# Permutation Feature Importance
# ---------------------------
def permutation_importance(model, X_test, y_test, feature_names, n_repeats: int = 5):
    """
    Shuffle each feature across the test set and measure MAE increase.
    For sequence input X_test: [n_samples, n_lags, n_features]
    Returns dict: {feature_name: mean_delta_mae}.
    """
    base_loss, base_mae = model.evaluate(X_test, y_test, verbose=0)
    importances = {}
    for j, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            # Shuffle sample axis for feature j (keeps within-sample time structure)
            perm = np.random.permutation(Xp.shape[0])
            Xp[:, :, j] = Xp[perm, :, j]
            loss, mae = model.evaluate(Xp, y_test, verbose=0)
            deltas.append(mae - base_mae)
        importances[name] = float(np.mean(deltas))
    return importances

def plot_feature_importance(importances: dict, fname: str = "feature_importance.png"):
    names = list(importances.keys())
    vals  = [importances[k] for k in names]
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals  = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], vals[::-1])  # largest at top
    ax.set_xlabel("MAE increase when permuted")
    ax.set_title("Permutation Feature Importance (test set)")
    fig.tight_layout()
    out = RESULTS_DIR / fname
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

def empirical_coverage(y_true, y_mean, y_std, z=1.28):
    lo = y_mean - z*y_std
    hi = y_mean + z*y_std
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())
