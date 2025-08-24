# src/train.py
import os
import json
import joblib
import argparse

from .data import load_and_prepare_data
from .model import build_model, train_model
from .config import N_LAGS, RESULTS_DIR, SAVED_MODELS_DIR, SELECTED_FEATURES
from .visualize import plot_learning_curves, plot_preds_vs_actuals, save_metrics_json, plot_preds_with_bands
from .uncertainty import mc_dropout_predict
from .utils import make_run_dir

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM inflation model")
    p.add_argument("--epochs", type=int, default=150, help="Training epochs")
    p.add_argument("--lags", type=int, default=None, help="Override N_LAGS")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    # keep features flag reserved if you later add 'full' feature set
    p.add_argument("--features", choices=["small", "full"], default="small", help="Feature preset (currently decorative)")
    return p.parse_args()

def main():
    args = parse_args()

    # Resolve hyperparams
    n_lags = args.lags if args.lags is not None else N_LAGS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir("train")  # results/run_YYYYmmdd_HHMMSS

    # ---------------------------
    # Data
    # ---------------------------
    X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(n_lags)
    # Use actual tensor shape to get feature count
    n_features = X_train.shape[-1]

    # ---------------------------
    # Model
    # ---------------------------
    # Try to pass custom lr if build_model supports it; otherwise fall back.
    try:
        model = build_model(n_lags, n_features, learning_rate=args.lr)
    except TypeError:
        model = build_model(n_lags, n_features)

    # Train (gracefully pass epochs if supported)
    try:
        history, loss, mae = train_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, epochs=args.epochs
        )
    except TypeError:
        history, loss, mae = train_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test, scaler
        )

    # ---------------------------
    # Save core artifacts
    # ---------------------------
    model.save(SAVED_MODELS_DIR / "inflation_model.keras")
    joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

    # also save into the per-run folder
    model.save(run_dir / "inflation_model.keras")
    joblib.dump(scaler, run_dir / "scaler.pkl")

    # ---------------------------
    # Plots + Metrics
    # ---------------------------
    if history is not None:
        plot_learning_curves(history, "learning_curves.png")

    plot_preds_vs_actuals(model, X_test, y_test, y_test_sr, "preds_vs_actuals.png")

    # Uncertainty band (MC Dropout)
    mean, std = mc_dropout_predict(model, X_test, n_samples=100)
    # Align dates to y_test length
    offset = len(y_test_sr) - len(y_test)
    idx = y_test_sr.index[offset:]
    plot_preds_with_bands(idx, y_test, mean, mean - 1.28 * std, mean + 1.28 * std, "uncertainty.png")

    # Save metrics to the shared results dir
    metrics = {
        "final_test_mse": float(loss),
        "final_test_mae": float(mae),
        "n_lags": int(n_lags),
        "n_features": int(n_features),
        "epochs": int(args.epochs),
        "learning_rate": (None if args.lr is None else float(args.lr)),
    }
    save_metrics_json(metrics, "metrics.json")

    # and also to the run-specific folder
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Quick summary in console
    print(
        f"[done] MAE={metrics['final_test_mae']:.4f}, "
        f"MSE={metrics['final_test_mse']:.4f}, "
        f"lags={n_lags}, feats={n_features}, epochs={args.epochs}, lr={args.lr}"
    )
    print(f"Artifacts:")
    print(f"  - Model  : {SAVED_MODELS_DIR / 'inflation_model.keras'}")
    print(f"  - Scaler : {SAVED_MODELS_DIR / 'scaler.pkl'}")
    print(f"  - Run dir: {run_dir}")

if __name__ == "__main__":
    main()
