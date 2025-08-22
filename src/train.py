# src/train.py
import os, joblib
from .data import load_and_prepare_data
from .model import build_model, train_model
from .config import N_LAGS, RESULTS_DIR, SAVED_MODELS_DIR
from .visualize import plot_learning_curves, plot_preds_vs_actuals, save_metrics_json, plot_preds_with_bands
from .uncertainty import mc_dropout_predict
from src.config import SELECTED_FEATURES, N_LAGS

X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)

n_features = len(SELECTED_FEATURES)
model = build_model(N_LAGS, n_features)
history, loss, mae = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler)

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
model.save(SAVED_MODELS_DIR / "inflation_model.keras")
joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

plot_learning_curves(history, "learning_curves.png")
plot_preds_vs_actuals(model, X_test, y_test, y_test_sr, "preds_vs_actuals.png")
save_metrics_json({"final_test_mae": float(mae), "final_test_mse": float(loss), "n_lags": int(N_LAGS)})
