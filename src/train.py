# src/train.py
import os, joblib
from .data import load_and_prepare_data
from .model import build_model, train_model
from .config import N_LAGS, RESULTS_DIR, SAVED_MODELS_DIR
from .visualize import plot_learning_curves, plot_preds_vs_actuals, save_metrics_json

# 1) data
X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)

# 2) model
model = build_model(N_LAGS, len(df.columns) - 1)

# 3) train
history, loss, mae = train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler)

# 4) save
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
model.save(SAVED_MODELS_DIR / "inflation_model.keras")
joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

# 5) visuals
plot_learning_curves(history, "learning_curves.png")
plot_preds_vs_actuals(model, X_test, y_test, y_test_sr, "preds_vs_actuals.png")
save_metrics_json({"final_test_mae": float(mae), "final_test_mse": float(loss), "n_lags": int(N_LAGS)})

# 6) metrics
save_metrics_json({"final_test_mae": float(mae), "final_test_mse": float(loss), "n_lags": int(N_LAGS)})
