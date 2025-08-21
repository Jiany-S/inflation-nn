# src/train.py

import os
import joblib
from .data import load_and_prepare_data
from .model import build_model, train_model
from .config import N_LAGS, RESULTS_DIR, SAVED_MODELS_DIR
from .visualize import plot_learning_curves, plot_preds_vs_actuals, save_metrics_json

# -------------------------------------------------------------------
# 1. Load & prepare data
# -------------------------------------------------------------------
X_train, y_train, X_val, y_val, X_test, y_test, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)

# -------------------------------------------------------------------
# 2. Build model
# -------------------------------------------------------------------
model = build_model(N_LAGS, len(df.columns) - 1)

# -------------------------------------------------------------------
# 3. Train model (train_model now returns history + test metrics)
# -------------------------------------------------------------------
history, loss, mae = train_model(
    model, X_train, y_train, X_val, y_val, X_test, y_test, scaler
)

# -------------------------------------------------------------------
# 4. Ensure output dirs
# -------------------------------------------------------------------
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 5. Save model + scaler
# -------------------------------------------------------------------
model.save(os.path.join(SAVED_MODELS_DIR, "inflation_model.keras"))
joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))

# -------------------------------------------------------------------
# 6. Visualizations
# -------------------------------------------------------------------
# Learning curves
plot_learning_curves(history, "learning_curves.png")

# Predictions vs Actuals (align with test index if available)
y_pred_test = model.predict(X_test, verbose=0).ravel()
test_index = y_test_sr.index if y_test_sr is not None else range(len(y_test))
plot_preds_vs_actuals(y_test, y_pred_test, index=test_index, fname="preds_vs_actuals.png")

# -------------------------------------------------------------------
# 7. Save metrics
# -------------------------------------------------------------------
save_metrics_json({
    "final_test_mae": float(mae),
    "final_test_mse": float(loss),
    "n_lags": int(N_LAGS),
})
