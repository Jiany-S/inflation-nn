# src/forecast.py
import joblib
from tensorflow.keras.models import load_model
from .data_prep import load_and_prepare_data
from .utils import forecast_next
from .config import N_LAGS

_, _, _, _, _, _, df, _, _ = load_and_prepare_data(N_LAGS)

model = load_model("saved_models/inflation_model.keras")
scaler = joblib.load("saved_models/scaler.pkl")

next_forecast = forecast_next(model, scaler, df.drop(columns=["Inflation"]), N_LAGS)
print(f"Next-month forecast: {next_forecast:.4f}")
