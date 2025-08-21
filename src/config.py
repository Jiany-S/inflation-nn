# src/config.py
import os
from pathlib import Path

# ==== task / data ====
N_LAGS = 60
TRAIN = True

# Engineered features created in data.py
SELECTED_FEATURES = [
    # base engineered
    "Unemp_d", "Rate_d", "Oil_ret", "PPI_yoy", "M2_yoy",
    "Retail_yoy", "Employment_d", "Housing_d", "T10Y", "Sentiment",
    # simple polynomials
    "PPI_yoy_sq", "Oil_ret_sq", "Rate_d_sq", "Unemp_d_sq",
    # interactions
    "Rate_Unemp", "Oil_PPI",
    # rolling/level features helpful for regime tracking
    "Infl_ma3", "Infl_ma6", "Infl_vol6", "Inflation_prev"
]

# Raw FRED series (transformed in data.py)
SERIES = {
    "CPI": "CPIAUCNS",
    "Unemployment": "UNRATE",
    "InterestRate": "FEDFUNDS",
    "M2": "M2SL",
    "Treasury10Y": "GS10",
    "OilPrice": "DCOILWTICO",
    "PPI": "PPIACO",
    "RetailSales": "RSXFS",
    "Sentiment": "UMCSENT",
    "Employment": "PAYEMS",
    "HousingStarts": "HOUST",
}

# API key from env (.env or shell). Do NOT hardcode in code.
API_KEY = os.getenv("FRED_API_KEY", "")

# ==== training hyperparams ====
EPOCHS = 250
BATCH_SIZE = 16
LR = 0.004              # small bump to help underfitting
PATIENCE = 15
MIN_LR = 1e-5

# ==== paths ====
ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
