import os
from pathlib import Path

# Hyperparams
N_LAGS = 36
TRAIN = True

# Features & FRED series
SELECTED_FEATURES = ["CPI", "Unemployment", "InterestRate", "OilPrice", "PPI"]
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

# API key (read from env; do NOT hardcode)
API_KEY = os.getenv("FRED_API_KEY", "")

# Project paths (root = repo/)
ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"

# Ensure folders exist at import time
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
