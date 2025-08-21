# src/config.py
import os
from pathlib import Path

# Hyperparams
N_LAGS = 36
TRAIN = True

# Engineered features we'll create in data.py
SELECTED_FEATURES = [
    "Unemp_d",       # Δ Unemployment
    "Rate_d",        # Δ Fed funds
    "Oil_ret",       # log return of WTI (%)
    "PPI_yoy",       # YoY PPI (%)
    "M2_yoy",        # YoY M2 (%)
    "Retail_yoy",    # YoY retail sales (%)
    "Employment_d",  # Δ nonfarm payrolls
    "Housing_d",     # Δ housing starts
    "T10Y",          # 10Y UST level
    "Sentiment"      # UMich level
]

# Raw FRED series (we'll transform them)
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
    "HousingStarts": "HOUST"
}

# API key (set in your shell: $env:FRED_API_KEY="..."; or export FRED_API_KEY=...)
API_KEY = os.getenv("FRED_API_KEY", "")

# Paths
ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = ROOT / "saved_models"
RESULTS_DIR = ROOT / "results"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
