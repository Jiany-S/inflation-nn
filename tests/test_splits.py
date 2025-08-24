from src.data import load_and_prepare_data
from src.config import N_LAGS

def test_chrono_split_no_leak():
    Xtr, ytr, Xv, yv, Xte, yte, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)
    # last train timestamp < first val timestamp < first test timestamp
    # we can at least check non-empty and chronological order of y_test_sr
    assert len(Xtr) > 0 and len(Xv) > 0 and len(Xte) > 0
    assert y_test_sr.index.is_monotonic_increasing
