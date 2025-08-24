# src/eval_baselines.py
import json

from .config import RESULTS_DIR, N_LAGS
from .data import load_dataset, load_and_prepare_data
from .baselines import (
    mae,
    naive_last,
    seasonal_naive,
    arima_forecast,
    ridge_mae,
    lasso_mae,
)


def main():
    # -------- Baselines on unsequenced Series (clean pandas workflow) --------
    df, features = load_dataset()
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    y = df["Inflation"]
    y_train = y.iloc[:val_end].copy()
    y_test = y.iloc[val_end:].copy()

    # Naive last (shift by 1), align to test length
    nl = naive_last(y_train).iloc[-len(y_test):]
    naive_last_mae = mae(y_test, nl)

    # Seasonal naive (default season=12), align to test length
    sn = seasonal_naive(y_train).iloc[-len(y_test):]
    seasonal_naive_mae = mae(y_test, sn)

    # ARIMA(1,0,0) forecast using train to predict test horizon
    ar = arima_forecast(y_train, y_test, order=(1, 0, 0))
    arima_100_mae = mae(y_test, ar)

    # -------- Ridge/Lasso on sequenced data (X/y numpy arrays) --------
    Xtr, ytr, Xv, yv, Xte, yte, _, _, y_test_sr = load_and_prepare_data(N_LAGS)
    ridge = ridge_mae(Xte, yte, Xtr, ytr)
    lasso = lasso_mae(Xte, yte, Xtr, ytr)

    # Combine and save once
    results = {
        "naive_last_mae": float(naive_last_mae),
        "seasonal_naive_mae": float(seasonal_naive_mae),
        "arima_100_mae": float(arima_100_mae),
        "ridge_mae": float(ridge),
        "lasso_mae": float(lasso),
    }

    (RESULTS_DIR / "baselines.json").write_text(json.dumps(results, indent=2))
    print(results)


if __name__ == "__main__":
    main()
