# src/eval_baselines.py
import json
import numpy as np
from .config import RESULTS_DIR, N_LAGS
from .data import load_dataset, load_and_prepare_data

from .baselines import mae, naive_last, seasonal_naive, arima_forecast, ridge_mae, lasso_mae

def main():
    df, features = load_dataset()
    n = len(df)
    train_end = int(n*0.70)
    val_end   = int(n*0.85)
    y = df["Inflation"]
    y_train, y_test = y.iloc[:val_end], y.iloc[val_end:]

    res = {}
    nl = naive_last(y_train).iloc[-len(y_test):]
    res["naive_last_mae"] = mae(y_test, nl)

    sn = seasonal_naive(y_train).iloc[-len(y_test):]
    res["seasonal_naive_mae"] = mae(y_test, sn)

    ar = arima_forecast(y_train, y_test, order=(1,0,0))
    res["arima_100_mae"] = mae(y_test, ar)

    (RESULTS_DIR / "baselines.json").write_text(json.dumps(res, indent=2))
    print(res)

    Xtr, ytr, Xv, yv, Xte, yte, df, scaler, y_test_sr = load_and_prepare_data(N_LAGS)
    metrics = {
        "naive_last_mae": mae(naive_last(y_test_sr.values)),
        "seasonal_naive_mae": mae(seasonal_naive(y_test_sr.values)),
        "arima_100_mae": mae(arima_forecast(y_test_sr.values, order=(1,0,0))),
        "ridge_mae": ridge_mae(Xte, yte, Xtr, ytr),
        "lasso_mae": lasso_mae(Xte, yte, Xtr, ytr),
    }
    print(metrics)

if __name__ == "__main__":
    main()
