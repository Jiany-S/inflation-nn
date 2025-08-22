# src/eval_baselines.py
import json
import numpy as np
from .config import RESULTS_DIR, N_LAGS
from .data import load_dataset

from .baselines import mae, naive_last, seasonal_naive, arima_forecast

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

if __name__ == "__main__":
    main()
