# src/backtest.py (replace main + rolling_splits with these improved versions)
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from .config import N_LAGS_BACKTEST, RESULTS_DIR
from .data import load_dataset
from .model import build_model
from sklearn.preprocessing import StandardScaler
from .utils import make_sequences

def rolling_splits(df, features, n_lags, k=5, train_ratio=0.6, val_ratio=0.2):
    """
    Expanding-origin splits. Each fold trains on a bigger window and validates/tests on the next chunks.
    """
    n = len(df)
    # choose k cutoff points between 60% and 85% of the series to ensure multiple folds
    starts = np.linspace(int(n * 0.60), int(n * 0.85), num=k, dtype=int)
    for cutoff in starts:
        window = df.iloc[:cutoff].copy()
        n_win = len(window)
        t_end = int(n_win * train_ratio)
        v_end = int(n_win * (train_ratio + val_ratio))

        feat = window[features]
        y = window["Inflation"]

        scaler = StandardScaler().fit(feat.iloc[:t_end].values)

        def seq(fX, yv):
            X = scaler.transform(fX.values)
            return make_sequences(X, yv.values, n_lags)

        Xtr, ytr = seq(feat.iloc[:t_end],      y.iloc[:t_end])
        Xv,  yv  = seq(feat.iloc[t_end:v_end], y.iloc[t_end:v_end])
        Xte, yte = seq(feat.iloc[v_end:],      y.iloc[v_end:])

        # Also pass date indices for logging
        dates = {
            "train": (feat.index[0], feat.index[t_end-1] if t_end>0 else None),
            "val":   (feat.index[t_end] if t_end < len(feat) else None,
                      feat.index[v_end-1] if v_end>t_end else None),
            "test":  (feat.index[v_end] if v_end < len(feat) else None,
                      feat.index[-1]),
        }

        yield (Xtr, ytr, Xv, yv, Xte, yte, dates)

def main(k=5, epochs=50):
    df, features = load_dataset()
    fold_metrics = []
    ok_folds = 0

    for i, (Xtr,ytr,Xv,yv,Xte,yte,dates) in enumerate(rolling_splits(df, features, N_LAGS_BACKTEST, k=k), 1):
        # Skip folds that don't have enough sequences
        if len(Xtr)==0 or len(Xv)==0 or len(Xte)==0:
            print(f"[fold {i}] skipped (not enough sequences) | "
                  f"lens: Xtr={len(Xtr)}, Xv={len(Xv)}, Xte={len(Xte)}")
            continue

        print(f"[fold {i}] "
              f"train={dates['train'][0].date()}→{dates['train'][1].date()} "
              f"val={dates['val'][0].date()}→{dates['val'][1].date()} "
              f"test={dates['test'][0].date()}→{dates['test'][1].date()} | "
              f"seqs: tr={len(Xtr)}, v={len(Xv)}, te={len(Xte)}")

        model = build_model(N_LAGS_BACKTEST, n_features=len(features))
        hist = model.fit(Xtr, ytr, validation_data=(Xv,yv),
                         epochs=epochs, batch_size=16, verbose=0)
        loss, mae = model.evaluate(Xte, yte, verbose=0)
        if len(yte) > 1:
            baseline_last_mae = float(np.mean(np.abs(yte[1:] - yte[:-1])))
        else:
            baseline_last_mae = float("nan")
        print(f"[fold {i}] model MAE={mae:.3f} | naive-last MAE={baseline_last_mae:.3f}")
        fold_metrics.append({
            "fold": i,
            "test_mae": float(mae),
            "test_mse": float(loss),
            "baseline_last_mae": baseline_last_mae
        })
        ok_folds += 1

    if ok_folds < 2:
        raise RuntimeError(f"Backtest produced only {ok_folds} valid fold(s). "
                           f"Reduce N_LAGS, reduce epochs, or adjust k/split ratios.")

    out = RESULTS_DIR / "backtest_metrics.json"
    out.write_text(json.dumps(fold_metrics, indent=2))
    maes = [m["test_mae"] for m in fold_metrics]
    print(f"saved: {out}")
    print(f"Backtest MAE: mean={np.mean(maes):.3f}, std={np.std(maes):.3f}, folds={ok_folds}")

if __name__ == "__main__":
    main(k=5, epochs=40)  # faster default
