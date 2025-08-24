import numpy as np
from src.utils import make_sequences

def test_make_sequences_shapes():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    n_lags = 12
    Xs, ys = make_sequences(X, y, n_lags)
    assert Xs.shape == (100 - n_lags, n_lags, 5)
    assert ys.shape == (100 - n_lags,)
