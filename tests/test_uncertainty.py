import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.uncertainty import mc_dropout_predict

def test_mc_dropout_lengths():
    model = Sequential([Input((3,)), Dropout(0.5), Dense(1)])
    X = np.random.randn(10, 3).astype("float32")
    mean, std = mc_dropout_predict(model, X, n_samples=5)
    assert mean.shape == (10,)
    assert std.shape == (10,)
