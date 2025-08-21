from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(n_lags, n_features):
    m = Sequential([
        LSTM(32, dropout=0.2, input_shape=(n_lags, n_features)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=0.003), loss='mse', metrics=['mae'])
    return m

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early],
        verbose=1
    )
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test MAE: {mae:.4f}")
    return history, loss, mae
