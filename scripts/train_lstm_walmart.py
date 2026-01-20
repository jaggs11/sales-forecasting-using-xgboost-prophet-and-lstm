# scripts/train_lstm_walmart.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils_walmart import check_walmart_dataset, print_metrics

SEQ_LEN = 10  # number of past weeks used to predict next

def build_sequences(values, seq_len):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i])
    return np.array(X), np.array(y)

def main():
    print("🔵 Loading processed Walmart dataset...")
    path = "data/processed_walmart.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess_walmart.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])

    check_walmart_dataset(df)

    # choose a single store series for LSTM (multiseries LSTM needs extra work)
    store_vals = sorted(df["Store"].unique())
    store_id = store_vals[0]
    print(f"🔵 Building LSTM for Store (label-encoded) = {store_id}")

    series = df[df["Store"] == store_id].sort_values("Date").reset_index(drop=True)
    sales = series["Sales"].values.astype(float)

    if len(sales) <= SEQ_LEN + 10:
        raise ValueError("Not enough data points for LSTM. Need more than SEQ_LEN + 10 rows for chosen store.")

    print("🔵 Preparing sequences...")
    X, y = build_sequences(sales, SEQ_LEN)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 80/20 split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test   = X[split:], y[split:]

    print("🔵 Building LSTM model...")
    model = Sequential([
        LSTM(128, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/lstm_walmart_best.h5", save_best_only=True, monitor="val_loss")
    early = EarlyStopping(patience=10, restore_best_weights=True)

    print("🔵 Training LSTM (this may take a while)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early],
        verbose=2
    )
    model.save("models/lstm_walmart_final.h5")
    print("✅ Saved models: models/lstm_walmart_best.h5 and models/lstm_walmart_final.h5")

    preds = model.predict(X_test).flatten()
    print_metrics(y_test, preds)

    # Save aligned output (dates correspond to end of each sequence)
    out_dates = series["Date"].iloc[SEQ_LEN + split:].reset_index(drop=True)
    out_df = pd.DataFrame({
        "Date": out_dates,
        "actual": y_test,
        "forecast_lstm": preds
    })
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv("outputs/forecast_lstm_walmart.csv", index=False)
    print("✅ Saved forecast: outputs/forecast_lstm_walmart.csv")


if __name__ == "__main__":
    main()
