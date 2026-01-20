# scripts/utils_walmart.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ----------------------------------------------------------
# Safe MAPE (avoids division by zero)
# ----------------------------------------------------------
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Only compute MAPE where actual values > 0
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan

    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100


# ----------------------------------------------------------
# Print all metrics (RMSE, MAE, MAPE)
# ----------------------------------------------------------
def print_metrics(y_true, y_pred):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (safe)
    mape_val = mape(y_true, y_pred)

    print(f"\n📊 Model Performance")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape_val:.2f}%")

    return {"rmse": rmse, "mae": mae, "mape": mape_val}



# ----------------------------------------------------------
# Time-based train/test split (for Walmart weekly data)
# ----------------------------------------------------------
def time_train_test_split(df, date_col="Date", split_date="2012-01-01"):
    """
    Training = everything before split_date
    Testing  = everything on/after split_date
    """

    df = df.sort_values(date_col)

    train = df[df[date_col] < split_date].copy()
    test  = df[df[date_col] >= split_date].copy()

    print(f"\n🗂️ Time Split Completed")
    print(f"Train range: {train[date_col].min()} → {train[date_col].max()}")
    print(f"Test range : {test[date_col].min()} → {test[date_col].max()}")
    print(f"Train rows : {len(train)}")
    print(f"Test rows  : {len(test)}")

    return train, test


# ----------------------------------------------------------
# Sanity check for Walmart dataset
# ----------------------------------------------------------
def check_walmart_dataset(df):
    print("\n🔍 Dataset Check:")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    missing = df.isna().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values found.")

    # Check required columns
    required = [
        "Store", "Date", "Sales", "Holiday_Flag",
        "Temperature", "Fuel_Price", "CPI", "Unemployment"
    ]

    print("\nRequired columns present:")
    for col in required:
        if col in df.columns:
            print(f"  ✔ {col}")
        else:
            print(f"  ❌ MISSING: {col}")
