# scripts/preprocess_walmart.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -----------------------------------------
# 1. Load Walmart Dataset
# -----------------------------------------
def load_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates safely
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Rename target
    if 'Weekly_Sales' in df.columns:
        df = df.rename(columns={'Weekly_Sales': 'Sales'})

    # Drop rows where date or sales is missing
    df = df.dropna(subset=['Date', 'Sales'])

    return df


# -----------------------------------------
# 2. Add time features
# -----------------------------------------
def add_time_features(df):
    df = df.copy()

    # Remove any bad date rows (causing the NA → int error)
    df = df.dropna(subset=['Date'])

    df['year']       = df['Date'].dt.year
    df['month']      = df['Date'].dt.month
    df['day']        = df['Date'].dt.day
    df['dayofweek']  = df['Date'].dt.dayofweek

    # FIX: safer conversion to avoid "cannot convert NA to integer"
    week = df['Date'].dt.isocalendar().week
    df['weekofyear'] = week.astype('int64')

    return df


# -----------------------------------------
# 3. Lag + Rolling Features
# -----------------------------------------
def add_lag_features(df, lags=[1, 7, 14, 28], windows=[7, 30]):
    df = df.sort_values(['Store', 'Date']).copy()

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

    for w in windows:
        df[f'rolling_mean_{w}'] = (
            df.groupby('Store')['Sales']
              .shift(1)
              .rolling(window=w, min_periods=1)
              .mean()
        )

    return df


# -----------------------------------------
# 4. Encode Store IDs (optional but good)
# -----------------------------------------
def encode_store(df):
    le = LabelEncoder()
    df['Store'] = le.fit_transform(df['Store'])
    joblib.dump(le, "models/le_store.joblib")
    return df


# -----------------------------------------
# 5. Save the output
# -----------------------------------------
def save_output(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_walmart.csv", index=False)
    print("\nSaved cleaned file → data/processed_walmart.csv")


# -----------------------------------------
# MAIN PIPELINE
# -----------------------------------------
if __name__ == "__main__":
    print("\n🔵 Loading Walmart dataset...")
    df = load_data("data/walmart.csv")   # <-- Make sure your file is named walmart.csv

    print("🔵 Adding time features...")
    df = add_time_features(df)

    print("🔵 Adding lag + rolling features...")
    df = add_lag_features(df)

    print("🔵 Encoding Store column...")
    df = encode_store(df)

    print("🔵 Final cleanup (dropping remaining NA rows)...")
    df = df.dropna().reset_index(drop=True)

    print("🔵 Saving output file...")
    save_output(df)

    print("\n✅ Preprocessing complete. Walmart dataset is ready!")
