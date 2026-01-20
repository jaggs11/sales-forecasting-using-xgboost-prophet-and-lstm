# scripts/train_prophet_walmart.py

import os
import pandas as pd
from prophet import Prophet
from utils_walmart import check_walmart_dataset

def main():
    print("🔵 Loading processed Walmart dataset...")
    path = "data/processed_walmart.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess_walmart.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])

    check_walmart_dataset(df)

    print("🔵 Building MINIMAL Prophet model (works with tiny datasets)...")

    # Aggregate stores
    global_series = (
        df.groupby("Date")["Sales"]
          .sum()
          .reset_index()
          .rename(columns={"Date": "ds", "Sales": "y"})
          .sort_values("ds")
          .reset_index(drop=True)
    )

    print(f"Rows available: {len(global_series)}")

    if len(global_series) < 5:
        raise ValueError("Dataset WAY too small. Prophet cannot run on <5 points.")

    # SIMPLE MODE: disable complex patterns
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.01
    )

    print("🔵 Fitting minimal Prophet model...")
    model.fit(global_series)

    print("🔵 Forecasting next 10 periods...")
    future = model.make_future_dataframe(periods=10, freq='W')
    forecast = model.predict(future)

    os.makedirs("outputs", exist_ok=True)
    forecast[["ds", "yhat"]].to_csv("outputs/forecast_prophet_walmart.csv", index=False)

    print("✅ Saved: outputs/forecast_prophet_walmart.csv (minimal Prophet model)")


if __name__ == "__main__":
    main()
