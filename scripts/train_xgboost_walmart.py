# scripts/train_xgboost_walmart.py
import os
import joblib
import pandas as pd
import xgboost as xgb
from utils_walmart import time_train_test_split, print_metrics, check_walmart_dataset

def load_processed(path="data/processed_walmart.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run preprocess_walmart.py first.")
    return pd.read_csv(path, parse_dates=["Date"])

def main():
    print("🔵 Loading processed Walmart dataset...")
    df = load_processed()

    # quick sanity check
    check_walmart_dataset(df)

    # features and target
    features = [
        'Store','Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment',
        'year','month','day','dayofweek','weekofyear',
        'lag_1','lag_7','lag_14','lag_28',
        'rolling_mean_7','rolling_mean_30'
    ]
    target = "Sales"

    # make sure features exist
    missing_feats = [c for c in features if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required features in processed file: {missing_feats}")

    # time split
    train, test = time_train_test_split(df, date_col="Date", split_date="2012-01-01")

    X_train, y_train = train[features], train[target]
    X_test, y_test   = test[features], test[target]

    print("\n🔵 Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )

    model.fit(X_train, y_train)


    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_walmart.joblib")
    print("✅ Saved model: models/xgb_walmart.joblib")

    preds = model.predict(X_test)
    print_metrics(y_test.values, preds)

    os.makedirs("outputs", exist_ok=True)
    out = test[["Store","Date","Sales"]].copy()
    out["forecast_xgb"] = preds
    out.to_csv("outputs/forecast_xgb_walmart.csv", index=False)
    print("✅ Saved forecast: outputs/forecast_xgb_walmart.csv")


if __name__ == "__main__":
    main()
