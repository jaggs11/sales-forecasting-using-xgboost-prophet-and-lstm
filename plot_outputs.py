import pandas as pd
import matplotlib.pyplot as plt

# =============================
#        CONFIG
# =============================
STORE_ID = 1      # choose store (0–44)
ACTUAL_PATH = "data/processed_walmart.csv"
XGB_PATH = "outputs/forecast_xgb_walmart.csv"
LSTM_PATH = "outputs/forecast_lstm_walmart.csv"
PROPHET_PATH = "outputs/forecast_prophet_walmart.csv"


# =============================
#    LOAD ACTUAL SALES
# =============================
df = pd.read_csv(ACTUAL_PATH, parse_dates=["Date"])
actual_store = (
    df[df["Store"] == STORE_ID][["Date", "Sales"]]
    .sort_values("Date")
    .rename(columns={"Sales": "actual"})
)


# =============================
#    LOAD XGBOOST PREDICTIONS
# =============================
xgb = pd.read_csv(XGB_PATH, parse_dates=["Date"])
xgb = (
    xgb[xgb["Store"] == STORE_ID][["Date", "forecast_xgb"]]
    .sort_values("Date")
)


# =============================
#    PLOT 1 — XGBOOST vs ACTUAL
# =============================
comp_xgb = actual_store.merge(xgb, on="Date", how="inner")

plt.figure(figsize=(12,5))
plt.plot(comp_xgb["Date"], comp_xgb["actual"], label="Actual", linewidth=2)
plt.plot(comp_xgb["Date"], comp_xgb["forecast_xgb"], label="XGBoost Forecast", linewidth=2)
plt.title(f"XGBoost Forecast vs Actual — Store {STORE_ID}")
plt.xlabel("Date"); plt.ylabel("Sales")
plt.legend(); plt.grid(); plt.tight_layout()
plt.show()


# =============================
#    LOAD LSTM (scaled space)
# =============================
lstm = pd.read_csv(LSTM_PATH, parse_dates=["Date"])
# Rename forecast column
for c in lstm.columns:
    if "forecast" in c.lower():
        lstm = lstm.rename(columns={c: "forecast_lstm"})
# LSTM has columns: Date, actual, forecast_lstm


# =============================
#    PLOT 2 — LSTM (scaled)
# =============================
plt.figure(figsize=(12,5))
plt.plot(lstm["Date"], lstm["actual"], label="Actual (scaled)", linewidth=2)
plt.plot(lstm["Date"], lstm["forecast_lstm"], label="LSTM Forecast (scaled)", linewidth=2)
plt.title("LSTM Forecast (Scaled Values)")
plt.xlabel("Date"); plt.ylabel("Scaled Value")
plt.legend(); plt.grid(); plt.tight_layout()
plt.show()


# =============================
#    LOAD PROPHET (global)
# =============================
prophet = pd.read_csv(PROPHET_PATH)

# Normalize date column
if "Date" in prophet.columns:
    prophet["Date"] = pd.to_datetime(prophet["Date"])
elif "ds" in prophet.columns:
    prophet = prophet.rename(columns={"ds": "Date"})
    prophet["Date"] = pd.to_datetime(prophet["Date"])

# normalize forecast column
if "forecast_prophet" not in prophet.columns:
    # maybe it is "yhat"
    for c in prophet.columns:
        if "yhat" in c.lower():
            prophet = prophet.rename(columns={c: "forecast_prophet"})
            break

# =============================
#    PLOT 3 — PROPHET (global)
# =============================
plt.figure(figsize=(12,5))
plt.plot(prophet["Date"], prophet["forecast_prophet"], 
         label="Prophet Global Forecast", linewidth=2)

plt.title("Prophet Forecast — Global Aggregated Sales")
plt.xlabel("Date"); plt.ylabel("Sales (Global)")
plt.legend(); plt.grid(); plt.tight_layout()
plt.show()
