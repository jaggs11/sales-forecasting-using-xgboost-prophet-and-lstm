# Sales Forecasting using Machine Learning (Walmart Dataset)

This project implements an end-to-end **sales forecasting system** using historical Walmart retail data.  
The goal is to predict future sales and compare different forecasting approaches, including machine learning, deep learning, and classical time-series models.

The pipeline starts from raw data preprocessing, moves through feature engineering and model training, and ends with visualizing forecasts and analyzing results.

The Walmart dataset contains weekly sales information along with external factors such as holidays, temperature, fuel price, CPI, and unemployment. These features are used to understand patterns and improve prediction accuracy.

Three different models were trained:

• **XGBoost** – a tree-based machine learning model trained per store using engineered time-based and lag features. This model produced the most accurate and realistic forecasts and is treated as the primary model.  
• **LSTM (Long Short-Term Memory)** – a deep learning model trained on sequential sales data to capture temporal dependencies. Predictions are shown in scaled form to demonstrate sequence learning behavior.  
• **Prophet** – a time-series forecasting model trained on globally aggregated sales data to analyze overall trends across all stores.

The project structure is organized as follows:

sales-forecasting/
│
├── data/
│   ├── walmart.csv
│   └── processed_walmart.csv
│
├── scripts/
│   ├── preprocess_walmart.py
│   ├── train_xgboost_walmart.py
│   ├── train_lstm_walmart.py
│   ├── train_prophet_walmart.py
│   ├── utils_walmart.py
│   └── plot_all_models.py
│
├── outputs/
│   ├── forecast_xgb_walmart.csv
│   ├── forecast_lstm_walmart.csv
│   └── forecast_prophet_walmart.csv
│
├── models/
│   ├── le_store.joblib
│   ├── xgb_walmart.joblib
│   ├── lstm_walmart_best.h5
│   └── lstm_walmart_final.h5
│
├── requirements.txt
├── README.md


To run the project, a Python virtual environment is created and dependencies are installed using `requirements.txt`.  
The data is first preprocessed, then all three models are trained, and finally the forecasting results are visualized using a plotting script.

The final output of the project consists of:
• Actual vs predicted sales graphs  
• Forecast CSV files for all models  
• A comparison of different forecasting approaches  

From the results, XGBoost was found to outperform the other models in terms of stability and prediction accuracy. LSTM demonstrated temporal learning capability but required more data and inverse scaling, while Prophet was effective in capturing long-term global trends.

The virtual environment (`venv/`) is intentionally excluded from the repository. All required dependencies are listed in `requirements.txt` so the environment can be recreated on any system.

This project is intended for academic and learning purposes.

**Author:** Nayan Jaggi
