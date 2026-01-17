# OpenPL Total Prediction

OpenPL Total Prediction is a Time-Series regression model within a Jupyter notebook project that builds per-lift regression models to predict next-meet totals (squat, bench, deadlift) from historical meet data. It implements a full pipeline: multi-federation data ingestion, feature engineering, train/validation splits, GPU-accelerated XGBoost training with early stopping, and metric reporting (MAE/RMSE/R²).

## Technical Overview
- Ingestion: loads `entries.csv` and `meet.csv` across federations, concatenates, and merges on `MeetID`.
- Feature prep: datetime parsing, age derivation, and target-specific masking for each lift.
- Modeling: three separate `XGBRegressor` models (squat/bench/deadlift) using GPU (`tree_method='gpu_hist'`).
- Optimization: early stopping on a validation set to select best iteration counts.
- Evaluation: validation predictions with MAE/RMSE/R² and baseline vs optimized comparison.

## Quickstart
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
   - If no requirements file is present, install `pandas`, `numpy`, `scikit-learn`, and `xgboost`.
2. Open and run the notebook:
   - `eda/totalpredict.ipynb`
3. GPU note:
   - If you do not have a compatible GPU, change `tree_method` to `hist` and remove `device='cuda'`.

## Key Files
- `eda/totalpredict.ipynb`: end-to-end pipeline implementation and evaluation.
