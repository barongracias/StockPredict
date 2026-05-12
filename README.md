# StockPredict

Train, evaluate, and deploy ML models for stock price prediction. Uses Yahoo Finance for data, scikit-learn/LightGBM/XGBoost for modelling, Optuna for hyperparameter tuning, Comet ML for experiment tracking, and Cerebrium for deployment.

## Pipeline

```
download_data.py  →  preprocess.py  →  features_pipeline.py  →  train_model.py  →  plot_model.py
```

1. **download_data.py** — fetch daily OHLC data from Yahoo Finance, save as Parquet in `data/`
2. **preprocess.py** — slice the time series into sliding (features, target) windows
3. **features_pipeline.py** — sklearn Pipeline: percentage returns + MACD indicators
4. **train_model.py** — chronological train/test split, model fitting, Comet ML logging, artifact saving
5. **plot_model.py** — residual, actual-vs-predicted, and learning-curve plots saved to `results/`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For deployment only: `pip install -r requirements-deploy.txt`

## Environment Variables

```bash
export COMET_ML_API_KEY="your-comet-key"
export COMET_ML_WORKSPACE="your-workspace"
export COMET_ML_MODEL_NAME="your-model-name"  # used by deployment only
export CEREBRIUM_API_KEY="your-cerebrium-key"  # deployment only
export CEREBRIUM_ENDPOINT_URL="your-endpoint-url"  # deployment only
```

See `environment_variables.sh` for a template.

## How to Run

### 1. Download data

```bash
python src/download_data.py
```

Default: Mastercard (`MC`), 730 days from 2022-01-01. Adjust `--ticker_list`, `--day`, and `--num_days` as needed.

### 2. Train a model

```bash
# Default: Lasso, no tuning
python src/train_model.py

# Specific model (lasso | light | boost | forest)
python src/train_model.py --model forest

# With Optuna hyperparameter tuning
python src/train_model.py --model boost --tune --trials 20

# Limit rows (useful for quick checks)
python src/train_model.py --model light --sample 300
```

Available models: `lasso` (Lasso), `light` (LightGBM), `boost` (XGBoost), `forest` (Random Forest).

### 3. Run baseline

```bash
python src/baseline_model.py
```

Predicts next-day price as equal to the previous day's close (naive baseline). Logs MAE to Comet ML.

## Feature Pipeline

Raw time-series windows are transformed by `src/features_pipeline.py` into:

| Feature | Description |
|---|---|
| `price_1_day_ago` | Most recent close price |
| `percentage_return_2_day` | 1-day percentage return |
| `percentage_return_5_day` | 4-day percentage return |
| `macd` | MACD line (short EMA − long EMA) |
| `signal` | MACD signal line (EMA of MACD) |

All features are computed from past data only — no look-ahead bias.

## Train/Test Split

Data is split **chronologically** (earliest 90% for training, most recent 10% for testing) to respect time ordering and prevent look-ahead bias. Hyperparameter search uses `TimeSeriesSplit` cross-validation.

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `yfinance` | Market data download |
| `scikit-learn` | ML models and pipelines |
| `lightgbm` / `xgboost` | Gradient boosting models |
| `optuna` | Hyperparameter tuning |
| `comet-ml` | Experiment tracking |
| `pandas` / `numpy` | Data processing |
| `matplotlib` | Plotting |

## Project Layout

| Path | Description |
|---|---|
| `src/download_data.py` | Downloads OHLC data from Yahoo Finance |
| `src/preprocess.py` | Slices time-series into sliding (features, target) windows |
| `src/features_pipeline.py` | sklearn Pipeline: percentage returns + MACD |
| `src/train_model.py` | Training, evaluation, and artifact saving |
| `src/hyperparams.py` | Optuna hyperparameter search for all four models |
| `src/baseline_model.py` | Naive previous-day baseline |
| `src/plot_model.py` | Residual, actual-vs-predicted, and learning-curve plots |
| `src/paths.py` | Directory constants (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`) |
| `src/logger.py` | Console logger helper |
| `deploy/deploy.py` | Cerebrium deployment entrypoint |
| `deploy/model_registry_api.py` | Load production model from Comet ML registry |
| `stock-predict-project/` | Cerebrium project config and inference entrypoint |
| `requirements.txt` | Training dependencies |
| `requirements-deploy.txt` | Deployment dependencies |
