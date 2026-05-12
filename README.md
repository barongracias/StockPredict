# StockPredict

Train, evaluate, and deploy ML models for stock price prediction. Uses Yahoo Finance for data, scikit-learn/LightGBM/XGBoost for modelling, Optuna for hyperparameter tuning, Comet ML for experiment tracking, and Cerebrium for deployment.

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
export CEREBRIUM_API_KEY="your-cerebrium-key"   # deployment only
```

See `environment_variables.sh` for a template.

## Data Download

Downloads daily OHLC data from Yahoo Finance and saves to `data/` as Parquet.

```bash
python src/download_data.py
```

Default: Mastercard (`MC`), 730 days from 2022-01-01. Adjust `--ticker_list`, `--day`, and `--num_days` as needed.

## Training

`src/train_model.py` reads preprocessed data, trains a model, saves it to `models/`, and writes performance plots to `results/`.

**Available models:** `lasso`, `light` (LightGBM), `boost` (XGBoost), `forest` (Random Forest)

```bash
# Default: Lasso, no tuning
python src/train_model.py

# Specific model
python src/train_model.py --model forest

# With Optuna hyperparameter tuning
python src/train_model.py --model boost --tune --trials 20

# Limit rows (useful for quick checks)
python src/train_model.py --model light --sample 300

# Combined
python src/train_model.py --model forest --tune --sample 500 --trials 20
```

## Baseline

```bash
python src/baseline_model.py
```

Predicts next-day price as equal to the previous day's close (naive baseline). Logs MAE to Comet ML.

## Feature Pipeline

Raw time-series windows are transformed by `src/features_pipepline.py` into:
- `price_1_day_ago` — most recent close
- `percentage_return_2_day` — 1-day percentage return
- `percentage_return_5_day` — 4-day percentage return
- `macd` — MACD line
- `signal` — MACD signal line

## Deployment

```bash
python deploy/deploy.py --local-pickle <model_filename>.pkl
```

`deploy/model_registry_api.py` provides `load_prod_model()` to pull a production model from the Comet ML registry.

The `stock-predict-project/` directory contains the Cerebrium deployment configuration (`cerebrium.toml`) and entrypoint (`main.py`).

## Project Layout

| Path | Description |
|---|---|
| `src/download_data.py` | Downloads OHLC data from Yahoo Finance |
| `src/preprocess.py` | Slices time-series into sliding (features, target) windows |
| `src/features_pipepline.py` | sklearn Pipeline: percentage returns + MACD |
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
