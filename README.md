# StockPredict

Train, evaluate, and deploy machine learning models for stock price prediction using scikit-learn, LightGBM, and XGBoost. Experiment tracking is handled by Comet ML; deployment targets Cerebrium.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: the repository previously used a `stock_venv` environment. Create your own with the name of your choice.

## API Keys Required

Set the following environment variables before running training scripts:

| Variable | Purpose |
|---|---|
| `COMET_ML_API_KEY` | Comet ML experiment tracking |
| `COMET_ML_WORKSPACE` | Comet ML workspace name |

```bash
export COMET_ML_API_KEY="your-comet-key"
export COMET_ML_WORKSPACE="your-workspace"
```

## Getting the Data

`src/download_data.py` fetches historical OHLC data from Yahoo Finance and saves it as Parquet files in the `data/` directory.

Currently configured to download Mastercard (`MC`) for 730 days from 2022-01-01. Adjust `ticker_list`, `day`, and `num_days` as needed.

```bash
python src/download_data.py
```

## Training a Model

`src/train_model.py` transforms the raw data into features and targets, trains a model, saves it to `models/`, and writes performance plots to `results/`.

**Model selection** (`lasso`, `light`, `boost`, `forest`):

```bash
python src/train_model.py --model forest
```

**Hyperparameter tuning** (Optuna cross-validation search):

```bash
python src/train_model.py --tune
```

**Limit training rows** (useful for quick checks):

```bash
python src/train_model.py --sample 500
```

**Number of Optuna trials:**

```bash
python src/train_model.py --trials 20
```

**Combined example:**

```bash
python src/train_model.py --model forest --tune --sample 500 --trials 20
```

## Project Layout

| Path | Description |
|---|---|
| `src/download_data.py` | Downloads OHLC data from Yahoo Finance |
| `src/preprocess.py` | Slices time-series into (features, target) windows |
| `src/features_pipepline.py` | sklearn Pipeline with MACD and percentage-return transformers |
| `src/train_model.py` | Model training, evaluation, and artifact saving |
| `src/hyperparams.py` | Optuna hyperparam search |
| `src/plot_model.py` | Residual, actual-vs-predicted, and learning-curve plots |
| `src/paths.py` | Directory path constants (`DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`) |
| `src/logger.py` | Logging helper |
| `deploy/` | Cerebrium deployment scripts (see below) |

## Deployment

The `deploy/` folder contains scripts for deploying trained models to [Cerebrium](https://www.cerebrium.ai/), a serverless ML platform. A Cerebrium account and `CEREBRIUM_API_KEY` environment variable are required.

```bash
python deploy/deploy.py --local-pickle <model_filename>.pkl
```

`deploy/model_registry_api.py` provides a helper to pull a production model directly from the Comet ML model registry.
