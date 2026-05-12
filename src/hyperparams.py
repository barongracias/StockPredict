# imports
from typing import Callable, Dict, Union, Tuple

import numpy as np
import pandas as pd
import optuna
from comet_ml import Experiment
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# project imports
from src.features_pipepline import preprocess_pipeline
from src.logger import get_console_logger

logger = get_console_logger()


def sample_hyperparams(
    model: Callable,
    trial: optuna.trial.Trial
) -> Dict[str, Union[str, int, float]]:
    """
    Returns a dict of model hyperparameters sampled by an Optuna trial.
    """
    if model == Lasso:
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=True),
        }
    elif model == LGBMRegressor:
        return {
            'verbose': -1,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        }
    elif model == XGBRegressor:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        }
    elif model == RandomForestRegressor:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
    else:
        raise ValueError(f"Unsupported model: {model}. Choose from Lasso, LGBMRegressor, XGBRegressor, RandomForestRegressor.")

def best_hyperparams(
    model: Callable,
    hyperparam_trials: int,
    X: pd.DataFrame,
    y: pd.Series,
    experiment: Experiment
) -> Tuple[Dict, Dict]:
    """
    Find best hyperparams using Optuna with TimeSeriesSplit cross-validation.
    Returns (preprocess_hyperparams, model_hyperparams).
    """
    if model not in {Lasso, LGBMRegressor, XGBRegressor, RandomForestRegressor}:
        raise ValueError(f"Unsupported model: {model}")

    def objective(trial: optuna.trial.Trial) -> float:
        preprocess_hyperparams = {
            'pp_macd_short_window': trial.suggest_int('pp_macd_short_window', 5, 30),
            'pp_macd_long_window': trial.suggest_int('pp_macd_long_window', 10, 60),
            'pp_macd_signal_window': trial.suggest_int('pp_macd_signal_window', 3, 20),
        }
        model_hyperparams = sample_hyperparams(model, trial)

        tss = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_index, val_index in tss.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            pipeline = make_pipeline(
                preprocess_pipeline(**preprocess_hyperparams),
                model(**model_hyperparams)
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)
            scores.append(mean_absolute_error(y_val, predictions))

        return float(np.mean(scores))

    logger.info('Starting hyperparameter search...')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=hyperparam_trials)

    best_params = study.best_params
    best_value = study.best_value

    preprocess_hyperparams = {k: v for k, v in best_params.items() if k.startswith('pp_')}
    model_hyperparams = {k: v for k, v in best_params.items() if not k.startswith('pp_')}

    logger.info(f'Best params: {best_params}')
    logger.info(f'Best CV MAE: {best_value}')
    experiment.log_metrics({'CV_MAE': best_value})

    return preprocess_hyperparams, model_hyperparams