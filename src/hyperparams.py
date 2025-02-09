# imports
import os
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

# log run
logger = get_console_logger()

# create function to set the initial hyperparams
def sample_hyperparams(
    model: Callable,
    trial: optuna.trial.Trial
) -> Dict[str, Union[str, int, float]]:
    """
    Function to grid search hyperparams using optuna trial framework
    """
    if model == Lasso:
        return {
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=True)
        }
    elif model == LGBMRegressor:
        return {
            'metric': 'mae',
            'verbose': -2,
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.2, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 100)
        }
    elif model == XGBRegressor:
        return {
            'metric': 'mae',
            'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        }
    elif model == RandomForestRegressor:
        return {
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 50)
        }
    else:
        raise NotImplementedError('TODO: implement other models')

# function to find best hyperparamters
def best_hyperparams(
    model: Callable,
    hyperparam_trials: int,
    X: pd.DataFrame,
    y: pd.Series,
    experiment: Experiment
) -> Tuple[Dict, Dict]:
    """
    Find best hyperparams using optuna trial search
    """
    
    # allowed models
    assert model in {Lasso, LGBMRegressor, XGBRegressor, RandomForestRegressor}
    
    # objective function to find best hyperparams
    def objective(trial: optuna.trial.Trial) -> float:
        """
        error function to minimize/maximize using hyperparam tuning
        """
        # sample hyperparams
        preprocess_hyperparams = {'pp_macd_short_window': trial.suggest_int('pp_macd_short_window', 1, 30),
                                  'pp_macd_long_window': trial.suggest_int('pp_macd_long_window', 1, 30),
                                  'pp_macd_signal_window': trial.suggest_int('pp_macd_signal_window', 1, 20)}
        model_hyperparams = sample_hyperparams(model, trial)

        # evaluate model using timeseriessplit cross-validation
        tss = TimeSeriesSplit(n_splits=3)
        scores = []
        logger.info(f'{trial.number=}')

        for split_number, (train_index, val_index) in enumerate(tss.split(X)):
            # split into training and validation
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            logger.info(f'{split_number=}')
            logger.info(f'{len(X_train)=}')
            logger.info(f'{len(X_val)=}')
            
            # train model
            pipeline = make_pipeline(
                preprocess_pipeline(**preprocess_hyperparams),
                model(**model_hyperparams)
            )
            pipeline.fit(X_train, y_train)
            
            # evaluate model
            predictions = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            scores.append(mae)
            
            logger.info(f'{mae=}')
        
        # calculate mean score
        score = np.array(scores).mean()
        
        return score
    
    # create optuna hyperparam study object and optimise using objective func
    logger.info('Starting hyper-parameter search...')
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=hyperparam_trials)
    
    # best hyperparams and mae val
    best_params = study.best_params
    best_value = study.best_value
    
    # split best params into preprocessing and model hyperparams
    preprocess_hyperparams = {key: value for key, value in best_params.items() if key.startswith('pp_')}
    model_hyperparams = {key: value for key, value in best_params.items() if not key.startswith('pp_')}
    
    # log best params and mae value
    logger.info('Best parameters')
    for key, value in best_params.items():
        logger.info(f'{key}: {value}')
    
    logger.info(f'Best MAE: {best_value}')
    
    experiment.log_metrics({'CV_MAE': best_value})
    
    return preprocess_hyperparams, model_hyperparams