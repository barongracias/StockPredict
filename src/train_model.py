import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.preprocess import transform_features_targets
from src.features_pipepline import preprocess_pipeline
from src.hyperparams import best_hyperparams
from src.plot_model import ModelPlotter
from src.paths import RESULTS_DIR, MODELS_DIR
from src.logger import get_console_logger

logger = get_console_logger()

MODEL_MAP = {
    'lasso': Lasso,
    'light': LGBMRegressor,
    'boost': XGBRegressor,
    'forest': RandomForestRegressor,
}


def choose_model(model: str) -> Callable:
    if model not in MODEL_MAP:
        raise ValueError(f"Unknown model: '{model}'. Choose from: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[model]


def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparams: Optional[bool] = False,
    hyperparam_trials: Optional[int] = 10,
) -> None:
    """Train a model, evaluate it, log to Comet ML, and save artifacts."""

    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name="stockpredict",
    )
    experiment.add_tag(f'{model}_tuned' if tune_hyperparams else model)

    model_func = choose_model(model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    if not tune_hyperparams:
        logger.info('Using default hyperparameters')
        pipeline = make_pipeline(preprocess_pipeline(), model_func())
        preprocess_hyperparams = {}
    else:
        logger.info('Finding best hyperparameters using cross-validation')
        preprocess_hyperparams, model_hyperparams = best_hyperparams(
            model_func, hyperparam_trials, X_train, y_train, experiment
        )
        logger.info(f'Best preprocessing hyperparameters: {preprocess_hyperparams}')
        logger.info(f'Best model hyperparameters: {model_hyperparams}')
        pipeline = make_pipeline(
            preprocess_pipeline(**preprocess_hyperparams),
            model_func(**model_hyperparams),
        )

    logger.info('Computing learning curve')
    train_sizes, train_scores, test_scores = learning_curve(
        make_pipeline(preprocess_pipeline(**preprocess_hyperparams), model_func()),
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    logger.info('Fitting model')
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae_error = mean_absolute_error(y_test, predictions)
    logger.info(f'Test MAE: {mae_error}')
    experiment.log_metrics({'Test_MAE': mae_error})

    timestamp = datetime.today().strftime("%Y-%m-%d")
    model_name = f"{model}_tuned_{timestamp}.pkl" if tune_hyperparams else f"{model}_{timestamp}.pkl"
    logger.info('Saving model to disk')
    with open(MODELS_DIR / model_name, "wb") as f:
        pickle.dump(pipeline, f)
    experiment.log_model(model_name, str(MODELS_DIR / model_name))

    plotter = ModelPlotter()
    plots = [
        (plotter.plot_residuals(y_test, predictions, model), 'res'),
        (plotter.plot_actual_predicted(y_test, predictions, model), 'avp'),
        (plotter.plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean, model), 'lc'),
    ]
    for plot, plot_name in plots:
        suffix = 'tuned_' if tune_hyperparams else ''
        file_name = f"{model}_{suffix}{plot_name}_{timestamp}.png"
        file_path = RESULTS_DIR / file_name
        if file_path.exists():
            logger.info(f'File {file_name} already exists, skipping')
        else:
            plotter.save_plot(plot, file_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    logger.info('Generating features and targets')
    features_df, target_df = transform_features_targets()

    if args.sample is not None:
        features_df = features_df.head(args.sample)
        target_df = target_df.head(args.sample)

    logger.info('Training model')
    train(
        features_df,
        target_df,
        model=args.model,
        tune_hyperparams=args.tune,
        hyperparam_trials=args.trials,
    )
