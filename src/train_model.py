# imports
import os
from typing import Optional, Callable

from datetime import datetime
from comet_ml import Experiment
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser

# project imports
from src.preprocess import transform_features_targets
from src.features_pipepline import preprocess_pipeline
from src.hyperparams import best_hyperparams
from src.plot_model import ModelPlotter

from src.paths import RESULTS_DIR
from src.paths import MODELS_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger()

# function to pass model name
def choose_model(model: str) -> Callable:
    """
    Returns model function given the model name
    """
    model_dict = {
        'lasso': Lasso,
        'light': LGBMRegressor,
        'boost': XGBRegressor,
        'forest': RandomForestRegressor
    }
    return model_dict.get(model, ValueError(f'Unknown model name: {model}'))

# function to train model
def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparams: Optional[bool] = False,
    hyperparam_trials: Optional[int] = 10
) -> None:
    """
    Train boosting tree model using input features X and target y
    """
    
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name="stockpredict"
    )
    
    tuned_model = f'{model}_tuned' if tune_hyperparams else f'{model}'
    experiment.add_tag(tuned_model)
    
    # pull desired model
    model_func = choose_model(model)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    
    train_sizes, train_scores, test_scores = learning_curve(
        model_func(),
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    if not tune_hyperparams:
        # create full pipeline with default hyperparams
        logger.info('Using default hyperparameters')
        
        pipeline = make_pipeline(
            preprocess_pipeline(),
            model_func()
        )
    else:
        # find best hyperparams using cross-validation
        logger.info('Finding best hyperparameters using cross-validation')
        
        preprocess_hyperparams, model_hyperparams = best_hyperparams(model_func, hyperparam_trials, X_train, y_train, experiment)
        logger.info(f'Best preprocessing hyperparameters: {preprocess_hyperparams}')
        logger.info(f'Best model hyperparameters: {model_hyperparams}')
        
        pipeline = make_pipeline(
            preprocess_pipeline(**preprocess_hyperparams),
            model_func(**model_hyperparams)
        )
        
        experiment.add_tag(tuned_model)
    
    # train model
    logger.info('Fitting model')
    pipeline.fit(X_train, y_train)
    
    # compute mae
    predictions = pipeline.predict(X_test)
    mae_error = mean_absolute_error(y_test, predictions)
    logger.info(f'Test MAE: {mae_error}')
    experiment.log_metrics({'Test_MAE': mae_error})
    
    # save model name and timestamp
    timestamp = datetime.today().strftime("%Y-%m-%d")
    logger.info('Saving model to disk')
    model_name = f"{model}_tuned_{timestamp}.pkl" if tune_hyperparams else f"{model}_{timestamp}.pkl"
    with open(MODELS_DIR / model_name, "wb") as f:
        pickle.dump(pipeline, f)
    
    # log model artifact
    experiment.log_model(model_name, str(MODELS_DIR / model_name))
    
    # plot and save performance metrics for each model
    plotter = ModelPlotter()
    plots = [
        (plotter.plot_residuals(y_test, predictions, model), 'res'),
        (plotter.plot_actual_predicted(y_test, predictions, model), 'avp'),
        (plotter.plot_learning_curve(train_sizes, train_scores_mean, test_scores_mean, model), 'lc')
    ]
    
    # create file if file does not already exist
    for plot, plot_name in plots:
        file_name = f"{model}_tuned_{plot_name}_{timestamp}.png" if tune_hyperparams else f"{model}_{plot_name}_{timestamp}.png"
        file_path = RESULTS_DIR / file_name
        if file_path.exists():
            logger.info(f'File {file_name} already exists')
        else:
            logger.info(f'Saving plots for {model}')
            plotter.save_plot(plot, file_name)
    
if __name__ == '__main__':
    # create arguments for CLI execution
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    # create features and target dataframes
    logger.info('Generating features and targets')
    features_df, target_df = transform_features_targets()
    
    # reduce input size to speed training
    if args.sample is not None:
        features_df = features_df.head(args.sample)
        target_df = target_df.head(args.sample)
        
    # train model
    logger.info('Training model')
    train(features_df, target_df,
          model=args.model,
          tune_hyperparams=args.tune,
          hyperparam_trials=args.trials
          )