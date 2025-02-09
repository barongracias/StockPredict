# imports
import os
import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# project imports
from src.preprocess import transform_features_targets
from src.logger import get_console_logger

# log run
logger = get_console_logger()

# function to get baseline error
def baseline_model_error(
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> float:
    predictions = X_test['price_1_day_ago']
    return mean_absolute_error(y_test, predictions)

# train baseline model
def train_model(
    X: pd.DataFrame,
    y: pd.Series
) -> None:
    """
    Split data into train/test and return baseline model error
    """
    
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name="stockpredict"
    )
    experiment.add_tag('baseline_model')
    
    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    
    # log results
    logger.info(f'Training set size: {len(X_train)}')
    logger.info(f'Testing set size: {len(X_test)}')
    
    # baseline model performance
    baseline_mae = baseline_model_error(X_test, y_test)
    logger.info(f'Test MAE: {baseline_mae}')
    experiment.log_metrics({'Test_MAE': baseline_mae})

if __name__ == '__main__':
    # pull features and target from transformed data
    logger.info('Generating features and target')
    features, target = transform_features_targets()
    
    # train model with the features and target dataframes
    logger.info('Starting training')
    train_model(features, target)