import os

import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.preprocess import transform_features_targets
from src.logger import get_console_logger

logger = get_console_logger()


def baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Naive baseline: predict next-day price == previous day's close."""
    return mean_absolute_error(y_test, X_test['price_1_day_ago'])


def train_model(X: pd.DataFrame, y: pd.Series) -> None:
    experiment = Experiment(
        api_key=os.environ["COMET_ML_API_KEY"],
        workspace=os.environ["COMET_ML_WORKSPACE"],
        project_name="stockpredict",
    )
    experiment.add_tag('baseline_model')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    logger.info(f'Training set size: {len(X_train)}')
    logger.info(f'Testing set size: {len(X_test)}')

    baseline_mae = baseline_model_error(X_test, y_test)
    logger.info(f'Test MAE: {baseline_mae}')
    experiment.log_metrics({'Test_MAE': baseline_mae})


if __name__ == '__main__':
    logger.info('Generating features and target')
    features, target = transform_features_targets()
    logger.info('Starting baseline evaluation')
    train_model(features, target)
