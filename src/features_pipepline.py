from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from src.logger import get_console_logger

logger = get_console_logger()


class MACD(BaseEstimator, TransformerMixin):
    """Adds MACD and signal columns computed from the most-recent close price."""

    def __init__(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MACD":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('Adding MACD to input dataframe')
        X = X.copy()
        short_ema = X['price_1_day_ago'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = X['price_1_day_ago'].ewm(span=self.long_window, adjust=False).mean()
        X['macd'] = short_ema - long_ema
        X['signal'] = X['macd'].ewm(span=self.signal_window, adjust=False).mean()
        return X


def price_percentage_return(X: pd.DataFrame, days: int) -> pd.DataFrame:
    X = X.copy()
    X[f'percentage_return_{days}_day'] = (
        (X['price_1_day_ago'] - X[f'price_{days}_day_ago']) / X[f'price_{days}_day_ago']
    )
    return X


def feature_subset(X: pd.DataFrame) -> pd.DataFrame:
    return X[['price_1_day_ago', 'percentage_return_2_day', 'percentage_return_5_day', 'macd', 'signal']]


def preprocess_pipeline(
    pp_macd_short_window: int = 12,
    pp_macd_long_window: int = 26,
    pp_macd_signal_window: int = 9,
) -> Pipeline:
    """Returns the preprocessing pipeline."""
    return make_pipeline(
        FunctionTransformer(price_percentage_return, kw_args={'days': 2}),
        FunctionTransformer(price_percentage_return, kw_args={'days': 5}),
        MACD(
            short_window=pp_macd_short_window,
            long_window=pp_macd_long_window,
            signal_window=pp_macd_signal_window,
        ),
        FunctionTransformer(feature_subset),
    )
