# imports
from typing import Optional, Union, Tuple, List

import pandas as pd
import ta
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# project imports
from src.paths import DATA_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger()

# function to pull price column
def price_column(X: pd.DataFrame) -> List[str]:
    return [col for col in X.columns if 'price' in col]

class MACD(BaseEstimator, TransformerMixin):
    """
    Adds MACD column to input dataframe from the close prices
    """
    def __init__(self, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
    
    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None) -> "MACD":
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('Adding MACD to input dataframe')
        
        # Calculate Short and Long EMA
        short_ema = X['price_1_day_ago'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = X['price_1_day_ago'].ewm(span=self.long_window, adjust=False).mean()
        
        # Calculate MACD Line and Signal Line
        X['macd'] = short_ema - long_ema
        X['signal'] = X['macd'].ewm(span=self.signal_window, adjust=False).mean()
        
        return X

# function to add price percentage return to dataframe
def price_percentage_return(X: pd.DataFrame, days: int) -> pd.DataFrame:
    X = X.copy()
    X.loc[:, f'percentage_return_{days}_day'] = (X['price_1_day_ago'] - X[f'price_{days}_day_ago']) / X[f'price_{days}_day_ago']
    return X

# feature subset function
def feature_subset(X: pd.DataFrame) -> pd.DataFrame:
    return X[['price_1_day_ago', 'percentage_return_2_day', 'percentage_return_5_day', 'macd', 'signal']]

# preprocess pipeline function
def preprocess_pipeline(
    pp_macd_short_window: int = 12,
    pp_macd_long_window: int = 26,
    pp_macd_signal_window: int = 9
) -> Pipeline:
    """
    Returns the preprocessing pipeline
    """
    return make_pipeline(
        # Calculate price percentage returns
        FunctionTransformer(price_percentage_return, kw_args={'days': 2}),
        FunctionTransformer(price_percentage_return, kw_args={'days': 5}),

        # Add MACD
        MACD(short_window=pp_macd_short_window, 
             long_window=pp_macd_long_window, 
             signal_window=pp_macd_signal_window),
        
        # Select feature subset
        FunctionTransformer(feature_subset)
    )