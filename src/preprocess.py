# imports
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import fire

# project imports
from src.paths import DATA_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger()

def transform_features_targets(
    input_path: Optional[Path] = DATA_DIR / 'MC_2023-01-01_to_2024-01-01.parquet',
    input_seq_len: Optional[int] = 7,
    step_size: Optional[int] = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slices and transposes data from time-series format into (features, target)
    """
    
    # load parquet file
    timeseries_data = pd.read_parquet(input_path)
    timeseries_data = timeseries_data[['timestamp', 'close']]
    timeseries_data.sort_values(by=['timestamp'], inplace=True)
    
    # create output features and targets dataframe
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    # get sliding window indices for features and target prices 
    indices = cutoff_indices(timeseries_data, input_seq_len, step_size)
    
    # number of windows in data range i.e. date range / input sequence length
    num_windows = len(indices)
    
    # slice and transpose data into numpy arrays
    x = np.ndarray(shape=(num_windows, input_seq_len), dtype=np.float32)
    y = np.ndarray(shape=(num_windows), dtype=np.float32)
    timestamps = []
    
    for i, idx in enumerate(indices): #e.g. idx = (0, 3, 4) -> feature window are indices 0,1,2 and target window is index 3
        # idx[0] to idx[1] pulls prices at feature between window start and end indices
        x[i, :] = timeseries_data.iloc[idx[0]: idx[1]]['close'].values
        
        # idx[1] to idx[2] pulls price at target window index
        y[i] = timeseries_data.iloc[idx[1]: idx[2]]['close'].values
        
        # store timestamp at window end index
        timestamps.append(timeseries_data.iloc[idx[1]]['timestamp'])
    
    # features dataframe with input sequence length columns
    features = pd.DataFrame(x, columns=[f'price_{i+1}_day_ago' for i in reversed(range(input_seq_len))])
    
    # target dataframe with target price after features window
    targets = pd.DataFrame(y, columns=[f'price_next_day'])
    
    return features, targets['price_next_day']

def cutoff_indices(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int
) -> List[Tuple[int, int, int]]:
    """
    Produces tuple of indices for feature and targets 
    i.e., sliding window based on input sequence length
    """
    
    # all days captured until final target day
    stop_position = len(data)
    
    subseq_first_index = 0                  # index of first day in feature window
    subseq_middle_index = input_seq_len     # index of last day in feature window
    subseq_last_index = input_seq_len + 1   # index of target day after feature window
    indices = []
    
    while subseq_last_index <= stop_position:
        # append each index as a tuple
        indices.append((subseq_first_index, subseq_middle_index, subseq_last_index))
        subseq_first_index += step_size
        subseq_middle_index += step_size
        subseq_last_index += step_size

    return indices

if __name__ == '__main__':
    features_df, target_df = fire.Fire(transform_features_targets)