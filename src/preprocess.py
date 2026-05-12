from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import fire

from src.paths import DATA_DIR
from src.logger import get_console_logger

logger = get_console_logger()


def transform_features_targets(
    input_path: Optional[Path] = DATA_DIR / 'MC_2023-01-01_to_2024-01-01.parquet',
    input_seq_len: int = 7,
    step_size: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Slices time-series data into sliding (features, target) windows."""
    timeseries_data = pd.read_parquet(input_path)[['timestamp', 'close']]
    timeseries_data = timeseries_data.sort_values('timestamp').reset_index(drop=True)

    indices = cutoff_indices(timeseries_data, input_seq_len, step_size)
    num_windows = len(indices)

    x = np.empty((num_windows, input_seq_len), dtype=np.float32)
    y = np.empty(num_windows, dtype=np.float32)

    for i, (start, mid, end) in enumerate(indices):
        x[i] = timeseries_data.iloc[start:mid]['close'].values
        y[i] = timeseries_data.iloc[mid:end]['close'].values[0]

    features = pd.DataFrame(x, columns=[f'price_{j+1}_day_ago' for j in reversed(range(input_seq_len))])
    target = pd.Series(y, name='price_next_day')

    return features, target


def cutoff_indices(
    data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
) -> List[Tuple[int, int, int]]:
    """Returns sliding window index triples (feature_start, feature_end, target_end)."""
    stop_position = len(data)
    first, mid, last = 0, input_seq_len, input_seq_len + 1
    indices = []
    while last <= stop_position:
        indices.append((first, mid, last))
        first += step_size
        mid += step_size
        last += step_size
    return indices


if __name__ == '__main__':
    features_df, target_df = fire.Fire(transform_features_targets)
