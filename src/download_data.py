from typing import List, Optional
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import fire
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

from src.paths import DATA_DIR
from src.logger import get_console_logger


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("config/yfinance.cache"),
)

logger = get_console_logger(name='dataset_generation')


def download_from_yfinance(
    ticker_list: Optional[List[str]] = ["MC"],
    day: Optional[str] = "2022-01-01",
    num_days: Optional[int] = 730,
) -> Path:
    """Downloads historical OHLC data from Yahoo Finance and saves as Parquet."""
    start = day
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=num_days)).strftime("%Y-%m-%d")

    frames = []
    for ticker in ticker_list:
        raw = pd.DataFrame(yf.download(ticker, start=start, end=end, interval="1d", session=session))
        raw = raw.reset_index()
        # yfinance >=0.2 no longer provides 'Adj Close' by default; use 'Close'
        close_col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        raw = raw[['Date', 'High', 'Low', close_col]].copy()
        raw['ticker'] = ticker
        raw['timestamp'] = raw['Date'].apply(lambda x: int(x.timestamp()))
        raw = raw.rename(columns={'High': 'high', 'Low': 'low', close_col: 'close'})
        raw = raw[['timestamp', 'ticker', 'high', 'low', 'close']]
        frames.append(raw)

    full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['timestamp', 'ticker', 'high', 'low', 'close']
    )

    tickers = "-".join(ticker_list)
    file_path = DATA_DIR / f"{tickers}_{start}_to_{end}.parquet"

    if file_path.exists():
        logger.info(f'File already exists: {file_path.name}')
    else:
        logger.info(f'Downloading {tickers} from {start} to {end}')
        full_df.to_parquet(file_path, index=False)

    return file_path


if __name__ == '__main__':
    fire.Fire(download_from_yfinance)
