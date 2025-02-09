# imports
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import fire

# project imports
from src.paths import DATA_DIR
from src.logger import get_console_logger

# rate-limiting cache imports
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

# initialise session with 2 requests per 5 seconds
session = CachedLimiterSession(
    limiter=Limiter(RequestRate(2, Duration.SECOND*5)),
    bucket_class=MemoryQueueBucket,
    backend=SQLiteCache("config/yfinance.cache"),
)

# log run
logger = get_console_logger(name='dataset_generation')

# function to download data from yfinance
def download_from_yfinance(
    ticker_list: Optional[List[str]] = ["MC"], #["TSLA", "AAPL", "AMZN"],
    day: Optional[str] = "2022-01-01",
    num_days: Optional[int] = 730
) -> Path:
    """
    Downloads historical data from yahoofinance api and saves to disk
    """
    
    # create start and end date strings
    start = day
    end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=num_days)).strftime("%Y-%m-%d")
    
    # create empty dataframe
    full_df = pd.DataFrame(columns=['timestamp', 'ticker', 'high', 'low', 'close'])
    
    # call yfinance api for each ticker
    for ticker in ticker_list:
        ticker_data = pd.DataFrame(yf.download(ticker, start=start, end=end, interval="1d", session=session))
        ticker_data = ticker_data.reset_index()
        ticker_data = ticker_data[['Date', 'High', 'Low', 'Adj Close']]
        ticker_data['Ticker'] = ticker
        ticker_data['Timestamp'] = ticker_data['Date'].apply(lambda x: int(x.timestamp()))
        ticker_data = ticker_data[['Timestamp', 'Ticker', 'High', 'Low', 'Adj Close']]
        ticker_data = ticker_data.rename(columns={
            'Timestamp': 'timestamp',
            'Ticker': 'ticker',
            'High': 'high',
            'Low': 'low',
            'Adj Close': 'close'
        })
        
        # union each dataframe
        full_df = pd.concat([full_df, ticker_data], ignore_index=True)
    
    # create ticker list and save to parquet
    tickers = "-".join(ticker_list)
    file_path = DATA_DIR / f"{tickers}_{start}_to_{end}.parquet"
    
    # create file if file does not already exist
    if file_path.exists():
        logger.info(f'File {tickers}_{start}_to_{end}.parquet already exists')
    else:
        logger.info(f'Downloading data for {tickers} between {start} and {end}')
        full_df.to_parquet(file_path, index=False)
    
    return file_path

if __name__== '__main__':
    fire.Fire(download_from_yfinance)