import json
import os
from typing import Optional
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")       # suppress yfinance warnings about missing data for some tickers (e.g. DJP)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Please install yfinance: pip install yfinance")


initial_portfolio_value = 10_000   # Starting portfolio value in USD

backtest_start = "2010-01-01"       # format: YYYY-MM-DD
backtest_end   = "2026-01-01"       # format: YYYY-MM-DD
backtest_years = 20                 # How many years back to backtest
                                    # Maximum available is ~20 years (ETF data goes back to ~2004–2007 for most tickers)

rebalance_threshold = 0.05          # Rebalance if any asset drifts > 10% from target

target_allocation = {
    "VTI":  0.30,   # US Total Stock Market (30%)
    "TLT":  0.40,   # Long-Term US Bonds 20+ yr (40%)
    "IEF":  0.15,   # Intermediate US Bonds 7-10 yr (15%)
    "GLD":  0.075,  # Gold (7.5%)
    "DJP":  0.075,  # Commodities (7.5%)
}

print(f"target_allocation: {target_allocation}\n")
print(f"target_allocation items: {target_allocation.items()}\n")

benchmark_ticker = "SPY"           # S&P 500 benchmark for comparison

start = datetime.strptime(backtest_start, "%Y-%m-%d")
end   = datetime.strptime(backtest_end,   "%Y-%m-%d")

tickers  = list(target_allocation.keys()) + [benchmark_ticker]
#tickers  = [benchmark_ticker]


#print(f"tickers: {tickers}")

raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

#print(f"raw:{raw}\n")
#print(f"raw_close:{raw['Close']}\n")