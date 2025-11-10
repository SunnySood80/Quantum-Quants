"""
EDA: NaN Value Analysis
Investigate which stocks/features have missing data and where it's concentrated.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from portfolio_data_pipeline import fetch_sp500_tickers, download_price_data
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("NaN VALUE ANALYSIS - Exploratory Data Analysis")
print("="*80)

# Step 1: Fetch data
print("\nStep 1: Fetching S&P 500 data...")
tickers_df = fetch_sp500_tickers()
# Extract ticker list from DataFrame
tickers = tickers_df['Ticker'].tolist()
print(f"Total tickers: {len(tickers)}")

close_df = download_price_data(tickers, period='2y', interval='1d')
print(f"Downloaded data shape: {close_df.shape}")
print(f"Date range: {close_df.index[0]} to {close_df.index[-1]}")

# Step 2: Calculate log returns (same as pipeline)
print("\nStep 2: Calculating log returns...")
log_ret = np.log(close_df / close_df.shift(1))
log_ret = log_ret.dropna(how='all')

print(f"Log returns shape: {log_ret.shape}")

# Step 3: Analyze NaN patterns
print("\n" + "="*80)
print("NaN PATTERN ANALYSIS")
print("="*80)

total_values = log_ret.shape[0] * log_ret.shape[1]
total_nans = log_ret.isna().sum().sum()
nan_percentage = (total_nans / total_values) * 100

print(f"\nOVERALL STATISTICS:")
print(f"  Total data points: {total_values:,}")
print(f"  Total NaN values:  {total_nans:,}")
print(f"  NaN percentage:    {nan_percentage:.2f}%")

# Step 4: NaN by stock (column-wise)
print("\n" + "-"*80)
print("NaN BY STOCK (Top 20 worst offenders)")
print("-"*80)

nan_by_stock = log_ret.isna().sum().sort_values(ascending=False)
nan_pct_by_stock = (nan_by_stock / len(log_ret)) * 100

print("\nTicker   NaN Count   % Missing   Status")
print("-"*50)
for ticker, count in nan_by_stock.head(20).items():
    pct = (count / len(log_ret)) * 100
    status = "CRITICAL" if pct > 10 else ("WARNING" if pct > 1 else "OK")
    print(f"{ticker:<8} {count:>6}     {pct:>6.2f}%     {status}")

# How many stocks have ANY NaN?
stocks_with_nans = (nan_by_stock > 0).sum()
stocks_no_nans = (nan_by_stock == 0).sum()
print(f"\nSummary:")
print(f"  Stocks with NO NaN:   {stocks_no_nans} ({stocks_no_nans/len(nan_by_stock)*100:.1f}%)")
print(f"  Stocks with ANY NaN:  {stocks_with_nans} ({stocks_with_nans/len(nan_by_stock)*100:.1f}%)")

# Step 5: NaN by date (row-wise)
print("\n" + "-"*80)
print("NaN BY DATE (Top 20 worst days)")
print("-"*80)

nan_by_date = log_ret.isna().sum(axis=1).sort_values(ascending=False)
nan_pct_by_date = (nan_by_date / log_ret.shape[1]) * 100

print("\nDate         NaN Count   % Missing   Status")
print("-"*50)
for date, count in nan_by_date.head(20).items():
    pct = (count / log_ret.shape[1]) * 100
    status = "CRITICAL" if pct > 10 else ("WARNING" if pct > 1 else "OK")
    print(f"{str(date)[:10]}   {count:>6}     {pct:>6.2f}%     {status}")

# How many days have ANY NaN?
days_with_nans = (nan_by_date > 0).sum()
days_no_nans = (nan_by_date == 0).sum()
print(f"\nSummary:")
print(f"  Days with NO NaN:   {days_no_nans} ({days_no_nans/len(nan_by_date)*100:.1f}%)")
print(f"  Days with ANY NaN:  {days_with_nans} ({days_with_nans/len(nan_by_date)*100:.1f}%)")

# Step 6: NaN clustering analysis
print("\n" + "="*80)
print("NaN CLUSTERING ANALYSIS")
print("="*80)

# Check if NaNs are isolated or in runs
print("\nChecking if NaNs are isolated or in consecutive runs...")
for ticker in nan_by_stock.head(10).index:
    stock_data = log_ret[ticker]
    nan_mask = stock_data.isna()
    
    # Find runs of NaN
    nan_runs = []
    in_run = False
    run_length = 0
    
    for is_nan in nan_mask:
        if is_nan:
            if in_run:
                run_length += 1
            else:
                in_run = True
                run_length = 1
        else:
            if in_run:
                nan_runs.append(run_length)
                in_run = False
                run_length = 0
    
    if in_run:
        nan_runs.append(run_length)
    
    if nan_runs:
        avg_run = np.mean(nan_runs)
        max_run = max(nan_runs)
        print(f"{ticker:<8}: {len(nan_runs)} gaps, avg {avg_run:.1f} days, max {max_run} days")

# Step 7: Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if nan_percentage < 0.5:
    print("\n[LOW IMPACT] < 0.5% missing data")
    print("  -> Forward/backward fill is SAFE and won't distort statistics")
    print("  -> Alternative: Drop stocks with >1% missing data")
    
elif nan_percentage < 2.0:
    print("\n[MODERATE IMPACT] 0.5-2% missing data")
    print("  -> Forward fill preferred (maintains trend)")
    print("  -> Consider: Interpolation for longer gaps")
    
else:
    print("\n[HIGH IMPACT] > 2% missing data")
    print("  -> WARNING: Filling may introduce bias")
    print("  -> Consider: Drop stocks with >5% missing")
    print("  -> Alternative: Use shorter time period")

# Specific recommendations
print("\nSPECIFIC OPTIONS:")

# Option 1: Drop worst offenders
stocks_over_1pct = nan_pct_by_stock[nan_pct_by_stock > 1.0]
if len(stocks_over_1pct) > 0:
    print(f"\nOption 1: Drop {len(stocks_over_1pct)} stocks with >1% missing")
    print(f"  Removes: {stocks_over_1pct.index.tolist()}")
    print(f"  Keeps: {len(log_ret.columns) - len(stocks_over_1pct)} stocks")
    print(f"  Remaining NaN: {log_ret[nan_pct_by_stock[nan_pct_by_stock <= 1.0].index].isna().sum().sum()}")

# Option 2: Forward fill only
print(f"\nOption 2: Forward fill (current fix)")
print(f"  Pros: Simple, preserves all stocks, maintains last known value")
print(f"  Cons: May lag during gaps, assumes price doesn't change")
print(f"  Impact: Fills {total_nans} values")

# Option 3: Interpolation
print(f"\nOption 3: Linear interpolation")
print(f"  Pros: Smoother transitions, more realistic for short gaps")
print(f"  Cons: Creates artificial data, may introduce false trends")
print(f"  Best for: Isolated 1-3 day gaps")

# Option 4: Market return imputation
print(f"\nOption 4: Use market average return for missing days")
print(f"  Pros: More accurate than zero, less bias than forward fill")
print(f"  Cons: More complex, requires S&P 500 index data")
print(f"  Impact: Fills with market-relative returns")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

