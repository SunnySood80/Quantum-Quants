"""
Portfolio Data Pipeline
Fetches S&P 500 tickers, downloads price data, calculates returns,
fetches fundamental features, and cleans via VIF analysis.
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import io
import time
from typing import Dict, Tuple, List
from statsmodels.stats.outliers_influence import variance_inflation_factor


def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Scrapes the S&P 500 ticker list from Wikipedia.
    
    Returns:
        DataFrame with 'Ticker' column containing S&P 500 symbols
    """
    print("Fetching S&P 500 ticker list from Wikipedia...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return pd.DataFrame({"Ticker": []})
    
    if not tables:
        print("Error: No tables found on Wikipedia page.")
        return pd.DataFrame({"Ticker": []})
    
    # Find the correct table
    sp500_df = None
    for table in tables:
        if 'Symbol' in table.columns and 'Security' in table.columns:
            sp500_df = table
            break
    
    if sp500_df is None:
        print("Error: Could not find the S&P 500 constituents table.")
        return pd.DataFrame({"Ticker": []})
    
    tickers = sp500_df['Symbol'].astype(str)
    
    # Clean tickers
    clean_tickers = (
        tickers
        .str.upper()
        .str.replace("-", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
        .drop_duplicates()
    )
    
    print(f"Successfully fetched {len(clean_tickers)} S&P 500 tickers.")
    
    return pd.DataFrame({"Ticker": clean_tickers}).reset_index(drop=True)


def download_price_data(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    max_retries: int = 3,
    initial_delay: float = 5.0
) -> pd.DataFrame:
    """
    Download historical price data with retry logic and rate limiting.
    
    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., "2y", "1y")
        interval: Data interval (e.g., "1d", "1h")
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay before first download attempt
        
    Returns:
        DataFrame with Close prices
    """
    print(f"\nDownloading {interval} price data for period {period}...")
    print(f"Processing {len(tickers)} tickers")
    
    # Add initial delay to avoid immediate rate limiting
    if initial_delay > 0:
        print(f"Waiting {initial_delay}s to avoid rate limiting...")
        time.sleep(initial_delay)
    
    for attempt in range(max_retries):
        try:
            close_df = yf.download(
                tickers,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False  # Disable threading to reduce load
            )["Close"]
            
            # Handle single vs multiple tickers
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(name=tickers[0])
            
            # Drop columns/rows that are all NaN
            close_df = close_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            print(f"Got data for {len(close_df.columns)} tickers")
            return close_df
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 10 * (2 ** attempt)  # Longer waits: 10s, 20s, 40s
                print(f"Download failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                raise
    
    return pd.DataFrame()


def calculate_returns_and_statistics(
    close_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Calculate log returns, mu (expected returns), and Sigma (covariance).
    Includes validation to prevent non-finite values.
    
    Args:
        close_df: DataFrame with close prices
        
    Returns:
        Tuple of (log_returns, mu, Sigma)
    """
    print("\nCalculating returns and statistics...")
    
    # Calculate log returns
    log_ret = np.log(close_df / close_df.shift(1))
    log_ret = log_ret.dropna(how='all')
    
    # Remove any columns with all NaN or infinite values
    valid_cols = []
    for col in log_ret.columns:
        if log_ret[col].notna().sum() > 0 and np.isfinite(log_ret[col].dropna()).all():
            valid_cols.append(col)
    
    log_ret = log_ret[valid_cols]
    print(f"Valid tickers after returns calculation: {len(valid_cols)}")
    
    # OPTIMAL FIX: Drop stocks with >10% missing data (recent IPOs/spinoffs)
    # This prevents NaN propagation without introducing artificial data
    nan_count_before = log_ret.isna().sum().sum()
    if nan_count_before > 0:
        print(f"\nDetected {nan_count_before} missing values across dataset")
        
        # Calculate % missing per stock
        nan_pct_by_stock = (log_ret.isna().sum() / len(log_ret)) * 100
        stocks_to_drop = nan_pct_by_stock[nan_pct_by_stock > 10.0].index.tolist()
        
        if stocks_to_drop:
            print(f"Dropping {len(stocks_to_drop)} stocks with >10% missing data:")
            for ticker in stocks_to_drop:
                pct = nan_pct_by_stock[ticker]
                print(f"  - {ticker}: {pct:.1f}% missing (likely recent IPO/spinoff)")
            
            # Drop problematic stocks
            log_ret = log_ret.drop(columns=stocks_to_drop)
            
            # Check remaining NaN (should be 0 or very few)
            nan_count_after = log_ret.isna().sum().sum()
            print(f"Remaining NaN after drop: {nan_count_after}")
            
            if nan_count_after > 0:
                # If any NaN remain, use conservative forward fill
                print(f"Filling {nan_count_after} remaining isolated gaps (forward fill)")
                log_ret = log_ret.ffill().bfill().fillna(0.0)
        else:
            # No major offenders, just fill isolated gaps
            print(f"No stocks with >10% missing. Filling isolated gaps (forward fill)")
            log_ret = log_ret.ffill().bfill().fillna(0.0)
        
        print(f"Final dataset: {len(log_ret.columns)} stocks, 0 NaN values")
    
    # Annualize (252 trading days)
    mu = log_ret.mean() * 252
    Sigma = log_ret.cov() * 252
    
    # Validate mu and Sigma
    valid_tickers = []
    for ticker in mu.index:
        if (np.isfinite(mu[ticker]) and 
            np.isfinite(Sigma.loc[ticker, ticker]) and
            Sigma.loc[ticker, ticker] > 0):
            valid_tickers.append(ticker)
    
    log_ret = log_ret[valid_tickers]
    mu = mu[valid_tickers]
    Sigma = Sigma.loc[valid_tickers, valid_tickers]
    
    print(f"Valid tickers after validation: {len(valid_tickers)}")
    print(f"Calculated mu and Sigma for {len(mu)} tickers")
    
    # Check for issues
    if not np.all(np.isfinite(mu)):
        raise ValueError("Non-finite values in mu after validation!")
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Non-finite values in Sigma after validation!")
    
    return log_ret, mu, Sigma


def fetch_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch fundamental features for given tickers.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        DataFrame with fundamental features (tickers as index)
    """
    print("\nFetching fundamental features...")
    
    fund_df = pd.DataFrame(index=tickers)
    
    feature_map = {
        # Valuation
        "priceToSalesTrailing12Months": "priceToSalesTTM",
        "priceToBook": "priceToBook",
        "forwardPE": "forwardPE",
        "trailingPE": "trailingPE",
        "pegRatio": "pegRatio",
        "enterpriseToRevenue": "enterpriseToRevenue",
        "enterpriseToEbitda": "enterpriseToEbitda",
        # Profitability
        "profitMargins": "profitMargin",
        "operatingMargins": "operatingMargin",
        "grossMargins": "grossMargin",
        "returnOnAssets": "returnOnAssets",
        "returnOnEquity": "returnOnEquity",
        # Growth
        "revenueGrowth": "revenueGrowth",
        "earningsGrowth": "earningsGrowth",
        "earningsQuarterlyGrowth": "earningsQuarterlyGrowth",
        # Financial Health
        "debtToEquity": "debtToEquity",
        "currentRatio": "currentRatio",
        "quickRatio": "quickRatio",
        # Dividends
        "dividendYield": "dividendYield",
        "payoutRatio": "payoutRatio",
        "trailingAnnualDividendYield": "trailingDividendYield",
        # Risk/Other
        "beta": "beta",
        "bookValue": "bookValue",
        "freeCashflow": "freeCashflow",
        "operatingCashflow": "operatingCashflow"
    }
    
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            for api_key, col_name in feature_map.items():
                value = info.get(api_key)
                if value is not None and np.isfinite(value):
                    # Normalize debt to equity if needed
                    if col_name == "debtToEquity" and value > 1:
                        value = value / 100
                    fund_df.at[ticker, col_name] = value
            
            # Log transform P/S if available
            ps = info.get("priceToSalesTrailing12Months")
            if ps and ps > 0:
                fund_df.at[ticker, "logPriceToSales"] = np.log(ps)
                
        except Exception as e:
            print(f"Warning: Error fetching {ticker}: {e}")
            continue
    
    print(f"Fetched fundamentals for {len(fund_df)} tickers")
    print(f"Total features before cleaning: {len(fund_df.columns)}")
    
    return fund_df


def clean_fundamentals_vif(fund_df: pd.DataFrame, vif_threshold: float = 10.0) -> pd.DataFrame:
    """
    Clean fundamental features using VIF (Variance Inflation Factor) analysis
    to remove multicollinear features.
    
    Args:
        fund_df: DataFrame with fundamental features
        vif_threshold: VIF threshold (features above this are removed)
        
    Returns:
        DataFrame with cleaned features
    """
    print("\n" + "="*60)
    print("CLEANING FUNDAMENTALS - VIF ANALYSIS")
    print("="*60)
    
    original_rows = fund_df.shape[0]
    
    # Keep columns with ≥50% non-null
    min_coverage = int(len(fund_df) * 0.5)
    fund_clean = fund_df.dropna(axis=1, thresh=min_coverage).copy()
    
    # Numeric columns only
    numeric_cols = fund_clean.select_dtypes(include=[np.number]).columns.tolist()
    X = fund_clean[numeric_cols].copy()
    
    # Impute dividend yields as 0 (structural zeros)
    for c in ['dividendYield', 'trailingDividendYield']:
        if c in X.columns:
            mask = X[c].isna()
            if mask.any():
                X.loc[mask, c] = 0.0
    
    # Mean-impute remaining NaNs
    X = X.fillna(X.mean())
    
    # Safety fallback
    if X.isna().any().any():
        X = X.fillna(X.median())
    
    # Remove zero-variance columns
    zero_var_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var_cols:
        print(f"Removing zero-variance features: {zero_var_cols}")
        X = X.drop(columns=zero_var_cols)
    
    print(f"Starting with {X.shape[1]} features (rows preserved: {X.shape[0]} / {original_rows})")
    
    # Manual feature drops (keep best versions)
    manual_drops = [
        'priceToSalesTTM',            # Keep logPriceToSales
        'earningsQuarterlyGrowth',    # Keep earningsGrowth
        'quickRatio',                 # Keep currentRatio
        'operatingMargin',            # Keep profitMargin
        'trailingPE',                 # Keep forwardPE
        'returnOnAssets',             # Keep returnOnEquity
        'enterpriseToEbitda',         # Keep enterpriseToRevenue
        'operatingCashflow',          # Keep freeCashflow
        'payoutRatio',                # Keep dividendYield
        'trailingDividendYield',      # Keep dividendYield
    ]
    X = X.drop(columns=[c for c in manual_drops if c in X.columns])
    print(f"After manual drops: {X.shape[1]} features")
    
    # Iteratively remove highest VIF
    max_iterations = 20
    for iteration in range(max_iterations):
        if X.shape[1] <= 2:
            print("Stopped: only 2 features left")
            break
            
        vif_series = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
            name="VIF"
        )
        max_vif = vif_series.max()
        
        if max_vif < vif_threshold:
            print(f"Converged after {iteration} iterations (max VIF = {max_vif:.2f})")
            break
        
        worst_feature = vif_series.idxmax()
        print(f"Iteration {iteration+1}: Removing {worst_feature} (VIF = {max_vif:.2f})")
        X = X.drop(columns=[worst_feature])
    
    # Final VIF
    vif_final = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    
    print("\n" + "="*60)
    print("FINAL VIF SCORES")
    print("="*60)
    print(vif_final.to_string(index=False))
    
    print("\n" + "="*60)
    print(f"FINAL FEATURE SET: {len(X.columns)} features")
    print("="*60)
    print(list(X.columns))
    
    assert X.shape[0] == original_rows, "Rows were dropped unexpectedly!"
    
    return X


def run_complete_data_pipeline(
    period: str = "2y",
    interval: str = "1d",
    max_tickers: int = None
) -> Dict:
    """
    Run the complete data pipeline.
    
    Args:
        period: Time period for price data
        interval: Data interval
        max_tickers: Maximum number of tickers to use (None for all S&P 500)
        
    Returns:
        Dictionary with all pipeline outputs
    """
    print("\n" + "="*60)
    print("STARTING PORTFOLIO DATA PIPELINE")
    print("="*60)
    
    # Fetch tickers
    tickers_df = fetch_sp500_tickers()
    if tickers_df.empty:
        raise ValueError("Failed to fetch S&P 500 tickers")
    
    tickers_list = tickers_df["Ticker"].tolist()
    
    # Limit tickers if specified
    if max_tickers is not None and max_tickers > 0:
        tickers_list = tickers_list[:max_tickers]
        print(f"Limiting to {max_tickers} tickers for testing")
    
    # Download price data
    close_df = download_price_data(tickers_list, period=period, interval=interval)
    if close_df.empty:
        raise ValueError("Failed to download price data")
    
    # Calculate returns and statistics
    log_ret, mu, Sigma = calculate_returns_and_statistics(close_df)
    
    # Get valid tickers after returns calculation
    valid_tickers = log_ret.columns.tolist()
    
    # Fetch fundamentals
    fund_df = fetch_fundamentals(valid_tickers)
    
    # Clean fundamentals with VIF
    fund_clean = clean_fundamentals_vif(fund_df)
    
    # Ensure alignment
    common_tickers = sorted(set(valid_tickers) & set(fund_clean.index))
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Final tickers: {len(common_tickers)}")
    print(f"Return data: {log_ret[common_tickers].shape}")
    print(f"Fundamentals: {fund_clean.loc[common_tickers].shape}")
    print(f"mu: {len(mu[common_tickers])}")
    print(f"Sigma: {Sigma.loc[common_tickers, common_tickers].shape}")
    
    return {
        'tickers': common_tickers,
        'close_df': close_df[common_tickers],
        'log_returns': log_ret[common_tickers],
        'mu': mu[common_tickers],
        'Sigma': Sigma.loc[common_tickers, common_tickers],
        'fundamentals': fund_clean.loc[common_tickers]
    }


if __name__ == "__main__":
    # Test the pipeline
    data = run_complete_data_pipeline()
    print("\nPipeline test successful!")
    print(f"Tickers: {len(data['tickers'])}")
    print(f"Features: {data['fundamentals'].shape[1]}")
