import numpy as np
import pandas as pd
from portfolio_data_pipeline import fetch_sp500_tickers, download_price_data, calculate_returns_and_statistics
from main_portfolio_optimization import run_autoencoder_compression, decode_portfolio_weights, calculate_portfolio_metrics

print('Starting quick test: 10 tickers, PCA fallback, no fundamentals')

# Get tickers
tickers_df = fetch_sp500_tickers()
tickers = tickers_df['Ticker'].tolist()[:10]
print('Tickers:', tickers)

# Download price data
close_df = download_price_data(tickers, period='2y', interval='1d')

# Calculate returns and stats
log_ret, mu, Sigma = calculate_returns_and_statistics(close_df)
print('Computed returns for', len(log_ret.columns), 'tickers')

# Empty fundamentals (PCA fallback will handle)
fundamentals = pd.DataFrame(index=log_ret.columns)

# Run compression (PCA fallback used if PyTorch unavailable)
compression = run_autoencoder_compression(log_ret, fundamentals, mu, Sigma, latent_dim=4)
print('Latent dim:', compression['n_latent'])

# Create a dummy selection: pick first 2 latent dims
n_latent = compression['n_latent']
qaoa_solution = np.zeros(n_latent, dtype=int)
qaoa_solution[:2] = 1

portfolio = decode_portfolio_weights(
    model=compression.get('model'),
    qaoa_solution=qaoa_solution,
    latent_codes=compression['latent_codes'],
    tickers=log_ret.columns.tolist(),
    mu_latent=compression['mu_latent']
)

metrics = calculate_portfolio_metrics(portfolio['weights'], mu, Sigma)
print('\nQuick test results:')
print('Top holdings:')
print(portfolio['portfolio_df'].head(10).to_string(index=False))
print('\nMetrics:', metrics)

# Save quick results
portfolio['portfolio_df'].to_csv('./results_quick_test/quick_portfolio.csv', index=False)
print('\nSaved quick portfolio to ./results_quick_test/quick_portfolio.csv')
