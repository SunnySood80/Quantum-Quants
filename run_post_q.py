import pickle
import os
import numpy as np
import pandas as pd

from main_portfolio_optimization import (
    run_autoencoder_compression,
    decode_portfolio_weights,
    run_classical_comparison,
    run_markowitz_optimization,
    calculate_portfolio_metrics
)

CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading cached data from', CACHE_PATH)
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

data = cache['data']

log_returns = data['log_returns']
fundamentals = data['fundamentals']
mu = data['mu']
Sigma = data['Sigma']
tickers = data['tickers']

print('Running PCA-based compression (via run_autoencoder_compression)')
compression = run_autoencoder_compression(
    log_returns=log_returns,
    fundamentals=fundamentals,
    mu=mu,
    Sigma=Sigma,
    latent_dim=16,
    epochs=1,
    device='cpu'
)

mu_latent = compression['mu_latent']
Sigma_latent = compression['Sigma_latent']
latent_codes = compression['latent_codes']

print('\nRunning classical QUBO solver (greedy/exhaustive based on latent size)')
classical = run_classical_comparison(
    mu_latent=mu_latent,
    Sigma_latent=Sigma_latent,
    risk_penalty=0.3,
    cardinality_penalty=12.0,
    target_cardinality=7,
    method='auto'
)

print('\nDecoding classical solution to stock weights')
classical_portfolio = decode_portfolio_weights(
    model=compression.get('model'),
    qaoa_solution=classical['solution'],
    latent_codes=latent_codes,
    tickers=tickers,
    mu_latent=mu_latent,
    device='cpu'
)

classical_metrics = calculate_portfolio_metrics(classical_portfolio['weights'], mu, Sigma)

print('\nRunning Markowitz optimizer on full universe')
markowitz = run_markowitz_optimization(mu, Sigma, risk_aversion=1.0, target_cardinality=None, allow_short=False)

# Save outputs
print('\nSaving results to', OUTPUT_DIR)
classical_portfolio['portfolio_df'].to_csv(os.path.join(OUTPUT_DIR, 'classical_portfolio_from_postq.csv'), index=False)

pd.DataFrame([classical_metrics]).to_csv(os.path.join(OUTPUT_DIR, 'classical_metrics_from_postq.csv'), index=False)

markowitz['portfolio_df'].to_csv(os.path.join(OUTPUT_DIR, 'markowitz_portfolio_from_postq.csv'), index=False)

pd.DataFrame([markowitz['metrics']]).to_csv(os.path.join(OUTPUT_DIR, 'markowitz_metrics_from_postq.csv'), index=False)

print('\nCLASSICAL QUBO SUMMARY:')
print('Method:', classical['method'])
print('Bitstring:', classical['bitstring'])
print('Objective:', classical['objective'])
print('Classical Sharpe:', classical_metrics['sharpe_ratio'])

print('\nMARKOWITZ SUMMARY:')
print('Expected Return:', markowitz['metrics']['expected_return'])
print('Volatility:', markowitz['metrics']['volatility'])
print('Sharpe:', markowitz['metrics']['sharpe_ratio'])

print('\nDone.')
