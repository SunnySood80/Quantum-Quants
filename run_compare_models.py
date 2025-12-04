"""
Run three separate methods for comparison on cached S&P 500 data:
 - Simulated QAOA (classical simulated annealing / local search)
 - Classical QUBO solver (existing `run_classical_comparison`)
 - Markowitz mean-variance optimizer

Saves results into `results/compare_{method}.csv` and prints summary.
"""
import os
import pickle
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
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

print('Loading cached data from', CACHE_PATH)
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

data = cache['data']
log_returns = data['log_returns']
fundamentals = data['fundamentals']
mu = data['mu']
Sigma = data['Sigma']
tickers = data['tickers']

# Compression (PCA fallback will be used if PyTorch missing)
print('\nComputing compression (PCA fallback if needed)')
compression = run_autoencoder_compression(
    log_returns=log_returns,
    fundamentals=fundamentals,
    mu=mu,
    Sigma=Sigma,
    latent_dim=16,
    epochs=1,
    device='cpu'
)

mu_latent = np.array(compression['mu_latent'])
Sigma_latent = np.array(compression['Sigma_latent'])
latent_codes = np.array(compression['latent_codes'])
n_latent = int(compression['n_latent'])
print(f'Latent dims: {n_latent}')

# Build QUBO matrix using internal fallback builder
from main_portfolio_optimization import _get_build_qubo_matrix
build_qubo = _get_build_qubo_matrix()
Q_df = build_qubo(mu_latent, Sigma_latent, risk_penalty=0.3, cardinality_penalty=12.0, target_cardinality=7)
Q = Q_df.values

# ---------- 1) Simulated QAOA (classical approximation) ----------
print('\nRunning simulated QAOA (simulated annealing + local search)')

def simulated_qaoa(Q, target_cardinality, n_restarts=2000, steps_per_restart=200):
    n = Q.shape[0]
    best_x = None
    best_val = float('inf')

    rng = np.random.default_rng(42)

    for r in range(n_restarts):
        # start from random bitstring with target_cardinality ones (if specified)
        if target_cardinality is not None and 0 < target_cardinality < n:
            ones = rng.choice(n, size=target_cardinality, replace=False)
            x = np.zeros(n, dtype=int)
            x[ones] = 1
        else:
            x = rng.integers(0, 2, size=n)

        val = x.T @ Q @ x
        T0 = max(1.0, abs(val))

        for step in range(steps_per_restart):
            # propose flip: if cardinality constrained, swap 1->0 and 0->1
            if target_cardinality is not None and 0 < target_cardinality < n:
                # swap a random 1 with random 0
                idx1 = rng.choice(np.where(x == 1)[0])
                idx0 = rng.choice(np.where(x == 0)[0])
                x_new = x.copy()
                x_new[idx1] = 0
                x_new[idx0] = 1
            else:
                i = rng.integers(0, n)
                x_new = x.copy()
                x_new[i] = 1 - x_new[i]

            val_new = x_new.T @ Q @ x_new
            delta = val_new - val

            # temperature schedule
            T = T0 * (0.995 ** step)
            accept = False
            if delta < 0:
                accept = True
            else:
                if rng.random() < np.exp(-delta / (T + 1e-12)):
                    accept = True

            if accept:
                x = x_new
                val = val_new

        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val

sim_qaoa_solution, sim_qaoa_obj = simulated_qaoa(Q, target_cardinality=7, n_restarts=1000, steps_per_restart=300)
print('Simulated QAOA objective:', sim_qaoa_obj)

sim_qaoa_portfolio = decode_portfolio_weights(None, sim_qaoa_solution, latent_codes, tickers, mu_latent=mu_latent)
sim_qaoa_metrics = calculate_portfolio_metrics(sim_qaoa_portfolio['weights'], mu, Sigma)

# Save
sim_qaoa_portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, 'qaoa_sim_portfolio.csv'), index=False)
pd.DataFrame([sim_qaoa_metrics]).to_csv(os.path.join(OUTDIR, 'qaoa_sim_metrics.csv'), index=False)

# ---------- 2) Classical QUBO solver (existing) ----------
print('\nRunning classical QUBO solver (greedy/exhaustive as appropriate)')
classical = run_classical_comparison(mu_latent, Sigma_latent, risk_penalty=0.3, cardinality_penalty=12.0, target_cardinality=7, method='auto')
print('Classical method used:', classical['method'])

classical_portfolio = decode_portfolio_weights(None, classical['solution'], latent_codes, tickers, mu_latent=mu_latent)
classical_metrics = calculate_portfolio_metrics(classical_portfolio['weights'], mu, Sigma)

classical_portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, 'classical_qubo_portfolio.csv'), index=False)
pd.DataFrame([classical_metrics]).to_csv(os.path.join(OUTDIR, 'classical_qubo_metrics.csv'), index=False)

# ---------- 3) Markowitz optimization ----------
print('\nRunning Markowitz optimizer (full universe)')
markowitz = run_markowitz_optimization(mu, Sigma, risk_aversion=1.0, target_cardinality=7, allow_short=False)
markowitz_metrics = markowitz['metrics']
markowitz['portfolio_df'].to_csv(os.path.join(OUTDIR, 'markowitz_portfolio.csv'), index=False)
pd.DataFrame([markowitz_metrics]).to_csv(os.path.join(OUTDIR, 'markowitz_metrics.csv'), index=False)

# Summary print
print('\nSummary:')
print('Simulated QAOA Sharpe:', sim_qaoa_metrics['sharpe_ratio'])
print('Classical QUBO Sharpe:', classical_metrics['sharpe_ratio'])
print('Markowitz Sharpe:', markowitz_metrics['sharpe_ratio'])

print('\nSaved CSVs to', OUTDIR)
print('Done.')
