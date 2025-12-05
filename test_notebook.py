#!/usr/bin/env python3
"""Quick test of the notebook by running key cells"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Change to root directory
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path.cwd()))

print("="*70)
print("QUICK NOTEBOOK TEST - Running All Three Models")
print("="*70)

# Import functions
from main_portfolio_optimization import (
    run_autoencoder_compression,
    decode_portfolio_weights,
    calculate_portfolio_metrics,
    _get_build_qubo_matrix,
    run_markowitz_optimization
)

# Configuration
CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
RESULTS_DIR = 'notebook_test_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

RISK_PENALTY = 3.0
CARDINALITY_PENALTY = 0.1
TARGET_CARDINALITY = 7
QAOA_P = 3
QAOA_MAX_ITER = 250
QAOA_SHOTS_EVAL = 3000
QAOA_SHOTS_FINAL = 20000
QAOA_NSTARTS = 5

print("\n[1] Loading data...")
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

data = cache['data']
mu = data['mu']  # Keep as Series
Sigma = data['Sigma']  # Keep as DataFrame
tickers = data['tickers']
print(f"✓ Loaded {len(mu)} stocks")
print(f"  mu type: {type(mu)}, shape: {mu.shape if hasattr(mu, 'shape') else len(mu)}")
print(f"  Sigma type: {type(Sigma)}, shape: {Sigma.shape}")

print("\n[2] Computing PCA compression...")
start = time.time()
compression = run_autoencoder_compression(
    log_returns=data['log_returns'],  # Keep as DataFrame
    fundamentals=data['fundamentals'],  # Keep as DataFrame
    mu=mu,
    Sigma=Sigma,
    latent_dim=16,
    epochs=1,
    device='cpu'
)
mu_latent = np.array(compression['mu_latent'])
Sigma_latent = np.array(compression['Sigma_latent'])
latent_codes = np.array(compression['latent_codes'])
elapsed = time.time() - start
print(f"✓ Compressed to {compression['n_latent']} dimensions in {elapsed:.2f}s")

print("\n[3] Running Markowitz...")
start = time.time()
result = run_markowitz_optimization(mu=mu, Sigma=Sigma, risk_aversion=1.0, allow_short=False)
time_mkt = time.time() - start
metrics_mkt = result['metrics']
print(f"✓ Sharpe: {metrics_mkt['sharpe_ratio']:.4f} | Return: {metrics_mkt['expected_return']:.2%} | Vol: {metrics_mkt['volatility']:.2%} | Time: {time_mkt:.2f}s")

print("\n[4] Running Classical QUBO...")
start = time.time()
build_qubo = _get_build_qubo_matrix()
Q_df = build_qubo(mu_latent, Sigma_latent, risk_penalty=RISK_PENALTY, cardinality_penalty=CARDINALITY_PENALTY, target_cardinality=TARGET_CARDINALITY)
Q = Q_df.values

# Find src directory
src_path = Path.cwd() / 'src' if Path('src').exists() else Path.cwd() / 'portfolio_optimization_package' / 'src'
sys.path.insert(0, str(src_path))
from classical_qubo_solver import simulated_annealing_qubo
x_best = simulated_annealing_qubo(Q, n_iter=1000, T_init=10.0, seed=42)

portfolio = decode_portfolio_weights(model=None, qaoa_solution=x_best, latent_codes=latent_codes, tickers=tickers, mu_latent=mu_latent)
weights_qubo = portfolio['weights']
metrics_qubo = calculate_portfolio_metrics(weights_qubo, mu, Sigma)
time_qubo = time.time() - start
print(f"✓ Sharpe: {metrics_qubo['sharpe_ratio']:.4f} | Return: {metrics_qubo['expected_return']:.2%} | Vol: {metrics_qubo['volatility']:.2%} | Time: {time_qubo:.2f}s")

print("\n[5] Running QAOA...")
start = time.time()
from qaoa_optimizer import QAOAPortfolioOptimizer

best_energy = np.inf
best_solution = None
for start_idx in range(QAOA_NSTARTS):
    print(f"  Start {start_idx + 1}/{QAOA_NSTARTS}...", end='', flush=True)
    qaoa = QAOAPortfolioOptimizer(Q, n_qubits=len(Q), p=QAOA_P, max_iter=QAOA_MAX_ITER, shots_eval=QAOA_SHOTS_EVAL, shots_final=QAOA_SHOTS_FINAL)
    solution, energy, result = qaoa.optimize()
    if energy < best_energy:
        best_energy = energy
        best_solution = solution
    print(f" energy={energy:.4f}")

portfolio = decode_portfolio_weights(model=None, qaoa_solution=best_solution, latent_codes=latent_codes, tickers=tickers, mu_latent=mu_latent)
weights_qaoa = portfolio['weights']
metrics_qaoa = calculate_portfolio_metrics(weights_qaoa, mu, Sigma)
time_qaoa = time.time() - start
print(f"✓ Sharpe: {metrics_qaoa['sharpe_ratio']:.4f} | Return: {metrics_qaoa['expected_return']:.2%} | Vol: {metrics_qaoa['volatility']:.2%} | Time: {time_qaoa:.2f}s")

print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
comparison = pd.DataFrame({
    'Method': ['Markowitz', 'Classical QUBO', 'QAOA'],
    'Sharpe': [metrics_mkt['sharpe_ratio'], metrics_qubo['sharpe_ratio'], metrics_qaoa['sharpe_ratio']],
    'Return': [metrics_mkt['expected_return'], metrics_qubo['expected_return'], metrics_qaoa['expected_return']],
    'Volatility': [metrics_mkt['volatility'], metrics_qubo['volatility'], metrics_qaoa['volatility']],
    'Time (s)': [time_mkt, time_qubo, time_qaoa]
})

print("\n" + comparison.to_string(index=False))
total_time = time_mkt + time_qubo + time_qaoa
print(f"\nTotal runtime: {total_time:.2f}s")
print(f"Best: {comparison.loc[comparison['Sharpe'].idxmax(), 'Method']} (Sharpe: {comparison['Sharpe'].max():.4f})")
print("\n✅ NOTEBOOK TEST SUCCESSFUL!")
