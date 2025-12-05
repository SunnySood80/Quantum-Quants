#!/usr/bin/env python3
"""
Run All Three Best Portfolio Optimization Models

This script runs the best versions of:
1. Markowitz (classical analytical)
2. Classical QUBO (simulated annealing heuristic)
3. QAOA (quantum approximate optimization with real Qiskit simulator)

Uses pre-cached S&P 500 data and reproduces exact same results as before.

Usage:
    python run_all_models.py [--verbose]

Example:
    python run_all_models.py --verbose
"""

import os
import sys
import pickle
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure we can import from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_portfolio_optimization import (
    run_autoencoder_compression,
    decode_portfolio_weights,
    calculate_portfolio_metrics,
    _get_build_qubo_matrix
)

# Configuration
CACHE_PATH = 'data/sp500_data_2y_1d_all.pkl'
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

# Best parameters found from optimization campaign
RISK_PENALTY = 3.0
CARDINALITY_PENALTY = 0.1
TARGET_CARDINALITY = 7

# QAOA settings (final best)
QAOA_P = 3
QAOA_MAX_ITER = 250
QAOA_SHOTS_EVAL = 3000
QAOA_SHOTS_FINAL = 20000
QAOA_NSTARTS = 5


def load_data(verbose=False):
    """Load and prepare cached S&P 500 data."""
    if verbose:
        print("[*] Loading cached S&P 500 data...")
    
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    
    data = cache['data']
    
    if verbose:
        print(f"    [OK] Loaded {len(data['tickers'])} stocks")
    
    return data


def prepare_latent_space(data, verbose=False):
    """Compute PCA compression to latent space."""
    if verbose:
        print("[*] Computing PCA compression (latent space)...")
    
    log_returns = data['log_returns']
    fundamentals = data['fundamentals']
    mu = data['mu']
    Sigma = data['Sigma']
    
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
    
    if verbose:
        print(f"    [OK] Latent dimensions: {n_latent}")
    
    return mu_latent, Sigma_latent, latent_codes, n_latent


def run_markowitz(data, verbose=False):
    """Run Markowitz optimizer on original (non-latent) space."""
    if verbose:
        print("\n" + "="*70)
        print("METHOD 1: MARKOWITZ (Classical Analytical)")
        print("="*70)
    
    start = time.time()
    
    mu = data['mu']
    Sigma = data['Sigma']
    tickers = data['tickers']
    
    # Use the optimized Markowitz function
    from main_portfolio_optimization import run_markowitz_optimization
    
    result = run_markowitz_optimization(
        mu=mu,
        Sigma=Sigma,
        risk_aversion=1.0,
        allow_short=False
    )
    
    elapsed = time.time() - start
    
    weights = result['weights']
    metrics = result['metrics']
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed:.2f}s")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        print(f"  Volatility:      {metrics['volatility']:.2%}")
        print(f"  Holdings:        {np.sum(weights > 0.001)}")
    
    # Save results
    portfolio_df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights
    })
    portfolio_df.to_csv(f'{OUTDIR}/1_markowitz_portfolio.csv', index=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTDIR}/1_markowitz_metrics.csv', index=False)
    
    return {'weights': weights, 'metrics': metrics, 'elapsed': elapsed}


def run_classical_qubo(mu_latent, Sigma_latent, latent_codes, data, verbose=False):
    """Run classical QUBO solver using simulated annealing."""
    if verbose:
        print("\n" + "="*70)
        print("METHOD 2: CLASSICAL QUBO (Simulated Annealing)")
        print("="*70)
    
    start = time.time()
    
    # Build QUBO matrix
    build_qubo = _get_build_qubo_matrix()
    Q_df = build_qubo(
        mu_latent, Sigma_latent,
        risk_penalty=RISK_PENALTY,
        cardinality_penalty=CARDINALITY_PENALTY,
        target_cardinality=TARGET_CARDINALITY
    )
    Q = Q_df.values
    n_latent = len(Q)
    
    # Use the optimized QUBO solver from src
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from classical_qubo_solver import simulated_annealing_qubo
    
    x_best = simulated_annealing_qubo(Q, n_iter=1000, T_init=10.0, seed=42)
    
    # Decode back to original space
    selected = np.where(x_best)[0]
    if len(selected) == 0:
        selected = np.argsort(-mu_latent)[:TARGET_CARDINALITY]
    
    portfolio = decode_portfolio_weights(
        model=None,
        qaoa_solution=x_best,
        latent_codes=latent_codes,
        tickers=data['tickers'],
        mu_latent=mu_latent
    )
    
    weights_decoded = portfolio['weights']
    
    # Compute metrics using ORIGINAL mu/Sigma (not latent)
    metrics = calculate_portfolio_metrics(
        weights_decoded, data['mu'], data['Sigma']
    )
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed:.2f}s")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        print(f"  Volatility:      {metrics['volatility']:.2%}")
        print(f"  Holdings:        {len(selected)}")
    
    # Save results
    portfolio_df = pd.DataFrame({
        'ticker': data['tickers'],
        'weight': weights_decoded
    })
    portfolio_df.to_csv(f'{OUTDIR}/2_classical_qubo_portfolio.csv', index=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTDIR}/2_classical_qubo_metrics.csv', index=False)
    
    return {'weights': weights_decoded, 'metrics': metrics, 'elapsed': elapsed}


def run_qaoa(mu_latent, Sigma_latent, latent_codes, data, verbose=False):
    """Run QAOA optimizer (quantum circuits + COBYLA)."""
    if verbose:
        print("\n" + "="*70)
        print("METHOD 3: QAOA (Quantum Approximate Optimization)")
        print("="*70)
        print(f"  Circuit depth (p):    {QAOA_P}")
        print(f"  Max iterations:       {QAOA_MAX_ITER}")
        print(f"  Shots (evaluation):   {QAOA_SHOTS_EVAL}")
        print(f"  Shots (final):        {QAOA_SHOTS_FINAL}")
        print(f"  Multistart runs:      {QAOA_NSTARTS}")
    
    start = time.time()
    
    # Build QUBO matrix
    build_qubo = _get_build_qubo_matrix()
    Q_df = build_qubo(
        mu_latent, Sigma_latent,
        risk_penalty=RISK_PENALTY,
        cardinality_penalty=CARDINALITY_PENALTY,
        target_cardinality=TARGET_CARDINALITY
    )
    Q = Q_df.values
    n_latent = len(Q)
    
    # Import QAOA optimizer from src
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from qaoa_optimizer import QAOAPortfolioOptimizer
    
    best_energy = np.inf
    best_solution = None
    
    # Run multistart
    for start_idx in range(QAOA_NSTARTS):
        if verbose:
            print(f"  Start {start_idx + 1}/{QAOA_NSTARTS}...", end='', flush=True)
        
        qaoa = QAOAPortfolioOptimizer(
            Q,
            n_qubits=n_latent,
            p=QAOA_P,
            max_iter=QAOA_MAX_ITER,
            shots_eval=QAOA_SHOTS_EVAL,
            shots_final=QAOA_SHOTS_FINAL
        )
        
        solution, energy, result = qaoa.optimize()
        
        if energy < best_energy:
            best_energy = energy
            best_solution = solution
        
        if verbose:
            print(f" energy={energy:.4f}")
    
    # Decode solution
    portfolio = decode_portfolio_weights(
        model=None,
        qaoa_solution=best_solution,
        latent_codes=latent_codes,
        tickers=data['tickers'],
        mu_latent=mu_latent
    )
    
    weights_decoded = portfolio['weights']
    
    # Compute metrics using ORIGINAL mu/Sigma (not latent)
    metrics = calculate_portfolio_metrics(
        weights_decoded, data['mu'], data['Sigma']
    )
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed:.2f}s")
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"  Expected Return: {metrics['expected_return']:.2%}")
        print(f"  Volatility:      {metrics['volatility']:.2%}")
        selected_count = np.sum(best_solution > 0.5)
        print(f"  Holdings:        {selected_count}")
    
    # Save results
    portfolio_df = pd.DataFrame({
        'ticker': data['tickers'],
        'weight': weights_decoded
    })
    portfolio_df.to_csv(f'{OUTDIR}/3_qaoa_final_portfolio.csv', index=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{OUTDIR}/3_qaoa_final_metrics.csv', index=False)
    
    return {'weights': weights_decoded, 'metrics': metrics, 'elapsed': elapsed}


def print_summary(results):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Sharpe':<12} {'Return':<12} {'Vol':<12} {'Time (s)':<10}")
    print("-" * 70)
    
    best_sharpe = -np.inf
    best_method = None
    
    for method, data in results.items():
        metrics = data['metrics']
        sharpe = metrics['sharpe_ratio']
        ret = metrics['expected_return']
        vol = metrics['volatility']
        elapsed = data['elapsed']
        
        print(f"{method:<20} {sharpe:<12.4f} {ret:<12.2%} {vol:<12.2%} {elapsed:<10.2f}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_method = method
    
    print("-" * 70)
    print(f"[OK] Best: {best_method} (Sharpe: {best_sharpe:.4f})")
    
    total_time = sum(d['elapsed'] for d in results.values())
    print(f"[OK] Total runtime: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Run all three best portfolio optimization models',
        epilog=__doc__
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', '-o', default='results', help='Output directory for results (default: results/)')
    
    args = parser.parse_args()
    verbose = args.verbose
    global OUTDIR
    OUTDIR = args.output_dir
    os.makedirs(OUTDIR, exist_ok=True)
    
    if verbose:
        print("\n" + "="*70)
        print("PORTFOLIO OPTIMIZATION: Running All Three Models")
        print("="*70)
    
    try:
        # Load data
        data = load_data(verbose=verbose)
        mu_latent, Sigma_latent, latent_codes, n_latent = prepare_latent_space(data, verbose=verbose)
        
        # Run all three
        results = {}
        results['Markowitz'] = run_markowitz(data, verbose=verbose)
        results['Classical QUBO'] = run_classical_qubo(mu_latent, Sigma_latent, latent_codes, data, verbose=verbose)
        results['QAOA'] = run_qaoa(mu_latent, Sigma_latent, latent_codes, data, verbose=verbose)
        
        # Print summary
        print_summary(results)
        
        print("\n[SUCCESS] All models completed successfully!")
        print(f"[OK] Results saved to: {OUTDIR}/\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
