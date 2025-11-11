"""
Quick runner for QAOA on real backend using cached data.
Skips data pipeline and autoencoder retraining.
"""

import numpy as np
import pandas as pd
import os
import pickle
from dotenv import load_dotenv

# Load from cache
from portfolio_data_pipeline import run_complete_data_pipeline
from autoencoder_compression import run_autoencoder_compression
from qubo_portfolio_optimizer import run_portfolio_qaoa_real, build_qubo_matrix

# Load env
load_dotenv()
api_key = os.getenv("IBM_QUANTUM_TOKEN")
backend_name = os.getenv("QUANTUM_BACKEND", "ibm_marrakesh")

# Parameters
latent_dim = 12
risk_penalty = 0.5
cardinality_penalty = 20.0
target_cardinality = 5
qaoa_depth = 5
maxiter = 1000

print("="*80)
print(" " * 20 + "QAOA ON REAL HARDWARE (Fast Mode)")
print("="*80)

# Step 1: Load cached data (fast)
print("\n[STEP 1] Loading cached market data...")
data = run_complete_data_pipeline(period="2y", interval="1d")
print(f"✓ Loaded {len(data['tickers'])} tickers from cache")

# Step 2: Run autoencoder compression (cache if needed)
print("\n[STEP 2] Running autoencoder compression...")
ae_cache = f"cache/ae_result_{latent_dim}d.pkl"
if os.path.exists(ae_cache):
    print(f"Loading cached autoencoder result from {ae_cache}")
    import pickle
    with open(ae_cache, 'rb') as f:
        compression_result = pickle.load(f)
    print(f"✓ Loaded cached compression (skipped training)")
else:
    print(f"Training autoencoder (will cache result to {ae_cache})...")
    compression_result = run_autoencoder_compression(
        log_returns=data['log_returns'],
        fundamentals=data['fundamentals'],
        mu=data['mu'],
        Sigma=data['Sigma'],
        latent_dim=latent_dim,
        epochs=200,
        device='cpu'
    )
    # Save for next time
    os.makedirs("cache", exist_ok=True)
    with open(ae_cache, 'wb') as f:
        pickle.dump(compression_result, f)
    print(f"✓ Cached compression result to {ae_cache}")

# Step 3: QAOA on real backend
print("\n[STEP 3] Running QAOA on real IBM Quantum hardware...")
print(f"Backend: {backend_name}")
print(f"QAOA depth: {qaoa_depth}")
print(f"Max iterations: {maxiter}")

qaoa_result = run_portfolio_qaoa_real(
    mu=compression_result['mu_latent'],
    Sigma=compression_result['Sigma_latent'],
    api_key=api_key,
    backend_name=backend_name,
    risk_penalty=risk_penalty,
    cardinality_penalty=cardinality_penalty,
    target_cardinality=target_cardinality,
    qaoa_depth=qaoa_depth,
    maxiter=maxiter
)

print("\n" + "="*80)
print(" " * 25 + "QAOA COMPLETE")
print("="*80)
print(f"Solution bitstring: {qaoa_result['optimal_bitstring']}")
print(f"Selected latent factors: {qaoa_result['optimal_solution'].sum() if qaoa_result['optimal_solution'] is not None else 'N/A'}/{latent_dim}")
print(f"Final energy: {qaoa_result['energy_real']:.6f}")
print("="*80)
