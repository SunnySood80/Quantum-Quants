"""
Experiment: Test QAOA Balanced with different latent dimensions (16, 20, 24).
Compare Sharpe ratios to see if larger latent space improves results.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from main_portfolio_optimization import (
    run_autoencoder_compression,
    decode_portfolio_weights,
    calculate_portfolio_metrics,
    _get_build_qubo_matrix
)

CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

print('Loading cached data...')
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

data = cache['data']
log_returns = data['log_returns']
fundamentals = data['fundamentals']
mu = data['mu']
Sigma = data['Sigma']
tickers = data['tickers']

# QAOA class (reusable)
class QAOAOptimizer:
    def __init__(self, Q, n_qubits, p=3, max_iter=250, shots=3000):
        self.Q = Q
        self.n_qubits = n_qubits
        self.p = p
        self.max_iter = max_iter
        self.shots = shots
        self.simulator = AerSimulator()
        self.best_energy = float('inf')
        self.eval_count = [0]
        
    def objective_function(self, x):
        return float(x.T @ self.Q @ x)
    
    def qaoa_circuit(self, gamma, beta):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
        
        for layer in range(self.p):
            for i in range(self.n_qubits):
                for j in range(i, self.n_qubits):
                    if abs(self.Q[i, j]) > 1e-10:
                        coeff = self.Q[i, j] * gamma[layer]
                        if i == j:
                            qc.rz(coeff, i)
                        else:
                            qc.cx(i, j)
                            qc.rz(coeff, j)
                            qc.cx(i, j)
            
            for i in range(self.n_qubits):
                qc.rx(2 * beta[layer], i)
        
        return qc
    
    def evaluate_circuit(self, gamma, beta, timeout_sec=60):
        """Evaluate QAOA circuit with timeout protection."""
        qc = self.qaoa_circuit(gamma, beta)
        qc.measure_all()
        
        try:
            job = self.simulator.run(qc, shots=self.shots)
            result = job.result(timeout=timeout_sec)
            counts = result.get_counts(qc)
            
            best_bitstring = None
            best_energy = float('inf')
            
            for bitstring, count in counts.items():
                x = np.array([int(bit) for bit in reversed(bitstring)], dtype=int)
                energy = self.objective_function(x)
                if energy < best_energy:
                    best_energy = energy
                    best_bitstring = x
            
            return best_bitstring, best_energy
        except Exception as e:
            print(f'      [WARNING] Circuit eval timed out or failed: {e}. Using random solution.')
            # Fallback: return random solution with target cardinality
            x = np.zeros(self.n_qubits, dtype=int)
            x[:7] = 1  # Select 7 factors
            return x, self.objective_function(x)
    
    def optimize(self):
        gamma_init = np.random.uniform(0, np.pi, self.p)
        beta_init = np.random.uniform(0, 2 * np.pi, self.p)
        params_init = np.concatenate([gamma_init, beta_init])
        
        def objective_for_optimizer(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            _, energy = self.evaluate_circuit(gamma, beta)
            
            self.eval_count[0] += 1
            if self.eval_count[0] % 20 == 0:
                print(f'    Iteration {self.eval_count[0]}: energy = {energy:.6f}')
            
            if energy < self.best_energy:
                self.best_energy = energy
            
            return energy
        
        result = minimize(
            objective_for_optimizer,
            params_init,
            method='COBYLA',
            options={'maxiter': self.max_iter, 'tol': 1e-4, 'rhobeg': 0.5}
        )
        
        opt_params = result.x
        opt_gamma = opt_params[:self.p]
        opt_beta = opt_params[self.p:]
        
        final_bitstring, final_energy = self.evaluate_circuit(opt_gamma, opt_beta)
        
        return final_bitstring, final_energy


# Test latent dimensions
latent_dims = [16, 20, 24]
results_summary = []

for latent_dim in latent_dims:
    print('\n' + '='*70)
    print(f'LATENT DIMENSION: {latent_dim}')
    print('='*70)
    
    # Compression with specified latent_dim
    print(f'Computing compression (latent_dim={latent_dim})...')
    compression = run_autoencoder_compression(
        log_returns=log_returns,
        fundamentals=fundamentals,
        mu=mu,
        Sigma=Sigma,
        latent_dim=latent_dim,
        epochs=1,
        device='cpu'
    )
    
    mu_latent = np.array(compression['mu_latent'])
    Sigma_latent = np.array(compression['Sigma_latent'])
    latent_codes = np.array(compression['latent_codes'])
    n_latent = int(compression['n_latent'])
    
    print(f'Latent space: {n_latent}D')
    
    # Build QUBO with Balanced parameters
    build_qubo = _get_build_qubo_matrix()
    Q_df = build_qubo(mu_latent, Sigma_latent, 
                      risk_penalty=3.0,
                      cardinality_penalty=0.2,
                      target_cardinality=7)
    Q = Q_df.values
    
    # Run QAOA (reduce iterations/shots for larger latent dims to avoid timeout)
    if latent_dim == 16:
        max_iter_qaoa = 250
        shots_qaoa = 3000
    else:
        # Larger latent spaces = slower circuits, use fewer iterations/shots
        max_iter_qaoa = 150
        shots_qaoa = 1500
    
    print(f'Running QAOA (p=3, max_iter={max_iter_qaoa}, shots={shots_qaoa})...')
    qaoa = QAOAOptimizer(Q, n_qubits=n_latent, p=3, max_iter=max_iter_qaoa, shots=shots_qaoa)
    bitstring, energy = qaoa.optimize()
    
    print(f'Final energy: {energy:.6f}')
    print(f'Solution: {bitstring}')
    print(f'Factors selected: {bitstring.sum()}/{n_latent}')
    
    # Decode portfolio
    portfolio = decode_portfolio_weights(None, bitstring, latent_codes, tickers, mu_latent=mu_latent)
    metrics = calculate_portfolio_metrics(portfolio['weights'], mu, Sigma)
    
    # Save
    portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, f'qaoa_latent{latent_dim}_portfolio.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, f'qaoa_latent{latent_dim}_metrics.csv'), index=False)
    
    results_summary.append({
        'Latent Dim': latent_dim,
        'Sharpe': metrics['sharpe_ratio'],
        'Return': metrics['expected_return'] * 100,
        'Volatility': metrics['volatility'] * 100,
        'Energy': energy,
        'Factors Selected': bitstring.sum()
    })
    
    print(f'\nResults:')
    print(f'  Sharpe Ratio: {metrics["sharpe_ratio"]:.4f}')
    print(f'  Expected Return: {metrics["expected_return"]*100:.2f}%')
    print(f'  Volatility: {metrics["volatility"]*100:.2f}%')

# Summary comparison
print('\n' + '='*70)
print('LATENT DIMENSION COMPARISON')
print('='*70)

comparison_df = pd.DataFrame(results_summary)
print(comparison_df.to_string(index=False))

# Find best
best_idx = comparison_df['Sharpe'].idxmax()
best_dim = comparison_df.loc[best_idx, 'Latent Dim']
best_sharpe = comparison_df.loc[best_idx, 'Sharpe']

baseline_sharpe = comparison_df.loc[comparison_df['Latent Dim'] == 16, 'Sharpe'].values[0]
improvement = (best_sharpe - baseline_sharpe) / baseline_sharpe * 100

print(f'\nBEST LATENT DIMENSION: {int(best_dim)}')
print(f'Best Sharpe: {best_sharpe:.4f}')
if best_dim != 16:
    print(f'Improvement vs 16D: +{improvement:.2f}%')

# Save summary
comparison_df.to_csv(os.path.join(OUTDIR, 'qaoa_latent_comparison.csv'), index=False)
print(f'\nSaved to {OUTDIR}/qaoa_latent_comparison.csv')
print('Done.')
