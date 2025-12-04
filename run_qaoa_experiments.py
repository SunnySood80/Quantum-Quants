"""
Experiment: Run Real QAOA with three different tuning strategies.
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

# Compression
print('Computing compression...')
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

# QAOA Implementation
class QAOAOptimizer:
    def __init__(self, Q, n_qubits, p=2, max_iter=500, shots=1000):
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
    
    def evaluate_circuit(self, gamma, beta):
        qc = self.qaoa_circuit(gamma, beta)
        qc.measure_all()
        
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
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
    
    def optimize(self, name="QAOA"):
        print(f'\nRunning {name} (p={self.p}, max_iter={self.max_iter}, shots={self.shots})')
        
        gamma_init = np.random.uniform(0, np.pi, self.p)
        beta_init = np.random.uniform(0, 2 * np.pi, self.p)
        params_init = np.concatenate([gamma_init, beta_init])
        
        def objective_for_optimizer(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            _, energy = self.evaluate_circuit(gamma, beta)
            
            self.eval_count[0] += 1
            if self.eval_count[0] % 20 == 0:
                print(f'  Iteration {self.eval_count[0]}: energy = {energy:.6f}')
            
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
        
        print(f'Final evaluation with optimized parameters...')
        final_bitstring, final_energy = self.evaluate_circuit(opt_gamma, opt_beta)
        
        print(f'Final energy: {final_energy:.6f}')
        print(f'Solution: {final_bitstring}')
        
        return final_bitstring, final_energy


# Define three tuning strategies
strategies = {
    'Conservative': {
        'risk_penalty': 2.0,
        'cardinality_penalty': 0.25,
        'p': 3,
        'max_iter': 200,
        'shots': 2000
    },
    'Balanced': {
        'risk_penalty': 3.0,
        'cardinality_penalty': 0.2,
        'p': 3,
        'max_iter': 250,
        'shots': 3000
    },
    'Aggressive': {
        'risk_penalty': 4.0,
        'cardinality_penalty': 0.1,
        'p': 3,
        'max_iter': 300,
        'shots': 5000
    }
}

# Run each strategy
results_summary = []

for strategy_name, params in strategies.items():
    print('\n' + '='*70)
    print(f'STRATEGY: {strategy_name}')
    print('='*70)
    print(f'Parameters:')
    print(f'  Risk Penalty: {params["risk_penalty"]}')
    print(f'  Cardinality Penalty: {params["cardinality_penalty"]}')
    print(f'  QAOA Depth (p): {params["p"]}')
    print(f'  Max Iterations: {params["max_iter"]}')
    print(f'  Shots: {params["shots"]}')
    
    # Build QUBO with strategy params
    build_qubo = _get_build_qubo_matrix()
    Q_df = build_qubo(mu_latent, Sigma_latent, 
                      risk_penalty=params['risk_penalty'],
                      cardinality_penalty=params['cardinality_penalty'],
                      target_cardinality=7)
    Q = Q_df.values
    
    # Run QAOA
    qaoa = QAOAOptimizer(Q, n_qubits=n_latent, p=params['p'], 
                        max_iter=params['max_iter'], shots=params['shots'])
    bitstring, energy = qaoa.optimize(name=strategy_name)
    
    # Decode portfolio
    portfolio = decode_portfolio_weights(None, bitstring, latent_codes, tickers, mu_latent=mu_latent)
    metrics = calculate_portfolio_metrics(portfolio['weights'], mu, Sigma)
    
    # Save
    portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, f'qaoa_{strategy_name.lower()}_portfolio.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, f'qaoa_{strategy_name.lower()}_metrics.csv'), index=False)
    
    results_summary.append({
        'Strategy': strategy_name,
        'Sharpe': metrics['sharpe_ratio'],
        'Return': metrics['expected_return'] * 100,
        'Volatility': metrics['volatility'] * 100,
        'Energy': energy,
        'Selections': bitstring.sum()
    })
    
    print(f'\n{strategy_name} Results:')
    print(f'  Sharpe Ratio: {metrics["sharpe_ratio"]:.4f}')
    print(f'  Expected Return: {metrics["expected_return"]*100:.2f}%')
    print(f'  Volatility: {metrics["volatility"]*100:.2f}%')
    print(f'  Factors Selected: {bitstring.sum()}/16')

# Summary comparison
print('\n' + '='*70)
print('STRATEGY COMPARISON')
print('='*70)

comparison_df = pd.DataFrame(results_summary)
print(comparison_df.to_string(index=False))

# Find best
best_idx = comparison_df['Sharpe'].idxmax()
best_strategy = comparison_df.loc[best_idx, 'Strategy']
best_sharpe = comparison_df.loc[best_idx, 'Sharpe']

print(f'\nBEST STRATEGY: {best_strategy}')
print(f'Best Sharpe Ratio: {best_sharpe:.4f}')

# Save summary
comparison_df.to_csv(os.path.join(OUTDIR, 'qaoa_strategies_comparison.csv'), index=False)
print(f'\nSaved to {OUTDIR}/qaoa_strategies_comparison.csv')
print('Done.')
