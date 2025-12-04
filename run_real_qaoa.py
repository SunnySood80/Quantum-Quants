"""
Real QAOA using Qiskit simulator on the latent QUBO problem.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from main_portfolio_optimization import (
    run_autoencoder_compression,
    decode_portfolio_weights,
    calculate_portfolio_metrics
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
print('Computing compression (PCA fallback if needed)...')
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

# Build QUBO with adjusted parameters to maximize Sharpe
# Key insight: we want to select high-return factors while minimizing covariance
# - risk_penalty: weight on covariance term (lower = more return-focused, higher = risk-averse)
# - cardinality_penalty: penalty for selecting factors (much lower = prefer non-empty selections)
# - We'll use: higher risk_penalty to properly penalize covariance, lower cardinality_penalty for selection
from main_portfolio_optimization import _get_build_qubo_matrix
build_qubo = _get_build_qubo_matrix()

# Tuned for Sharpe maximization
risk_penalty = 1.0       # Higher weight on covariance (better risk control)
cardinality_penalty = 0.5  # Much lower penalty for selection (encourage non-trivial solutions)
target_cardinality = 7   # Target 7 factors

Q_df = build_qubo(mu_latent, Sigma_latent, risk_penalty=risk_penalty, 
                  cardinality_penalty=cardinality_penalty, target_cardinality=target_cardinality)
Q = Q_df.values

print(f'QUBO shape: {Q.shape}')
print(f'QUBO Parameters:')
print(f'  - Risk Penalty: {risk_penalty}')
print(f'  - Cardinality Penalty: {cardinality_penalty}')
print(f'  - Target Cardinality: {target_cardinality}')

# Real QAOA Implementation
class QAOAPortfolioOptimizer:
    def __init__(self, Q, n_qubits, p=2, max_iter=500):
        """
        QAOA optimizer for QUBO problem.
        
        Args:
            Q: QUBO matrix
            n_qubits: Number of qubits
            p: QAOA circuit depth (number of layers)
            max_iter: Max iterations for classical optimizer
        """
        self.Q = Q
        self.n_qubits = n_qubits
        self.p = p
        self.max_iter = max_iter
        self.simulator = AerSimulator()
        self.best_bitstring = None
        self.best_energy = float('inf')
        
    def objective_function(self, x):
        """Evaluate QUBO for binary vector x."""
        return float(x.T @ self.Q @ x)
    
    def qaoa_circuit(self, gamma, beta):
        """
        Build QAOA circuit for QUBO problem.
        
        Args:
            gamma: QAOA phase separation angles (array of length p)
            beta: QAOA mixing angles (array of length p)
            
        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
        
        # QAOA layers
        for layer in range(self.p):
            # Phase separation: e^{-i gamma H}
            # For QUBO: H = 0.5 * sum_ij Q_ij Z_i Z_j
            for i in range(self.n_qubits):
                for j in range(i, self.n_qubits):
                    if abs(self.Q[i, j]) > 1e-10:
                        coeff = self.Q[i, j] * gamma[layer]
                        if i == j:
                            # Single qubit: Z rotation
                            qc.rz(coeff, i)
                        else:
                            # Two-qubit: ZZ interaction
                            qc.cx(i, j)
                            qc.rz(coeff, j)
                            qc.cx(i, j)
            
            # Mixing: e^{-i beta B} where B = sum X_i
            for i in range(self.n_qubits):
                qc.rx(2 * beta[layer], i)
        
        return qc
    
    def evaluate_circuit(self, gamma, beta, shots=1000):
        """
        Evaluate QAOA circuit and return best bitstring and energy.
        
        Args:
            gamma: Phase separation angles
            beta: Mixing angles
            shots: Number of measurement shots
            
        Returns:
            Tuple of (best_bitstring, best_energy, counts_dict)
        """
        qc = self.qaoa_circuit(gamma, beta)
        
        # Measure all qubits
        qc.measure_all()
        
        # Run on simulator
        job = self.simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Find best bitstring from measurements
        best_bitstring = None
        best_energy = float('inf')
        
        for bitstring, count in counts.items():
            # Convert bitstring (from measurement) to binary array
            x = np.array([int(bit) for bit in reversed(bitstring)], dtype=int)
            energy = self.objective_function(x)
            
            if energy < best_energy:
                best_energy = energy
                best_bitstring = x
        
        return best_bitstring, best_energy, counts
    
    def optimize(self):
        """
        Run QAOA optimization using classical optimizer.
        """
        print(f'\nRunning QAOA (p={self.p}) with Qiskit simulator')
        print(f'  - Qubits: {self.n_qubits}')
        print(f'  - Classical optimizer: scipy.optimize.minimize (COBYLA)')
        print(f'  - Max iterations: {self.max_iter}')
        print(f'  - Shots per evaluation: 1000')
        
        # Initial parameters
        gamma_init = np.random.uniform(0, np.pi, self.p)
        beta_init = np.random.uniform(0, 2 * np.pi, self.p)
        params_init = np.concatenate([gamma_init, beta_init])
        
        # Objective function for classical optimizer
        eval_count = [0]
        
        def objective_for_optimizer(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            
            _, energy, _ = self.evaluate_circuit(gamma, beta, shots=1000)
            
            eval_count[0] += 1
            if eval_count[0] % 10 == 0:
                print(f'  Iteration {eval_count[0]}: best energy = {energy:.6f}')
            
            if energy < self.best_energy:
                self.best_energy = energy
            
            return energy
        
        # Run optimizer
        result = minimize(
            objective_for_optimizer,
            params_init,
            method='COBYLA',
            options={'maxiter': self.max_iter, 'tol': 1e-4, 'rhobeg': 0.5}
        )
        
        opt_params = result.x
        opt_gamma = opt_params[:self.p]
        opt_beta = opt_params[self.p:]
        
        # Final evaluation with optimized parameters
        print('\nFinal evaluation with optimized parameters...')
        final_bitstring, final_energy, final_counts = self.evaluate_circuit(opt_gamma, opt_beta, shots=10000)
        
        print(f'Optimized QAOA energy: {final_energy:.6f}')
        print(f'Optimized solution bitstring: {final_bitstring}')
        
        return final_bitstring, final_energy, result, final_counts

# Run QAOA
print('\n' + '='*70)
print('REAL QAOA WITH QISKIT SIMULATOR')
print('='*70)

qaoa = QAOAPortfolioOptimizer(Q, n_qubits=n_latent, p=2, max_iter=150)
qaoa_bitstring, qaoa_energy, opt_result, counts = qaoa.optimize()

print('\nDecoding QAOA solution to portfolio...')
qaoa_portfolio = decode_portfolio_weights(
    model=None,
    qaoa_solution=qaoa_bitstring,
    latent_codes=latent_codes,
    tickers=tickers,
    mu_latent=mu_latent
)
qaoa_metrics = calculate_portfolio_metrics(qaoa_portfolio['weights'], mu, Sigma)

# Save results
print('\nSaving results...')
qaoa_portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, 'qaoa_real_portfolio.csv'), index=False)
pd.DataFrame([qaoa_metrics]).to_csv(os.path.join(OUTDIR, 'qaoa_real_metrics.csv'), index=False)

print('\nQAOA Real Results:')
print(f'  Sharpe Ratio: {qaoa_metrics["sharpe_ratio"]:.4f}')
print(f'  Expected Return: {qaoa_metrics["expected_return"]*100:.2f}%')
print(f'  Volatility: {qaoa_metrics["volatility"]*100:.2f}%')
print(f'  QUBO Energy: {qaoa_energy:.6f}')
print(f'\nSaved to {OUTDIR}/')
print('Done.')
