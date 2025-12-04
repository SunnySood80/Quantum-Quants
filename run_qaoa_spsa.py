"""
Run QAOA using SPSA optimizer (with optional SLSQP refine) for portfolio QUBO.
Saves portfolio and metrics to `results/`.
"""
import os
import pickle
import time
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

# Config
CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

# QUBO params (Balanced baseline)
risk_penalty = 3.0
cardinality_penalty = 0.2
target_cardinality = 7

# QAOA params (can override via CLI)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=3, help='QAOA depth p')
parser.add_argument('--n_iter_spsa', type=int, default=200, help='SPSA iterations')
parser.add_argument('--shots_eval', type=int, default=1500, help='Shots per evaluation')
parser.add_argument('--shots_final', type=int, default=10000, help='Shots for final evaluation')
args = parser.parse_args()

p = args.p
n_iter_spsa = args.n_iter_spsa
shots_eval = args.shots_eval
shots_final = args.shots_final

# SPSA hyperparams (Spall defaults scaled)
a0 = 0.2
c0 = 0.1
A = max(10, n_iter_spsa/10)
alpha = 0.602
gamma = 0.101

print('Loading cached data...')
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)

data = cache['data']
log_returns = data['log_returns']
fundamentals = data['fundamentals']
mu = data['mu']
Sigma = data['Sigma']
tickers = data['tickers']

print('Computing compression (PCA fallback if needed)')
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

# Build QUBO
build_qubo = _get_build_qubo_matrix()
Q_df = build_qubo(mu_latent, Sigma_latent, risk_penalty=risk_penalty,
                  cardinality_penalty=cardinality_penalty, target_cardinality=target_cardinality)
Q = Q_df.values

# QAOA helper
class QAOAPortfolioSPSA:
    def __init__(self, Q, n_qubits, p=3, shots_eval=1500, shots_final=10000):
        self.Q = Q
        self.n_qubits = n_qubits
        self.p = p
        self.shots_eval = shots_eval
        self.shots_final = shots_final
        self.simulator = AerSimulator()

    def objective_function_binary(self, x):
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

    def evaluate_params(self, params, shots=None):
        shots = shots or self.shots_eval
        gamma = params[:self.p]
        beta = params[self.p:]
        qc = self.qaoa_circuit(gamma, beta)
        qc.measure_all()
        job = self.simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        best_energy = float('inf')
        best_bit = None
        for bit, c in counts.items():
            x = np.array([int(b) for b in reversed(bit)], dtype=int)
            e = self.objective_function_binary(x)
            if e < best_energy:
                best_energy = e
                best_bit = x
        return best_energy, best_bit, counts

    def spsa_optimize(self, n_iter=200, a0=0.2, c0=0.1, A=10.0, alpha=0.602, gamma=0.101):
        dim = 2 * self.p
        # initialize params
        params = np.concatenate([np.random.uniform(0, np.pi, self.p), np.random.uniform(0, 2*np.pi, self.p)])
        best_params = params.copy()
        best_energy, _, _ = self.evaluate_params(params)
        print(f'Initial energy: {best_energy:.6f}')

        for k in range(1, n_iter+1):
            ak = a0 / ((k + A) ** alpha)
            ck = c0 / (k ** gamma)
            # delta vector ±1
            delta = 2 * (np.random.rand(dim) > 0.5) - 1
            params_plus = params + ck * delta
            params_minus = params - ck * delta

            f_plus, _, _ = self.evaluate_params(params_plus, shots=self.shots_eval)
            f_minus, _, _ = self.evaluate_params(params_minus, shots=self.shots_eval)

            ghat = (f_plus - f_minus) / (2.0 * ck) * (1.0 / delta)
            params = params - ak * ghat

            if (k % 10) == 0:
                cur_energy, _, _ = self.evaluate_params(params, shots=self.shots_eval)
                print(f'  SPSA iter {k}: energy={cur_energy:.6f}')
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_params = params.copy()

        # Final evaluation
        final_energy, final_bit, final_counts = self.evaluate_params(best_params, shots=self.shots_final)
        return best_params, final_energy, final_bit, final_counts


if __name__ == '__main__':
    print('Running QAOA with SPSA optimizer (p=%d, iters=%d)' % (p, n_iter_spsa))
    qaoa = QAOAPortfolioSPSA(Q, n_qubits=n_latent, p=p, shots_eval=shots_eval, shots_final=shots_final)
    s_params, s_energy, s_bitstring, s_counts = qaoa.spsa_optimize(n_iter=n_iter_spsa, a0=a0, c0=c0, A=A, alpha=alpha, gamma=gamma)

    print('\nDecoding SPSA QAOA solution to portfolio...')
    s_portfolio = decode_portfolio_weights(
        model=None,
        qaoa_solution=s_bitstring,
        latent_codes=latent_codes,
        tickers=tickers,
        mu_latent=mu_latent
    )
    s_metrics = calculate_portfolio_metrics(s_portfolio['weights'], mu, Sigma)

    # Save results
    s_portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, 'qaoa_spsa_portfolio.csv'), index=False)
    pd.DataFrame([s_metrics]).to_csv(os.path.join(OUTDIR, 'qaoa_spsa_metrics.csv'), index=False)

    print('\nSPSA QAOA Results:')
    print(f'  Sharpe Ratio: {s_metrics["sharpe_ratio"]:.4f}')
    print(f'  Expected Return: {s_metrics["expected_return"]*100:.2f}%')
    print(f'  Volatility: {s_metrics["volatility"]*100:.2f}%')
    print(f'  QUBO Energy: {s_energy:.6f}')
    print(f'  Saved to {OUTDIR}/qaoa_spsa_metrics.csv')
