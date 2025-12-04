"""
Run QAOA using Nelder-Mead optimizer (SciPy) for portfolio QUBO.
Saves portfolio and metrics to `results/`.
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

# Config
CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

# QUBO params (Balanced baseline)
risk_penalty = 3.0
cardinality_penalty = 0.2
target_cardinality = 7

# QAOA params
p = 3
max_iter = 200
shots_eval = 1500
shots_final = 10000

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

# QAOA class (Nelder-Mead)
class QAOAPortfolioNM:
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

    def optimize_nm(self, maxiter=200):
        params_init = np.concatenate([np.random.uniform(0, np.pi, self.p), np.random.uniform(0, 2*np.pi, self.p)])
        eval_count = [0]
        def obj(params):
            eval_count[0] += 1
            energy, _, _ = self.evaluate_params(params, shots=self.shots_eval)
            if eval_count[0] % 10 == 0:
                print(f'  Eval {eval_count[0]}: energy={energy:.6f}')
            return energy

        res = minimize(obj, params_init, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol':1e-3, 'fatol':1e-3})
        opt_params = res.x
        final_energy, final_bit, final_counts = self.evaluate_params(opt_params, shots=self.shots_final)
        return opt_params, final_energy, final_bit, final_counts

if __name__ == '__main__':
    print('Running QAOA with Nelder-Mead optimizer (p=%d, maxiter=%d)' % (p, max_iter))
    qaoa = QAOAPortfolioNM(Q, n_qubits=n_latent, p=p, shots_eval=shots_eval, shots_final=shots_final)
    opt_params, opt_energy, opt_bit, opt_counts = qaoa.optimize_nm(maxiter=max_iter)

    print('\nDecoding Nelder-Mead QAOA solution to portfolio...')
    portfolio = decode_portfolio_weights(
        model=None,
        qaoa_solution=opt_bit,
        latent_codes=latent_codes,
        tickers=tickers,
        mu_latent=mu_latent
    )
    metrics = calculate_portfolio_metrics(portfolio['weights'], mu, Sigma)

    # Save results
    portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, 'qaoa_neldermead_portfolio.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, 'qaoa_neldermead_metrics.csv'), index=False)

    print('\nNelder-Mead QAOA Results:')
    print(f'  Sharpe Ratio: {metrics["sharpe_ratio"]:.4f}')
    print(f'  Expected Return: {metrics["expected_return"]*100:.2f}%')
    print(f'  Volatility: {metrics["volatility"]*100:.2f}%')
    print(f'  QUBO Energy: {opt_energy:.6f}')
    print(f'  Saved to {OUTDIR}/qaoa_neldermead_metrics.csv')
