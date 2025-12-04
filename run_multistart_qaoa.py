"""
Multi-start QAOA runner.
Runs several independent QAOA optimizations with different random seeds,
collects metrics and saves best portfolio.

Usage: python run_multistart_qaoa.py --nstarts 3
"""
import os
import pickle
import argparse
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

# Settings
CACHE_PATH = 'data_cache/sp500_data_2y_1d_all.pkl'
OUTDIR = 'results'
os.makedirs(OUTDIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--nstarts', type=int, default=3, help='Number of independent QAOA starts')
parser.add_argument('--p', type=int, default=2, help='QAOA depth p')
parser.add_argument('--max_iter', type=int, default=150, help='Max iterations for classical optimizer')
parser.add_argument('--shots_eval', type=int, default=1000, help='Shots per evaluation')
parser.add_argument('--shots_final', type=int, default=10000, help='Shots for final evaluation')
args = parser.parse_args()

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
# Keep Balanced params (good baseline)
risk_penalty = 1.0
cardinality_penalty = 0.5
target_cardinality = 7

Q_df = build_qubo(mu_latent, Sigma_latent, risk_penalty=risk_penalty, 
                  cardinality_penalty=cardinality_penalty, target_cardinality=target_cardinality)
Q = Q_df.values

# QAOA class (copied/adapted)
class QAOAPortfolioOptimizer:
    def __init__(self, Q, n_qubits, p=2, max_iter=150, simulator=None, shots_eval=1000, shots_final=10000):
        self.Q = Q
        self.n_qubits = n_qubits
        self.p = p
        self.max_iter = max_iter
        self.simulator = simulator or AerSimulator()
        self.shots_eval = shots_eval
        self.shots_final = shots_final
        self.best_energy = float('inf')

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

    def evaluate_circuit(self, gamma, beta, shots=None):
        shots = shots or self.shots_eval
        qc = self.qaoa_circuit(gamma, beta)
        qc.measure_all()
        job = self.simulator.run(qc, shots=shots)
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
        return best_bitstring, best_energy, counts

    def optimize(self):
        gamma_init = np.random.uniform(0, np.pi, self.p)
        beta_init = np.random.uniform(0, 2 * np.pi, self.p)
        params_init = np.concatenate([gamma_init, beta_init])
        eval_count = [0]
        def objective_for_optimizer(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            _, energy, _ = self.evaluate_circuit(gamma, beta, shots=self.shots_eval)
            eval_count[0] += 1
            if eval_count[0] % 10 == 0:
                print(f'  Eval {eval_count[0]}: energy={energy:.6f}')
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
        final_bitstring, final_energy, final_counts = self.evaluate_circuit(opt_gamma, opt_beta, shots=self.shots_final)
        return final_bitstring, final_energy, result, final_counts


def run_multistart(nstarts=3, p=2, max_iter=150):
    summary = []
    best_sharpe = -999
    best_artifact = None
    start_time = time.time()
    for s in range(nstarts):
        print('\n' + '='*60)
        print(f'Start {s+1}/{nstarts} (seed={s})')
        np.random.seed(s)
        qaoa = QAOAPortfolioOptimizer(Q, n_qubits=n_latent, p=p, max_iter=max_iter,
                                      shots_eval=args.shots_eval, shots_final=args.shots_final)
        try:
            bitstring, energy, res, counts = qaoa.optimize()
        except Exception as e:
            print(f'  Start {s} failed: {e}')
            continue

        print('Decoding portfolio...')
        portfolio = decode_portfolio_weights(
            model=None,
            qaoa_solution=bitstring,
            latent_codes=latent_codes,
            tickers=tickers,
            mu_latent=mu_latent
        )
        metrics = calculate_portfolio_metrics(portfolio['weights'], mu, Sigma)

        # Save per-start
        portfolio['portfolio_df'].to_csv(os.path.join(OUTDIR, f'qaoa_multistart_{s+1}_portfolio.csv'), index=False)
        pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR, f'qaoa_multistart_{s+1}_metrics.csv'), index=False)

        summary.append({
            'start': s+1,
            'sharpe': metrics['sharpe_ratio'],
            'expected_return': metrics['expected_return'],
            'volatility': metrics['volatility'],
            'qubo_energy': float(energy)
        })

        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_artifact = {
                'start': s+1,
                'portfolio': portfolio,
                'metrics': metrics,
                'energy': energy
            }

    # Save summary
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(OUTDIR, 'qaoa_multistart_summary.csv'), index=False)

    if best_artifact is not None:
        # Save best
        best_artifact['portfolio']['portfolio_df'].to_csv(os.path.join(OUTDIR, 'qaoa_multistart_best_portfolio.csv'), index=False)
        pd.DataFrame([best_artifact['metrics']]).to_csv(os.path.join(OUTDIR, 'qaoa_multistart_best_metrics.csv'), index=False)
        print('\nBEST START:', best_artifact['start'])
        print('Best Sharpe:', best_artifact['metrics']['sharpe_ratio'])
    else:
        print('No successful starts')

    elapsed = time.time() - start_time
    print(f'Completed {len(summary)} starts in {elapsed:.1f}s')


if __name__ == '__main__':
    run_multistart(nstarts=args.nstarts, p=args.p, max_iter=args.max_iter)
