"""
QAOA (Quantum Approximate Optimization Algorithm) for Portfolio QUBO
Uses real quantum circuits via Qiskit simulator.
"""
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


class QAOAPortfolioOptimizer:
    """QAOA optimizer for binary portfolio optimization (QUBO)."""
    
    def __init__(self, Q, n_qubits, p=3, max_iter=150, shots_eval=1500, shots_final=10000):
        """
        Args:
            Q: QUBO matrix (n x n)
            n_qubits: Number of qubits (problem dimension)
            p: QAOA depth (number of layers)
            max_iter: Max iterations for classical optimizer
            shots_eval: Shots per evaluation during optimization
            shots_final: Shots for final solution evaluation
        """
        self.Q = Q
        self.n_qubits = n_qubits
        self.p = p
        self.max_iter = max_iter
        self.shots_eval = shots_eval
        self.shots_final = shots_final
        self.simulator = AerSimulator()
        self.best_energy = float('inf')
    
    def objective_function(self, x):
        """Evaluate QUBO for binary vector."""
        return float(x.T @ self.Q @ x)
    
    def qaoa_circuit(self, gamma, beta):
        """Build QAOA circuit with given parameters."""
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial Hadamard layer
        for i in range(self.n_qubits):
            qc.h(i)
        
        # QAOA layers
        for layer in range(self.p):
            # Phase separation: e^{-i gamma H}
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
            
            # Mixing: e^{-i beta B}
            for i in range(self.n_qubits):
                qc.rx(2 * beta[layer], i)
        
        return qc
    
    def evaluate_circuit(self, gamma, beta, shots=None):
        """Evaluate circuit and return best bitstring."""
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
        """Run QAOA optimization with classical optimizer."""
        # Random initialization
        gamma_init = np.random.uniform(0, np.pi, self.p)
        beta_init = np.random.uniform(0, 2 * np.pi, self.p)
        params_init = np.concatenate([gamma_init, beta_init])
        
        eval_count = [0]
        
        def objective_for_optimizer(params):
            gamma = params[:self.p]
            beta = params[self.p:]
            _, energy, _ = self.evaluate_circuit(gamma, beta, shots=self.shots_eval)
            eval_count[0] += 1
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
        
        # Final evaluation
        final_bitstring, final_energy, final_counts = self.evaluate_circuit(opt_gamma, opt_beta, shots=self.shots_final)
        
        return final_bitstring, final_energy, result
