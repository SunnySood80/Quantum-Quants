"""
QUBO Portfolio Optimizer - Real IBM Backend Implementation
Builds QUBO matrix in latent space and runs QAOA on real hardware.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator as IBMEstimator, Sampler as IBMSampler
from qiskit_algorithms.optimizers import SPSA


def get_ibm_backend(api_key: str, backend_name: str = "ibm_oslo"):
    """
    Authenticate and get IBM Quantum backend.
    
    Args:
        api_key: IBM Quantum API key
        backend_name: Name of real quantum backend (e.g., 'ibm_marrakesh')
        
    Returns:
        IBM Quantum backend object
    """
    service = QiskitRuntimeService(channel="ibm_cloud", token=api_key)
    return service.backend(backend_name)


def build_qubo_matrix(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 20.0,
    target_cardinality: int = 5
) -> pd.DataFrame:
    """
    Build QUBO matrix from portfolio returns and covariance.
    
    Q(x) = -mu^T x + lambda*x^T Sigma x + penalty*(sum(x)-K)^2
    
    Args:
        mu: Expected returns vector
        Sigma: Covariance matrix
        risk_penalty: Weight on risk term
        cardinality_penalty: Weight on cardinality constraint
        target_cardinality: Target portfolio size
        
    Returns:
        QUBO matrix as DataFrame
    """
    n = len(mu)
    Q = np.zeros((n, n))
    
    print(f"Building QUBO matrix: Q(x) = -mu^T x + lambda*x^T Sigma x + penalty*(sum(x)-K)^2")
    print(f"  - Dimensions: {n}x{n}")
    print(f"  - Objective: Maximize returns, minimize risk, enforce cardinality")
    print(f"  - Risk penalty (lambda): {risk_penalty}")
    print(f"  - Cardinality penalty: {cardinality_penalty}")
    print(f"  - Target cardinality (K): {target_cardinality}")
    
    # Diagonal: -mu + lambda*diag(Sigma) + penalty*(2K - 2*sum(...)⟹ +penalty for each x_i due to the square
    for i in range(n):
        Q[i, i] = -mu[i] + risk_penalty * Sigma[i, i] + cardinality_penalty * (1 - 2 * target_cardinality)
    
    # Off-diagonal: lambda*Sigma (accounts for covariance)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2 * risk_penalty * Sigma[i, j]
            Q[j, i] = Q[i, j]
    
    Q_df = pd.DataFrame(Q, index=range(n), columns=range(n))
    print(f"  - QUBO value range: [{Q.min():.4f}, {Q.max():.4f}]")
    
    return Q_df


def normalize_qubo(Q_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize QUBO to [-1, 1] range."""
    Q = Q_df.values
    Q_min, Q_max = Q.min(), Q.max()
    if Q_max == Q_min:
        return Q_df
    Q_normalized = 2 * (Q - Q_min) / (Q_max - Q_min) - 1
    print(f"  - Normalized QUBO range: [{Q_normalized.min():.4f}, {Q_normalized.max():.4f}]")
    return pd.DataFrame(Q_normalized, index=Q_df.index, columns=Q_df.columns)


def qubo_to_ising(Q_df: pd.DataFrame) -> Tuple[SparsePauliOp, float]:
    """
    Convert QUBO to Ising Hamiltonian.
    x_i ∈ {0, 1} → z_i ∈ {-1, +1} via x_i = (1 - z_i) / 2
    """
    Q = Q_df.values
    n = len(Q)
    
    # Compute Ising coefficients
    h_dict = {}  # Linear terms
    j_dict = {}  # Quadratic terms
    
    for i in range(n):
        h_dict[i] = Q[i, i] / 2
        for j in range(i + 1, n):
            j_dict[(i, j)] = Q[i, j] / 2
    
    # Build Pauli string
    pauli_strings = []
    coefficients = []
    
    # Linear terms: h_i * Z_i
    for i, h_i in h_dict.items():
        if abs(h_i) > 1e-10:
            pauli_str = 'I' * i + 'Z' + 'I' * (n - i - 1)
            pauli_strings.append(pauli_str)
            coefficients.append(h_i)
    
    # Quadratic terms: J_ij * Z_i * Z_j
    for (i, j), J_ij in j_dict.items():
        if abs(J_ij) > 1e-10:
            pauli_list = ['I'] * n
            pauli_list[i] = 'Z'
            pauli_list[j] = 'Z'
            pauli_str = ''.join(pauli_list)
            pauli_strings.append(pauli_str)
            coefficients.append(J_ij)
    
    if not pauli_strings:
        pauli_strings = ['I' * n]
        coefficients = [0]
    
    hamiltonian = SparsePauliOp(pauli_strings, coefficients)
    
    # Offset (constant term from the transformation)
    offset = sum(Q[i, i] / 4 for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            offset += Q[i, j] / 4
    
    print(f"Ising Hamiltonian: {len(pauli_strings)} terms")
    print(f"Offset: {offset:.4f}")
    
    return hamiltonian, offset


def run_qaoa_real_backend(
    hamiltonian,
    qaoa_depth: int = 5,
    api_key: str = None,
    backend_name: str = "ibm_marrakesh",
    maxiter: int = 100
) -> Dict:
    """
    Run QAOA on real IBM Quantum backend.
    
    Args:
        hamiltonian: Ising Hamiltonian (SparsePauliOp)
        qaoa_depth: QAOA layers (p)
        api_key: IBM Quantum API key
        backend_name: Backend name (e.g., 'ibm_marrakesh')
        maxiter: Max SPSA iterations
        
    Returns:
        Dict with optimal solution and energy
    """
    print(f"Setting up QAOA circuit for real IBM Quantum backend '{backend_name}'")
    print(f"  - Qubits: {hamiltonian.num_qubits}")
    print(f"  - QAOA layers (p): {qaoa_depth}")
    print(f"  - Max iterations: {maxiter}")
    
    backend = get_ibm_backend(api_key, backend_name)
    qaoa_circuit = QAOAAnsatz(hamiltonian, reps=qaoa_depth)
    print(f"  - Circuit parameters: {qaoa_circuit.num_parameters} (2p = 2x{qaoa_depth})")
    
    # Decompose to basic gates
    qaoa_circuit_decomposed = qaoa_circuit.decompose()
    
    # Instantiate primitives
    estimator = IBMEstimator(mode=backend)
    
    def _extract_energy_from_result(res):
        if hasattr(res, 'values'):
            return float(res.values[0])
        try:
            return float(res[0].data.values[0])
        except:
            raise RuntimeError("Unable to extract energy from Estimator result")
    
    def cost_function(params):
        # Create concrete circuit
        concrete_circuit = qaoa_circuit_decomposed.assign_parameters(params)
        # Transpile with optimization_level 1
        transpiled = transpile(concrete_circuit, backend=backend, optimization_level=1)
        # Skip ISA validation using EstimatorPub with validate=False
        from qiskit.primitives.containers.estimator_pub import EstimatorPub
        pub = EstimatorPub(transpiled, hamiltonian, validate=False)
        job = estimator.run([pub])
        res = job.result()
        energy = _extract_energy_from_result(res)
        return energy
    
    # Optimize
    print(f"\nOptimizing QAOA parameters (SPSA optimizer)...")
    initial_params = np.random.uniform(0, 2*np.pi, qaoa_circuit.num_parameters)
    spsa = SPSA(maxiter=maxiter)
    result = spsa.minimize(fun=cost_function, x0=initial_params)
    optimal_params = result.x
    energy_real = result.fun
    
    print(f"  - Optimal energy: {energy_real:.6f}")
    if hasattr(result, 'nfev'):
        print(f"  - Function evaluations: {result.nfev}")
    
    # Sample final bitstring
    try:
        sampler = IBMSampler(mode=backend)
        concrete_sample_circuit = qaoa_circuit_decomposed.assign_parameters(optimal_params)
        transpiled_sample = transpile(concrete_sample_circuit, backend=backend, optimization_level=1)
        transpiled_sample.measure_all()
        
        from qiskit.primitives.containers.sampler_pub import SamplerPub
        pub = SamplerPub(transpiled_sample, validate=False)
        sample_job = sampler.run([pub])
        sample_res = sample_job.result()
        
        # Extract counts
        meas = sample_res[0].data.meas
        counts = meas.get_counts()
        optimal_bitstring = max(counts, key=counts.get)
        print(f"  - Most frequent bitstring: {optimal_bitstring} (count: {counts[optimal_bitstring]})")
        optimal_solution = np.array([int(b) for b in optimal_bitstring])
    except Exception as e:
        print(f"  - Warning: unable to sample: {e}")
        optimal_bitstring = None
        optimal_solution = np.zeros(hamiltonian.num_qubits, dtype=int)
    
    return {
        'optimal_params': optimal_params,
        'optimal_solution': optimal_solution,
        'optimal_bitstring': optimal_bitstring,
        'energy_real': energy_real,
        'circuit': qaoa_circuit,
        'hamiltonian': hamiltonian
    }


def run_portfolio_qaoa_real(
    mu: np.ndarray,
    Sigma: np.ndarray,
    api_key: str,
    backend_name: str = "ibm_marrakesh",
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 20.0,
    target_cardinality: int = 5,
    qaoa_depth: int = 5,
    maxiter: int = 100
) -> Dict:
    """
    Run full QAOA portfolio optimization on real backend.
    """
    print("\n" + "="*80)
    print("PORTFOLIO QAOA OPTIMIZATION - Real Backend")
    print("="*80)
    
    # Build QUBO
    Q_df = build_qubo_matrix(mu, Sigma, risk_penalty, cardinality_penalty, target_cardinality)
    Q_normalized = normalize_qubo(Q_df)
    
    # Convert to Ising
    hamiltonian, offset = qubo_to_ising(Q_normalized)
    
    # Run QAOA
    result = run_qaoa_real_backend(
        hamiltonian,
        qaoa_depth=qaoa_depth,
        api_key=api_key,
        backend_name=backend_name,
        maxiter=maxiter
    )
    
    return result
