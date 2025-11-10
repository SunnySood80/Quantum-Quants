"""
QUBO Portfolio Optimizer
Builds QUBO matrix in latent space and runs QAOA with Zero-Noise Extrapolation (ZNE).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import SPSA


def build_qubo_matrix(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 0.1,
    target_cardinality: int = 5
) -> pd.DataFrame:
    """
    Build QUBO matrix for portfolio optimization.
    
    Q(x) = -mu^T x + lambda * x^T Sigma x + penalty * (sum(x) - K)^2
    
    Args:
        mu: Expected returns [8]
        Sigma: Covariance matrix [8 × 8]
        risk_penalty: Weight for risk term (lambda)
        cardinality_penalty: Weight for cardinality constraint
        target_cardinality: Target number of assets (K)
        
    Returns:
        QUBO matrix as DataFrame [8 × 8]
    """
    n = len(mu)
    Q = np.zeros((n, n))
    
    print(f"Building QUBO matrix: Q(x) = -mu^T x + lambda*x^T Sigma x + penalty*(sum(x)-K)^2")
    print(f"  - Dimensions: {n}x{n}")
    print(f"  - Objective: Maximize returns, minimize risk, enforce cardinality")
    
    # Linear term: -mu (maximize returns)
    for i in range(n):
        Q[i, i] -= mu[i]
    
    # Quadratic risk term: lambda * Sigma
    Q += risk_penalty * Sigma
    
    # Cardinality constraint: penalty * (sum(x) - K)^2
    # Expands to: penalty * (sum(x_i) - K)^2 = penalty * [sum(x_i^2) + 2*sum(x_i*x_j) - 2K*sum(x_i) + K^2]
    # For binary x: x^2 = x, so diagonal gets: penalty * (1 - 2K)
    # Off-diagonal gets: 2 * penalty
    for i in range(n):
        Q[i, i] += cardinality_penalty * (1 - 2 * target_cardinality)
    
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * cardinality_penalty
    
    # Convert to DataFrame
    Q_df = pd.DataFrame(Q, index=range(n), columns=range(n))
    
    print(f"  - Risk penalty (lambda): {risk_penalty}")
    print(f"  - Cardinality penalty: {cardinality_penalty}")
    print(f"  - Target cardinality (K): {target_cardinality}")
    print(f"  - QUBO value range: [{Q_df.values.min():.4f}, {Q_df.values.max():.4f}]")
    
    return Q_df


def normalize_qubo(Q_df: pd.DataFrame, scale_factor: float = 100.0) -> pd.DataFrame:
    """
    Normalize QUBO matrix to improve QAOA convergence.
    
    Args:
        Q_df: QUBO matrix
        scale_factor: Scaling factor
        
    Returns:
        Normalized QUBO matrix
    """
    Q_norm = Q_df.copy()
    
    # Find max absolute value
    max_val = np.abs(Q_df.values).max()
    
    if max_val > 0:
        Q_norm = Q_df / max_val * scale_factor
    
    print(f"Normalized QUBO range: [{Q_norm.values.min():.4f}, {Q_norm.values.max():.4f}]")
    
    return Q_norm


def qubo_to_ising(Q_df: pd.DataFrame) -> Tuple[SparsePauliOp, float]:
    """
    Convert QUBO to Ising Hamiltonian for QAOA.
    
    Binary to spin: x_i = (1 - z_i) / 2
    
    Args:
        Q_df: QUBO matrix [n × n]
        
    Returns:
        Tuple of (Ising Hamiltonian, offset)
    """
    n = len(Q_df)
    Q = Q_df.values
    
    # Make symmetric
    Q_sym = (Q + Q.T) / 2
    
    # Build Ising Hamiltonian
    pauli_list = []
    coeffs = []
    
    # Diagonal terms (Z_i)
    for i in range(n):
        coeff = -0.5 * Q_sym[i, i]
        if abs(coeff) > 1e-10:
            pauli_str = 'I' * i + 'Z' + 'I' * (n - i - 1)
            pauli_list.append(pauli_str)
            coeffs.append(coeff)
    
    # Off-diagonal terms (Z_i Z_j)
    for i in range(n):
        for j in range(i + 1, n):
            coeff = -0.25 * Q_sym[i, j]
            if abs(coeff) > 1e-10:
                pauli_str = 'I' * n
                pauli_str = pauli_str[:i] + 'Z' + pauli_str[i+1:]
                pauli_str = pauli_str[:j] + 'Z' + pauli_str[j+1:]
                pauli_list.append(pauli_str)
                coeffs.append(coeff)
    
    # Calculate offset
    offset = 0.25 * Q_sym.sum()
    
    # Create SparsePauliOp
    if pauli_list:
        hamiltonian = SparsePauliOp(pauli_list, coeffs)
    else:
        hamiltonian = SparsePauliOp(['I' * n], [0.0])
    
    print(f"\nIsing Hamiltonian: {len(pauli_list)} terms")
    print(f"Offset: {offset:.4f}")
    
    return hamiltonian, offset


def create_noise_model(noise_level: float = 0.01) -> NoiseModel:
    """
    Create a noise model for realistic simulation.
    
    Args:
        noise_level: Depolarizing error probability
        
    Returns:
        NoiseModel
    """
    noise_model = NoiseModel()
    error = depolarizing_error(noise_level, 1)
    error_2q = depolarizing_error(noise_level * 2, 2)
    
    # Add errors to gates
    noise_model.add_all_qubit_quantum_error(error, ['rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    return noise_model


def run_qaoa_with_zne(
    hamiltonian: SparsePauliOp,
    qaoa_depth: int = 3,
    noise_level: float = 0.01,
    use_zne: bool = True,
    noise_scales: list = None,
    maxiter: int = 500
) -> Dict:
    """
    Run QAOA with optional Zero-Noise Extrapolation.
    
    Args:
        hamiltonian: Ising Hamiltonian
        qaoa_depth: QAOA circuit depth (p)
        noise_level: Noise level for simulation
        use_zne: Whether to use ZNE
        noise_scales: Noise scale factors for ZNE
        maxiter: Max optimizer iterations
        
    Returns:
        Dictionary with QAOA results
    """
    print(f"Setting up QAOA circuit with {'ZNE (Zero-Noise Extrapolation)' if use_zne else 'basic noise model'}")
    print(f"  - Qubits: {hamiltonian.num_qubits}")
    print(f"  - QAOA layers (p): {qaoa_depth}")
    print(f"  - Noise level: {noise_level*100:.1f}% depolarizing error")
    print(f"  - Max iterations: {maxiter}")
    
    n_qubits = hamiltonian.num_qubits
    
    # Create QAOA ansatz
    qaoa_circuit = QAOAAnsatz(hamiltonian, reps=qaoa_depth)
    print(f"  - Circuit parameters: {qaoa_circuit.num_parameters} (2p = 2x{qaoa_depth})")
    
    # Setup noisy backend
    noise_model = create_noise_model(noise_level)
    backend = AerSimulator(noise_model=noise_model)
    
    # Transpile
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled_circuit = pm.run(qaoa_circuit)
    measured_circuit = transpiled_circuit.copy()
    measured_circuit.measure_all()
    print(f"  - Circuit compiled (optimization level 3)")
    
    # Estimator
    estimator = AerEstimator(backend_options={"noise_model": noise_model})
    
    # Cost function
    def cost_function(params):
        job = estimator.run(
            circuits=[transpiled_circuit],
            observables=[hamiltonian],
            parameter_values=[params]
        )
        energy = job.result().values[0]
        return energy
    
    # Initial parameters
    initial_params = np.random.uniform(0, 2*np.pi, qaoa_circuit.num_parameters)
    
    print(f"\nOptimizing QAOA parameters (SPSA optimizer)...")
    print(f"  - SPSA is a gradient-free stochastic optimizer")
    print(f"  - Better for noisy/non-smooth objective functions")
    
    # Use Qiskit's SPSA optimizer
    spsa = SPSA(maxiter=maxiter)
    result = spsa.minimize(
        fun=cost_function,
        x0=initial_params
    )
    
    optimal_params = result.x
    energy_noisy = result.fun
    
    iterations = result.nfev if hasattr(result, 'nfev') else None
    
    converged_msg = "[OK] optimization complete"
    print(f"  {converged_msg}")
    print(f"  - Final energy (noisy): {energy_noisy:.6f}")
    if iterations is not None:
        print(f"  - Function evaluations: {iterations}")
    
    # Zero-Noise Extrapolation
    energy_zne = energy_noisy
    energies_by_scale = {}
    
    if use_zne:
        if noise_scales is None:
            noise_scales = [1, 3]
        
        print(f"\nApplying Zero-Noise Extrapolation (ZNE)...")
        print(f"  - Noise scales: {noise_scales}")
        print(f"  - Extrapolating to zero-noise limit")
        
        for scale in noise_scales:
            scaled_noise = noise_level * scale
            scaled_noise_model = create_noise_model(scaled_noise)
            scaled_estimator = AerEstimator(backend_options={"noise_model": scaled_noise_model})
            
            scaled_result = scaled_estimator.run(
                circuits=[transpiled_circuit],
                observables=[hamiltonian],
                parameter_values=[optimal_params]
            ).result()
            
            energy_scaled = scaled_result.values[0]
            energies_by_scale[scale] = energy_scaled
            print(f"    E(noise_scale={scale}): {energy_scaled:.6f}")
        
        # Linear extrapolation to s=0
        scales = np.array(list(energies_by_scale.keys()))
        energies = np.array(list(energies_by_scale.values()))
        
        # Fit line: E(s) = a*s + b, extrapolate to s=0
        coeffs = np.polyfit(scales, energies, 1)
        energy_zne = coeffs[1]  # Intercept = E(0)
        slope = coeffs[0]
        
        print(f"\n  - ZNE extrapolated energy: {energy_zne:.6f}")
        print(f"  - Noise correction: {abs(energy_noisy - energy_zne):.6f} ({abs(energy_noisy - energy_zne)/abs(energy_noisy)*100:.1f}%)")
        print(f"  - Extrapolation slope: {slope:.6f}")
    
    # Get optimal bitstring
    from qiskit_aer.primitives import Sampler as AerSampler
    sampler = AerSampler()
    
    result_samples = sampler.run(
        circuits=[measured_circuit],
        parameter_values=[optimal_params]
    ).result()
    
    quasi_dist = result_samples.quasi_dists[0]
    optimal_bitstring = max(quasi_dist, key=quasi_dist.get)
    if isinstance(optimal_bitstring, str):
        optimal_bitstring_str = optimal_bitstring.zfill(n_qubits)
    else:
        optimal_bitstring_str = format(int(optimal_bitstring), f'0{n_qubits}b')
    optimal_solution = np.array([int(b) for b in optimal_bitstring_str])
    
    print(f"\nOptimal solution: {optimal_bitstring_str}")
    print(f"Selected assets: {optimal_solution.sum()}/{n_qubits}")
    
    return {
        'optimal_params': optimal_params,
        'optimal_solution': optimal_solution,
        'optimal_bitstring': optimal_bitstring_str,
        'energy_noisy': energy_noisy,
        'energy_zne': energy_zne,
        'energies_by_scale': energies_by_scale,
        'circuit': transpiled_circuit,
        'hamiltonian': hamiltonian
    }


def run_portfolio_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 0.1,
    target_cardinality: int = 5,
    qaoa_depth: int = 3,
    noise_level: float = 0.01,
    use_zne: bool = True,
    maxiter: int = 500
) -> Dict:
    """
    Complete QAOA portfolio optimization pipeline.
    
    Args:
        mu: Expected returns (latent space)
        Sigma: Covariance matrix (latent space)
        risk_penalty: Risk penalty weight
        cardinality_penalty: Cardinality constraint weight
        target_cardinality: Target number of assets
        qaoa_depth: QAOA circuit depth
        noise_level: Noise level for simulation
        use_zne: Use Zero-Noise Extrapolation
        maxiter: Max optimizer iterations
        
    Returns:
        Dictionary with QAOA results
    """
    print("\n" + "="*80)
    print("PORTFOLIO QAOA OPTIMIZATION - Complete Pipeline")
    print("="*80)
    
    # Build QUBO
    Q_df = build_qubo_matrix(
        mu, Sigma,
        risk_penalty=risk_penalty,
        cardinality_penalty=cardinality_penalty,
        target_cardinality=target_cardinality
    )
    
    # Normalize QUBO
    Q_norm = normalize_qubo(Q_df)
    
    # Convert to Ising
    hamiltonian, offset = qubo_to_ising(Q_norm)
    
    # Run QAOA
    qaoa_result = run_qaoa_with_zne(
        hamiltonian,
        qaoa_depth=qaoa_depth,
        noise_level=noise_level,
        use_zne=use_zne,
        maxiter=maxiter
    )
    
    return {
        **qaoa_result,
        'Q_matrix': Q_df,
        'Q_normalized': Q_norm,
        'offset': offset
    }


if __name__ == "__main__":
    print("QUBO Portfolio Optimizer loaded.")
    print("Use run_portfolio_qaoa() to optimize your portfolio.")
