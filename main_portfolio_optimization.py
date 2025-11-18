"""
Main Portfolio Optimization Pipeline
Orchestrates the complete quantum portfolio optimization workflow:
1. Data Pipeline (fetch tickers, prices, returns, fundamentals)
2. Autoencoder Compression (compress to 8D latent space)
3. QAOA Optimization (8-qubit quantum optimization)
4. Portfolio Decoding (map back to stock weights)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

# Import our modules
from portfolio_data_pipeline import run_complete_data_pipeline
from autoencoder_compression import run_autoencoder_compression, decode_portfolio_weights
from qubo_portfolio_optimizer import run_portfolio_qaoa, build_qubo_matrix


def calculate_portfolio_metrics(
    weights: np.ndarray,
    mu: pd.Series,
    Sigma: pd.DataFrame
) -> Dict:
    """
    Calculate portfolio performance metrics.
    
    Args:
        weights: Portfolio weights
        mu: Expected returns
        Sigma: Covariance matrix
        
    Returns:
        Dictionary with portfolio metrics
    """
    portfolio_return = np.dot(weights, mu.values)
    portfolio_variance = np.dot(weights, np.dot(Sigma.values, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
    
    return {
        'expected_return': portfolio_return,
        'volatility': portfolio_std,
        'sharpe_ratio': sharpe_ratio,
        'variance': portfolio_variance
    }


def evaluate_qubo_solution(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Evaluate QUBO objective for a given solution.
    
    Args:
        Q: QUBO matrix [n × n]
        x: Binary solution [n]
        
    Returns:
        Objective value
    """
    return x.T @ Q @ x


def solve_qubo_exhaustive(
    Q: np.ndarray,
    target_cardinality: int = None
) -> Tuple[np.ndarray, float]:
    """
    Solve QUBO exhaustively by trying all feasible solutions.
    Only practical for small problems (n ≤ 10).
    
    Args:
        Q: QUBO matrix [n × n]
        target_cardinality: If specified, only consider solutions with this many 1s
        
    Returns:
        Tuple of (best_solution, best_value)
    """
    n = len(Q)
    best_solution = None
    best_value = float('inf')
    
    if target_cardinality is not None:
        # Only try solutions with exactly target_cardinality ones
        for indices in combinations(range(n), target_cardinality):
            x = np.zeros(n, dtype=int)
            x[list(indices)] = 1
            value = evaluate_qubo_solution(Q, x)
            if value < best_value:
                best_value = value
                best_solution = x
    else:
        # Try all 2^n combinations
        for i in range(2**n):
            x = np.array([int(b) for b in format(i, f'0{n}b')], dtype=int)
            value = evaluate_qubo_solution(Q, x)
            if value < best_value:
                best_value = value
                best_solution = x
    
    return best_solution, best_value


def solve_qubo_greedy(
    Q: np.ndarray,
    target_cardinality: int
) -> Tuple[np.ndarray, float]:
    """
    Solve QUBO using a greedy heuristic.
    Iteratively selects the dimension that minimally increases the objective.
    
    Args:
        Q: QUBO matrix [n × n]
        target_cardinality: Number of dimensions to select
        
    Returns:
        Tuple of (solution, objective_value)
    """
    n = len(Q)
    x = np.zeros(n, dtype=int)
    
    for _ in range(target_cardinality):
        best_idx = None
        best_delta = float('inf')
        
        for i in range(n):
            if x[i] == 0:  # Not yet selected
                x_test = x.copy()
                x_test[i] = 1
                value = evaluate_qubo_solution(Q, x_test)
                
                if value < best_delta:
                    best_delta = value
                    best_idx = i
        
        if best_idx is not None:
            x[best_idx] = 1
    
    final_value = evaluate_qubo_solution(Q, x)
    return x, final_value


def run_classical_comparison(
    mu_latent: np.ndarray,
    Sigma_latent: np.ndarray,
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 0.1,
    target_cardinality: int = 5,
    method: str = 'auto'
) -> Dict:
    """
    Run classical optimization for comparison with QAOA.
    
    Args:
        mu_latent: Expected returns in latent space
        Sigma_latent: Covariance matrix in latent space
        risk_penalty: Risk penalty weight
        cardinality_penalty: Cardinality constraint weight
        target_cardinality: Target number of assets
        method: 'exhaustive', 'greedy', or 'auto' (chooses based on problem size)
        
    Returns:
        Dictionary with classical optimization results
    """
    print("Solving QUBO with classical methods for baseline comparison...")
    print(f"  - QUBO Problem Size: {len(mu_latent)}D latent space")
    print(f"  - Target Cardinality: {target_cardinality} factors")
    print(f"  - Risk Penalty: {risk_penalty}")
    print(f"  - Cardinality Penalty: {cardinality_penalty}")
    
    # Build QUBO matrix (same as quantum approach)
    Q_df = build_qubo_matrix(
        mu_latent,
        Sigma_latent,
        risk_penalty=risk_penalty,
        cardinality_penalty=cardinality_penalty,
        target_cardinality=target_cardinality
    )
    Q = Q_df.values
    n = len(Q)
    
    # Choose method
    if method == 'auto':
        method = 'exhaustive' if n <= 10 else 'greedy'
    
    num_combinations = len(list(combinations(range(n), target_cardinality)))
    
    if method == 'exhaustive':
        print(f"\n  - Using EXHAUSTIVE search (guaranteed optimal)")
        print(f"  - Evaluating all C({n},{target_cardinality}) = {num_combinations:,} combinations")
        solution, objective = solve_qubo_exhaustive(Q, target_cardinality=target_cardinality)
        print(f"  - Search complete: found global optimum")
    elif method == 'greedy':
        print(f"\n  - Using GREEDY heuristic (fast approximation)")
        print(f"  - Iteratively selecting {target_cardinality} best factors")
        solution, objective = solve_qubo_greedy(Q, target_cardinality=target_cardinality)
        print(f"  - Heuristic complete")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    bitstring = ''.join(map(str, solution))
    
    return {
        'solution': solution,
        'bitstring': bitstring,
        'objective': objective,
        'method': method,
        'Q_matrix': Q_df
    }


def run_complete_pipeline(
    # Data parameters
    period: str = "2y",
    interval: str = "1d",
    max_tickers: int = None,
    force_refresh: bool = False,
    cache_max_age_hours: int = 24,
    
    # Autoencoder parameters
    latent_dim: int = 16,
    ae_epochs: int = 200,
    
    # QAOA parameters
    risk_penalty: float = 0.5,
    cardinality_penalty: float = 0.1,
    target_cardinality: int = 5,
    qaoa_depth: int = 3,
    noise_level: float = 0.01,
    use_zne: bool = True,
    maxiter: int = 500,
    
    # Portfolio parameters
    top_n_stocks: int = None,
    
    # Classical comparison
    run_classical: bool = True,
    classical_method: str = 'auto',
    
    # Device
    device: str = 'cpu'
) -> Dict:
    """
    Run the complete quantum portfolio optimization pipeline with classical comparison.
    
    Args:
        period: Historical data period (e.g., "2y", "1y")
        interval: Data interval (e.g., "1d", "1h")
        max_tickers: Maximum number of tickers (None for all S&P 500)
        force_refresh: If True, ignore cache and fetch fresh data
        cache_max_age_hours: Maximum age of cache in hours (default: 24)
        latent_dim: Latent dimension for compression (default 8)
        ae_epochs: Autoencoder training epochs
        risk_penalty: QAOA risk penalty weight
        cardinality_penalty: QAOA cardinality constraint weight
        target_cardinality: Target number of latent factors
        qaoa_depth: QAOA circuit depth
        noise_level: Simulation noise level
        use_zne: Use Zero-Noise Extrapolation
        maxiter: Max QAOA optimizer iterations
        run_classical: Whether to run classical comparison
        classical_method: Classical solver method ('exhaustive', 'greedy', or 'auto')
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with complete pipeline results
    """
    print("\n" + "="*80)
    print(" " * 15 + "QUANTUM PORTFOLIO OPTIMIZATION PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Period: {period} | Interval: {interval} | Max Tickers: {max_tickers or 'All S&P 500'}")
    print(f"  Latent Dim: {latent_dim} | AE Epochs: {ae_epochs} | QAOA Depth: {qaoa_depth}")
    print(f"  Risk Penalty: {risk_penalty} | Cardinality Target: {target_cardinality}")
    print(f"  ZNE: {use_zne} | Noise Level: {noise_level}")
    print("="*80)
    
    # ========================================
    # STEP 1: Data Pipeline
    # ========================================
    print("\n[STEP 1/5] DATA PIPELINE - Fetching & Processing Market Data")
    print("-" * 80)
    data = run_complete_data_pipeline(
        period=period,
        interval=interval,
        max_tickers=max_tickers,
        force_refresh=force_refresh,
        cache_max_age_hours=cache_max_age_hours
    )
    
    print(f"\n[OK] Data Pipeline Complete:")
    print(f"  - Stocks: {len(data['tickers'])} tickers")
    print(f"  - Time Series: {data['log_returns'].shape[0]} days x {data['log_returns'].shape[1]} stocks")
    print(f"  - Fundamental Features: {data['fundamentals'].shape[1]} cleaned features")
    print(f"  - Expected Returns (mu): range [{data['mu'].min()*100:.2f}%, {data['mu'].max()*100:.2f}%] annualized")
    print(f"  - Covariance Matrix: {data['Sigma'].shape[0]}x{data['Sigma'].shape[1]}")
    
    # ========================================
    # STEP 2: Autoencoder Compression
    # ========================================
    print("\n[STEP 2/5] AUTOENCODER - Compressing to Latent Space")
    print("-" * 80)
    compression_result = run_autoencoder_compression(
        log_returns=data['log_returns'],
        fundamentals=data['fundamentals'],
        mu=data['mu'],
        Sigma=data['Sigma'],
        latent_dim=latent_dim,
        epochs=ae_epochs,
        device=device
    )
    
    print(f"\n[OK] Compression Complete:")
    print(f"  - Original Dimension: {len(data['tickers'])} stocks -> Latent Dimension: {compression_result['n_latent']}")
    print(f"  - Compression Ratio: {len(data['tickers']) / compression_result['n_latent']:.1f}x reduction")
    print(f"  - Latent mu range: [{compression_result['mu_latent'].min():.4f}, {compression_result['mu_latent'].max():.4f}]")
    print(f"  - Latent Sigma eigenvalues: [{np.linalg.eigvalsh(compression_result['Sigma_latent']).min():.2e}, {np.linalg.eigvalsh(compression_result['Sigma_latent']).max():.2e}]")
    
    # ========================================
    # STEP 3: QAOA Optimization
    # ========================================
    print("\n[STEP 3/5] QAOA - Quantum Approximate Optimization")
    print("-" * 80)
    qaoa_result = run_portfolio_qaoa(
        mu=compression_result['mu_latent'],
        Sigma=compression_result['Sigma_latent'],
        risk_penalty=risk_penalty,
        cardinality_penalty=cardinality_penalty,
        target_cardinality=target_cardinality,
        qaoa_depth=qaoa_depth,
        noise_level=noise_level,
        use_zne=use_zne,
        maxiter=maxiter
    )
    
    print(f"\n[OK] QAOA Optimization Complete:")
    print(f"  - Solution Bitstring: {qaoa_result['optimal_bitstring']}")
    print(f"  - Selected Latent Factors: {qaoa_result['optimal_solution'].sum()}/{latent_dim}")
    print(f"  - Energy (with noise): {qaoa_result['energy_noisy']:.6f}")
    if use_zne:
        print(f"  - Energy (ZNE corrected): {qaoa_result['energy_zne']:.6f}")
        print(f"  - Noise Mitigation Gain: {abs(qaoa_result['energy_noisy'] - qaoa_result['energy_zne']):.6f}")
    
    # ========================================
    # STEP 4: Classical Comparison (Optional)
    # ========================================
    classical_result = None
    classical_portfolio = None
    classical_metrics = None
    
    if run_classical:
        print("\n[STEP 4/5] CLASSICAL BASELINE - Traditional Optimization")
        print("-" * 80)
        classical_result = run_classical_comparison(
            mu_latent=compression_result['mu_latent'],
            Sigma_latent=compression_result['Sigma_latent'],
            risk_penalty=risk_penalty,
            cardinality_penalty=cardinality_penalty,
            target_cardinality=target_cardinality,
            method=classical_method
        )
        
        print(f"\n[OK] Classical Optimization Complete:")
        print(f"  - Method: {classical_result['method'].upper()}")
        print(f"  - Solution: {classical_result['bitstring']}")
        print(f"  - QUBO Objective: {classical_result['objective']:.6f}")
        
        # Decode classical portfolio
        print("\n  Decoding classical portfolio to stock weights...")
        classical_portfolio = decode_portfolio_weights(
            model=compression_result['model'],
            qaoa_solution=classical_result['solution'],
            latent_codes=compression_result['latent_codes'],
            tickers=data['tickers'],
            device=device
        )
        
        # Calculate classical metrics
        classical_metrics = calculate_portfolio_metrics(
            classical_portfolio['weights'],
            data['mu'],
            data['Sigma']
        )
        print(f"  - Portfolio Sharpe Ratio: {classical_metrics['sharpe_ratio']:.4f}")
    
    # ========================================
    # STEP 5: Decode Quantum Portfolio
    # ========================================
    print("\n[STEP 5/5] PORTFOLIO DECODING - Mapping to Stock Weights")
    print("-" * 80)
    portfolio = decode_portfolio_weights(
        model=compression_result['model'],
        qaoa_solution=qaoa_result['optimal_solution'],
        latent_codes=compression_result['latent_codes'],
        tickers=data['tickers'],
        device=device
    )
    
    # Calculate quantum portfolio metrics
    quantum_metrics = calculate_portfolio_metrics(
        portfolio['weights'],
        data['mu'],
        data['Sigma']
    )
    
    print(f"\n[OK] Quantum Portfolio Complete:")
    print(f"  - Total Stocks: {len(portfolio['tickers'])}")
    print(f"  - Stocks with >1% weight: {(portfolio['weights'] > 0.01).sum()}")
    print(f"  - Stocks with >5% weight: {(portfolio['weights'] > 0.05).sum()}")
    print(f"  - Portfolio Sharpe Ratio: {quantum_metrics['sharpe_ratio']:.4f}")
    
    # ========================================
    # Print Results
    # ========================================
    print("\n" + "="*80)
    print(" " * 25 + "FINAL RESULTS")
    print("="*80)
    
    if run_classical and classical_metrics is not None:
        print("\nPERFORMANCE COMPARISON: Quantum (QAOA) vs Classical")
        print("-"*80)
        print(f"{'Metric':<35} {'Quantum':>18} {'Classical':>18} {'Diff (Q-C)':>12}")
        print("-"*80)
        print(f"{'Solution Bitstring':<35} {qaoa_result['optimal_bitstring']:>18} {classical_result['bitstring']:>18}")
        print(f"{'Selected Latent Factors':<35} {qaoa_result['optimal_solution'].sum():>18} {classical_result['solution'].sum():>18}")
        print("-"*80)
        print(f"{'Expected Annual Return':<35} {quantum_metrics['expected_return']*100:>17.2f}% {classical_metrics['expected_return']*100:>17.2f}% {(quantum_metrics['expected_return']-classical_metrics['expected_return'])*100:>11.2f}%")
        print(f"{'Annual Volatility':<35} {quantum_metrics['volatility']*100:>17.2f}% {classical_metrics['volatility']*100:>17.2f}% {(quantum_metrics['volatility']-classical_metrics['volatility'])*100:>11.2f}%")
        print(f"{'Sharpe Ratio':<35} {quantum_metrics['sharpe_ratio']:>18.4f} {classical_metrics['sharpe_ratio']:>18.4f} {quantum_metrics['sharpe_ratio']-classical_metrics['sharpe_ratio']:>12.4f}")
        print(f"{'Variance':<35} {quantum_metrics['variance']:>18.4f} {classical_metrics['variance']:>18.4f} {quantum_metrics['variance']-classical_metrics['variance']:>12.4f}")
        print("-"*80)
        
        # Determine winner
        sharpe_diff = quantum_metrics['sharpe_ratio'] - classical_metrics['sharpe_ratio']
        if sharpe_diff > 0.001:
            print(f"\n*** WINNER: QUANTUM! Sharpe ratio is {sharpe_diff:.4f} higher (+{sharpe_diff/classical_metrics['sharpe_ratio']*100:.1f}%)")
        elif sharpe_diff < -0.001:
            print(f"\n*** WINNER: CLASSICAL! Sharpe ratio is {abs(sharpe_diff):.4f} higher (+{abs(sharpe_diff)/quantum_metrics['sharpe_ratio']*100:.1f}%)")
        else:
            print(f"\n*** TIE! Both approaches achieved essentially the same Sharpe ratio (diff: {abs(sharpe_diff):.4f})")
    else:
        print("\nQUANTUM PORTFOLIO PERFORMANCE:")
        print("-"*80)
        print(f"  Expected Annual Return: {quantum_metrics['expected_return']*100:>6.2f}%")
        print(f"  Annual Volatility:      {quantum_metrics['volatility']*100:>6.2f}%")
        print(f"  Sharpe Ratio:           {quantum_metrics['sharpe_ratio']:>6.4f}")
        print(f"  Variance:               {quantum_metrics['variance']:>6.4f}")
    
    print("\n" + "="*80)
    print(" " * 25 + "QUANTUM TOP 15 HOLDINGS")
    print("="*80)
    print(portfolio['portfolio_df'].head(15).to_string(index=False))
    
    if run_classical and classical_portfolio is not None:
        print("\n" + "="*80)
        print(" " * 25 + "CLASSICAL TOP 15 HOLDINGS")
        print("="*80)
        print(classical_portfolio['portfolio_df'].head(15).to_string(index=False))
    
    # ========================================
    # Package Results
    # ========================================
    return {
        # Data
        'tickers': data['tickers'],
        'mu': data['mu'],
        'Sigma': data['Sigma'],
        'log_returns': data['log_returns'],
        'fundamentals': data['fundamentals'],
        
        # Compression
        'autoencoder': compression_result['model'],
        'latent_codes': compression_result['latent_codes'],
        'mu_latent': compression_result['mu_latent'],
        'Sigma_latent': compression_result['Sigma_latent'],
        
        # Quantum QAOA
        'qaoa_solution': qaoa_result['optimal_solution'],
        'qaoa_bitstring': qaoa_result['optimal_bitstring'],
        'qaoa_energy': qaoa_result['energy_zne'] if use_zne else qaoa_result['energy_noisy'],
        'selected_dimensions': portfolio['selected_dimensions'],
        
        # Quantum Portfolio
        'weights': portfolio['weights'],
        'portfolio_df': portfolio['portfolio_df'],
        'metrics': quantum_metrics,
        
        # Classical Comparison
        'classical_solution': classical_result['solution'] if classical_result else None,
        'classical_bitstring': classical_result['bitstring'] if classical_result else None,
        'classical_weights': classical_portfolio['weights'] if classical_portfolio else None,
        'classical_portfolio_df': classical_portfolio['portfolio_df'] if classical_portfolio else None,
        'classical_metrics': classical_metrics,
        
        # Full results for reference
        'data_pipeline': data,
        'compression_full': compression_result,
        'qaoa_full': qaoa_result,
        'portfolio_full': portfolio,
        'classical_full': classical_result
    }


def save_results(results: Dict, output_dir: str = "./results") -> None:
    """
    Save pipeline results to files.
    
    Args:
        results: Results dictionary from run_complete_pipeline
        output_dir: Output directory
    """
    import os
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    
    # Save quantum portfolio
    results['portfolio_df'].to_csv(f"{output_dir}/quantum_portfolio.csv", index=False)
    
    # Save quantum metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(f"{output_dir}/quantum_metrics.csv", index=False)
    
    # Save QAOA solution
    qaoa_df = pd.DataFrame({
        'dimension': range(len(results['qaoa_solution'])),
        'selected': results['qaoa_solution']
    })
    qaoa_df.to_csv(f"{output_dir}/qaoa_solution.csv", index=False)
    
    # Save classical results if available
    if results['classical_metrics'] is not None:
        results['classical_portfolio_df'].to_csv(f"{output_dir}/classical_portfolio.csv", index=False)
        
        classical_metrics_df = pd.DataFrame([results['classical_metrics']])
        classical_metrics_df.to_csv(f"{output_dir}/classical_metrics.csv", index=False)
        
        classical_solution_df = pd.DataFrame({
            'dimension': range(len(results['classical_solution'])),
            'selected': results['classical_solution']
        })
        classical_solution_df.to_csv(f"{output_dir}/classical_solution.csv", index=False)
        
        # Save comparison
        comparison_df = pd.DataFrame({
            'Metric': ['Expected_Return', 'Volatility', 'Sharpe_Ratio'],
            'Quantum': [
                results['metrics']['expected_return'],
                results['metrics']['volatility'],
                results['metrics']['sharpe_ratio']
            ],
            'Classical': [
                results['classical_metrics']['expected_return'],
                results['classical_metrics']['volatility'],
                results['classical_metrics']['sharpe_ratio']
            ]
        })
        comparison_df['Difference'] = comparison_df['Quantum'] - comparison_df['Classical']
        comparison_df.to_csv(f"{output_dir}/comparison.csv", index=False)
    
    # Save full results as pickle for visualization
    print(f"\nSaving full results for visualization...")
    pickle_results = {
        'tickers': results['tickers'],
        'latent_codes': results['latent_codes'],
        'selected_dimensions': results['selected_dimensions'],
        'data_pipeline': {
            'close_df': results['data_pipeline']['close_df'],
            'log_returns': results['data_pipeline']['log_returns']
        }
    }
    with open(f"{output_dir}/full_results.pkl", 'wb') as f:
        pickle.dump(pickle_results, f)
    
    print(f"[OK] Results saved to {output_dir}/")
    print(f"[OK] Full results saved to {output_dir}/full_results.pkl")


if __name__ == "__main__":
    print("Running complete quantum portfolio optimization pipeline...")
    print("This will take several minutes.\n")
    
    # ========================================================================
    # CONFIGURATION: SIMULATOR vs REAL QUANTUM HARDWARE
    # ========================================================================
    #
    # DATA CACHING (NEW!):
    # -----------------------------------------------------------------------
    # - First run: Downloads S&P 500 data (~1-2 min)
    # - Subsequent runs: Loads from cache (INSTANT!)
    # - Cache expires after 24 hours (refreshes automatically)
    # - To force refresh: set force_refresh=True
    # - Cache location: ./data_cache/
    #
    # CURRENT MODE: SIMULATOR (for testing & development)
    # -----------------------------------------------------------------------
    # latent_dim = 16         ->  16 qubits (GOOD for real hardware!)
    # qaoa_depth = 3          ->  ~5-7 min on simulator
    # maxiter = 300           ->  Good convergence
    #
    # FOR REAL HARDWARE (IBM Brisbane/Kyoto): Uncomment section below
    # -----------------------------------------------------------------------
    # qaoa_depth = 2          ->  Shallower circuit = less errors!
    # maxiter = 5             ->  Minimal but safe (~2 min runtime)
    # 
    # Why these changes?
    #   - Real HW has ~0.1 sec per shot (vs instant on simulator)
    #   - 10 min limit = need FAST execution for debugging
    #   - Shallower circuits = fewer gate errors
    #   - ZNE helps compensate for noise with fewer shots
    #
    # ========================================================================
    
    # Run with default parameters - FULL S&P 500 (all ~503 tickers)
    results = run_complete_pipeline(
        # Data parameters
        period="2y",
        interval="1d",
        max_tickers=None,  # Use ALL S&P 500 tickers (no limit!)
        force_refresh=False,  # Set to True to ignore cache and fetch fresh data
        cache_max_age_hours=24,  # Cache expires after 24 hours
        
        # Autoencoder parameters
        latent_dim=16,  # 16 qubits -> Low AE loss (~0.40-0.50) + Real HW ready!
        ae_epochs=200,
        
        # QAOA parameters
        risk_penalty=0.5,
        cardinality_penalty=20.0,  # High penalty to enforce exactly 5 selections
        target_cardinality=5,
        
        # SIMULATOR SETTINGS (current):
        qaoa_depth=3,    # Good for testing (~5-7 min)
        maxiter=300,     # Strong convergence
        
        # REAL HARDWARE SETTINGS (uncomment when ready):
        # qaoa_depth=2,  # Faster + less errors on real HW
        # maxiter=5,     # Safe for 10-min limit (~2 min runtime)
        
        noise_level=0.01,
        use_zne=True,
        run_classical=True,  # Enable classical comparison
        classical_method='auto'  # Use exhaustive for n<=10, greedy otherwise
    )
    
    # Save results
    save_results(results)
    
    print("\n" + "="*70)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("="*70)
