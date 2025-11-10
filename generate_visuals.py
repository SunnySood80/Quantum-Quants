"""
Generate All Visualizations
Loads results and creates comprehensive visualization suite.
"""

import os
import numpy as np
import pandas as pd
import pickle
from visualization import *

def load_results(results_dir: str = "results"):
    """Load all results from CSV files and pickle files."""
    print("\n" + "="*80)
    print(" " * 25 + "LOADING RESULTS")
    print("="*80)
    
    results = {}
    
    # Load CSV files
    try:
        results['quantum_portfolio'] = pd.read_csv(f"{results_dir}/quantum_portfolio.csv")
        print("OK Loaded quantum_portfolio.csv")
    except:
        print("WARNING: quantum_portfolio.csv not found, trying legacy portfolio.csv")
        results['quantum_portfolio'] = pd.read_csv(f"{results_dir}/portfolio.csv")
    
    try:
        results['quantum_metrics'] = pd.read_csv(f"{results_dir}/quantum_metrics.csv")
        print("OK Loaded quantum_metrics.csv")
    except:
        print("WARNING: quantum_metrics.csv not found, trying legacy metrics.csv")
        results['quantum_metrics'] = pd.read_csv(f"{results_dir}/metrics.csv")
    
    try:
        results['qaoa_solution'] = pd.read_csv(f"{results_dir}/qaoa_solution.csv")
        print("OK Loaded qaoa_solution.csv")
    except Exception as e:
        print(f"WARNING: Could not load qaoa_solution.csv: {e}")
    
    try:
        results['classical_portfolio'] = pd.read_csv(f"{results_dir}/classical_portfolio.csv")
        results['classical_metrics'] = pd.read_csv(f"{results_dir}/classical_metrics.csv")
        print("OK Loaded classical comparison files")
    except:
        print("WARNING: Classical comparison files not found (optional)")
    
    try:
        results['comparison'] = pd.read_csv(f"{results_dir}/comparison.csv")
        print("OK Loaded comparison.csv")
    except:
        print("WARNING: comparison.csv not found (optional)")
    
    # Try to load pickle file with full results
    try:
        with open(f"{results_dir}/full_results.pkl", 'rb') as f:
            full_results = pickle.load(f)
            results['full_results'] = full_results
            print("OK Loaded full_results.pkl (contains prices, latent codes, etc.)")
    except:
        print("WARNING: full_results.pkl not found - some visualizations may be limited")
    
    return results


def generate_all_plots(results_dir: str = "results", output_dir: str = "results/plots"):
    """
    Generate all visualizations from results.
    
    Args:
        results_dir: Directory containing result CSV files
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(" " * 20 + "GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Load results
    results = load_results(results_dir)
    
    # Extract data
    quantum_portfolio = results['quantum_portfolio']
    qaoa_solution = results.get('qaoa_solution')
    
    # Get selected stocks (non-zero weights)
    selected_stocks = quantum_portfolio[quantum_portfolio['Weight'] > 0].copy()
    selected_tickers = selected_stocks['Ticker'].tolist()
    
    print(f"\nFound {len(selected_tickers)} stocks with non-zero weights")
    print(f"Selected stocks: {', '.join(selected_tickers[:10])}{'...' if len(selected_tickers) > 10 else ''}")
    
    # ========================================
    # 1. QAOA Bitstring Solution
    # ========================================
    if qaoa_solution is not None:
        print("\n[1/10] Generating bitstring visualization...")
        solution_array = qaoa_solution['selected'].values
        plot_bitstring_solution(solution_array, f"{output_dir}/bitstring_solution.png")
    
    # ========================================
    # 2. Weight Distribution
    # ========================================
    print("\n[2/10] Generating weight distribution...")
    plot_weight_distribution(quantum_portfolio, 
                            title="Quantum Portfolio Weight Distribution",
                            save_path=f"{output_dir}/quantum_weight_distribution.png")
    
    # ========================================
    # 3-7. Price-based visualizations (if we have price data)
    # ========================================
    if 'full_results' in results and 'close_df' in results['full_results'].get('data_pipeline', {}):
        prices = results['full_results']['data_pipeline']['close_df']
        log_returns = results['full_results']['data_pipeline']['log_returns']
        
        print("\n[3/10] Generating individual stock charts...")
        plot_individual_stocks(prices, selected_tickers[:15],  # Limit to top 15
                             f"{output_dir}/individual_stocks.png")
        
        print("\n[4/10] Generating stock overlay...")
        weights_dict = dict(zip(quantum_portfolio['Ticker'], quantum_portfolio['Weight']))
        plot_stock_overlay(prices, selected_tickers[:20],  # Limit to top 20
                          weights_dict, f"{output_dir}/stock_overlay.png")
        
        print("\n[5/10] Generating portfolio performance...")
        weights_series = pd.Series(quantum_portfolio['Weight'].values, 
                                  index=quantum_portfolio['Ticker'].values)
        plot_portfolio_performance(prices, weights_series, 
                                  save_path=f"{output_dir}/portfolio_performance.png")
        
        print("\n[6/10] Generating correlation matrix...")
        plot_correlation_matrix(log_returns, selected_tickers[:15],
                               f"{output_dir}/correlation_matrix.png")
        
        print("\n[7/10] Generating risk-return scatter...")
        plot_risk_return_scatter(log_returns, weights_series,
                                f"{output_dir}/risk_return_scatter.png")
        
        # ========================================
        # 8. Quantum vs Classical Comparison
        # ========================================
        if 'classical_portfolio' in results:
            print("\n[8/10] Generating quantum vs classical comparison...")
            classical_portfolio = results['classical_portfolio']
            classical_weights = pd.Series(classical_portfolio['Weight'].values,
                                         index=classical_portfolio['Ticker'].values)
            plot_quantum_vs_classical(prices, weights_series, classical_weights,
                                    f"{output_dir}/quantum_vs_classical.png")
            
            print("\n       Generating classical weight distribution...")
            plot_weight_distribution(classical_portfolio,
                                   title="Classical Portfolio Weight Distribution",
                                   save_path=f"{output_dir}/classical_weight_distribution.png")
        else:
            print("\n[8/10] Skipping quantum vs classical (no classical data)")
    else:
        print("\n[3-8] Skipping price-based visualizations (no price data in pickle)")
        print("       To enable these, save 'close_df' in full_results.pkl")
    
    # ========================================
    # 9. Latent Space Heatmap
    # ========================================
    if 'full_results' in results and 'latent_codes' in results['full_results']:
        print("\n[9/10] Generating latent space heatmap...")
        latent_codes = results['full_results']['latent_codes']
        tickers = results['full_results']['tickers']
        selected_dims = results['full_results']['selected_dimensions']
        plot_latent_heatmap(latent_codes, tickers, selected_dims,
                           f"{output_dir}/latent_heatmap.png")
    else:
        print("\n[9/10] Skipping latent heatmap (no latent codes in results)")
    
    # ========================================
    # 10. QAOA Circuit Diagram
    # ========================================
    print("\n[10/10] Generating QAOA circuit diagram...")
    plot_qaoa_circuit(n_qubits=8, qaoa_depth=5, 
                     save_path=f"{output_dir}/qaoa_circuit.png")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print(" " * 25 + "COMPLETE!")
    print("="*80)
    print(f"All visualizations saved to: {output_dir}/")
    print("\nGenerated plots:")
    
    plot_files = [
        "bitstring_solution.png",
        "quantum_weight_distribution.png",
        "individual_stocks.png",
        "stock_overlay.png",
        "portfolio_performance.png",
        "correlation_matrix.png",
        "risk_return_scatter.png",
        "quantum_vs_classical.png",
        "latent_heatmap.png",
        "qaoa_circuit.png"
    ]
    
    for i, plot_file in enumerate(plot_files, 1):
        full_path = f"{output_dir}/{plot_file}"
        status = "[OK]" if os.path.exists(full_path) else "[  ]"
        print(f"  {status} {i:2d}. {plot_file}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print(" " * 15 + "QUANTUM PORTFOLIO VISUALIZATION SUITE")
    print("="*80)
    
    # Generate all plots
    generate_all_plots()
    
    print("\nAll done! Check results/plots/ folder for your visualizations.")

