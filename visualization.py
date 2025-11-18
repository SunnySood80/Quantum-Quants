"""
Visualization Module for Quantum Portfolio Optimization
Generates comprehensive plots and charts for analysis and presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_individual_stocks(
    prices: pd.DataFrame,
    selected_tickers: List[str],
    save_path: str = "results/plots/individual_stocks.png"
):
    """
    Plot individual price charts for each selected stock.
    
    Args:
        prices: DataFrame with historical prices [dates × tickers]
        selected_tickers: List of selected stock tickers
        save_path: Path to save the plot
    """
    n_stocks = len(selected_tickers)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_stocks > 1 else [axes]
    
    for idx, ticker in enumerate(selected_tickers):
        ax = axes[idx]
        
        if ticker in prices.columns:
            stock_prices = prices[ticker].dropna()
            
            # Calculate returns
            returns = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100
            
            # Plot
            ax.plot(stock_prices.index, stock_prices.values, linewidth=2, color='#2E86AB')
            ax.fill_between(stock_prices.index, stock_prices.values, alpha=0.3, color='#2E86AB')
            
            # Formatting
            ax.set_title(f'{ticker} - Return: {returns:+.1f}%', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add start/end markers
            ax.scatter(stock_prices.index[0], stock_prices.iloc[0], 
                      color='green', s=100, zorder=5, marker='^', label='Start')
            ax.scatter(stock_prices.index[-1], stock_prices.iloc[-1], 
                      color='red', s=100, zorder=5, marker='v', label='End')
            ax.legend(loc='best')
    
    # Hide empty subplots
    for idx in range(n_stocks, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Individual Stock Price Charts - Selected Portfolio', 
                 fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_stock_overlay(
    prices: pd.DataFrame,
    selected_tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
    save_path: str = "results/plots/stock_overlay.png"
):
    """
    Plot all selected stocks on one chart (normalized to 100).
    
    Args:
        prices: DataFrame with historical prices
        selected_tickers: List of selected tickers
        weights: Optional dict of ticker -> weight
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_tickers)))
    
    for idx, ticker in enumerate(selected_tickers):
        if ticker in prices.columns:
            stock_prices = prices[ticker].dropna()
            normalized = (stock_prices / stock_prices.iloc[0]) * 100
            
            label = ticker
            if weights and ticker in weights:
                label = f"{ticker} ({weights[ticker]*100:.1f}%)"
            
            ax.plot(normalized.index, normalized.values, 
                   linewidth=2.5, label=label, color=colors[idx], alpha=0.8)
    
    # Formatting
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Starting Value')
    ax.set_title('Normalized Stock Performance (Base = 100)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Price', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_portfolio_performance(
    prices: pd.DataFrame,
    weights: pd.Series,
    benchmark_ticker: str = 'SPY',
    save_path: str = "results/plots/portfolio_performance.png"
):
    """
    Plot portfolio performance vs benchmark.
    
    Args:
        prices: DataFrame with historical prices
        weights: Series with portfolio weights (ticker -> weight)
        benchmark_ticker: Ticker for benchmark comparison
        save_path: Path to save the plot
    """
    # Calculate portfolio value
    portfolio_tickers = weights[weights > 0].index.tolist()
    portfolio_prices = prices[portfolio_tickers].fillna(method='ffill')
    
    # Calculate daily portfolio value
    normalized_prices = portfolio_prices / portfolio_prices.iloc[0]
    portfolio_value = (normalized_prices * weights[portfolio_tickers].values).sum(axis=1)
    
    # Calculate returns
    portfolio_returns = (portfolio_value / portfolio_value.iloc[0] - 1) * 100
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Main performance plot
    ax1.plot(portfolio_value.index, portfolio_returns, 
            linewidth=3, label='Quantum Portfolio', color='#E63946')
    ax1.fill_between(portfolio_value.index, portfolio_returns, 0, 
                     alpha=0.3, color='#E63946')
    
    # Add benchmark if available
    if benchmark_ticker in prices.columns:
        benchmark = prices[benchmark_ticker].dropna()
        benchmark_returns = (benchmark / benchmark.iloc[0] - 1) * 100
        ax1.plot(benchmark.index, benchmark_returns, 
                linewidth=2.5, label=f'{benchmark_ticker} (Benchmark)', 
                color='#457B9D', linestyle='--', alpha=0.8)
    
    # Formatting
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_title('Portfolio Performance vs Benchmark', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add final return annotation
    final_return = portfolio_returns.iloc[-1]
    ax1.text(0.02, 0.98, f'Final Return: {final_return:+.2f}%', 
            transform=ax1.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    # Drawdown plot
    running_max = portfolio_value.expanding().max()
    drawdown = ((portfolio_value - running_max) / running_max) * 100
    
    ax2.fill_between(drawdown.index, drawdown, 0, 
                     alpha=0.5, color='#E63946', label='Drawdown')
    ax2.plot(drawdown.index, drawdown, linewidth=1.5, color='#A4161A')
    
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    max_drawdown = drawdown.min()
    ax2.text(0.02, 0.05, f'Max Drawdown: {max_drawdown:.2f}%', 
            transform=ax2.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_quantum_vs_classical(
    prices: pd.DataFrame,
    quantum_weights: pd.Series,
    classical_weights: pd.Series,
    save_path: str = "results/plots/quantum_vs_classical.png"
):
    """
    Compare quantum and classical portfolio performance.
    
    Args:
        prices: DataFrame with historical prices
        quantum_weights: Quantum portfolio weights
        classical_weights: Classical portfolio weights
        save_path: Path to save the plot
    """
    # Calculate quantum portfolio
    q_tickers = quantum_weights[quantum_weights > 0].index
    q_prices = prices[q_tickers].fillna(method='ffill')
    q_normalized = q_prices / q_prices.iloc[0]
    q_portfolio = (q_normalized * quantum_weights[q_tickers].values).sum(axis=1)
    q_returns = (q_portfolio / q_portfolio.iloc[0] - 1) * 100
    
    # Calculate classical portfolio
    c_tickers = classical_weights[classical_weights > 0].index
    c_prices = prices[c_tickers].fillna(method='ffill')
    c_normalized = c_prices / c_prices.iloc[0]
    c_portfolio = (c_normalized * classical_weights[c_tickers].values).sum(axis=1)
    c_returns = (c_portfolio / c_portfolio.iloc[0] - 1) * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(q_returns.index, q_returns, linewidth=3, 
           label='Quantum (QAOA)', color='#E63946')
    ax.plot(c_returns.index, c_returns, linewidth=3, 
           label='Classical (Exhaustive)', color='#457B9D')
    
    # Fill between
    ax.fill_between(q_returns.index, q_returns, c_returns, 
                    where=(q_returns >= c_returns), 
                    interpolate=True, alpha=0.3, color='#E63946', 
                    label='Quantum Advantage')
    ax.fill_between(q_returns.index, q_returns, c_returns, 
                    where=(q_returns < c_returns), 
                    interpolate=True, alpha=0.3, color='#457B9D', 
                    label='Classical Advantage')
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_title('Quantum vs Classical Portfolio Performance', 
                fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add final comparison
    q_final = q_returns.iloc[-1]
    c_final = c_returns.iloc[-1]
    winner = "QUANTUM" if q_final > c_final else "CLASSICAL"
    diff = abs(q_final - c_final)
    
    ax.text(0.02, 0.98, 
           f'Quantum: {q_final:+.2f}%\nClassical: {c_final:+.2f}%\n{winner} WINS by {diff:.2f}%', 
           transform=ax.transAxes, fontsize=12, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_heatmap(
    latent_codes: np.ndarray,
    tickers: List[str],
    selected_dims: List[int],
    save_path: str = "results/plots/latent_heatmap.png"
):
    """
    Plot heatmap of stock loadings in latent space.
    
    Args:
        latent_codes: Latent representations [N × 8]
        tickers: List of stock tickers
        selected_dims: Selected latent dimensions
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    
    # Create heatmap
    im = ax.imshow(latent_codes, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.abs(latent_codes).max(), 
                   vmax=np.abs(latent_codes).max())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading Strength', rotation=270, labelpad=20, fontsize=12)
    
    # Highlight selected dimensions
    for dim in selected_dims:
        ax.axvline(x=dim-0.5, color='lime', linewidth=3, alpha=0.8)
        ax.axvline(x=dim+0.5, color='lime', linewidth=3, alpha=0.8)
    
    # Formatting
    ax.set_xticks(range(latent_codes.shape[1]))
    ax.set_xticklabels([f'Dim {i}' for i in range(latent_codes.shape[1])])
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    
    ax.set_xlabel('Latent Dimensions', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title('Stock Loadings in Latent Space\n(Green borders = QAOA selected dimensions)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_qaoa_circuit(
    n_qubits: int = 8,
    qaoa_depth: int = 3,
    save_path: str = "results/plots/qaoa_circuit.png"
):
    """
    Plot QAOA circuit diagram.
    
    Args:
        n_qubits: Number of qubits
        qaoa_depth: QAOA depth (p)
        save_path: Path to save the plot
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import QAOAAnsatz
        from qiskit.quantum_info import SparsePauliOp
        
        # Create dummy Hamiltonian
        pauli_list = ['Z' * n_qubits, 'I' * (n_qubits-2) + 'ZZ']
        coeffs = [1.0, 0.5]
        hamiltonian = SparsePauliOp(pauli_list, coeffs)
        
        # Create QAOA circuit
        qaoa_circuit = QAOAAnsatz(hamiltonian, reps=qaoa_depth)
        
        # Draw circuit
        fig = qaoa_circuit.decompose().draw('mpl', style='iqp', fold=-1)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    except Exception as e:
        print(f"WARNING: Could not generate circuit diagram: {e}")


def plot_weight_distribution(
    portfolio_df: pd.DataFrame,
    title: str = "Portfolio Weight Distribution",
    save_path: str = "results/plots/weight_distribution.png"
):
    """
    Plot portfolio weight distribution.
    
    Args:
        portfolio_df: DataFrame with Ticker and Weight columns
        title: Plot title
        save_path: Path to save the plot
    """
    # Filter non-zero weights
    portfolio_df = portfolio_df[portfolio_df['Weight'] > 0].sort_values('Weight', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Horizontal bar chart
    colors = plt.cm.Spectral(np.linspace(0, 1, len(portfolio_df)))
    ax1.barh(portfolio_df['Ticker'], portfolio_df['Weight'] * 100, color=colors)
    ax1.set_xlabel('Weight (%)', fontsize=12)
    ax1.set_title('Portfolio Weights by Stock', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Pie chart (top 10 + others)
    top_10 = portfolio_df.head(10)
    others_weight = portfolio_df.iloc[10:]['Weight'].sum() if len(portfolio_df) > 10 else 0
    
    pie_data = list(top_10['Weight'].values)
    pie_labels = list(top_10['Ticker'].values)
    
    if others_weight > 0:
        pie_data.append(others_weight)
        pie_labels.append(f'Others ({len(portfolio_df)-10})')
    
    ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
           startangle=90, colors=colors[:len(pie_data)])
    ax2.set_title('Portfolio Concentration\n(Top 10 Holdings)', 
                 fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_matrix(
    returns: pd.DataFrame,
    selected_tickers: List[str],
    save_path: str = "results/plots/correlation_matrix.png"
):
    """
    Plot correlation matrix of selected stocks.
    
    Args:
        returns: DataFrame with stock returns
        selected_tickers: List of selected tickers
        save_path: Path to save the plot
    """
    # Calculate correlation
    corr = returns[selected_tickers].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Stock Correlation Matrix - Selected Portfolio', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_risk_return_scatter(
    returns: pd.DataFrame,
    weights: pd.Series,
    save_path: str = "results/plots/risk_return_scatter.png"
):
    """
    Plot risk-return scatter plot for selected stocks.
    
    Args:
        returns: DataFrame with stock returns
        weights: Series with portfolio weights
        save_path: Path to save the plot
    """
    # Calculate metrics
    selected = weights[weights > 0].index
    mean_returns = returns[selected].mean() * 252 * 100  # Annualized %
    volatilities = returns[selected].std() * np.sqrt(252) * 100  # Annualized %
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot with size based on weight
    scatter = ax.scatter(volatilities, mean_returns, 
                        s=weights[selected].values * 10000,
                        alpha=0.6, c=range(len(selected)), cmap='viridis',
                        edgecolors='black', linewidth=1.5)
    
    # Add labels
    for ticker, vol, ret in zip(selected, volatilities, mean_returns):
        ax.annotate(ticker, (vol, ret), fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile\n(Bubble size = Portfolio weight)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_bitstring_solution(
    solution: np.ndarray,
    save_path: str = "results/plots/bitstring_solution.png"
):
    """
    Visualize QAOA bitstring solution.
    
    Args:
        solution: Binary solution array
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#E63946' if x == 1 else '#457B9D' for x in solution]
    bars = ax.bar(range(len(solution)), solution, color=colors, 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, solution)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'{int(val)}', ha='center', va='center', 
               fontsize=16, fontweight='bold', color='white')
    
    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Selected (1) / Not Selected (0)', fontsize=12)
    ax.set_title(f'QAOA Solution Bitstring: {"".join(map(str, solution.astype(int)))}\n' + 
                f'Selected {solution.sum():.0f} dimensions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(solution)))
    ax.set_xticklabels([f'Dim {i}' for i in range(len(solution))])
    ax.set_ylim(-0.2, 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Import this module and use the plotting functions.")

