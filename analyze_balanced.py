"""Analyze QAOA Balanced solution."""
import pandas as pd
import numpy as np

# Load balanced metrics and portfolio
metrics = pd.read_csv('results/qaoa_balanced_metrics.csv').iloc[0]
portfolio = pd.read_csv('results/qaoa_balanced_portfolio.csv')

print('QAOA BALANCED ANALYSIS')
print('='*60)
print(f'Sharpe Ratio: {metrics["sharpe_ratio"]:.4f}')
print(f'Expected Return: {metrics["expected_return"]*100:.2f}%')
print(f'Volatility: {metrics["volatility"]*100:.2f}%')
print()

# Concentration analysis
print('CONCENTRATION ANALYSIS:')
print(f'  Stocks with >5% weight: {(portfolio["Weight"] > 0.05).sum()}')
print(f'  Stocks with >1% weight: {(portfolio["Weight"] > 0.01).sum()}')
print(f'  Herfindahl Index (concentration): {(portfolio["Weight"]**2).sum():.4f}')
print(f'  Top 3 cumulative: {portfolio["Weight"].head(3).sum()*100:.1f}%')
print(f'  Top 10 cumulative: {portfolio["Weight"].head(10).sum()*100:.1f}%')
print()

# Return distribution
print('TOP 10 HOLDINGS:')
for idx, row in portfolio.head(10).iterrows():
    print(f'  {row["Ticker"]}: {row["Weight"]*100:.2f}%')

print()
print('IMPROVEMENT OPPORTUNITIES:')
print('='*60)
print('1. QAOA Depth: Currently p=3, try p=4-5 for richer circuit expressiveness')
print('2. Optimizer: Use SLSQP or L-BFGS-B instead of COBYLA for better convergence')
print('3. Warm Start: Initialize params near known-good solutions')
print('4. Shots: Increase to 10000 for lower measurement noise')
print('5. Iterations: Increase to 400-500 for thorough parameter search')
print('6. QUBO tuning: Fine-tune risk_penalty (3.0->3.5) and cardinality_penalty (0.2->0.15)')
print()
print('Expected improvements: +1-3% Sharpe potential')
