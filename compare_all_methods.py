"""
Summary comparison of all three portfolio optimization methods.
"""
import os
import pandas as pd

OUTDIR = 'results'

# Load metrics for each method
qaoa_real = pd.read_csv(os.path.join(OUTDIR, 'qaoa_real_metrics.csv')).iloc[0].to_dict()
classical = pd.read_csv(os.path.join(OUTDIR, 'classical_qubo_metrics.csv')).iloc[0].to_dict()
markowitz = pd.read_csv(os.path.join(OUTDIR, 'markowitz_metrics.csv')).iloc[0].to_dict()

print('\n' + '='*80)
print(' '*20 + 'PORTFOLIO OPTIMIZATION COMPARISON')
print('='*80)

# Create comparison table
comparison = pd.DataFrame({
    'Metric': ['Expected Return', 'Annual Volatility', 'Sharpe Ratio', 'Variance'],
    'Real QAOA': [
        f"{qaoa_real['expected_return']*100:.2f}%",
        f"{qaoa_real['volatility']*100:.2f}%",
        f"{qaoa_real['sharpe_ratio']:.4f}",
        f"{qaoa_real['variance']:.6f}"
    ],
    'Classical QUBO': [
        f"{classical['expected_return']*100:.2f}%",
        f"{classical['volatility']*100:.2f}%",
        f"{classical['sharpe_ratio']:.4f}",
        f"{classical['variance']:.6f}"
    ],
    'Markowitz': [
        f"{markowitz['expected_return']*100:.2f}%",
        f"{markowitz['volatility']*100:.2f}%",
        f"{markowitz['sharpe_ratio']:.4f}",
        f"{markowitz['variance']:.6f}"
    ]
})

print('\n' + comparison.to_string(index=False))

# Determine winner by Sharpe ratio
methods = {
    'Real QAOA': qaoa_real['sharpe_ratio'],
    'Classical QUBO': classical['sharpe_ratio'],
    'Markowitz': markowitz['sharpe_ratio']
}

winner = max(methods, key=methods.get)
sharpe_vals = [methods[m] for m in methods]
spread = max(sharpe_vals) - min(sharpe_vals)

print('\n' + '='*80)
print(f'WINNER (by Sharpe Ratio): {winner}')
print(f'  Sharpe Ratio: {methods[winner]:.4f}')
print(f'  Performance spread: {spread:.4f}')
print('='*80)

# Load and show top holdings for each
print('\n' + '='*80)
print(' '*25 + 'TOP 10 HOLDINGS BY METHOD')
print('='*80)

for method, csv_file in [
    ('Real QAOA', 'qaoa_real_portfolio.csv'),
    ('Classical QUBO', 'classical_qubo_portfolio.csv'),
    ('Markowitz', 'markowitz_portfolio.csv')
]:
    print(f'\n{method}:')
    print('-'*50)
    df = pd.read_csv(os.path.join(OUTDIR, csv_file))
    
    # Handle different column names (QAOA/Classical use 'Ticker'/'Weight', Markowitz uses 'ticker'/'weight')
    ticker_col = 'Ticker' if 'Ticker' in df.columns else 'ticker'
    weight_col = 'Weight' if 'Weight' in df.columns else 'weight'
    
    top10 = df.head(10)[[ticker_col, weight_col]].copy()
    top10.columns = ['Ticker', 'Weight']
    top10['Weight'] = top10['Weight'].apply(lambda x: f'{x*100:.2f}%')
    print(top10.to_string(index=False))

print('\n' + '='*80)
print('Done.')
