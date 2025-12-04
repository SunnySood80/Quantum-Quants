"""
Final comprehensive comparison: all QAOA variants + Classical QUBO + Markowitz
"""
import os
import pandas as pd

OUTDIR = 'results'

# Load all metrics
methods = {
    'QAOA (Original)': 'qaoa_real_metrics.csv',
    'QAOA Conservative': 'qaoa_conservative_metrics.csv',
    'QAOA Balanced': 'qaoa_balanced_metrics.csv',
    'QAOA Aggressive': 'qaoa_aggressive_metrics.csv',
    'Classical QUBO': 'classical_qubo_metrics.csv',
    'Markowitz': 'markowitz_metrics.csv'
}

results = {}
for name, csv_file in methods.items():
    path = os.path.join(OUTDIR, csv_file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df.iloc[0]
        results[name] = {
            'Return': row['expected_return'] * 100,
            'Volatility': row['volatility'] * 100,
            'Sharpe': row['sharpe_ratio'],
            'Variance': row['variance']
        }

# Create comparison table
print('\n' + '='*100)
print(' '*30 + 'COMPREHENSIVE COMPARISON - ALL METHODS')
print('='*100)

comparison_data = []
for method in sorted(results.keys()):
    r = results[method]
    comparison_data.append({
        'Method': method,
        'Expected Return': f"{r['Return']:.2f}%",
        'Volatility': f"{r['Volatility']:.2f}%",
        'Sharpe Ratio': f"{r['Sharpe']:.4f}",
        'Variance': f"{r['Variance']:.6f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Find best by Sharpe
sharpe_dict = {name: results[name]['Sharpe'] for name in results}
best_method = max(sharpe_dict, key=sharpe_dict.get)
best_sharpe = sharpe_dict[best_method]

# Find second best
sorted_methods = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)
second_best_method = sorted_methods[1][0]
second_best_sharpe = sorted_methods[1][1]

print('\n' + '='*100)
print(f'🏆 WINNER: {best_method}')
print(f'   Sharpe Ratio: {best_sharpe:.4f}')
print(f'\n🥈 SECOND: {second_best_method}')
print(f'   Sharpe Ratio: {second_best_sharpe:.4f}')
print(f'\nPerformance Gap: {(best_sharpe - second_best_sharpe):.4f}')
print('='*100)

# Show top holdings for best QAOA variant
print('\nTOP 10 HOLDINGS:')
print('-'*100)

# Determine best QAOA variant
qaoa_variants = {
    'QAOA (Original)': 'qaoa_real_portfolio.csv',
    'QAOA Conservative': 'qaoa_conservative_portfolio.csv',
    'QAOA Balanced': 'qaoa_balanced_portfolio.csv',
    'QAOA Aggressive': 'qaoa_aggressive_portfolio.csv',
}

qaoa_sharpe = {name: results[name]['Sharpe'] for name in qaoa_variants}
best_qaoa = max(qaoa_sharpe, key=qaoa_sharpe.get)

print(f'\nBEST QAOA VARIANT: {best_qaoa} (Sharpe: {qaoa_sharpe[best_qaoa]:.4f})')
print('-'*50)

portfolio_path = os.path.join(OUTDIR, qaoa_variants[best_qaoa])
if os.path.exists(portfolio_path):
    df_port = pd.read_csv(portfolio_path)
    ticker_col = 'Ticker' if 'Ticker' in df_port.columns else 'ticker'
    weight_col = 'Weight' if 'Weight' in df_port.columns else 'weight'
    
    top10 = df_port.head(10)[[ticker_col, weight_col]].copy()
    top10.columns = ['Ticker', 'Weight']
    top10['Weight'] = top10['Weight'].apply(lambda x: f'{x*100:.2f}%')
    print(top10.to_string(index=False))

# Show best overall top holdings
print(f'\n\nBEST OVERALL METHOD: {best_method} (Sharpe: {best_sharpe:.4f})')
print('-'*50)

# Map method name to portfolio CSV
if 'QAOA' in best_method:
    variant = best_method.replace('QAOA ', '').lower()
    portfolio_csv = f'qaoa_{variant}_portfolio.csv'
elif 'Classical' in best_method:
    portfolio_csv = 'classical_qubo_portfolio.csv'
elif 'Markowitz' in best_method:
    portfolio_csv = 'markowitz_portfolio.csv'

portfolio_path = os.path.join(OUTDIR, portfolio_csv)
if os.path.exists(portfolio_path):
    df_port = pd.read_csv(portfolio_path)
    ticker_col = 'Ticker' if 'Ticker' in df_port.columns else 'ticker'
    weight_col = 'Weight' if 'Weight' in df_port.columns else 'weight'
    
    top10 = df_port.head(10)[[ticker_col, weight_col]].copy()
    top10.columns = ['Ticker', 'Weight']
    top10['Weight'] = top10['Weight'].apply(lambda x: f'{x*100:.2f}%')
    print(top10.to_string(index=False))

print('\n' + '='*100)
print('Done.')
