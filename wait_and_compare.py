"""
Wait for Advanced QAOA to complete, then compare all methods including Advanced.
This script polls for the result and prints final comparison when ready.
"""
import os
import time
import pandas as pd

OUTDIR = 'results'

# Poll for Advanced QAOA results (max 30 min wait)
advanced_ready = False
elapsed = 0
max_wait = 1800  # 30 minutes

print('Waiting for Advanced QAOA to complete...')
while not advanced_ready and elapsed < max_wait:
    if os.path.exists(os.path.join(OUTDIR, 'qaoa_advanced_metrics.csv')):
        try:
            df = pd.read_csv(os.path.join(OUTDIR, 'qaoa_advanced_metrics.csv'))
            if len(df) > 0:
                advanced_ready = True
                print(f'Advanced QAOA ready after {elapsed}s')
        except:
            pass
    
    if not advanced_ready:
        time.sleep(30)
        elapsed += 30

if not advanced_ready:
    print(f'Advanced QAOA did not complete after {max_wait}s. Skipping comparison.')
    exit(1)

# Load all methods
print('\nFinal Comprehensive Comparison (all methods including Advanced):')
print('='*100)

methods = {
    'QAOA (Original)': 'qaoa_real_metrics.csv',
    'QAOA Conservative': 'qaoa_conservative_metrics.csv',
    'QAOA Balanced': 'qaoa_balanced_metrics.csv',
    'QAOA Aggressive': 'qaoa_aggressive_metrics.csv',
    'QAOA Advanced': 'qaoa_advanced_metrics.csv',
    'Classical QUBO': 'classical_qubo_metrics.csv',
    'Markowitz': 'markowitz_metrics.csv'
}

results = {}
for name, csv_file in methods.items():
    path = os.path.join(OUTDIR, csv_file)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            row = df.iloc[0]
            results[name] = {
                'Return': row['expected_return'] * 100,
                'Volatility': row['volatility'] * 100,
                'Sharpe': row['sharpe_ratio'],
            }
        except:
            pass

# Create comparison
comparison_data = []
for method in sorted(results.keys()):
    r = results[method]
    comparison_data.append({
        'Method': method,
        'Expected Return': f"{r['Return']:.2f}%",
        'Volatility': f"{r['Volatility']:.2f}%",
        'Sharpe Ratio': f"{r['Sharpe']:.4f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Find best
sharpe_dict = {name: results[name]['Sharpe'] for name in results}
best_method = max(sharpe_dict, key=sharpe_dict.get)

print('\n' + '='*100)
print(f'🏆 BEST METHOD: {best_method} (Sharpe: {sharpe_dict[best_method]:.4f})')
print('='*100)

# Top holdings for Advanced
if 'QAOA Advanced' in results:
    print('\nTop 10 Holdings - QAOA Advanced:')
    print('-'*50)
    portfolio_path = os.path.join(OUTDIR, 'qaoa_advanced_portfolio.csv')
    if os.path.exists(portfolio_path):
        portfolio = pd.read_csv(portfolio_path)
        ticker_col = 'Ticker' if 'Ticker' in portfolio.columns else 'ticker'
        weight_col = 'Weight' if 'Weight' in portfolio.columns else 'weight'
        
        top10 = portfolio.head(10)[[ticker_col, weight_col]].copy()
        top10.columns = ['Ticker', 'Weight']
        top10['Weight'] = top10['Weight'].apply(lambda x: f'{x*100:.2f}%')
        print(top10.to_string(index=False))

print('\nDone.')
