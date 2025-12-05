import json

# Load notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 6 (load data) - convert numpy arrays to DataFrames
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'log_returns = np.array' in source:
            # Replace this cell with corrected version
            cell['source'] = '''data = cache['data']
log_returns = data['log_returns']  # Keep as DataFrame
fundamentals = data['fundamentals']  # Keep as DataFrame
mu = data['mu']  # Keep as Series
Sigma = data['Sigma']  # Keep as DataFrame
tickers = data['tickers']

print(f"✓ Loaded {len(mu)} stocks")
print(f"  Log returns shape: {log_returns.shape}")
print(f"  Fundamentals shape: {fundamentals.shape}")
print(f"  Covariance matrix shape: {Sigma.shape}")
print(f"\\nSample tickers: {list(tickers[:10])}")'''
            cell['source'] = [cell['source']]

# Save updated notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Updated notebook with correct data types")
