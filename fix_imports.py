import json

# Load notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'r') as f:
    nb = json.load(f)

# Fix QUBO import cell
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'sys.path.insert(0, str(Path.cwd()' in source and 'src' in source and 'classical_qubo_solver' in source:
            # Replace import section
            cell['source'] = '''# Find src directory
src_path = Path.cwd() / 'src' if Path('src').exists() else Path.cwd() / 'portfolio_optimization_package' / 'src'
sys.path.insert(0, str(src_path))
from classical_qubo_solver import simulated_annealing_qubo

x_best = simulated_annealing_qubo(Q, n_iter=1000, T_init=10.0, seed=42)'''
            cell['source'] = [cell['source']]

# Save updated notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Fixed import paths in notebook")
