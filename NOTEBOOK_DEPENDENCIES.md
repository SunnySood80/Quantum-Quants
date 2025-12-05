# Portfolio Optimization Notebook - External Dependencies

## Overview
The notebook `Portfolio_Optimization_All_Models.ipynb` requires several external components to run successfully. Below is a comprehensive breakdown.

---

## 1. PYTHON PACKAGES (via pip)

### Standard Libraries (Built-in)
- `os` - File system operations
- `sys` - System parameters and path manipulation
- `pickle` - Data serialization
- `time` - Timing operations
- `warnings` - Warning suppression

### Third-Party Packages (Must Install)
**Core Data Science:**
- `numpy` - Numerical computations
- `pandas` - DataFrame operations and data manipulation
- `scipy` - Scientific computing (used internally by modules)
- `scikit-learn` - Machine learning utilities (PCA used in compression)

**Visualization:**
- `matplotlib` - Creating comparison plots and charts

**Quantum Computing:**
- `qiskit` - Quantum circuit framework
- `qiskit-aer` - Quantum circuit simulator (AerSimulator)
- `qiskit-ibm-runtime` - IBM quantum services (optional, for real hardware)

**Optional:**
- `torch` / `pytorch` - Deep learning (gracefully falls back to PCA if unavailable)

### Installation Command
```bash
pip install numpy pandas scipy scikit-learn matplotlib qiskit qiskit-aer
```

---

## 2. LOCAL FILES/DIRECTORIES REQUIRED

### Data Files
- **`data_cache/sp500_data_2y_1d_all.pkl`** (REQUIRED)
  - Pre-cached S&P 500 data
  - Contains: 496 stocks, 2 years daily returns, fundamental features
  - Must exist at `data_cache/` relative to notebook location
  - Size: ~several MB

### Python Modules (Parent Directory)
- **`main_portfolio_optimization.py`** (REQUIRED)
  - Contains: `run_autoencoder_compression()`, `run_markowitz_optimization()`, `decode_portfolio_weights()`, `calculate_portfolio_metrics()`, `_get_build_qubo_matrix()`
  - Must be in parent directory of notebook

### Source Modules (src/)
- **`src/classical_qubo_solver.py`** (REQUIRED)
  - Contains: `simulated_annealing_qubo()` function
  - Solves QUBO using simulated annealing heuristic
  - Located at either `src/` or `portfolio_optimization_package/src/`

- **`src/qaoa_optimizer.py`** (REQUIRED)
  - Contains: `QAOAPortfolioOptimizer` class
  - Implements quantum circuits using Qiskit
  - Located at either `src/` or `portfolio_optimization_package/src/`

---

## 3. DIRECTORY STRUCTURE (Expected Layout)

```
Quantum-Quants/
├── Portfolio_Optimization_All_Models.ipynb    ← Your notebook
├── main_portfolio_optimization.py              ← Required module
├── data_cache/
│   └── sp500_data_2y_1d_all.pkl              ← Required data
├── src/
│   ├── classical_qubo_solver.py               ← Required module
│   └── qaoa_optimizer.py                      ← Required module
└── notebook_results/                          ← Output directory (auto-created)
    ├── 1_markowitz_portfolio.csv
    ├── 1_markowitz_metrics.csv
    ├── 2_classical_qubo_portfolio.csv
    ├── 2_classical_qubo_metrics.csv
    ├── 3_qaoa_final_portfolio.csv
    ├── 3_qaoa_final_metrics.csv
    └── COMPARISON.csv
```

---

## 4. RUNTIME REQUIREMENTS

### Minimum System Resources
- **RAM**: 2-4 GB (for data + QAOA quantum simulation)
- **CPU**: Multi-core recommended (QAOA can use parallel computation)
- **Disk**: ~100 MB for results and temporary files

### Execution Time
- **Markowitz**: ~1 second
- **Classical QUBO**: ~0.5 seconds
- **QAOA**: ~18-20 seconds (with 5 multistart runs)
- **Total**: ~20-25 seconds

---

## 5. EXTERNAL DEPENDENCIES SUMMARY TABLE

| Component | Type | Required | Location | Purpose |
|-----------|------|----------|----------|---------|
| numpy | Package | YES | pip | Numerical computation |
| pandas | Package | YES | pip | Data handling |
| scipy | Package | YES | pip | Scientific computing |
| scikit-learn | Package | YES | pip | Machine learning utilities |
| matplotlib | Package | YES | pip | Visualization |
| qiskit | Package | YES | pip | Quantum circuits |
| qiskit-aer | Package | YES | pip | Quantum simulator |
| torch/pytorch | Package | NO | pip | Optional deep learning |
| main_portfolio_optimization.py | Module | YES | Parent dir | Core portfolio functions |
| classical_qubo_solver.py | Module | YES | src/ | Classical optimizer |
| qaoa_optimizer.py | Module | YES | src/ | Quantum optimizer |
| sp500_data_2y_1d_all.pkl | Data | YES | data_cache/ | S&P 500 dataset |

---

## 6. BEFORE RUNNING THE NOTEBOOK

### Setup Checklist
- [ ] Install Python 3.8+
- [ ] Create virtual environment (recommended): `python -m venv venv`
- [ ] Activate virtual environment
- [ ] Install required packages: `pip install -r requirements.txt`
- [ ] Verify `data_cache/sp500_data_2y_1d_all.pkl` exists
- [ ] Verify `main_portfolio_optimization.py` in parent directory
- [ ] Verify `src/classical_qubo_solver.py` exists
- [ ] Verify `src/qaoa_optimizer.py` exists
- [ ] Open notebook in Jupyter: `jupyter notebook Portfolio_Optimization_All_Models.ipynb`

### Verification Script
```python
import os
from pathlib import Path

# Check required files
required_files = [
    'data_cache/sp500_data_2y_1d_all.pkl',
    'main_portfolio_optimization.py',
    'src/classical_qubo_solver.py',
    'src/qaoa_optimizer.py',
]

for file in required_files:
    if Path(file).exists():
        print(f"✓ {file}")
    else:
        print(f"✗ MISSING: {file}")

# Check packages
packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'qiskit']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} installed")
    except ImportError:
        print(f"✗ MISSING: {pkg}")
```

---

## 7. TROUBLESHOOTING

### Missing Data File Error
```
FileNotFoundError: data_cache/sp500_data_2y_1d_all.pkl not found
```
**Solution**: Ensure data cache file is in correct location relative to notebook

### Module Import Error
```
ModuleNotFoundError: No module named 'main_portfolio_optimization'
```
**Solution**: Notebook must be in `Quantum-Quants/` root directory

### Quantum Simulator Error
```
ModuleNotFoundError: No module named 'qiskit_aer'
```
**Solution**: `pip install qiskit-aer`

### Memory Error
```
MemoryError: Unable to allocate X.XX GiB for an array
```
**Solution**: Close other applications or reduce QAOA multistart runs

---

## Summary

**Minimum to Run Notebook:**
1. Python 3.8+ with packages: numpy, pandas, scipy, scikit-learn, matplotlib, qiskit, qiskit-aer
2. Data file: `data_cache/sp500_data_2y_1d_all.pkl`
3. Three Python modules: `main_portfolio_optimization.py`, `src/classical_qubo_solver.py`, `src/qaoa_optimizer.py`

**Total Setup Time**: ~5-10 minutes
**Total Execution Time**: ~20-25 seconds
