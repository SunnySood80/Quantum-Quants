# Quantum Portfolio Optimization

Complete pipeline for quantum portfolio optimization using autoencoder compression and QAOA.

## System Overview

This system solves the portfolio optimization problem for S&P 500 stocks using quantum computing:

```
S&P 500 Tickers (501) 
    ↓
Price Data + Fundamentals
    ↓
Autoencoder Compression (501 → 8)
    ↓
QAOA Optimization (8 qubits)
    ↓
Portfolio Decoding (8 → 501 weights)
```

## Files

1. **portfolio_data_pipeline.py** - Data fetching and preprocessing
   - Fetches S&P 500 tickers from Wikipedia
   - Downloads 2 years of daily price data
   - Calculates log returns, mu, Sigma
   - Fetches fundamental features (P/E, margins, etc.)
   - VIF cleaning to remove multicollinear features
   - **Fixes the non-finite mu error** with robust validation

2. **autoencoder_compression.py** - Neural network compression
   - Trains autoencoder on return matrix [501 × T]
   - Uses fundamentals as auxiliary input
   - Compresses to 8-dimensional latent space
   - Projects mu/Sigma to latent space
   - Decodes QAOA solution back to stock weights

3. **qubo_portfolio_optimizer.py** - Quantum optimization
   - Builds 8×8 QUBO matrix in latent space
   - Converts to Ising Hamiltonian
   - Runs QAOA with 8 qubits (2^8 = 256 states - totally feasible!)
   - Includes Zero-Noise Extrapolation (ZNE)
   - Uses noisy simulator for realistic results

4. **main_portfolio_optimization.py** - Complete pipeline orchestration
   - Runs all steps in sequence
   - Calculates portfolio metrics (Sharpe ratio, etc.)
   - Saves results to files

## Quick Start

### Option 1: Run Complete Pipeline

```python
from main_portfolio_optimization import run_complete_pipeline

# Run with default settings
results = run_complete_pipeline(
    period="2y",           # 2 years of data
    interval="1d",         # Daily data
    latent_dim=8,          # 8-dimensional compression
    ae_epochs=200,         # Autoencoder training epochs
    risk_penalty=0.5,      # Risk weight in QUBO
    target_cardinality=5,  # Target 5 latent factors
    qaoa_depth=3,          # QAOA circuit depth
    use_zne=True          # Use Zero-Noise Extrapolation
)

# View results
print(results['portfolio_df'].head(15))
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")
```

### Option 2: Run Steps Individually

```python
# Step 1: Data Pipeline
from portfolio_data_pipeline import run_complete_data_pipeline

data = run_complete_data_pipeline(period="2y", interval="1d")
print(f"Got {len(data['tickers'])} tickers")
print(f"Features: {data['fundamentals'].shape[1]}")

# Step 2: Autoencoder Compression
from autoencoder_compression import run_autoencoder_compression

compression = run_autoencoder_compression(
    log_returns=data['log_returns'],
    fundamentals=data['fundamentals'],
    mu=data['mu'],
    Sigma=data['Sigma'],
    latent_dim=8,
    epochs=200
)

print(f"Compressed to {compression['n_latent']} dimensions")

# Step 3: QAOA Optimization
from qubo_portfolio_optimizer import run_portfolio_qaoa

qaoa_result = run_portfolio_qaoa(
    mu=compression['mu_latent'],
    Sigma=compression['Sigma_latent'],
    risk_penalty=0.5,
    target_cardinality=5,
    qaoa_depth=3,
    use_zne=True
)

print(f"QAOA solution: {qaoa_result['optimal_bitstring']}")

# Step 4: Decode Portfolio
from autoencoder_compression import decode_portfolio_weights

portfolio = decode_portfolio_weights(
    model=compression['model'],
    qaoa_solution=qaoa_result['optimal_solution'],
    latent_codes=compression['latent_codes'],
    tickers=data['tickers']
)

print(portfolio['portfolio_df'].head(10))
```

## Key Parameters

### Data Pipeline
- `period`: "2y" (2 years), "1y", "6mo", etc.
- `interval`: "1d" (daily), "1h", "1m" (not recommended)

### Autoencoder
- `latent_dim`: 8 (fixed for 8-qubit QAOA)
- `epochs`: 200 (more = better compression, but slower)

### QAOA
- `risk_penalty`: 0.5 (higher = prefer lower risk)
- `cardinality_penalty`: 0.1 (enforce factor selection)
- `target_cardinality`: 5 (target number of latent factors)
- `qaoa_depth`: 3 (circuit depth, higher = better but slower)
- `noise_level`: 0.01 (1% depolarizing noise)
- `use_zne`: True (Zero-Noise Extrapolation)
- `maxiter`: 500 (max optimizer iterations)

## What's Fixed

### Non-Finite Mu Error ✅
The original code had issues with:
- Invalid tickers from Yahoo Finance
- NaN propagation in log returns
- No validation before QUBO building

**Solution:**
- Multi-layer validation in `calculate_returns_and_statistics()`
- Automatic removal of bad tickers
- Checks that mu and Sigma are finite before proceeding

### Autoencoder Integration ✅
The original code couldn't handle 501 stocks (2^501 states = impossible).

**Solution:**
- Compress 501 stocks → 8 latent factors
- Run QAOA on 8 qubits (2^8 = 256 states = feasible)
- Decode back to 501 stock weights

### Everything That Worked from Your Original ✅
- 2-year daily data (not 1-minute)
- Fundamental features with VIF cleaning
- QAOA with ZNE
- Proper annualization (252 trading days)
- Log returns calculation

## Output Structure

```python
results = {
    # Portfolio
    'weights': np.ndarray,              # Stock weights [501]
    'portfolio_df': pd.DataFrame,       # Ranked holdings
    'metrics': {
        'expected_return': float,       # Annual return
        'volatility': float,            # Annual volatility
        'sharpe_ratio': float          # Sharpe ratio
    },
    
    # QAOA
    'qaoa_solution': np.ndarray,        # Binary solution [8]
    'qaoa_bitstring': str,              # e.g., "11001100"
    'selected_dimensions': np.ndarray,  # Which factors selected
    
    # Latent Space
    'mu_latent': np.ndarray,            # Expected returns [8]
    'Sigma_latent': np.ndarray,         # Covariance [8×8]
    'latent_codes': np.ndarray,         # Latent representations [501×8]
    
    # Raw Data
    'tickers': list,                    # Stock symbols
    'mu': pd.Series,                    # Original mu [501]
    'Sigma': pd.DataFrame,              # Original Sigma [501×501]
    'log_returns': pd.DataFrame,        # Return history
    'fundamentals': pd.DataFrame        # Cleaned features
}
```

## Requirements

```bash
pip install numpy pandas yfinance statsmodels
pip install torch  # PyTorch for autoencoder
pip install qiskit qiskit-aer scipy matplotlib
```

## Performance Notes

- **Data Pipeline**: ~2-3 minutes (downloads 501 stocks)
- **Autoencoder**: ~5-10 minutes (200 epochs)
- **QAOA**: ~3-5 minutes (8 qubits with ZNE)
- **Total Runtime**: ~10-20 minutes

## Troubleshooting

**Issue: "Non-finite values in mu"**
- Fixed in the new code! The validation now automatically removes bad tickers.

**Issue: "Out of memory" during autoencoder training**
- Reduce `ae_epochs` to 100
- Use smaller batch size in `train_autoencoder()`

**Issue: QAOA not converging**
- Increase `maxiter` to 1000
- Try different `risk_penalty` values (0.3 - 0.7)
- Increase `qaoa_depth` to 4 or 5

**Issue: Too few/many stocks selected**
- Adjust `target_cardinality` (default 5)
- Tune `cardinality_penalty` (higher = stricter enforcement)

## How It Works

### Why 8 Dimensions?
- 8 qubits = 2^8 = 256 possible solutions (totally feasible for QAOA)
- Captures main portfolio "themes" (growth, value, defensive, etc.)
- Enough to represent complex portfolio strategies

### Why Autoencoder?
- Direct 501-qubit QAOA = 2^501 states = IMPOSSIBLE
- Autoencoder learns which stocks move together
- Compression preserves the important structure

### Why Fundamentals?
- Guide the autoencoder to learn meaningful factors
- E.g., stocks with similar P/E ratios should cluster together
- Results in interpretable latent dimensions

### Why ZNE?
- Real quantum computers are noisy
- ZNE extrapolates to "perfect" zero-noise result
- Improves QAOA accuracy significantly

## Example Output

```
PORTFOLIO METRICS
Expected Annual Return: 18.45%
Annual Volatility: 22.31%
Sharpe Ratio: 0.8270

TOP 15 HOLDINGS
     Ticker    Weight     Score
       NVDA  0.045231  0.089234
       AAPL  0.032156  0.063421
      GOOGL  0.028943  0.057123
       MSFT  0.026781  0.052834
       AMZN  0.024532  0.048392
        ...       ...       ...
```

## Notes

- Uses daily data (not 1-minute) for better signal-to-noise
- 2 years of history balances recency with statistical power
- VIF cleaning ensures features are independent
- Annualized using 252 trading days (standard)
- Portfolio weights are continuous (not binary stock selection)
- Sharpe ratio assumes risk-free rate = 0

## References

- QAOA: Farhi et al. (2014)
- ZNE: Temme et al. (2017)
- Portfolio Optimization: Markowitz (1952)
