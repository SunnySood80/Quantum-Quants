"""
Markowitz Mean-Variance Portfolio Optimizer
Selects an efficient frontier portfolio that maximizes Sharpe ratio.
"""
import numpy as np
from scipy.optimize import minimize


def optimize_markowitz(mu, Sigma, risk_free_rate=0.02):
    """
    Markowitz mean-variance optimization to maximize Sharpe ratio.
    
    Args:
        mu: Expected returns vector (n_assets,)
        Sigma: Covariance matrix (n_assets, n_assets)
        risk_free_rate: Risk-free rate (default 0.02 = 2%)
    
    Returns:
        Dict with 'weights' (optimal portfolio weights), 'sharpe_ratio', 'expected_return', 'volatility'
    """
    n = len(mu)
    
    def negative_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        return -sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.ones(n) / n
    
    result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    w = result.x
    
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = (ret - risk_free_rate) / vol
    
    return {
        'weights': w,
        'expected_return': ret,
        'volatility': vol,
        'sharpe_ratio': sharpe
    }
