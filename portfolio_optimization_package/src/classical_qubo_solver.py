"""
Classical QUBO Solver for Portfolio Optimization
Uses greedy/simulated-annealing heuristics to solve the binary QUBO problem.
"""
import numpy as np
from scipy.optimize import minimize


def greedy_qubo_solver(Q, seed=0):
    """
    Greedy heuristic for QUBO problem: start with zero, iteratively flip bits.
    
    Args:
        Q: QUBO matrix (n x n)
        seed: Random seed for reproducibility
    
    Returns:
        Binary solution vector (n,)
    """
    np.random.seed(seed)
    n = Q.shape[0]
    x = np.zeros(n, dtype=int)
    
    for _ in range(n * 5):  # Multiple passes
        for i in range(n):
            x_flip = x.copy()
            x_flip[i] = 1 - x_flip[i]
            E_current = float(x @ Q @ x)
            E_flip = float(x_flip @ Q @ x_flip)
            if E_flip < E_current:
                x = x_flip
    
    return x


def simulated_annealing_qubo(Q, n_iter=1000, T_init=10.0, seed=0):
    """
    Simulated annealing heuristic for QUBO problem.
    
    Args:
        Q: QUBO matrix (n x n)
        n_iter: Number of iterations
        T_init: Initial temperature
        seed: Random seed
    
    Returns:
        Binary solution vector (n,)
    """
    np.random.seed(seed)
    n = Q.shape[0]
    x = np.random.randint(0, 2, n)
    x_best = x.copy()
    E_best = float(x_best @ Q @ x_best)
    
    for iteration in range(n_iter):
        T = T_init * (1 - iteration / n_iter)
        i = np.random.randint(0, n)
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        E_old = float(x @ Q @ x)
        E_new = float(x_new @ Q @ x_new)
        delta = E_new - E_old
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-6)):
            x = x_new
            if E_new < E_best:
                E_best = E_new
                x_best = x_new.copy()
    
    return x_best
