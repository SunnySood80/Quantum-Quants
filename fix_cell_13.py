import json

# Load notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'r') as f:
    nb = json.load(f)

# Find cell #VSC-dcf72481 (cell 13) and replace its source
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == '#VSC-dcf72481':
        print(f"Found cell {i}: {cell.get('id')}")
        # Replace with complete QUBO cell
        cell['source'] = [
            "# Run Classical QUBO optimizer\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"METHOD 2: CLASSICAL QUBO (Simulated Annealing)\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "start = time.time()\n",
            "\n",
            "# Build QUBO matrix\n",
            "build_qubo = _get_build_qubo_matrix()\n",
            "Q_df = build_qubo(\n",
            "    mu_latent, Sigma_latent,\n",
            "    risk_penalty=RISK_PENALTY,\n",
            "    cardinality_penalty=CARDINALITY_PENALTY,\n",
            "    target_cardinality=TARGET_CARDINALITY\n",
            ")\n",
            "Q = Q_df.values\n",
            "n_latent = len(Q)\n",
            "\n",
            "# Find src directory and import QUBO solver\n",
            "src_path = Path.cwd() / 'src' if Path('src').exists() else Path.cwd() / 'portfolio_optimization_package' / 'src'\n",
            "sys.path.insert(0, str(src_path))\n",
            "from classical_qubo_solver import simulated_annealing_qubo\n",
            "\n",
            "x_best = simulated_annealing_qubo(Q, n_iter=1000, T_init=10.0, seed=42)\n",
            "\n",
            "# Decode solution\n",
            "portfolio = decode_portfolio_weights(\n",
            "    model=None,\n",
            "    qaoa_solution=x_best,\n",
            "    latent_codes=latent_codes,\n",
            "    tickers=tickers,\n",
            "    mu_latent=mu_latent\n",
            ")\n",
            "\n",
            "weights_qubo = portfolio['weights']\n",
            "metrics_qubo = calculate_portfolio_metrics(weights_qubo, mu, Sigma)\n",
            "time_qubo = time.time() - start\n",
            "\n",
            "print(f\"\\n[OK] Completed in {time_qubo:.2f}s\")\n",
            "print(f\"  Sharpe Ratio:    {metrics_qubo['sharpe_ratio']:.4f}\")\n",
            "print(f\"  Expected Return: {metrics_qubo['expected_return']:.2%}\")\n",
            "print(f\"  Volatility:      {metrics_qubo['volatility']:.2%}\")\n",
            "print(f\"  Holdings:        {np.sum(x_best > 0.5)}\")\n",
            "\n",
            "# Save results\n",
            "portfolio_df = pd.DataFrame({\n",
            "    'ticker': tickers,\n",
            "    'weight': weights_qubo\n",
            "})\n",
            "portfolio_df = portfolio_df[portfolio_df['weight'] > 1e-6].sort_values('weight', ascending=False)\n",
            "portfolio_df.to_csv(f'{RESULTS_DIR}/2_classical_qubo_portfolio.csv', index=False)\n",
            "pd.DataFrame([metrics_qubo]).to_csv(f'{RESULTS_DIR}/2_classical_qubo_metrics.csv', index=False)\n",
            "\n",
            "print(f\"\\n  Top holdings:\")\n",
            "print(portfolio_df.head(10).to_string(index=False))\n",
        ]
        # Clear outputs/execution
        cell['outputs'] = []
        cell.pop('execution_count', None)
        print(f"✓ Fixed cell - added {len(cell['source'])} lines of code")

# Save updated notebook
with open('Portfolio_Optimization_All_Models.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook fixed successfully!")
