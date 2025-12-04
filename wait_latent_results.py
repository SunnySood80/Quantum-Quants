"""Poll and wait for latent experiment to complete, then print results."""
import os
import time
import pandas as pd

OUTDIR = 'results'
comparison_file = os.path.join(OUTDIR, 'qaoa_latent_comparison.csv')

# Poll for up to 20 minutes
max_wait = 1200  # 20 minutes
elapsed = 0
poll_interval = 15  # Check every 15 seconds

print('Waiting for latent dimension experiment to complete...')

while elapsed < max_wait:
    if os.path.exists(comparison_file):
        try:
            df = pd.read_csv(comparison_file)
            if len(df) == 3:  # All 3 dimensions complete
                print(f'\nExperiment complete! Results after {elapsed}s:\n')
                print(df.to_string(index=False))
                
                # Find best
                best_idx = df['Sharpe'].idxmax()
                best_dim = int(df.loc[best_idx, 'Latent Dim'])
                best_sharpe = df.loc[best_idx, 'Sharpe']
                baseline_sharpe = df.loc[df['Latent Dim'] == 16, 'Sharpe'].values[0]
                
                print(f'\nBEST DIMENSION: {best_dim}D')
                print(f'Sharpe: {best_sharpe:.4f}')
                if best_dim != 16:
                    improvement = (best_sharpe - baseline_sharpe) / baseline_sharpe * 100
                    print(f'Improvement vs 16D: +{improvement:.2f}%')
                
                exit(0)
        except:
            pass
    
    time.sleep(poll_interval)
    elapsed += poll_interval
    if elapsed % 60 == 0:
        print(f'  ... still running ({elapsed}s elapsed)')

print(f'Experiment did not complete after {max_wait}s')
