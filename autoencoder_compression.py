"""
Autoencoder Compression Module
Compresses stock returns from N-dimensional space to 8-dimensional latent space.
Uses fundamentals as auxiliary input to guide learning.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler


class PortfolioAutoencoder(nn.Module):
    """
    Autoencoder that compresses return patterns with fundamental guidance.
    
    Architecture (6-layer deeper network for better capacity):
    - Encoder: Returns [N x T] + Fundamentals [N x F] -> 384 -> 256 -> 128 -> 64 -> 32 -> Latent [N x D]
    - Decoder: Latent [N x D] -> 32 -> 64 -> 128 -> 256 -> 384 -> Returns [N x T]
    
    Where D = latent_dim (typically 8-20)
    """
    
    def __init__(self, n_stocks: int, n_timesteps: int, n_features: int, latent_dim: int = 8):
        super().__init__()
        
        self.n_stocks = n_stocks
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Encoder: Returns + Fundamentals -> Latent
        # Deeper 6-layer architecture for higher capacity
        encoder_input_dim = n_timesteps + n_features
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: Latent -> Returns
        # Mirror architecture
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, n_timesteps)
        )
    
    def encode(self, returns: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Encode returns + features to latent space."""
        x = torch.cat([returns, features], dim=1)
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to returns."""
        return self.decoder(latent)
    
    def forward(self, returns: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        latent = self.encode(returns, features)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def prepare_training_data(
    log_returns: pd.DataFrame,
    fundamentals: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]:
    """
    Prepare and normalize data for autoencoder training.
    
    Args:
        log_returns: DataFrame [N stocks × T timesteps]
        fundamentals: DataFrame [N stocks × F features]
        
    Returns:
        Tuple of (returns_tensor, features_tensor, returns_scaler, features_scaler)
    """
    print("\nPreparing training data...")
    
    # Ensure alignment
    common_idx = sorted(set(log_returns.columns) & set(fundamentals.index))
    log_returns = log_returns[common_idx]
    fundamentals = fundamentals.loc[common_idx]
    
    # Transpose returns so stocks are rows
    returns_array = log_returns.T.values  # [N × T]
    features_array = fundamentals.values  # [N × F]
    
    # Standardize
    returns_scaler = StandardScaler()
    features_scaler = StandardScaler()
    
    returns_scaled = returns_scaler.fit_transform(returns_array)
    
    # Handle case when no fundamental features are available
    if features_array.shape[1] == 0:
        print("WARNING: No fundamental features available. Creating dummy features.")
        # Create simple dummy features: mean return and volatility
        mean_returns = returns_array.mean(axis=1, keepdims=True)
        std_returns = returns_array.std(axis=1, keepdims=True)
        features_array = np.hstack([mean_returns, std_returns])
        features_scaled = features_scaler.fit_transform(features_array)
    else:
        features_scaled = features_scaler.fit_transform(features_array)
    
    # Convert to tensors
    returns_tensor = torch.FloatTensor(returns_scaled)
    features_tensor = torch.FloatTensor(features_scaled)
    
    print(f"Training data shape:")
    print(f"  Returns: {returns_tensor.shape}")
    print(f"  Features: {features_tensor.shape}")
    
    return returns_tensor, features_tensor, returns_scaler, features_scaler


def train_autoencoder(
    returns_tensor: torch.Tensor,
    features_tensor: torch.Tensor,
    latent_dim: int = 8,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cpu'
) -> PortfolioAutoencoder:
    """
    Train the autoencoder.
    
    Args:
        returns_tensor: Normalized returns [N × T]
        features_tensor: Normalized features [N × F]
        latent_dim: Dimension of latent space
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
        
    Returns:
        Trained autoencoder model
    """
    # Fix random 
    # s for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Get dimensions FIRST
    n_stocks, n_timesteps = returns_tensor.shape
    n_features = features_tensor.shape[1]
    
    print("Training autoencoder neural network...")
    print(f"  - Architecture: {n_timesteps+n_features} -> 384 -> 256 -> 128 -> 64 -> 32 -> {latent_dim} -> 32 -> 64 -> 128 -> 256 -> 384 -> {n_timesteps}")
    print(f"  - Training samples: {n_stocks} stocks")
    print(f"  - Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
    print(f"  - Weight decay: 1e-5 (L2 regularization)")
    print(f"  - Gradient clipping: max_norm=1.0")
    
    # ========== DEBUG: INPUT DATA DIAGNOSTICS ==========
    print("\n" + "="*60)
    print("DEBUG: INPUT DATA STATISTICS")
    print("="*60)
    
    print(f"\nRETURNS TENSOR:")
    print(f"  Shape: {returns_tensor.shape}")
    print(f"  Min:   {returns_tensor.min().item():.6f}")
    print(f"  Max:   {returns_tensor.max().item():.6f}")
    print(f"  Mean:  {returns_tensor.mean().item():.6f}")
    print(f"  Std:   {returns_tensor.std().item():.6f}")
    print(f"  NaN:   {torch.isnan(returns_tensor).sum().item()}")
    print(f"  Inf:   {torch.isinf(returns_tensor).sum().item()}")
    
    print(f"\nFEATURES TENSOR:")
    print(f"  Shape: {features_tensor.shape}")
    print(f"  Min:   {features_tensor.min().item():.6f}")
    print(f"  Max:   {features_tensor.max().item():.6f}")
    print(f"  Mean:  {features_tensor.mean().item():.6f}")
    print(f"  Std:   {features_tensor.std().item():.6f}")
    print(f"  NaN:   {torch.isnan(features_tensor).sum().item()}")
    print(f"  Inf:   {torch.isinf(features_tensor).sum().item()}")
    
    # Check scale mismatch
    returns_scale = returns_tensor.std().item()
    features_scale = features_tensor.std().item()
    scale_ratio = returns_scale / (features_scale + 1e-10)
    print(f"\nSCALE COMPARISON:")
    print(f"  Returns std:  {returns_scale:.6f}")
    print(f"  Features std: {features_scale:.6f}")
    print(f"  Scale ratio:  {scale_ratio:.6f}")
    if scale_ratio > 10 or scale_ratio < 0.1:
        print("  [WARNING] Large scale mismatch detected! This could hurt training.")
    
    # Check for NaN/Inf in input data
    if torch.isnan(returns_tensor).any() or torch.isinf(returns_tensor).any():
        print("\n[CRITICAL] NaN or Inf detected in returns_tensor! Cleaning...")
        returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
        print("[CRITICAL] NaN or Inf detected in features_tensor! Cleaning...")
        features_tensor = torch.nan_to_num(features_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    print("="*60)
    
    # Create model
    model = PortfolioAutoencoder(n_stocks, n_timesteps, n_features, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    returns_tensor = returns_tensor.to(device)
    features_tensor = features_tensor.to(device)
    
    # Create dataset and dataloader for proper batching
    dataset = torch.utils.data.TensorDataset(returns_tensor, features_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("DEBUG: STARTING TRAINING LOOP")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Track diagnostics per epoch
        batch_losses = []
        grad_norms = []
        latent_stds = []
        recon_errors = []
        
        for batch_idx, (batch_returns, batch_features) in enumerate(dataloader):
            # Forward pass
            reconstructed, latent = model(batch_returns, batch_features)
            
            # Loss (reconstruction error)
            loss = criterion(reconstructed, batch_returns)
            
            # DEBUG: First batch of first epoch
            if epoch == 0 and batch_idx == 0:
                print(f"\nDEBUG: FIRST BATCH ANALYSIS")
                print(f"  Batch returns shape:  {batch_returns.shape}")
                print(f"  Batch returns range:  [{batch_returns.min().item():.6f}, {batch_returns.max().item():.6f}]")
                print(f"  Batch features shape: {batch_features.shape}")
                print(f"  Batch features range: [{batch_features.min().item():.6f}, {batch_features.max().item():.6f}]")
                print(f"  Latent shape:         {latent.shape}")
                print(f"  Latent range:         [{latent.min().item():.6f}, {latent.max().item():.6f}]")
                print(f"  Latent std:           {latent.std().item():.6f}")
                print(f"  Reconstructed shape:  {reconstructed.shape}")
                print(f"  Reconstructed range:  [{reconstructed.min().item():.6f}, {reconstructed.max().item():.6f}]")
                print(f"  Initial loss (MSE):   {loss.item():.6f}")
                
                # Check if model is outputting reasonable values
                if latent.std().item() < 0.01:
                    print("  [WARNING] Latent codes have very low variance! Model may be collapsing.")
                if reconstructed.std().item() < 0.01:
                    print("  [WARNING] Reconstructed outputs have very low variance! Model may be stuck.")
                if abs(reconstructed.mean().item() - batch_returns.mean().item()) > 1.0:
                    print(f"  [WARNING] Large mean mismatch! Recon mean: {reconstructed.mean().item():.6f} vs Target mean: {batch_returns.mean().item():.6f}")
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[CRITICAL] NaN loss at epoch {epoch+1}, batch {batch_idx}. Skipping.")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Calculate gradient norm BEFORE clipping
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())
            grad_norms.append(total_grad_norm)
            latent_stds.append(latent.std().item())
            
            # Per-sample reconstruction error
            per_sample_error = ((reconstructed - batch_returns) ** 2).mean(dim=1)
            recon_errors.extend(per_sample_error.detach().cpu().numpy())
        
        # Average loss for the epoch
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
        else:
            avg_loss = float('inf')
        
        # Detailed progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            avg_latent_std = np.mean(latent_stds) if latent_stds else 0.0
            loss_std = np.std(batch_losses) if len(batch_losses) > 1 else 0.0
            
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  Loss:            {avg_loss:.6f} (std: {loss_std:.6f})")
            print(f"  Gradient norm:   {avg_grad_norm:.6f}")
            print(f"  Latent std:      {avg_latent_std:.6f}")
            print(f"  Best loss so far: {best_loss:.6f}")
            
            # Check for training issues
            if avg_grad_norm < 1e-6:
                print(f"  [WARNING] Vanishing gradients detected! (norm: {avg_grad_norm:.2e})")
            if avg_latent_std < 0.01:
                print(f"  [WARNING] Latent collapse detected! (std: {avg_latent_std:.6f})")
            if epoch > 50 and abs(avg_loss - best_loss) < 1e-6:
                print(f"  [INFO] Loss plateaued. Consider early stopping or LR decay.")
        
        if avg_loss < best_loss and not np.isnan(avg_loss) and not np.isinf(avg_loss):
            best_loss = avg_loss
    
    # ========== DEBUG: FINAL DIAGNOSTICS ==========
    print("\n" + "="*60)
    print("DEBUG: FINAL TRAINING SUMMARY")
    print("="*60)
    print(f"\nFinal best loss: {best_loss:.6f}")
    
    # Run final evaluation pass to check reconstruction quality
    model.eval()
    with torch.no_grad():
        full_reconstructed, full_latent = model(returns_tensor, features_tensor)
        final_loss = criterion(full_reconstructed, returns_tensor).item()
        
        print(f"\nFull dataset reconstruction:")
        print(f"  Final MSE loss:        {final_loss:.6f}")
        print(f"  Target (returns) mean: {returns_tensor.mean().item():.6f}")
        print(f"  Target (returns) std:  {returns_tensor.std().item():.6f}")
        print(f"  Recon mean:            {full_reconstructed.mean().item():.6f}")
        print(f"  Recon std:             {full_reconstructed.std().item():.6f}")
        print(f"  Latent mean:           {full_latent.mean().item():.6f}")
        print(f"  Latent std:            {full_latent.std().item():.6f}")
        
        # Element-wise error analysis
        abs_error = (full_reconstructed - returns_tensor).abs()
        print(f"\nReconstruction error breakdown:")
        print(f"  Mean abs error:   {abs_error.mean().item():.6f}")
        print(f"  Median abs error: {abs_error.median().item():.6f}")
        print(f"  Max abs error:    {abs_error.max().item():.6f}")
        print(f"  95th percentile:  {torch.quantile(abs_error, 0.95).item():.6f}")
        
        # Check for variance preservation
        original_var = returns_tensor.var(dim=0).mean().item()
        recon_var = full_reconstructed.var(dim=0).mean().item()
        var_ratio = recon_var / (original_var + 1e-10)
        print(f"\nVariance preservation:")
        print(f"  Original variance: {original_var:.6f}")
        print(f"  Recon variance:    {recon_var:.6f}")
        print(f"  Ratio:             {var_ratio:.6f}")
        if var_ratio < 0.5:
            print(f"  [WARNING] Model is under-fitting! Only capturing {var_ratio*100:.1f}% of variance.")
        elif var_ratio > 1.5:
            print(f"  [WARNING] Model may be overfitting or amplifying noise.")
        
        # Correlation check
        flat_target = returns_tensor.flatten()
        flat_recon = full_reconstructed.flatten()
        corr = torch.corrcoef(torch.stack([flat_target, flat_recon]))[0, 1].item()
        print(f"\nTarget-Reconstruction correlation: {corr:.6f}")
        if corr < 0.5:
            print(f"  [CRITICAL] Very low correlation! Model is not learning the data.")
        elif corr < 0.7:
            print(f"  [WARNING] Moderate correlation. Model could be better.")
        else:
            print(f"  [OK] Good correlation.")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUGGESTIONS:")
    print("="*60)
    
    # Provide actionable recommendations
    if best_loss > 1.0:
        print("- Loss > 1.0 suggests poor reconstruction. Check:")
        print("  1. Data normalization (should have mean=0, std=1)")
        print("  2. Feature scale mismatch between returns and fundamentals")
        print("  3. Learning rate (try 0.0001 or 0.01)")
    
    if corr < 0.7:
        print("- Low correlation suggests:")
        print("  1. Model capacity too small (latent_dim=8 may be too compressed)")
        print("  2. Insufficient training (try more epochs)")
        print("  3. Bad initialization (already fixed with seed=42)")
    
    if var_ratio < 0.5:
        print("- Low variance preservation suggests:")
        print("  1. Model is outputting near-constant values")
        print("  2. Activation functions may be saturating")
        print("  3. Learning rate too high or too low")
    
    print("="*60)
    
    return model


def project_to_latent_space(
    model: PortfolioAutoencoder,
    returns_tensor: torch.Tensor,
    features_tensor: torch.Tensor,
    mu: pd.Series,
    Sigma: pd.DataFrame,
    tickers: list,
    device: str = 'cpu'
) -> Dict:
    """
    Project mu and Sigma to latent space.
    
    Args:
        model: Trained autoencoder
        returns_tensor: Normalized returns [N × T]
        features_tensor: Normalized features [N × F]
        mu: Expected returns [N]
        Sigma: Covariance matrix [N × N]
        tickers: List of tickers
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with latent space projections
    """
    print("\n" + "="*60)
    print("PROJECTING TO LATENT SPACE")
    print("="*60)
    
    model.eval()
    returns_tensor = returns_tensor.to(device)
    features_tensor = features_tensor.to(device)
    
    with torch.no_grad():
        # Get encoder weights (maps returns to latent)
        latent_codes = model.encode(returns_tensor, features_tensor)
        
        # Move back to CPU
        latent_codes_np = latent_codes.cpu().numpy()  # [N × 8]
        
        # Check for NaN/Inf in latent codes
        if np.isnan(latent_codes_np).any() or np.isinf(latent_codes_np).any():
            print("ERROR: NaN or Inf detected in latent codes! This should not happen.")
            print("Replacing with small random values.")
            latent_codes_np = np.random.randn(*latent_codes_np.shape) * 0.01
        
        # Project mu to latent space
        # mu_latent = W^T @ mu where W are the "factor loadings"
        # We approximate this by taking weighted average using latent codes
        mu_array = mu.values.reshape(-1, 1)  # [N × 1]
        
        # Check for NaN/Inf in mu
        if np.isnan(mu_array).any() or np.isinf(mu_array).any():
            print("WARNING: NaN or Inf detected in mu! Replacing with zeros.")
            mu_array = np.nan_to_num(mu_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        mu_latent = latent_codes_np.T @ mu_array  # [8 × 1]
        mu_latent = mu_latent.flatten()
        
        # Final check on mu_latent
        if np.isnan(mu_latent).any() or np.isinf(mu_latent).any():
            print("WARNING: NaN in mu_latent after projection! Using zeros.")
            mu_latent = np.zeros(latent_codes_np.shape[1])
        
        # Project Sigma to latent space
        # Sigma_latent = W^T @ Sigma @ W
        # Using latent codes as factor loadings
        Sigma_array = Sigma.values  # [N × N]
        
        # Check for NaN/Inf in Sigma
        if np.isnan(Sigma_array).any() or np.isinf(Sigma_array).any():
            print("WARNING: NaN or Inf detected in Sigma! Cleaning...")
            Sigma_array = np.nan_to_num(Sigma_array, nan=0.0, posinf=1.0, neginf=-1.0)
            # Make it positive definite
            Sigma_array = Sigma_array @ Sigma_array.T / len(Sigma_array)
            Sigma_array += np.eye(len(Sigma_array)) * 0.01
        
        # Normalize latent codes to prevent numerical issues
        # For large N (500+), use more aggressive normalization
        n_stocks = latent_codes_np.shape[0]
        latent_codes_norm = latent_codes_np / (np.linalg.norm(latent_codes_np, axis=0, keepdims=True) + 1e-8)
        
        # Scale down for large portfolios to prevent overflow
        if n_stocks > 100:
            scaling_factor = np.sqrt(n_stocks / 100.0)
            latent_codes_norm = latent_codes_norm / scaling_factor
            print(f"Large portfolio detected ({n_stocks} stocks). Applying scaling factor: {scaling_factor:.2f}")
        
        Sigma_latent = latent_codes_norm.T @ Sigma_array @ latent_codes_norm  # [8 × 8]
        
        # Check for NaN in Sigma_latent
        if np.isnan(Sigma_latent).any() or np.isinf(Sigma_latent).any():
            print("ERROR: NaN in Sigma_latent after projection! Using identity.")
            n_latent = latent_codes_np.shape[1]
            Sigma_latent = np.eye(n_latent) * 0.01
            eigenvalues = np.ones(n_latent) * 0.01
        else:
            # Ensure Sigma is symmetric and positive definite
            Sigma_latent = (Sigma_latent + Sigma_latent.T) / 2
            
            # Add stronger regularization for large portfolios
            n_latent = latent_codes_np.shape[1]  # Should be 8
            reg_strength = 1e-3 if n_stocks > 100 else 1e-4
            Sigma_latent += np.eye(n_latent) * reg_strength
            
            try:
                eigenvalues = np.linalg.eigvalsh(Sigma_latent)
                print(f"Sigma_latent eigenvalues: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")
                
                if eigenvalues.min() < 1e-6:
                    print(f"Warning: Adding extra regularization (min eigenvalue: {eigenvalues.min():.2e})")
                    Sigma_latent += np.eye(n_latent) * 1e-2
                    eigenvalues = np.linalg.eigvalsh(Sigma_latent)
            except np.linalg.LinAlgError:
                print("ERROR: Eigenvalue computation failed. Using identity matrix.")
                Sigma_latent = np.eye(n_latent) * 0.01
                eigenvalues = np.ones(n_latent) * 0.01
    
    print(f"Latent space dimensions: {n_latent}")
    print(f"Original dimensions: {len(mu)}")
    print(f"Compression ratio: {len(mu) / n_latent:.1f}x")
    print(f"mu_latent range: [{mu_latent.min():.4f}, {mu_latent.max():.4f}]")
    print(f"Sigma_latent final eigenvalues: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")
    
    return {
        'mu_latent': mu_latent,
        'Sigma_latent': Sigma_latent,
        'latent_codes': latent_codes_np,
        'n_latent': n_latent,
        'tickers': tickers
    }


def decode_portfolio_weights(
    model: PortfolioAutoencoder,
    qaoa_solution: np.ndarray,
    latent_codes: np.ndarray,
    tickers: list,
    mu_latent: np.ndarray = None,
    device: str = 'cpu'
) -> Dict:
    """
    Decode QAOA solution from latent space to actual stock weights.
    
    Args:
        model: Trained autoencoder
        qaoa_solution: Binary solution from QAOA [8]
        latent_codes: Latent representations [N × 8]
        tickers: List of tickers
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with decoded portfolio
    """
    print("Decoding latent solution to stock weights...")
    
    # Selected latent dimensions
    selected_dims = np.where(qaoa_solution == 1)[0]
    print(f"  - Selected latent dimensions: {selected_dims.tolist()} ({len(selected_dims)}/{len(qaoa_solution)})")
    print(f"  - Mapping {len(tickers)} stocks to portfolio weights...")
    
    # Score each stock by return-weighted strength in selected dimensions
    # Weight by mu_latent (expected returns) in selected dims for better Sharpe
    if mu_latent is not None and len(selected_dims) > 0:
        # Get mu values for selected dimensions
        mu_selected = mu_latent[selected_dims]
        # Weight latent codes by expected returns in selected dims
        scores = (latent_codes[:, selected_dims] * mu_selected).sum(axis=1)
        # Only keep positive scores (stocks with positive expected return contribution)
        scores = np.maximum(scores, 0)
    else:
        # Fallback: use squared values for concentration
        scores = (latent_codes[:, selected_dims] ** 2).sum(axis=1)
        scores = np.maximum(scores, 0)
    
    # Apply power function for concentration (higher power = more concentration)
    # Cube the scores to strongly favor top stocks (creates high concentration)
    scores_concentrated = scores ** 3
    
    # Normalize to get weights
    if scores_concentrated.sum() > 0:
        weights = scores_concentrated / scores_concentrated.sum()
    else:
        # Fallback: equal weights if all scores zero
        weights = np.ones(len(scores)) / len(scores)
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Score': scores
    }).sort_values('Weight', ascending=False)
    
    print(f"\n  Top 10 holdings by weight:")
    for idx, row in portfolio_df.head(10).iterrows():
        print(f"    {row['Ticker']:<6} {row['Weight']*100:>6.2f}%")
    
    print(f"\n  Portfolio concentration:")
    print(f"    - Stocks with weight > 1%:  {(weights > 0.01).sum()}")
    print(f"    - Stocks with weight > 5%:  {(weights > 0.05).sum()}")
    print(f"    - Largest position:         {weights.max()*100:.2f}%")
    print(f"    - Total weight (check):     {weights.sum():.6f}")
    
    return {
        'weights': weights,
        'tickers': tickers,
        'portfolio_df': portfolio_df,
        'selected_dimensions': selected_dims
    }


def run_autoencoder_compression(
    log_returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    mu: pd.Series,
    Sigma: pd.DataFrame,
    latent_dim: int = 8,
    epochs: int = 200,
    device: str = 'cpu'
) -> Dict:
    """
    Complete autoencoder compression pipeline.
    
    Args:
        log_returns: Log returns DataFrame
        fundamentals: Cleaned fundamentals DataFrame
        mu: Expected returns
        Sigma: Covariance matrix
        latent_dim: Latent dimension (default 8)
        epochs: Training epochs
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with all compression outputs
    """
    print("\n" + "="*60)
    print("AUTOENCODER COMPRESSION PIPELINE")
    print("="*60)
    
    # Prepare data
    returns_tensor, features_tensor, returns_scaler, features_scaler = prepare_training_data(
        log_returns, fundamentals
    )
    
    # Train autoencoder
    model = train_autoencoder(
        returns_tensor, features_tensor,
        latent_dim=latent_dim,
        epochs=epochs,
        device=device
    )
    
    # Project to latent space
    tickers = log_returns.columns.tolist()
    latent_projection = project_to_latent_space(
        model, returns_tensor, features_tensor,
        mu, Sigma, tickers, device
    )
    
    return {
        'model': model,
        'returns_scaler': returns_scaler,
        'features_scaler': features_scaler,
        'returns_tensor': returns_tensor,
        'features_tensor': features_tensor,
        **latent_projection
    }


if __name__ == "__main__":
    print("Autoencoder compression module loaded.")
    print("Use run_autoencoder_compression() to compress your portfolio data.")

