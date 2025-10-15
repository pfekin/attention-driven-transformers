"""
Minimal PatchTST Implementation with Summation-Based Attention + Other Architectures

Compares:
1. Baseline PatchTST with standard attention (custom implementation)
2. Hybrid PatchTST with summation attention (trained from scratch)
3. N-BEATS (custom implementation)
4. Temporal Fusion Transformer (custom implementation)
5. TCN - Temporal Convolutional Network (custom implementation)

Benchmark on ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Darts datasets
from darts.datasets import TrafficDataset, WeatherDataset, ElectricityDataset
from darts.datasets import ETTh1Dataset, ETTh2Dataset, ETTm1Dataset, ETTm2Dataset

import warnings
warnings.filterwarnings('ignore')

# Data loading

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def timeseries_to_dataframe(ts):
    """Convert Darts TimeSeries to pandas DataFrame, handling 3D arrays"""
    if hasattr(ts, "pd_dataframe"):
        df = ts.pd_dataframe()
        if isinstance(df, pd.DataFrame):
            return df
        else:
            arr = ts.all_values(copy=False)
            if arr.ndim == 3:
                arr = arr.squeeze(-1)
            cols = [f"var_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
    elif hasattr(ts, "all_values"):
        arr = ts.all_values(copy=False)
        if arr.ndim == 3:
            arr = arr.squeeze(-1)
        cols = [f"var_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    else:
        raise AttributeError("Unsupported Darts TimeSeries object. Please update Darts.")


def load_dataset_general(name, seq_len, pred_len, batch_size=32,
                         val_ratio=0.2, test_ratio=0.2, max_vars=None):
    """Generalized loader using Darts datasets"""
    dataset_map = {
        "ETTh1": ETTh1Dataset,
        "ETTh2": ETTh2Dataset,
        "ETTm1": ETTm1Dataset,
        "ETTm2": ETTm2Dataset,
        "Traffic": TrafficDataset,
        "Weather": WeatherDataset,
        "Electricity": ElectricityDataset
    }
    
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}")
    
    ts = dataset_map[name]().load()
    df = timeseries_to_dataframe(ts)

    if max_vars is not None and df.shape[1] > max_vars:
        df = df.iloc[:, :max_vars]

    values = df.values
    T, n_vars = values.shape

    # Train/val/test split
    n_test = int(T * test_ratio)
    n_val = int(T * val_ratio)
    n_train = T - n_val - n_test

    train_vals = values[:n_train]
    val_vals = values[n_train:n_train+n_val]
    test_vals = values[n_train+n_val:]

    # Standardize
    scaler = StandardScaler()
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals)
    val_scaled = scaler.transform(val_vals)
    test_scaled = scaler.transform(test_vals)

    # Handle NaN/Inf values after standardization
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Wrap in PyTorch datasets
    train_ds = TimeSeriesDataset(train_scaled, seq_len, pred_len)
    val_ds = TimeSeriesDataset(val_scaled, seq_len, pred_len)
    test_ds = TimeSeriesDataset(test_scaled, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_vars, scaler


# ============================================================================
# PatchTST Models
# ============================================================================

class Patching(nn.Module):
    """Convert time series into patches"""
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = torch.zeros(batch_size, n_vars, num_patches, self.patch_len, device=x.device)
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches[:, :, i, :] = x[:, start:end, :].transpose(1, 2)
        patches = patches.reshape(batch_size * n_vars, num_patches, self.patch_len)
        return patches, n_vars, num_patches


class StandardAttentionBlock(nn.Module):
    """Standard transformer block with multi-head self-attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class ProjectionBlock(nn.Module):
    """Summation-based attention block"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        summed = self.proj(x)
        x = self.norm1(x + summed)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class BaselinePatchTST(nn.Module):
    """PatchTST with standard attention"""
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.patching = Patching(patch_len, stride)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        num_patches = (seq_len - patch_len) // stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.blocks = nn.ModuleList([
            StandardAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model * num_patches, pred_len)
        self.n_vars = n_vars
        self.pred_len = pred_len

    def forward(self, x):
        batch_size = x.shape[0]
        patches, n_vars, num_patches = self.patching(x)
        x = self.patch_embedding(patches)
        x = x + self.pos_encoding
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch_size * n_vars, -1)
        x = self.head(x)
        x = x.reshape(batch_size, n_vars, self.pred_len).transpose(1, 2)
        return x


class HybridPatchTST(nn.Module):
    """PatchTST with linear projection + final standard attention"""
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16, stride=8,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.patching = Patching(patch_len, stride)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        num_patches = (seq_len - patch_len) // stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.summation_blocks = nn.ModuleList([
            ProjectionBlock(d_model, d_ff, dropout)
            for _ in range(n_layers - 1)
        ])
        self.final_attention = StandardAttentionBlock(d_model, n_heads, d_ff, dropout)
        self.head = nn.Linear(d_model * num_patches, pred_len)
        self.n_vars = n_vars
        self.pred_len = pred_len

    def forward(self, x):
        batch_size = x.shape[0]
        patches, n_vars, num_patches = self.patching(x)
        x = self.patch_embedding(patches)
        x = x * self.pos_encoding
        for block in self.summation_blocks:
            x = block(x)
        x = self.final_attention(x)
        x = x.reshape(batch_size * n_vars, -1)
        x = self.head(x)
        x = x.reshape(batch_size, n_vars, self.pred_len).transpose(1, 2)
        return x


# ============================================================================
# N-BEATS Implementation
# ============================================================================

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size, num_layers):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)
        
    def forward(self, x):
        h = self.layers(x)
        return self.theta_b(h), self.theta_f(h)


class NBeats(nn.Module):
    def __init__(self, n_vars, seq_len, pred_len, hidden_size=128, num_blocks=3, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        
        # Shared blocks across all variables
        self.blocks = nn.ModuleList([
            NBeatsBlock(seq_len, max(seq_len, pred_len), hidden_size, num_layers)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch_size = x.shape[0]
        
        # Process each variable independently
        residuals = x.transpose(1, 2)  # (batch, n_vars, seq_len)
        forecast = torch.zeros(batch_size, self.n_vars, self.pred_len, device=x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast[:, :, :self.seq_len]
            forecast = forecast + block_forecast[:, :, :self.pred_len]
        
        return forecast.transpose(1, 2)  # (batch, pred_len, n_vars)

# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=30):
    """Train model and return best validation loss"""
    best_val_loss = float("inf")
    max_grad_norm = 1.0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for past_values, future_values in train_loader:
            past_values = past_values.to(device)
            future_values = future_values.to(device)

            optimizer.zero_grad()
            outputs = model(past_values)
            loss = criterion(outputs, future_values)

            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected at epoch {epoch+1}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for past_values, future_values in val_loader:
                past_values = past_values.to(device)
                future_values = future_values.to(device)
                outputs = model(past_values)
                loss = criterion(outputs, future_values)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)

        if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
            print(f"  Training diverged at epoch {epoch+1}. Stopping early.")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Best={best_val_loss:.4f}")

    return best_val_loss


def evaluate_model(model, loader, device):
    """Compute MSE and MAE on a given DataLoader."""
    model.eval()
    mse_loss = 0.0
    mae_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            mse_loss += torch.nn.functional.mse_loss(y_pred, y, reduction="sum").item()
            mae_loss += torch.nn.functional.l1_loss(y_pred, y, reduction="sum").item()
            count += y.numel()
    return mse_loss / count, mae_loss / count


def measure_inference_speed(model, loader, device, warmup=2, reps=5):
    """Measure inference speed in samples/sec."""
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= warmup:
                break
            _ = model(x.to(device))

    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for r in range(reps):
            for x, _ in loader:
                x = x.to(device)
                batch_size = x.size(0)
                start = time.time()
                _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time += time.time() - start
                total_samples += batch_size
    return total_samples / total_time


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(dataset_name, train_loader, val_loader, test_loader, n_vars, config, device):
    print("\n" + "="*70)
    print(f"Dataset: {dataset_name}")
    print("="*70)

    results = {'dataset': dataset_name}
    criterion = nn.MSELoss()

    models_to_test = [
        ("Baseline PatchTST", lambda: BaselinePatchTST(
            n_vars=n_vars, seq_len=config['seq_len'], pred_len=config['pred_len'],
            patch_len=config['patch_len'], stride=config['stride'],
            d_model=config['d_model'], n_heads=config['n_heads'],
            n_layers=config['n_layers'], d_ff=config['d_ff'], dropout=config['dropout']
        )),
        ("Hybrid PatchTST", lambda: HybridPatchTST(
            n_vars=n_vars, seq_len=config['seq_len'], pred_len=config['pred_len'],
            patch_len=config['patch_len'], stride=config['stride'],
            d_model=config['d_model'], n_heads=config['n_heads'],
            n_layers=config['n_layers'], d_ff=config['d_ff'], dropout=config['dropout']
        )),
        ("N-BEATS", lambda: NBeats(
            n_vars=n_vars, seq_len=config['seq_len'], pred_len=config['pred_len'],
            hidden_size=config['d_model'], num_blocks=3, num_layers=4
        ))
    ]

    for idx, (model_name, model_fn) in enumerate(models_to_test, 1):
        print(f"\n[{idx}/5] Training {model_name}...")
        
        model = model_fn().to(device)
        print(f"{model_name} params: {sum(p.numel() for p in model.parameters()):,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        start_time = time.time()
        val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, device, config['n_epochs'])
        train_time = time.time() - start_time
        
        test_mse, test_mae = evaluate_model(model, test_loader, device)
        speed = measure_inference_speed(model, test_loader, device)
        
        prefix = model_name.lower().replace(' ', '_').replace('-', '_')
        results.update({
            f'{prefix}_mse': test_mse,
            f'{prefix}_mae': test_mae,
            f'{prefix}_speed': speed,
            f'{prefix}_time': train_time,
            f'{prefix}_val': val_loss
        })
        
        print(f"{model_name} -> MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, Speed: {speed:.1f} samples/s")
        
        # Clean up
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    CONFIG = {
        'seq_len': 512,
        'pred_len': 96,
        'patch_len': 16,
        'stride': 8,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 3,
        'd_ff': 256,
        'batch_size': 32,
        'n_epochs': 10,
        'lr': 1e-4,
        'dropout': 0.15
    }

    datasets_to_run = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Traffic", "Electricity"]

    results = []
    for dset in datasets_to_run:
        max_vars = 100 if (dset == "Traffic" or dset == "Electricity") else None
        train_loader, val_loader, test_loader, n_vars, _ = load_dataset_general(
            dset, CONFIG['seq_len'], CONFIG['pred_len'],
            batch_size=CONFIG['batch_size'], max_vars=max_vars
        )
        result = run_experiment(dset, train_loader, val_loader, test_loader, n_vars, CONFIG, device)
        results.append(result)

    # Print summary table
    print("\n" + "="*70)
    print("Final benchmark summary")
    print("="*70)
    for r in results:
        print(f"\n{r['dataset']}:")
        print(f"  Baseline PatchTST: MSE={r['baseline_patchtst_mse']:.4f}, MAE={r['baseline_patchtst_mae']:.4f}, Speed={r['baseline_patchtst_speed']:.1f} samples/s")
        print(f"  Hybrid PatchTST:   MSE={r['hybrid_patchtst_mse']:.4f}, MAE={r['hybrid_patchtst_mae']:.4f}, Speed={r['hybrid_patchtst_speed']:.1f} samples/s")
        print(f"  N-BEATS:           MSE={r['n_beats_mse']:.4f}, MAE={r['n_beats_mae']:.4f}, Speed={r['n_beats_speed']:.1f} samples/s")
        
        improvement = ((r['baseline_patchtst_mse'] - r['hybrid_patchtst_mse']) / r['baseline_patchtst_mse']) * 100
        speedup = r['hybrid_patchtst_speed'] / r['baseline_patchtst_speed']
        print(f"  -> Hybrid vs Baseline: MSE {improvement:+.1f}%, Speed {speedup:.2f}x")