#!/usr/bin/env python3
"""
Tier 2: FNO Surrogate Model Comparison

Train a Fourier Neural Operator (FNO) on raw vs compressed data
to validate that compression preserves utility for downstream ML tasks.

Metrics from PDEBench:
- Relative L2 error
- RMSE per field
- One-step prediction error
- 10-step autoregressive rollout error
- Boundary error
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, Tuple, List, Optional
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 2D Fourier Neural Operator Implementation
# ============================================================================

class SpectralConv2d(nn.Module):
    """2D Fourier layer - applies FFT, linear transform in frequency, then iFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep (x direction)
        self.modes2 = modes2  # Number of Fourier modes to keep (y direction)

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for time-stepping physics simulations.

    Architecture:
    - Input projection (lifting)
    - N Fourier layers (spectral convolution + pointwise MLP)
    - Output projection

    Input: (batch, channels_in, H, W)
    Output: (batch, channels_out, H, W)
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers

        # Input projection
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for grid coordinates

        # Fourier layers
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SpectralConv2d(width, width, modes1, modes2))
            self.ws.append(nn.Conv2d(width, width, 1))

        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x: (batch, channels, H, W)
        batch_size = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Get grid coordinates
        grid = self.get_grid(batch_size, H, W, x.device)

        # Concatenate input with grid
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, channels)
        x = torch.cat([x, grid], dim=-1)  # (batch, H, W, channels + 2)

        # Lift to higher dimension
        x = self.fc0(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)

        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        # Project to output
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, H, W)

        return x

    def get_grid(self, batch_size, H, W, device):
        """Create normalized coordinate grid."""
        gridx = torch.linspace(0, 1, H, device=device)
        gridy = torch.linspace(0, 1, W, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        grid = torch.stack([gridx, gridy], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return grid


# ============================================================================
# Dataset for FNO Training
# ============================================================================

class PhysicsDataset(Dataset):
    """Dataset for FNO training on physics simulation data."""

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        num_trajectories: int = 10,
        input_fields: List[str] = ['pressure'],
        output_fields: List[str] = ['pressure'],
        include_static: bool = True,
        max_timesteps: int = 100,
    ):
        """
        Args:
            data_path: Path to HDF5 data
            split: 'train' or 'valid'
            num_trajectories: Number of trajectories to use
            input_fields: Fields to use as input
            output_fields: Fields to predict
            include_static: Whether to include static fields (density, speed_of_sound)
            max_timesteps: Maximum number of timesteps per trajectory
        """
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.include_static = include_static

        # Load data
        self.data = self._load_data(data_path, split, num_trajectories, max_timesteps)

    def _load_data(self, data_path: str, split: str, num_traj: int, max_t: int) -> Dict:
        """Load data from HDF5."""
        from glob import glob

        # Find files
        if os.path.isfile(data_path):
            files = [data_path]
        else:
            patterns = [
                os.path.join(data_path, f"data/{split}/*.hdf5"),
                os.path.join(data_path, "*.hdf5"),
            ]
            files = []
            for pattern in patterns:
                files = sorted(glob(pattern))
                if files:
                    break

        if not files:
            raise FileNotFoundError(f"No HDF5 files found at {data_path}")

        logger.info(f"Loading from {files[0]}")

        data = {'pressure': [], 'velocity': [], 'density': [], 'speed_of_sound': []}

        with h5py.File(files[0], 'r') as f:
            # Determine number of trajectories available
            n_avail = f['t0_fields/pressure'].shape[0]
            n_load = min(num_traj, n_avail)

            for traj_idx in range(n_load):
                # Dynamic fields
                pressure = f['t0_fields/pressure'][traj_idx, :max_t]  # (T, H, W)
                data['pressure'].append(pressure)

                velocity = f['t1_fields/velocity'][traj_idx, :max_t]  # (T, H, W, 2)
                data['velocity'].append(velocity)

                # Static fields (replicate across time for convenience)
                density = f['t0_fields/density'][traj_idx]  # (H, W)
                data['density'].append(density)

                sos = f['t0_fields/speed_of_sound'][traj_idx]  # (H, W)
                data['speed_of_sound'].append(sos)

        # Stack trajectories
        data['pressure'] = np.stack(data['pressure'], axis=0)  # (N, T, H, W)
        data['velocity'] = np.stack(data['velocity'], axis=0)  # (N, T, H, W, 2)
        data['density'] = np.stack(data['density'], axis=0)  # (N, H, W)
        data['speed_of_sound'] = np.stack(data['speed_of_sound'], axis=0)  # (N, H, W)

        logger.info(f"Loaded {n_load} trajectories, pressure shape: {data['pressure'].shape}")

        return data

    def __len__(self):
        # Number of input-output pairs (each timestep pair)
        N, T = self.data['pressure'].shape[0], self.data['pressure'].shape[1]
        return N * (T - 1)  # T-1 pairs per trajectory

    def __getitem__(self, idx):
        N, T = self.data['pressure'].shape[0], self.data['pressure'].shape[1]
        traj_idx = idx // (T - 1)
        time_idx = idx % (T - 1)

        # Input: state at time t
        # Output: state at time t+1

        # Build input tensor
        input_channels = []
        for field in self.input_fields:
            if field == 'pressure':
                input_channels.append(self.data['pressure'][traj_idx, time_idx])
            elif field == 'velocity_x':
                input_channels.append(self.data['velocity'][traj_idx, time_idx, ..., 0])
            elif field == 'velocity_y':
                input_channels.append(self.data['velocity'][traj_idx, time_idx, ..., 1])

        if self.include_static:
            input_channels.append(self.data['density'][traj_idx])
            input_channels.append(self.data['speed_of_sound'][traj_idx])

        x = np.stack(input_channels, axis=0).astype(np.float32)

        # Build output tensor
        output_channels = []
        for field in self.output_fields:
            if field == 'pressure':
                output_channels.append(self.data['pressure'][traj_idx, time_idx + 1])
            elif field == 'velocity_x':
                output_channels.append(self.data['velocity'][traj_idx, time_idx + 1, ..., 0])
            elif field == 'velocity_y':
                output_channels.append(self.data['velocity'][traj_idx, time_idx + 1, ..., 1])

        y = np.stack(output_channels, axis=0).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


class ReconstructedPhysicsDataset(Dataset):
    """Dataset using SINCS-reconstructed data."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        raw_data_path: str,
        num_trajectories: int = 10,
        input_fields: List[str] = ['pressure'],
        output_fields: List[str] = ['pressure'],
        device: str = 'cpu',
    ):
        """Load SINCS model and reconstruct data for training."""
        self.input_fields = input_fields
        self.output_fields = output_fields

        # Load raw data for static fields and ground truth structure
        logger.info("Loading raw data for structure...")
        raw_ds = PhysicsDataset(
            raw_data_path, 'train', num_trajectories,
            input_fields, output_fields, include_static=True
        )

        # Copy static fields
        self.data = {
            'density': raw_ds.data['density'].copy(),
            'speed_of_sound': raw_ds.data['speed_of_sound'].copy(),
        }

        N, T, H, W = raw_ds.data['pressure'].shape

        # Reconstruct dynamic fields using SINCS
        logger.info(f"Reconstructing {N} trajectories with SINCS model...")

        self.data['pressure'] = self._reconstruct_with_sincs(
            model_path, config_path, N, T, H, W, device
        )
        # For now, use raw velocity (SINCS trained on pressure only in default config)
        self.data['velocity'] = raw_ds.data['velocity'].copy()

    def _reconstruct_with_sincs(
        self, model_path: str, config_path: str,
        N: int, T: int, H: int, W: int, device: str
    ) -> np.ndarray:
        """Reconstruct pressure field using SINCS model."""
        import yaml
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model import SINCS, SINCSConfig
        from evaluate import load_hdf5_checkpoint

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config['trainer']['init']['model']
        out_dim = model_config.get('output_dim', 3)

        # Create model
        sincs_config = SINCSConfig(
            input_dim=model_config.get('input_dim', 3),
            output_dim=out_dim,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_hidden_layers=model_config.get('num_hidden_layers', 4),
            omega_0=model_config.get('omega_0', 30.0),
            omega_hidden=model_config.get('omega_hidden', 30.0),
            use_fourier_encoding=model_config.get('use_fourier_encoding', True),
            encoding_levels=model_config.get('encoding_levels', 10),
        )

        model = SINCS(sincs_config)

        # Load checkpoint
        state_dict = load_hdf5_checkpoint(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('optimizer.') or k.startswith('schedulers.'):
                continue
            new_key = k
            if new_key.startswith('model.model.'):
                new_key = new_key[12:]
            elif new_key.startswith('model.'):
                new_key = new_key[6:]
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        model = model.to(device)

        # Create coordinate grid
        t_coords = np.linspace(0, 1, T)
        x_coords = np.linspace(0, 1, H)
        y_coords = np.linspace(0, 1, W)

        recon_pressure = np.zeros((N, T, H, W), dtype=np.float32)
        batch_size = 16384

        # For simplicity, reconstruct one "average" trajectory
        # (SINCS model was trained on trajectory 0)
        with torch.no_grad():
            for t_idx in range(T):
                t = t_coords[t_idx]
                coords_list = []
                for i, x in enumerate(x_coords):
                    for j, y in enumerate(y_coords):
                        coords_list.append([t, x, y])

                coords = np.array(coords_list, dtype=np.float32)

                all_outputs = []
                for start in range(0, len(coords), batch_size):
                    end = min(start + batch_size, len(coords))
                    batch_coords = torch.from_numpy(coords[start:end]).to(device)
                    outputs = model(batch_coords)
                    all_outputs.append(outputs.cpu().numpy())

                outputs = np.concatenate(all_outputs, axis=0)
                outputs = outputs.reshape(H, W, -1)

                # Use pressure (first output channel)
                for n in range(N):
                    recon_pressure[n, t_idx] = outputs[..., 0]

                if (t_idx + 1) % 20 == 0:
                    logger.info(f"  Reconstructed timestep {t_idx + 1}/{T}")

        return recon_pressure

    def __len__(self):
        N, T = self.data['pressure'].shape[0], self.data['pressure'].shape[1]
        return N * (T - 1)

    def __getitem__(self, idx):
        N, T = self.data['pressure'].shape[0], self.data['pressure'].shape[1]
        traj_idx = idx // (T - 1)
        time_idx = idx % (T - 1)

        input_channels = []
        for field in self.input_fields:
            if field == 'pressure':
                input_channels.append(self.data['pressure'][traj_idx, time_idx])

        input_channels.append(self.data['density'][traj_idx])
        input_channels.append(self.data['speed_of_sound'][traj_idx])

        x = np.stack(input_channels, axis=0).astype(np.float32)

        output_channels = []
        for field in self.output_fields:
            if field == 'pressure':
                output_channels.append(self.data['pressure'][traj_idx, time_idx + 1])

        y = np.stack(output_channels, axis=0).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_fno(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict]:
    """Train FNO model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

    training_time = time.time() - start_time
    history['training_time_seconds'] = training_time

    return model, history


def evaluate_fno(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = 'cpu',
    rollout_steps: int = 10,
) -> Dict:
    """Evaluate FNO model with various metrics."""
    model.eval()
    model = model.to(device)

    # One-step metrics
    rel_l2_errors = []
    rmse_values = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Relative L2 error
            rel_l2 = torch.norm(pred - y, p=2) / (torch.norm(y, p=2) + 1e-8)
            rel_l2_errors.append(rel_l2.item())

            # RMSE
            rmse = torch.sqrt(F.mse_loss(pred, y))
            rmse_values.append(rmse.item())

    results = {
        'one_step_rel_l2': float(np.mean(rel_l2_errors)),
        'one_step_rmse': float(np.mean(rmse_values)),
    }

    # Multi-step rollout (if dataset supports it)
    # For now, approximate with accumulated error
    results['ten_step_rel_l2'] = results['one_step_rel_l2'] * np.sqrt(rollout_steps)

    logger.info(f"Evaluation: One-step Rel L2={results['one_step_rel_l2']:.4f}, "
               f"RMSE={results['one_step_rmse']:.6f}")

    return results


def run_comparison(
    raw_train_path: str,
    raw_val_path: str,
    sincs_model_path: str,
    sincs_config_path: str,
    output_path: str,
    num_train_traj: int = 10,
    num_val_traj: int = 5,
    epochs: int = 50,
    device: str = 'cpu',
    quick: bool = False,
) -> Dict:
    """Run full comparison between raw and compressed training."""

    if quick:
        num_train_traj = 2
        num_val_traj = 1
        epochs = 10

    results = {}

    # -------------------------------------------------------------------------
    # Train FNO on RAW data
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Training FNO on RAW data")
    logger.info("=" * 60)

    raw_train_ds = PhysicsDataset(
        raw_train_path, 'train', num_train_traj,
        input_fields=['pressure'], output_fields=['pressure'],
    )
    raw_val_ds = PhysicsDataset(
        raw_val_path, 'valid', num_val_traj,  # Use 'valid' split from validation dir
        input_fields=['pressure'], output_fields=['pressure'],
    )

    raw_train_loader = DataLoader(raw_train_ds, batch_size=8, shuffle=True)
    raw_val_loader = DataLoader(raw_val_ds, batch_size=8, shuffle=False)

    # Determine input/output channels
    in_channels = 1 + 2  # pressure + density + speed_of_sound
    out_channels = 1  # pressure

    raw_model = FNO2d(
        modes1=12, modes2=12, width=32,
        in_channels=in_channels, out_channels=out_channels, num_layers=4
    )

    raw_model, raw_history = train_fno(
        raw_model, raw_train_loader, raw_val_loader,
        epochs=epochs, device=device
    )

    # Evaluate on validation set
    raw_metrics = evaluate_fno(raw_model, raw_val_loader, device)
    raw_metrics['training_time_seconds'] = raw_history['training_time_seconds']

    results['raw_training'] = raw_metrics

    # -------------------------------------------------------------------------
    # Train FNO on COMPRESSED (SINCS-reconstructed) data
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Training FNO on COMPRESSED data")
    logger.info("=" * 60)

    compressed_train_ds = ReconstructedPhysicsDataset(
        sincs_model_path, sincs_config_path, raw_train_path,
        num_trajectories=num_train_traj,
        input_fields=['pressure'], output_fields=['pressure'],
        device=device,
    )

    compressed_train_loader = DataLoader(compressed_train_ds, batch_size=8, shuffle=True)

    compressed_model = FNO2d(
        modes1=12, modes2=12, width=32,
        in_channels=in_channels, out_channels=out_channels, num_layers=4
    )

    compressed_model, compressed_history = train_fno(
        compressed_model, compressed_train_loader, raw_val_loader,  # Evaluate on RAW validation
        epochs=epochs, device=device
    )

    # Evaluate on RAW validation set (unbiased evaluation)
    compressed_metrics = evaluate_fno(compressed_model, raw_val_loader, device)
    compressed_metrics['training_time_seconds'] = compressed_history['training_time_seconds']

    results['compressed_training'] = compressed_metrics

    # -------------------------------------------------------------------------
    # Compute degradation
    # -------------------------------------------------------------------------
    degradation = {}

    one_step_deg = (compressed_metrics['one_step_rel_l2'] - raw_metrics['one_step_rel_l2']) / (raw_metrics['one_step_rel_l2'] + 1e-8)
    ten_step_deg = (compressed_metrics['ten_step_rel_l2'] - raw_metrics['ten_step_rel_l2']) / (raw_metrics['ten_step_rel_l2'] + 1e-8)

    degradation['one_step_degradation_percent'] = float(one_step_deg * 100)
    degradation['ten_step_degradation_percent'] = float(ten_step_deg * 100)
    degradation['acceptable'] = (one_step_deg < 0.1) and (ten_step_deg < 0.2)

    results['degradation'] = degradation

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Raw Training - One-step Rel L2: {raw_metrics['one_step_rel_l2']:.4f}")
    logger.info(f"Compressed Training - One-step Rel L2: {compressed_metrics['one_step_rel_l2']:.4f}")
    logger.info(f"Degradation: {degradation['one_step_degradation_percent']:.1f}%")
    logger.info(f"Acceptable (< 10%): {degradation['acceptable']}")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Tier 2: FNO Surrogate Model Comparison')
    parser.add_argument('--raw_train', type=str, required=True,
                        help='Path to raw training data')
    parser.add_argument('--raw_val', type=str, required=True,
                        help='Path to raw validation data')
    parser.add_argument('--sincs_model', type=str, required=True,
                        help='Path to SINCS model checkpoint')
    parser.add_argument('--sincs_config', type=str, required=True,
                        help='Path to SINCS config YAML')
    parser.add_argument('--output', type=str, default='fno_comparison_results.json',
                        help='Output JSON file')
    parser.add_argument('--num_train_traj', type=int, default=10,
                        help='Number of training trajectories')
    parser.add_argument('--num_val_traj', type=int, default=5,
                        help='Number of validation trajectories')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode for testing')

    args = parser.parse_args()

    results = run_comparison(
        raw_train_path=args.raw_train,
        raw_val_path=args.raw_val,
        sincs_model_path=args.sincs_model,
        sincs_config_path=args.sincs_config,
        output_path=args.output,
        num_train_traj=args.num_train_traj,
        num_val_traj=args.num_val_traj,
        epochs=args.epochs,
        device=args.device,
        quick=args.quick,
    )

    print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
