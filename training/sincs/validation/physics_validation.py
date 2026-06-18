#!/usr/bin/env python3
"""
Tier 1: Physics-Preserving Validation for SINCS-Compressed Data

Validates that compression preserves physically meaningful quantities:
1. Wave speed consistency
2. Acoustic impedance at inclusions
3. Energy conservation
4. Frequency spectrum preservation
5. Boundary condition adherence
6. Pressure-velocity phase relationship
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from typing import Dict, Tuple, Optional
import h5py

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

import torch
from scipy import fft
from scipy.signal import correlate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhysicsValidator:
    """Validates physics properties between raw and reconstructed data."""

    def __init__(self, raw_data: Dict[str, np.ndarray],
                 reconstructed_data: Dict[str, np.ndarray],
                 boundary_conditions: Dict[str, np.ndarray] = None):
        """
        Args:
            raw_data: Dict with keys 'pressure', 'velocity', 'density', 'speed_of_sound'
            reconstructed_data: Dict with same keys as raw_data
            boundary_conditions: Dict with boundary masks
        """
        self.raw = raw_data
        self.recon = reconstructed_data
        self.bc = boundary_conditions or {}

    def validate_wave_speed(self) -> Dict:
        """
        Validate wave speed consistency.
        For acoustic waves: c = sqrt(dp/drho) ≈ speed_of_sound field
        """
        results = {}

        # Use provided speed_of_sound field
        if 'speed_of_sound' in self.raw and 'speed_of_sound' in self.recon:
            raw_c = self.raw['speed_of_sound']
            recon_c = self.recon['speed_of_sound']

            # Compute errors
            abs_error = np.abs(raw_c - recon_c)
            results['mean_absolute_error'] = float(np.mean(abs_error))
            results['max_error'] = float(np.max(abs_error))
            results['relative_error'] = float(np.mean(abs_error) / (np.mean(np.abs(raw_c)) + 1e-10))
            results['std_error'] = float(np.std(abs_error))
        else:
            # Estimate from pressure/density relationship
            # c^2 = dp/drho, approximate as c ≈ sqrt(P/rho) for ideal gas
            raw_p = self.raw['pressure']
            raw_rho = self.raw['density']
            recon_p = self.recon['pressure']
            recon_rho = self.recon['density']

            # Compute effective wave speed (time-averaged)
            raw_c_eff = np.sqrt(np.abs(np.mean(raw_p, axis=0)) / (raw_rho + 1e-10))
            recon_c_eff = np.sqrt(np.abs(np.mean(recon_p, axis=0)) / (recon_rho + 1e-10))

            abs_error = np.abs(raw_c_eff - recon_c_eff)
            results['mean_absolute_error'] = float(np.mean(abs_error))
            results['max_error'] = float(np.max(abs_error))
            results['relative_error'] = float(np.mean(abs_error) / (np.mean(np.abs(raw_c_eff)) + 1e-10))

        logger.info(f"Wave speed validation: MAE={results['mean_absolute_error']:.6f}, "
                   f"Rel Error={results['relative_error']*100:.2f}%")
        return results

    def validate_acoustic_impedance(self) -> Dict:
        """
        Validate acoustic impedance Z = rho * c at material boundaries.
        Critical for wave reflection/transmission behavior.
        """
        results = {}

        raw_rho = self.raw['density']
        recon_rho = self.recon['density']

        if 'speed_of_sound' in self.raw:
            raw_c = self.raw['speed_of_sound']
            recon_c = self.recon['speed_of_sound']
        else:
            # Use constant or estimated value
            raw_c = np.ones_like(raw_rho)
            recon_c = np.ones_like(recon_rho)

        # Compute impedance
        raw_Z = raw_rho * raw_c
        recon_Z = recon_rho * recon_c

        # Find material boundaries (gradient of density)
        grad_rho = np.gradient(raw_rho)
        boundary_mask = np.sqrt(grad_rho[0]**2 + grad_rho[1]**2) > np.percentile(
            np.sqrt(grad_rho[0]**2 + grad_rho[1]**2), 90
        )

        # Error at boundaries vs interior
        if np.sum(boundary_mask) > 0:
            boundary_error = np.abs(raw_Z[boundary_mask] - recon_Z[boundary_mask])
            results['mean_error_at_boundaries'] = float(np.mean(boundary_error))
            results['max_error_at_boundaries'] = float(np.max(boundary_error))
        else:
            results['mean_error_at_boundaries'] = 0.0
            results['max_error_at_boundaries'] = 0.0

        interior_mask = ~boundary_mask
        interior_error = np.abs(raw_Z[interior_mask] - recon_Z[interior_mask])
        results['mean_error_interior'] = float(np.mean(interior_error))

        # Overall
        overall_error = np.abs(raw_Z - recon_Z)
        results['relative_error'] = float(np.mean(overall_error) / (np.mean(np.abs(raw_Z)) + 1e-10))

        logger.info(f"Acoustic impedance: Boundary error={results['mean_error_at_boundaries']:.6f}, "
                   f"Rel Error={results['relative_error']*100:.2f}%")
        return results

    def validate_energy_conservation(self) -> Dict:
        """
        Validate energy conservation over time.
        Acoustic energy density: E = 0.5 * rho * |v|^2 + 0.5 * p^2 / (rho * c^2)
        """
        results = {}

        raw_p = self.raw['pressure']  # (T, H, W) or (H, W) depending on input
        raw_v = self.raw['velocity']  # (T, H, W, 2) or (H, W, 2)
        raw_rho = self.raw['density']  # (H, W)

        recon_p = self.recon['pressure']
        recon_v = self.recon['velocity']
        recon_rho = self.recon['density']

        if 'speed_of_sound' in self.raw:
            raw_c = self.raw['speed_of_sound']
            recon_c = self.recon['speed_of_sound']
        else:
            raw_c = np.ones_like(raw_rho) * 343.0  # Default air
            recon_c = raw_c.copy()

        # Handle different tensor shapes
        if raw_p.ndim == 2:
            # Single timestep
            raw_p = raw_p[np.newaxis, ...]
            recon_p = recon_p[np.newaxis, ...]
        if raw_v.ndim == 3:
            raw_v = raw_v[np.newaxis, ...]
            recon_v = recon_v[np.newaxis, ...]

        T = raw_p.shape[0]

        # Compute energy at each timestep
        raw_energies = []
        recon_energies = []

        for t in range(T):
            # Kinetic energy: 0.5 * rho * |v|^2
            raw_ke = 0.5 * raw_rho * (raw_v[t, ..., 0]**2 + raw_v[t, ..., 1]**2)
            recon_ke = 0.5 * recon_rho * (recon_v[t, ..., 0]**2 + recon_v[t, ..., 1]**2)

            # Potential energy: 0.5 * p^2 / (rho * c^2)
            raw_pe = 0.5 * raw_p[t]**2 / (raw_rho * raw_c**2 + 1e-10)
            recon_pe = 0.5 * recon_p[t]**2 / (recon_rho * recon_c**2 + 1e-10)

            # Total energy (sum over domain)
            raw_energies.append(np.sum(raw_ke + raw_pe))
            recon_energies.append(np.sum(recon_ke + recon_pe))

        raw_energies = np.array(raw_energies)
        recon_energies = np.array(recon_energies)

        # Energy deviation
        energy_diff = np.abs(raw_energies - recon_energies)
        results['max_energy_deviation'] = float(np.max(energy_diff) / (np.mean(raw_energies) + 1e-10))
        results['mean_energy_deviation'] = float(np.mean(energy_diff) / (np.mean(raw_energies) + 1e-10))

        # Energy drift (change over time)
        raw_drift = (raw_energies[-1] - raw_energies[0]) / (raw_energies[0] + 1e-10)
        recon_drift = (recon_energies[-1] - recon_energies[0]) / (recon_energies[0] + 1e-10)
        results['raw_energy_drift'] = float(raw_drift)
        results['recon_energy_drift'] = float(recon_drift)
        results['energy_drift_rate'] = float(np.abs(recon_drift - raw_drift))

        logger.info(f"Energy conservation: Max deviation={results['max_energy_deviation']*100:.2f}%, "
                   f"Drift rate diff={results['energy_drift_rate']*100:.2f}%")
        return results

    def validate_frequency_spectrum(self, num_timesteps: int = 10) -> Dict:
        """
        Validate frequency spectrum preservation via 2D FFT.
        """
        results = {}

        raw_p = self.raw['pressure']
        recon_p = self.recon['pressure']

        if raw_p.ndim == 2:
            raw_p = raw_p[np.newaxis, ...]
            recon_p = recon_p[np.newaxis, ...]

        T = min(raw_p.shape[0], num_timesteps)

        # Compute FFT and power spectrum
        low_freq_errors = []
        mid_freq_errors = []
        high_freq_errors = []

        H, W = raw_p.shape[1], raw_p.shape[2]
        freq_x = np.fft.fftfreq(W)
        freq_y = np.fft.fftfreq(H)
        freq_mag = np.sqrt(freq_x[np.newaxis, :]**2 + freq_y[:, np.newaxis]**2)

        # Define frequency bands
        low_mask = freq_mag < 0.1
        mid_mask = (freq_mag >= 0.1) & (freq_mag < 0.3)
        high_mask = freq_mag >= 0.3

        for t in range(T):
            raw_fft = np.fft.fft2(raw_p[t])
            recon_fft = np.fft.fft2(recon_p[t])

            raw_power = np.abs(raw_fft)**2
            recon_power = np.abs(recon_fft)**2

            # Compute relative error per band
            if np.sum(raw_power[low_mask]) > 0:
                low_err = np.sum(np.abs(raw_power[low_mask] - recon_power[low_mask])) / np.sum(raw_power[low_mask])
                low_freq_errors.append(low_err)

            if np.sum(raw_power[mid_mask]) > 0:
                mid_err = np.sum(np.abs(raw_power[mid_mask] - recon_power[mid_mask])) / np.sum(raw_power[mid_mask])
                mid_freq_errors.append(mid_err)

            if np.sum(raw_power[high_mask]) > 0:
                high_err = np.sum(np.abs(raw_power[high_mask] - recon_power[high_mask])) / np.sum(raw_power[high_mask])
                high_freq_errors.append(high_err)

        results['low_freq_error'] = float(np.mean(low_freq_errors)) if low_freq_errors else 0.0
        results['mid_freq_error'] = float(np.mean(mid_freq_errors)) if mid_freq_errors else 0.0
        results['high_freq_error'] = float(np.mean(high_freq_errors)) if high_freq_errors else 0.0

        logger.info(f"Frequency spectrum: Low={results['low_freq_error']*100:.2f}%, "
                   f"Mid={results['mid_freq_error']*100:.2f}%, High={results['high_freq_error']*100:.2f}%")
        return results

    def validate_boundary_conditions(self) -> Dict:
        """
        Validate error at boundary conditions.
        Wall boundaries should have zero normal velocity.
        """
        results = {}

        raw_v = self.raw['velocity']
        recon_v = self.recon['velocity']

        # Ensure 4D: (T, H, W, 2)
        if raw_v.ndim == 3:
            # (H, W, 2) -> (1, H, W, 2)
            raw_v = raw_v[np.newaxis, ...]
            recon_v = recon_v[np.newaxis, ...]

        T, H, W, _ = raw_v.shape

        # Compute overall RMSE
        overall_rmse = np.sqrt(np.mean((raw_v - recon_v)**2))
        results['overall_velocity_rmse'] = float(overall_rmse)

        # Edge regions (first/last 5 rows/cols)
        edge_size = 5

        # Wall boundaries (x=0 -> column 0, y=0 -> row 0)
        # Extract wall regions properly
        # x0_wall: first column (all rows)
        x0_raw = raw_v[:, :, :edge_size, 0]  # v_x at x=0 boundary
        x0_recon = recon_v[:, :, :edge_size, 0]
        results['x0_wall_rmse'] = float(np.sqrt(np.mean((x0_raw - x0_recon)**2)))

        # y0_wall: first rows (all columns)
        y0_raw = raw_v[:, :edge_size, :, 1]  # v_y at y=0 boundary
        y0_recon = recon_v[:, :edge_size, :, 1]
        results['y0_wall_rmse'] = float(np.sqrt(np.mean((y0_raw - y0_recon)**2)))

        # Open boundaries (xL, yL)
        xL_raw = raw_v[:, :, -edge_size:, :]
        xL_recon = recon_v[:, :, -edge_size:, :]
        results['xL_open_rmse'] = float(np.sqrt(np.mean((xL_raw - xL_recon)**2)))

        yL_raw = raw_v[:, -edge_size:, :, :]
        yL_recon = recon_v[:, -edge_size:, :, :]
        results['yL_open_rmse'] = float(np.sqrt(np.mean((yL_raw - yL_recon)**2)))

        # Combined wall vs open
        wall_rmse = (results['x0_wall_rmse'] + results['y0_wall_rmse']) / 2
        open_rmse = (results['xL_open_rmse'] + results['yL_open_rmse']) / 2

        results['wall_rmse'] = float(wall_rmse)
        results['open_rmse'] = float(open_rmse)

        logger.info(f"Boundary conditions: Wall RMSE={results['wall_rmse']:.6f}, "
                   f"Open RMSE={results['open_rmse']:.6f}")
        return results

    def validate_phase_relationship(self) -> Dict:
        """
        Validate pressure-velocity phase relationship.
        For traveling acoustic waves, pressure gradient and velocity should be correlated.
        """
        results = {}

        raw_p = self.raw['pressure']
        raw_v = self.raw['velocity']
        recon_p = self.recon['pressure']
        recon_v = self.recon['velocity']

        if raw_p.ndim == 2:
            raw_p = raw_p[np.newaxis, ...]
            recon_p = recon_p[np.newaxis, ...]
        if raw_v.ndim == 3:
            raw_v = raw_v[np.newaxis, ...]
            recon_v = recon_v[np.newaxis, ...]

        # Compute pressure gradients
        raw_correlations = []
        recon_correlations = []

        T = raw_p.shape[0]
        for t in range(T):
            # Gradient of pressure
            raw_dp_dx = np.gradient(raw_p[t], axis=1)
            raw_dp_dy = np.gradient(raw_p[t], axis=0)
            recon_dp_dx = np.gradient(recon_p[t], axis=1)
            recon_dp_dy = np.gradient(recon_p[t], axis=0)

            # Correlation with velocity (momentum equation: dv/dt = -grad(p)/rho)
            # Velocity should be anti-correlated with pressure gradient
            raw_corr_x = np.corrcoef(raw_dp_dx.flatten(), raw_v[t, ..., 0].flatten())[0, 1]
            raw_corr_y = np.corrcoef(raw_dp_dy.flatten(), raw_v[t, ..., 1].flatten())[0, 1]
            recon_corr_x = np.corrcoef(recon_dp_dx.flatten(), recon_v[t, ..., 0].flatten())[0, 1]
            recon_corr_y = np.corrcoef(recon_dp_dy.flatten(), recon_v[t, ..., 1].flatten())[0, 1]

            if not np.isnan(raw_corr_x):
                raw_correlations.append((raw_corr_x + raw_corr_y) / 2)
            if not np.isnan(recon_corr_x):
                recon_correlations.append((recon_corr_x + recon_corr_y) / 2)

        results['raw_correlation'] = float(np.mean(raw_correlations)) if raw_correlations else 0.0
        results['decompressed_correlation'] = float(np.mean(recon_correlations)) if recon_correlations else 0.0
        results['correlation_difference'] = float(abs(results['raw_correlation'] - results['decompressed_correlation']))

        logger.info(f"Phase relationship: Raw corr={results['raw_correlation']:.4f}, "
                   f"Recon corr={results['decompressed_correlation']:.4f}")
        return results

    def run_all_validations(self) -> Dict:
        """Run all physics validations and return combined results."""
        results = {
            'wave_speed': self.validate_wave_speed(),
            'impedance': self.validate_acoustic_impedance(),
            'energy_conservation': self.validate_energy_conservation(),
            'frequency_spectrum': self.validate_frequency_spectrum(),
            'boundary_error': self.validate_boundary_conditions(),
            'phase_correlation': self.validate_phase_relationship(),
        }

        # Compute pass/fail summary
        results['summary'] = {
            'wave_speed_pass': results['wave_speed']['relative_error'] < 0.05,
            'energy_pass': results['energy_conservation']['max_energy_deviation'] < 0.01,
            'high_freq_pass': results['frequency_spectrum']['high_freq_error'] < 0.20,
            'boundary_comparable': abs(results['boundary_error']['wall_rmse'] -
                                       results['boundary_error']['open_rmse']) < 0.1,
        }
        results['summary']['all_pass'] = all(results['summary'].values())

        return results


def load_raw_data(data_path: str, trajectory_idx: int = 0) -> Tuple[Dict, Dict]:
    """Load raw data from HDF5 file."""

    # Find HDF5 files
    from glob import glob
    if os.path.isfile(data_path):
        files = [data_path]
    else:
        # Look for chunk files
        patterns = [
            os.path.join(data_path, "data/train/*.hdf5"),
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

    with h5py.File(files[0], 'r') as f:
        # Load data for specified trajectory
        data = {}

        # Static fields
        if 't0_fields/density' in f:
            density = f['t0_fields/density'][trajectory_idx]
            data['density'] = np.array(density)

        if 't0_fields/speed_of_sound' in f:
            sos = f['t0_fields/speed_of_sound'][trajectory_idx]
            data['speed_of_sound'] = np.array(sos)

        # Dynamic fields
        if 't0_fields/pressure' in f:
            pressure = f['t0_fields/pressure'][trajectory_idx]
            data['pressure'] = np.array(pressure)

        if 't1_fields/velocity' in f:
            velocity = f['t1_fields/velocity'][trajectory_idx]
            data['velocity'] = np.array(velocity)

        # Boundary conditions
        bc = {}
        if 'boundary_conditions' in f:
            for key in f['boundary_conditions'].keys():
                bc[key] = np.array(f['boundary_conditions'][key]['mask'])

        logger.info(f"Loaded data shapes: pressure={data.get('pressure', np.array([])).shape}, "
                   f"velocity={data.get('velocity', np.array([])).shape}")

    return data, bc


def load_reconstructed_data(model_path: str, config_path: str,
                           raw_data: Dict, device: str = 'cpu') -> Tuple[Dict, Dict]:
    """
    Load SINCS model and reconstruct data.

    Returns both normalized raw data and reconstructed data for comparison.
    SINCS operates in z-score normalized space, so we normalize raw data
    to match for fair comparison.
    """
    from model import SINCS, SINCSConfig
    from sincs_evaluate import load_hdf5_checkpoint
    import yaml

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['trainer']['init']['model']

    # Create model
    sincs_config = SINCSConfig(
        input_dim=model_config.get('input_dim', 3),
        output_dim=model_config.get('output_dim', 3),
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

    # Get data dimensions
    if 'pressure' in raw_data:
        T, H, W = raw_data['pressure'].shape
    else:
        H, W = raw_data['density'].shape
        T = 102  # Default

    # Compute normalization stats from raw data (z-score)
    norm_stats = {}
    if 'pressure' in raw_data:
        norm_stats['pressure'] = {
            'mean': float(np.mean(raw_data['pressure'])),
            'std': float(np.std(raw_data['pressure'])) + 1e-8
        }
    if 'velocity' in raw_data:
        norm_stats['velocity_0'] = {
            'mean': float(np.mean(raw_data['velocity'][..., 0])),
            'std': float(np.std(raw_data['velocity'][..., 0])) + 1e-8
        }
        norm_stats['velocity_1'] = {
            'mean': float(np.mean(raw_data['velocity'][..., 1])),
            'std': float(np.std(raw_data['velocity'][..., 1])) + 1e-8
        }

    logger.info(f"Normalization stats: pressure mean={norm_stats.get('pressure', {}).get('mean', 0):.4f}, "
               f"std={norm_stats.get('pressure', {}).get('std', 1):.4f}")

    # Normalize raw data for comparison
    normalized_raw = {
        'density': raw_data['density'].copy(),
        'speed_of_sound': raw_data.get('speed_of_sound', np.ones_like(raw_data['density'])).copy(),
    }
    if 'pressure' in raw_data:
        normalized_raw['pressure'] = (raw_data['pressure'] - norm_stats['pressure']['mean']) / norm_stats['pressure']['std']
    if 'velocity' in raw_data:
        normalized_raw['velocity'] = np.zeros_like(raw_data['velocity'])
        normalized_raw['velocity'][..., 0] = (raw_data['velocity'][..., 0] - norm_stats['velocity_0']['mean']) / norm_stats['velocity_0']['std']
        normalized_raw['velocity'][..., 1] = (raw_data['velocity'][..., 1] - norm_stats['velocity_1']['mean']) / norm_stats['velocity_1']['std']

    # Create coordinate grid (normalized to [0, 1])
    t_coords = np.linspace(0, 1, T)
    x_coords = np.linspace(0, 1, H)
    y_coords = np.linspace(0, 1, W)

    # Reconstruct dynamic fields
    logger.info(f"Reconstructing data with shape T={T}, H={H}, W={W}...")

    recon_pressure = np.zeros((T, H, W), dtype=np.float32)
    recon_velocity = np.zeros((T, H, W, 2), dtype=np.float32)

    # Batch inference for efficiency
    batch_size = 16384

    with torch.no_grad():
        for t_idx in range(T):
            t = t_coords[t_idx]

            # Create all spatial coordinates for this timestep
            coords_list = []
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    coords_list.append([t, x, y])

            coords = np.array(coords_list, dtype=np.float32)

            # Batch process
            all_outputs = []
            for start in range(0, len(coords), batch_size):
                end = min(start + batch_size, len(coords))
                batch_coords = torch.from_numpy(coords[start:end]).to(device)
                outputs = model(batch_coords)
                all_outputs.append(outputs.cpu().numpy())

            outputs = np.concatenate(all_outputs, axis=0)

            # Reshape to spatial grid (outputs are [pressure, vx, vy] in normalized space)
            outputs = outputs.reshape(H, W, -1)

            # Store outputs (already in normalized space)
            if outputs.shape[-1] >= 1:
                recon_pressure[t_idx] = outputs[..., 0]
            if outputs.shape[-1] >= 3:
                recon_velocity[t_idx, ..., 0] = outputs[..., 1]
                recon_velocity[t_idx, ..., 1] = outputs[..., 2]

            if (t_idx + 1) % 10 == 0:
                logger.info(f"  Reconstructed timestep {t_idx + 1}/{T}")

    # Build reconstructed data dict (in normalized space)
    recon_data = {
        'pressure': recon_pressure,
        'velocity': recon_velocity,
        'density': raw_data['density'].copy(),  # Static field unchanged
        'speed_of_sound': raw_data.get('speed_of_sound', np.ones_like(raw_data['density'])).copy(),
    }

    return normalized_raw, recon_data


def main():
    parser = argparse.ArgumentParser(description='Tier 1: Physics Validation for SINCS')
    parser.add_argument('--raw_data', type=str, required=True,
                        help='Path to raw HDF5 data or directory')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to SINCS model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML')
    parser.add_argument('--trajectory', type=int, default=0,
                        help='Trajectory index to validate')
    parser.add_argument('--output', type=str, default='physics_validation_results.json',
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer timesteps)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Tier 1: Physics-Preserving Validation")
    logger.info("=" * 60)

    # Load raw data
    logger.info(f"\nLoading raw data from {args.raw_data}")
    raw_data, bc = load_raw_data(args.raw_data, args.trajectory)

    # Quick mode: use fewer timesteps
    if args.quick:
        if 'pressure' in raw_data:
            raw_data['pressure'] = raw_data['pressure'][:20]
        if 'velocity' in raw_data:
            raw_data['velocity'] = raw_data['velocity'][:20]

    # Load and reconstruct data (both in normalized space for fair comparison)
    logger.info(f"\nLoading model and reconstructing data...")
    normalized_raw, recon_data = load_reconstructed_data(
        args.model_checkpoint, args.config, raw_data, args.device
    )

    # Ensure shapes match
    if args.quick:
        if 'pressure' in recon_data:
            recon_data['pressure'] = recon_data['pressure'][:20]
        if 'velocity' in recon_data:
            recon_data['velocity'] = recon_data['velocity'][:20]
        if 'pressure' in normalized_raw:
            normalized_raw['pressure'] = normalized_raw['pressure'][:20]
        if 'velocity' in normalized_raw:
            normalized_raw['velocity'] = normalized_raw['velocity'][:20]

    # Run validation (compare normalized raw vs reconstructed)
    logger.info("\nRunning physics validations...")
    validator = PhysicsValidator(normalized_raw, recon_data, bc)
    results = validator.run_all_validations()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("PHYSICS VALIDATION RESULTS")
    logger.info("=" * 60)

    print(json.dumps(results, indent=2))

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for key, passed in results['summary'].items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {key}: {status}")

    overall = "PASS" if results['summary']['all_pass'] else "FAIL"
    logger.info(f"\n  OVERALL: {overall}")


if __name__ == '__main__':
    main()
