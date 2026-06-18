#!/usr/bin/env python3
"""
Tier 1 Physics-Based Validation for SINCS Models

Uses class-specific physics validators based on the dataset type.
Each physics class has specific tests derived from the governing equations.

Usage:
    python run_tier1_physics_validation.py \
        --dataset acoustic_scattering_inclusions \
        --checkpoint path/to/checkpoint.mdl \
        --config path/to/config.yaml \
        --output_dir ./validation_results
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import numpy as np
import h5py
from glob import glob

def find_data_path_case_insensitive(data_dir: str, dataset_name: str) -> str:
    """
    Find the actual data path using case-insensitive matching.

    Handles cases where dataset names in model directories are lowercase
    but actual data directories have mixed case (e.g., 'mhd_64' vs 'MHD_64').
    """
    # First try exact match
    exact_path = os.path.join(data_dir, dataset_name)
    if os.path.exists(exact_path):
        return exact_path

    # Try case-insensitive search
    if os.path.isdir(data_dir):
        for entry in os.listdir(data_dir):
            if entry.lower() == dataset_name.lower():
                return os.path.join(data_dir, entry)

    # Return original path (will fail with clear error message)
    return exact_path

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

import torch
import yaml

from physics_class_validators import (
    get_physics_class,
    get_validator,
    run_physics_validation,
    DATASET_TO_CLASS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Field Name Mapping by Dataset Class
# =============================================================================

# Map internal field names to expected validator field names
FIELD_NAME_MAPS = {
    'acoustic_scattering': {
        'pressure': 'pressure',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'density': 'density',
        'speed_of_sound': 'speed_of_sound',
    },
    'euler': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'pressure': 'pressure',
        'energy': 'energy',
        'total_energy': 'total_energy',
    },
    'gray_scott': {
        'A': 'A',
        'B': 'B',
        'concentration_A': 'A',
        'concentration_B': 'B',
    },
    'mhd': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'magnetic_field_0': 'magnetic_field_x',
        'magnetic_field_1': 'magnetic_field_y',
        'magnetic_field_2': 'magnetic_field_z',
        'pressure': 'pressure',
    },
    'rayleigh_benard': {
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'temperature': 'temperature',
        'pressure': 'pressure',
    },
    'rayleigh_taylor': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'pressure': 'pressure',
    },
    'shallow_water': {
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'height': 'height',
        'eta': 'eta',
    },
    'supernova': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'energy': 'energy',
        'temperature': 'temperature',
        'pressure': 'pressure',
    },
    'shear_flow': {
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'tracer': 'tracer',
        'concentration': 'concentration',
    },
    'convective_envelope': {
        'density': 'density',
        'velocity_0': 'velocity_r',
        'velocity_1': 'velocity_theta',
        'velocity_2': 'velocity_phi',
        'temperature': 'temperature',
        'pressure': 'pressure',
    },
    'turbulent_radiative': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'temperature': 'temperature',
    },
    'turbulence_gravity': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'temperature': 'temperature',
        'pressure': 'pressure',
    },
    'viscoelastic': {
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'pressure': 'pressure',
        'polymer_stress': 'polymer_stress',
    },
    'neutron_star_merger': {
        'density': 'density',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
        'velocity_2': 'velocity_z',
        'magnetic_field_0': 'magnetic_field_x',
        'magnetic_field_1': 'magnetic_field_y',
        'magnetic_field_2': 'magnetic_field_z',
        'electron_fraction': 'electron_fraction',
        'Y_e': 'electron_fraction',
    },
    'active_matter': {
        'concentration': 'concentration',
        'c': 'concentration',
        'director': 'director',
        'orientation': 'orientation',
        'velocity_0': 'velocity_x',
        'velocity_1': 'velocity_y',
    },
    'helmholtz': {
        'u': 'u',
        'field': 'field',
        'pressure': 'pressure',
    },
}


def map_field_names(data: Dict[str, np.ndarray], physics_class: str) -> Dict[str, np.ndarray]:
    """Map field names to validator-expected names based on physics class."""
    field_map = FIELD_NAME_MAPS.get(physics_class, {})
    mapped_data = {}

    for original_name, array in data.items():
        # Check if we have a mapping for this field
        mapped_name = field_map.get(original_name, original_name)
        mapped_data[mapped_name] = array

        # Also keep original name for fallback
        if original_name != mapped_name:
            mapped_data[original_name] = array

    return mapped_data


def split_vector_fields(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Split multi-component vector fields into separate arrays to match SINCS training format."""
    new_data = {}

    for name, arr in data.items():
        if arr is None:
            continue

        # Check if this is a vector field (last dim is 2 or 3 for components)
        if arr.ndim >= 3 and arr.shape[-1] in [2, 3]:
            # Could be (T, H, W, C) or (H, W, C) for 2D
            # Or (T, X, Y, Z, C) or (X, Y, Z, C) for 3D
            # Split into individual components
            n_components = arr.shape[-1]
            for c in range(n_components):
                component_name = f"{name}_{c}"
                new_data[component_name] = arr[..., c]
                logger.info(f"  Split {name} component {c} -> {component_name}, shape={new_data[component_name].shape}")
        else:
            new_data[name] = arr

    return new_data


def load_raw_data_flexible(data_path: str, dataset_name: str,
                           trajectory_idx: int = 0,
                           num_timesteps: int = 20) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load raw data from HDF5 with flexible field detection.

    Returns:
        Tuple of (data dict, metadata dict)
    """
    # Find HDF5 files
    if os.path.isfile(data_path):
        files = [data_path]
    else:
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

    data = {}
    metadata = {}

    with h5py.File(files[0], 'r') as f:
        # Load all fields from t0_fields and t1_fields
        # Note: the_well uses t0_fields for current time, t1_fields for next time
        # Both can have full time series data

        for group_name in ['t0_fields', 't1_fields']:
            if group_name in f:
                for field_name in f[group_name].keys():
                    if field_name in data:
                        continue  # Skip if already loaded
                    try:
                        field_data = np.array(f[f'{group_name}/{field_name}'][trajectory_idx])

                        # Limit timesteps for fields with time dimension
                        if field_data.ndim >= 3:
                            # Check if last dim is small (vector components) vs spatial
                            if field_data.shape[-1] <= 3 and field_data.ndim >= 4:
                                # Shape like (T, H, W, C) - limit T
                                field_data = field_data[:num_timesteps]
                            elif field_data.ndim == 3:
                                # Shape like (T, H, W) - limit T
                                field_data = field_data[:num_timesteps]

                        data[field_name] = field_data
                        is_dynamic = field_data.ndim >= 3
                        logger.info(f"  Loaded {'dynamic' if is_dynamic else 'static'} field: {field_name}, shape={data[field_name].shape}")
                    except Exception as e:
                        logger.warning(f"  Could not load {field_name}: {e}")

        # Extract metadata
        if 'grid' in f:
            try:
                metadata['grid_type'] = f['grid'].attrs.get('type', 'unknown')
            except:
                pass

        # Get coordinate information
        for coord_name in ['x', 'y', 'z', 't', 'r', 'theta', 'phi']:
            if coord_name in f:
                try:
                    coords = np.array(f[coord_name])
                    if len(coords) > 1:
                        if coord_name == 't':
                            metadata['dt'] = float(coords[1] - coords[0])
                        else:
                            metadata[f'd{coord_name}'] = float(coords[1] - coords[0])
                except:
                    pass

    # Set default grid spacing if not found
    metadata.setdefault('dx', 1.0)
    metadata.setdefault('dy', 1.0)
    metadata.setdefault('dz', 1.0)
    metadata.setdefault('dt', 1.0)

    # Split vector fields into components to match SINCS training format
    data = split_vector_fields(data)

    return data, metadata


def load_sincs_model(checkpoint_path: str, config_path: str, device: str = 'cpu'):
    """Load SINCS model from checkpoint."""
    from model import SINCS, SINCSConfig
    from sincs_evaluate import load_hdf5_checkpoint

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
    state_dict = load_hdf5_checkpoint(checkpoint_path)
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

    return model, model_config


def reconstruct_data(model, raw_data: Dict[str, np.ndarray],
                     config: Dict, device: str = 'cpu',
                     max_timesteps: int = 20) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Reconstruct data using SINCS model.

    Returns:
        normalized_raw: Normalized raw data for fair comparison with model outputs
        recon_data: Reconstructed data (in normalized space)
        norm_stats: Normalization statistics (mean, std) for each field
    """
    # Determine data dimensions - prioritize fields with time dimension
    spatial_shape = None
    T = None

    # First pass: look for 3D/4D fields (with time dimension)
    for field_name, field_data in raw_data.items():
        if field_data is not None:
            if field_data.ndim == 3:  # (T, H, W)
                T = min(field_data.shape[0], max_timesteps)
                spatial_shape = field_data.shape[1:]
                logger.info(f"Using field '{field_name}' for dimensions: T={T}, spatial={spatial_shape}")
                break
            elif field_data.ndim == 4:  # (T, H, W, D) or (T, X, Y, Z)
                T = min(field_data.shape[0], max_timesteps)
                # Check if last dim is small (velocity components) vs spatial
                if field_data.shape[-1] <= 3:  # Likely (T, H, W, components)
                    spatial_shape = field_data.shape[1:-1]
                else:  # Likely (T, X, Y, Z)
                    spatial_shape = field_data.shape[1:]
                logger.info(f"Using field '{field_name}' for dimensions: T={T}, spatial={spatial_shape}")
                break

    # Second pass: if no time-dependent field found, use 2D static field
    if spatial_shape is None:
        for field_name, field_data in raw_data.items():
            if field_data is not None and field_data.ndim == 2:
                spatial_shape = field_data.shape
                logger.info(f"Using static field '{field_name}' for spatial dimensions: {spatial_shape}")
                break

    if spatial_shape is None:
        raise ValueError("Could not determine spatial dimensions from data")

    if T is None:
        T = max_timesteps  # Use max_timesteps as default for static data

    logger.info(f"Data dimensions: T={T}, spatial_shape={spatial_shape}")

    # Compute normalization stats
    norm_stats = {}
    for field_name, field_data in raw_data.items():
        if field_data is not None:
            norm_stats[field_name] = {
                'mean': float(np.mean(field_data)),
                'std': float(np.std(field_data)) + 1e-8
            }

    # Normalize raw data
    normalized_raw = {}
    for field_name, field_data in raw_data.items():
        if field_data is not None:
            stats = norm_stats[field_name]
            normalized_raw[field_name] = (field_data - stats['mean']) / stats['std']
            # Limit timesteps
            if normalized_raw[field_name].ndim >= 3:
                normalized_raw[field_name] = normalized_raw[field_name][:T]

    # Create coordinate grid
    ndim = len(spatial_shape) + 1  # +1 for time

    if ndim == 3:  # 2D spatial
        t_coords = np.linspace(0, 1, T)
        x_coords = np.linspace(0, 1, spatial_shape[0])
        y_coords = np.linspace(0, 1, spatial_shape[1])
    elif ndim == 4:  # 3D spatial
        t_coords = np.linspace(0, 1, T)
        x_coords = np.linspace(0, 1, spatial_shape[0])
        y_coords = np.linspace(0, 1, spatial_shape[1])
        z_coords = np.linspace(0, 1, spatial_shape[2])
    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")

    # Determine output dimension from config
    output_dim = config.get('output_dim', len(raw_data))

    # Reconstruct using batched inference
    logger.info(f"Reconstructing with output_dim={output_dim}...")

    batch_size = 16384
    recon_data = {}

    with torch.no_grad():
        if ndim == 3:  # 2D spatial
            recon_array = np.zeros((T, spatial_shape[0], spatial_shape[1], output_dim), dtype=np.float32)

            # Pre-compute spatial grid (vectorized - much faster than nested loops)
            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            spatial_coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
            n_spatial = len(spatial_coords)

            for t_idx in range(T):
                t = t_coords[t_idx]
                # Prepend time coordinate to all spatial coords
                t_col = np.full((n_spatial, 1), t, dtype=np.float32)
                coords = np.concatenate([t_col, spatial_coords], axis=1)

                all_outputs = []
                for start in range(0, len(coords), batch_size):
                    end = min(start + batch_size, len(coords))
                    batch_coords = torch.from_numpy(coords[start:end]).to(device)
                    outputs = model(batch_coords)
                    all_outputs.append(outputs.cpu().numpy())

                outputs = np.concatenate(all_outputs, axis=0)
                recon_array[t_idx] = outputs.reshape(spatial_shape[0], spatial_shape[1], -1)

                if (t_idx + 1) % 5 == 0:
                    logger.info(f"  Reconstructed timestep {t_idx + 1}/{T}")

        elif ndim == 4:  # 3D spatial
            recon_array = np.zeros((T, spatial_shape[0], spatial_shape[1], spatial_shape[2], output_dim),
                                   dtype=np.float32)

            # Pre-compute spatial grid (vectorized - much faster than nested loops)
            xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            spatial_coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
            n_spatial = len(spatial_coords)
            logger.info(f"  3D grid: {spatial_shape} = {n_spatial} coordinates per timestep")

            for t_idx in range(T):
                t = t_coords[t_idx]
                # Prepend time coordinate to all spatial coords
                t_col = np.full((n_spatial, 1), t, dtype=np.float32)
                coords = np.concatenate([t_col, spatial_coords], axis=1)

                all_outputs = []
                for start in range(0, len(coords), batch_size):
                    end = min(start + batch_size, len(coords))
                    batch_coords = torch.from_numpy(coords[start:end]).to(device)
                    outputs = model(batch_coords)
                    all_outputs.append(outputs.cpu().numpy())

                outputs = np.concatenate(all_outputs, axis=0)
                recon_array[t_idx] = outputs.reshape(spatial_shape + (output_dim,))

                if (t_idx + 1) % 5 == 0:
                    logger.info(f"  Reconstructed timestep {t_idx + 1}/{T}")

    # Map reconstructed outputs to field names
    # SINCS sorts field names alphabetically during training, so we must match that order
    # Only include dynamic fields (those with time dimension or matching output_dim count)
    dynamic_field_names = []
    static_field_names = []

    for name, arr in raw_data.items():
        if arr is not None:
            if arr.ndim >= 3:  # Has time dimension -> dynamic
                dynamic_field_names.append(name)
            else:
                static_field_names.append(name)

    # Sort alphabetically to match SINCS training order
    dynamic_field_names = sorted(dynamic_field_names)

    logger.info(f"Dynamic fields (sorted): {dynamic_field_names}")
    logger.info(f"Static fields: {static_field_names}")

    # Map SINCS outputs to dynamic field names
    for idx, field_name in enumerate(dynamic_field_names):
        if idx < output_dim:
            recon_data[field_name] = recon_array[..., idx]
            logger.info(f"  Mapped output {idx} -> {field_name}")

    # Copy static fields directly (not reconstructed)
    for field_name in static_field_names:
        if field_name in normalized_raw:
            recon_data[field_name] = normalized_raw[field_name].copy()

    return normalized_raw, recon_data, norm_stats


def unnormalize_data(data: Dict[str, np.ndarray], norm_stats: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    """
    Un-normalize data using stored normalization statistics.

    Args:
        data: Dictionary of normalized field arrays
        norm_stats: Dictionary with mean/std for each field

    Returns:
        Dictionary of un-normalized field arrays (in original physical units)
    """
    unnorm_data = {}
    for field_name, field_data in data.items():
        if field_data is not None and field_name in norm_stats:
            stats = norm_stats[field_name]
            unnorm_data[field_name] = field_data * stats['std'] + stats['mean']
        elif field_data is not None:
            # No stats available, copy as-is
            unnorm_data[field_name] = field_data.copy()
    return unnorm_data


def run_tier1_validation(
    dataset_name: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    data_dir: str = "/cra-1272/PhysicsAlchemists/datasets",
    device: str = 'cpu',
    trajectory_idx: int = 0,
    num_timesteps: int = 20,
) -> Dict[str, Any]:
    """
    Run Tier 1 physics validation for a SINCS model.

    Args:
        dataset_name: Name of the dataset
        checkpoint_path: Path to SINCS checkpoint
        config_path: Path to config YAML
        output_dir: Directory to save results
        data_dir: Base directory for datasets
        device: Device to use
        trajectory_idx: Trajectory to validate
        num_timesteps: Number of timesteps to use

    Returns:
        Validation results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"TIER 1 PHYSICS VALIDATION: {dataset_name}")
    logger.info("=" * 70)

    # Get physics class
    physics_class = get_physics_class(dataset_name)
    logger.info(f"Physics class: {physics_class}")

    results = {
        'metadata': {
            'dataset': dataset_name,
            'physics_class': physics_class,
            'checkpoint': checkpoint_path,
            'config': config_path,
            'timestamp': datetime.now().isoformat(),
            'trajectory_idx': trajectory_idx,
            'num_timesteps': num_timesteps,
        }
    }

    # Load raw data
    logger.info("\nLoading raw data...")
    data_path = find_data_path_case_insensitive(data_dir, dataset_name)
    try:
        raw_data, metadata = load_raw_data_flexible(data_path, dataset_name,
                                                     trajectory_idx, num_timesteps)
        logger.info(f"Loaded {len(raw_data)} fields")
        results['metadata']['grid_metadata'] = metadata
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        results['error'] = f"Data loading failed: {e}"
        return results

    # Load model and reconstruct
    logger.info("\nLoading SINCS model...")
    try:
        model, model_config = load_sincs_model(checkpoint_path, config_path, device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        results['error'] = f"Model loading failed: {e}"
        return results

    logger.info("\nReconstructing data...")
    try:
        normalized_raw, recon_data, norm_stats = reconstruct_data(model, raw_data, model_config,
                                                                   device, num_timesteps)
        logger.info(f"Reconstructed {len(recon_data)} fields")
    except Exception as e:
        logger.error(f"Failed to reconstruct: {e}")
        results['error'] = f"Reconstruction failed: {e}"
        return results

    # Map field names for validator (apply to both normalized and unnormalized)
    normalized_raw = map_field_names(normalized_raw, physics_class)
    recon_data = map_field_names(recon_data, physics_class)

    # Un-normalize data for physics validation (conservation metrics need physical units)
    # Note: We need to map norm_stats keys as well to match the mapped field names
    mapped_norm_stats = {}
    for old_name, stats in norm_stats.items():
        # Try to find the mapped name
        temp_dict = {old_name: None}
        temp_mapped = map_field_names(temp_dict, physics_class)
        new_name = list(temp_mapped.keys())[0] if temp_mapped else old_name
        mapped_norm_stats[new_name] = stats

    unnorm_raw = unnormalize_data(normalized_raw, mapped_norm_stats)
    unnorm_recon = unnormalize_data(recon_data, mapped_norm_stats)
    logger.info(f"Un-normalized data for physics conservation metrics")

    # Run physics validation with UN-NORMALIZED data
    # Conservation metrics need physical units to make sense
    logger.info("\nRunning physics validation...")
    try:
        validation_results = run_physics_validation(
            dataset_name=dataset_name,
            raw_data=unnorm_raw,
            reconstructed_data=unnorm_recon,
            metadata=metadata
        )
        results['validation'] = validation_results
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        results['error'] = f"Validation failed: {e}"
        return results

    # Save results
    results_path = os.path.join(output_dir, f'tier1_{dataset_name}_physics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    print_validation_summary(results)

    return results


def print_validation_summary(results: Dict[str, Any]):
    """Print a formatted validation summary.

    Shows universal metrics (PSNR, Absolute L2 Error, Spectral Error)
    and physics-specific metrics (conservation laws, etc.).
    """
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    meta = results.get('metadata', {})
    print(f"\nDataset: {meta.get('dataset', 'unknown')}")
    print(f"Physics Class: {meta.get('physics_class', 'unknown')}")

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        return

    validation = results.get('validation', {})

    # Universal metrics
    print("\n--- Universal Metrics (Absolute Errors) ---")
    universal = validation.get('universal_metrics', {})
    if 'error' in universal:
        print(f"  Error: {universal['error']}")
    else:
        if 'field_used' in universal:
            print(f"  Field used: {universal['field_used']}")

        # PSNR
        if 'psnr' in universal:
            psnr_val = universal['psnr']
            if isinstance(psnr_val, float):
                if psnr_val == float('inf'):
                    print(f"  PSNR: inf (perfect reconstruction)")
                else:
                    print(f"  PSNR: {psnr_val:.4f} dB")

        # Absolute L2 Error
        if 'abs_l2_error' in universal:
            print(f"  Absolute L2 Error: {universal['abs_l2_error']:.6f}")

        # L2 Error Percentage (range-normalized)
        if 'l2_error_pct' in universal:
            print(f"  L2 Error %: {universal['l2_error_pct']:.4f}%")

        # Spectral Error (absolute and percentage)
        if 'spectral_error' in universal:
            print(f"  Spectral Error: {universal['spectral_error']:.6f}")
        if 'spectral_error_pct' in universal:
            print(f"  Spectral Error %: {universal['spectral_error_pct']:.4f}%")

    # Primary metrics (physics-specific conservation laws)
    primary = validation.get('primary_metrics', {})
    if primary and 'error' not in primary and 'skipped' not in primary:
        print("\n--- Primary Metrics (Physics-Specific) ---")
        for key, value in primary.items():
            if isinstance(value, float):
                if 'abs_error' in key:
                    print(f"  {key}: {value:.6f}")
                elif 'pct' in key or 'error_pct' in key:
                    print(f"  {key}: {value:.4f}%")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    # Secondary metrics (additional physics metrics)
    secondary = validation.get('secondary_metrics', {})
    if secondary and 'error' not in secondary and 'skipped' not in secondary:
        print("\n--- Secondary Metrics ---")
        for key, value in secondary.items():
            if isinstance(value, float):
                if 'abs_error' in key:
                    print(f"  {key}: {value:.6f}")
                elif 'pct' in key or 'error_pct' in key:
                    print(f"  {key}: {value:.4f}%")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    # Summary
    summary = validation.get('summary', {})
    print("\n--- Summary ---")
    print(f"  Status: {summary.get('status', 'unknown')}")

    if summary.get('notes'):
        for note in summary['notes']:
            print(f"  Note: {note}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Tier 1 Physics Validation for SINCS')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to SINCS checkpoint')
    parser.add_argument('--config', '-p', type=str, required=True,
                        help='Path to SINCS config YAML')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='/cra-1272/PhysicsAlchemists/WellImplementation/Well/modelzoo/validation_results_all',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str,
                        default='/cra-1272/PhysicsAlchemists/datasets',
                        help='Base directory for datasets')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--trajectory', type=int, default=0,
                        help='Trajectory index')
    parser.add_argument('--num_timesteps', type=int, default=20,
                        help='Number of timesteps to validate')

    args = parser.parse_args()

    run_tier1_validation(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        device=args.device,
        trajectory_idx=args.trajectory,
        num_timesteps=args.num_timesteps,
    )


if __name__ == '__main__':
    main()
