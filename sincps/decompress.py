"""
SINCPS Decompression Utility

Load trained checkpoints and reconstruct simulation data.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import yaml

from .model import SINCPS


def load_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load model weights from HDF5 checkpoint file.

    Args:
        checkpoint_path: Path to .mdl checkpoint file

    Returns:
        State dictionary with model weights
    """
    state_dict = {}
    with h5py.File(checkpoint_path, 'r') as f:
        for key in f.keys():
            if key.startswith('model.'):
                data = torch.from_numpy(np.array(f[key]))
                # Remove 'model.' or 'model.model.' prefix
                new_key = key
                if new_key.startswith('model.model.'):
                    new_key = new_key[12:]
                elif new_key.startswith('model.'):
                    new_key = new_key[6:]
                state_dict[new_key] = data
    return state_dict


def load_model(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: str = 'cpu'
) -> SINCPS:
    """
    Load SINCPS model from checkpoint.

    Args:
        checkpoint_path: Path to .mdl checkpoint file
        config: Model configuration dict. If None, uses default config.
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded SINCPS model in eval mode
    """
    if config is None:
        config = {
            'input_dim': 3,
            'output_dim': 1,
            'hidden_dim': 1024,
            'num_hidden_layers': 4,
            'omega_0': 30.0,
            'omega_hidden': 30.0,
            'use_fourier_encoding': True,
            'encoding_levels': 10,
        }

    model = SINCPS.from_config(config)
    state_dict = load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def reconstruct(
    model: SINCPS,
    spatial_shape: Tuple[int, ...],
    num_timesteps: int,
    timestep_indices: Optional[List[int]] = None,
    batch_size: int = 65536,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Reconstruct simulation data from SINCPS model.

    Args:
        model: Loaded SINCPS model
        spatial_shape: Spatial dimensions (e.g., (256, 256) for 2D, (64, 64, 64) for 3D)
        num_timesteps: Total number of timesteps in the simulation
        timestep_indices: Which timesteps to reconstruct. If None, reconstructs all.
        batch_size: Batch size for inference
        device: Device to run inference on

    Returns:
        Reconstructed data with shape (T, *spatial_shape, num_fields)
        where T is len(timestep_indices) or num_timesteps
    """
    model = model.to(device)
    model.eval()

    if timestep_indices is None:
        timestep_indices = list(range(num_timesteps))

    output_dim = model.output_dim
    n_spatial = len(spatial_shape)

    # Create spatial coordinate grid
    spatial_coords = [np.linspace(0, 1, dim) for dim in spatial_shape]
    mesh = np.meshgrid(*spatial_coords, indexing='ij')
    spatial_flat = np.stack([m.flatten() for m in mesh], axis=-1).astype(np.float32)
    n_points = spatial_flat.shape[0]

    results = []

    for ts_idx in timestep_indices:
        t_val = ts_idx / (num_timesteps - 1) if num_timesteps > 1 else 0.0
        t_coords = np.full((n_points, 1), t_val, dtype=np.float32)
        coords = np.concatenate([t_coords, spatial_flat], axis=1)

        predictions = []
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                batch_coords = torch.from_numpy(coords[i:i + batch_size]).to(device)
                batch_preds = model(batch_coords).cpu().numpy()
                predictions.append(batch_preds)

        predictions = np.concatenate(predictions, axis=0)
        reconstructed = predictions.reshape(spatial_shape + (output_dim,))
        results.append(reconstructed)

    return np.stack(results, axis=0)


def denormalize(
    data: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """
    Denormalize z-score normalized data.

    Args:
        data: Normalized data
        mean: Original mean
        std: Original standard deviation

    Returns:
        Denormalized data in original units
    """
    return data * std + mean


class SINCPSDecompressor:
    """
    High-level interface for decompressing SINCPS checkpoints.

    Example:
        >>> decompressor = SINCPSDecompressor(
        ...     checkpoint_path='checkpoints/shear_flow.mdl',
        ...     config={'input_dim': 3, 'output_dim': 4, 'hidden_dim': 1024}
        ... )
        >>> data = decompressor.reconstruct(
        ...     spatial_shape=(256, 256),
        ...     num_timesteps=100,
        ...     timestep_indices=[0, 50, 99]
        ... )
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[dict] = None,
        normalization_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize decompressor.

        Args:
            checkpoint_path: Path to .mdl checkpoint
            config: Model configuration
            normalization_stats: Dict mapping field names to (mean, std) tuples
            device: Device for inference
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.normalization_stats = normalization_stats or {}
        self.device = device
        self.model = load_model(checkpoint_path, config, device)

    def reconstruct(
        self,
        spatial_shape: Tuple[int, ...],
        num_timesteps: int,
        timestep_indices: Optional[List[int]] = None,
        batch_size: int = 65536,
        denormalize_output: bool = False,
        field_names: Optional[List[str]] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reconstruct simulation data.

        Args:
            spatial_shape: Spatial grid dimensions
            num_timesteps: Total timesteps in simulation
            timestep_indices: Which timesteps to reconstruct
            batch_size: Inference batch size
            denormalize_output: Whether to apply denormalization
            field_names: Names of output fields (for dict output)

        Returns:
            Reconstructed data as array or dict of arrays
        """
        data = reconstruct(
            self.model,
            spatial_shape,
            num_timesteps,
            timestep_indices,
            batch_size,
            self.device
        )

        if field_names is not None:
            result = {}
            for i, name in enumerate(field_names):
                field_data = data[..., i]
                if denormalize_output and name in self.normalization_stats:
                    mean, std = self.normalization_stats[name]
                    field_data = denormalize(field_data, mean, std)
                result[name] = field_data
            return result

        return data
