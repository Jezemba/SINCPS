# Copyright 2024 Cerebras Systems.
#
# SINCS Data Processor - Optimized for Cerebras CSX
# Supports static/dynamic field splitting for improved compression
# Supports multiple coordinate systems (Cartesian, spherical, log-spherical, etc.)

import logging
import os
from glob import glob
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig

# Import the_well for metadata extraction
try:
    from the_well.data.datasets import WellDataset
    THE_WELL_AVAILABLE = True
except ImportError:
    THE_WELL_AVAILABLE = False
    logging.warning("the_well library not available. Using fallback coordinate detection.")


class SINCSDataProcessorConfig(DataConfig):
    """Configuration for SINCS data processor."""

    data_processor: Literal["SINCSDataProcessor"] = "SINCSDataProcessor"
    """Data processor name for registry."""

    data_dir: str = "/cra-1272/PhysicsAlchemists/datasets"
    """Path to the datasets directory."""

    dataset_name: str = "acoustic_scattering_inclusions"
    """Name of the dataset (subdirectory name)."""

    split: str = "train"
    """Data split to use: train, valid, or test."""

    trajectory_idx: int = 0
    """Which trajectory to use from the dataset."""

    num_timesteps: Optional[int] = None
    """Number of timesteps to use. None = all available."""

    batch_size: int = 4096
    """Global batch size (number of coordinate samples)."""

    num_samples_per_epoch: int = 100000
    """Total samples per epoch."""

    shuffle: bool = True
    """Whether to shuffle the data."""

    shuffle_seed: Optional[int] = 42
    """Random seed for shuffling."""

    num_workers: int = 0
    """Number of data loading workers. Use 0 for CSX."""

    prefetch_factor: Optional[int] = 2
    """Prefetch factor for data loading."""

    persistent_workers: bool = False
    """Keep workers alive between epochs."""

    drop_last: bool = True
    """Drop last incomplete batch."""

    use_fake_data: bool = False
    """Use synthetic data for testing compilation."""

    fake_data_shape: Tuple[int, int, int] = (64, 64, 10)
    """Shape of fake data (nx, ny, nt)."""

    # Static/Dynamic field splitting options
    split_static_dynamic: bool = True
    """Whether to separate static and dynamic fields."""

    dynamic_fields_only: bool = True
    """Only train SINCS on dynamic fields (static stored as float16)."""

    normalize_fields: bool = True
    """Apply z-score normalization to field values."""


class CoordinateDataset(Dataset):
    """Dataset that yields (coordinate, field_value) pairs."""

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        num_samples: int,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            coords: [N, input_dim] array of coordinates
            values: [N, output_dim] array of field values
            num_samples: Number of samples per epoch
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.coords = torch.from_numpy(coords).float()
        self.values = torch.from_numpy(values).float()
        self.num_points = len(coords)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Pre-generate indices for this epoch
        self._generate_indices()

    def _generate_indices(self):
        """Generate random indices for sampling."""
        if self.shuffle:
            self.indices = self.rng.choice(
                self.num_points,
                size=self.num_samples,
                replace=True  # Allow replacement for large num_samples
            )
        else:
            # Cycle through points sequentially
            self.indices = np.arange(self.num_samples) % self.num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        point_idx = self.indices[idx]
        return self.coords[point_idx], self.values[point_idx]


class FakeCoordinateDataset(Dataset):
    """Synthetic dataset for testing CSX compilation."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        input_dim: int,
        output_dim: int,
        num_samples: int,
        seed: Optional[int] = None
    ):
        """
        Args:
            shape: (nx, ny, nt) grid shape
            input_dim: Number of input coordinate dimensions
            output_dim: Number of output field dimensions
            num_samples: Samples per epoch
            seed: Random seed
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = np.random.default_rng(seed)

        # Generate random coordinates in [0, 1]
        self.coords = torch.rand(num_samples, input_dim)
        # Generate random target values
        self.values = torch.rand(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.coords[idx], self.values[idx]


class SINCSDataProcessor:
    """
    Data processor for SINCS training on Cerebras CSX.

    Supports the PhysicsAlchemists dataset structure:
        {data_dir}/{dataset_name}/data/{split}/{dataset_name}_chunk_*.hdf5

    HDF5 structure:
        - t0_fields: Fields without extra dimensions (scalar fields)
          - Static fields have shape (n_traj, H, W)
          - Dynamic fields have shape (n_traj, T, H, W)
        - t1_fields: Fields with one extra dimension (vector fields)
          - Shape: (n_traj, T, H, W, n_components)
        - dimensions: time, x, y coordinate arrays

    Static vs Dynamic Detection:
        - Static fields: No time dimension (constant across timesteps)
        - Dynamic fields: Have time dimension (vary across timesteps)
    """

    def __init__(self, config: SINCSDataProcessorConfig):
        if isinstance(config, dict):
            config = SINCSDataProcessorConfig(**config)

        self.config = config
        self.data_dir = config.data_dir
        self.dataset_name = config.dataset_name
        self.split = config.split
        self.trajectory_idx = config.trajectory_idx
        self.num_timesteps = config.num_timesteps
        self.num_samples = config.num_samples_per_epoch

        # Batch size for streaming
        self.global_batch_size = config.batch_size
        self.batch_size = get_streaming_batch_size(self.global_batch_size)

        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers
        self.drop_last = config.drop_last

        self.use_fake_data = config.use_fake_data
        self.fake_data_shape = config.fake_data_shape

        self.split_static_dynamic = config.split_static_dynamic
        self.dynamic_fields_only = config.dynamic_fields_only
        self.normalize_fields = config.normalize_fields

        # Will be populated after loading
        self.static_fields: Dict[str, np.ndarray] = {}
        self.dynamic_fields: Dict[str, np.ndarray] = {}
        self.normalization_stats: Dict[str, Dict[str, float]] = {}

        # Metadata from the_well (populated in _load_metadata)
        self.grid_type: Optional[str] = None
        self.n_spatial_dims: Optional[int] = None
        self.spatial_resolution: Optional[Tuple[int, ...]] = None
        self.coord_keys: Optional[List[str]] = None  # e.g., ['r', 'theta', 'phi'] or ['x', 'y']
        self.field_names_by_order: Optional[Dict[int, List[str]]] = None

        # Computed dimensions (populated after loading data)
        self.input_dim: Optional[int] = None  # n_spatial_dims + 1 (time)
        self.output_dim: Optional[int] = None  # number of output fields

        logging.info(f"SINCSDataProcessor initialized:")
        logging.info(f"  Dataset: {self.dataset_name}")
        logging.info(f"  Split: {self.split}")
        logging.info(f"  Batch size: {self.batch_size} (global: {self.global_batch_size})")
        logging.info(f"  Samples per epoch: {self.num_samples}")
        logging.info(f"  Use fake data: {self.use_fake_data}")
        logging.info(f"  Split static/dynamic: {self.split_static_dynamic}")

    def create_dataloader(self):
        """Create the data loader for training."""
        if self.use_fake_data:
            dataset = self._create_fake_dataset()
        else:
            dataset = self._create_real_dataset()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Dataset handles shuffling internally
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            pin_memory=False,
        )

        return dataloader

    def _create_fake_dataset(self):
        """Create synthetic dataset for testing."""
        logging.info("Creating fake dataset for CSX compilation testing")
        nx, ny, nt = self.fake_data_shape
        return FakeCoordinateDataset(
            shape=self.fake_data_shape,
            input_dim=3,  # x, y, t
            output_dim=1,  # field value
            num_samples=self.num_samples,
            seed=self.shuffle_seed
        )

    def _load_metadata(self, split_dir: str) -> None:
        """
        Load dataset metadata using the_well library.

        Extracts grid_type, n_spatial_dims, spatial_resolution, and field_names
        to handle different coordinate systems (Cartesian, spherical, etc.).
        """
        if THE_WELL_AVAILABLE:
            try:
                well_ds = WellDataset(path=split_dir)
                self.grid_type = well_ds.metadata.grid_type
                self.n_spatial_dims = well_ds.metadata.n_spatial_dims
                self.spatial_resolution = well_ds.metadata.spatial_resolution
                self.field_names_by_order = well_ds.metadata.field_names

                logging.info(f"Loaded metadata via the_well:")
                logging.info(f"  Grid type: {self.grid_type}")
                logging.info(f"  Spatial dims: {self.n_spatial_dims}")
                logging.info(f"  Resolution: {self.spatial_resolution}")
                logging.info(f"  Field names: {self.field_names_by_order}")

                # Determine coordinate keys from HDF5 dimensions
                self._detect_coord_keys(split_dir)
                return
            except Exception as e:
                logging.warning(f"the_well metadata loading failed: {e}. Using fallback.")

        # Fallback: detect coordinates directly from HDF5
        self._detect_coord_keys_fallback(split_dir)

    def _detect_coord_keys(self, split_dir: str) -> None:
        """Detect coordinate keys from HDF5 dimensions group."""
        chunk_files = sorted(glob(os.path.join(split_dir, "*.hdf5")))
        if not chunk_files:
            chunk_files = sorted(glob(os.path.join(split_dir, "*.h5")))

        if chunk_files:
            with h5py.File(chunk_files[0], 'r') as f:
                dim_keys = [k for k in f['dimensions'].keys() if k != 'time']
                self.coord_keys = dim_keys
                logging.info(f"  Coordinate keys: {self.coord_keys}")

    def _detect_coord_keys_fallback(self, split_dir: str) -> None:
        """Fallback method to detect coordinate system without the_well."""
        chunk_files = sorted(glob(os.path.join(split_dir, "*.hdf5")))
        if not chunk_files:
            chunk_files = sorted(glob(os.path.join(split_dir, "*.h5")))

        if not chunk_files:
            raise FileNotFoundError(f"No HDF5 files found in {split_dir}")

        with h5py.File(chunk_files[0], 'r') as f:
            dim_keys = list(f['dimensions'].keys())
            self.coord_keys = [k for k in dim_keys if k != 'time']
            self.n_spatial_dims = len(self.coord_keys)
            self.spatial_resolution = tuple(
                f[f'dimensions/{k}'].shape[0] for k in self.coord_keys
            )

            # Infer grid_type from coordinate keys
            if set(self.coord_keys) <= {'x', 'y', 'z'}:
                self.grid_type = 'cartesian'
            elif set(self.coord_keys) <= {'r', 'theta', 'phi'}:
                self.grid_type = 'spherical'
            elif set(self.coord_keys) <= {'log_r', 'theta', 'phi'}:
                self.grid_type = 'log_spherical'
            elif set(self.coord_keys) <= {'theta', 'phi'}:
                self.grid_type = 'equiangular'
            else:
                self.grid_type = 'unknown'

            logging.info(f"Fallback metadata detection:")
            logging.info(f"  Grid type: {self.grid_type}")
            logging.info(f"  Spatial dims: {self.n_spatial_dims}")
            logging.info(f"  Resolution: {self.spatial_resolution}")
            logging.info(f"  Coordinate keys: {self.coord_keys}")

    def _create_real_dataset(self):
        """Load real dataset from HDF5 files."""
        # Find the data directory
        split_dir = os.path.join(
            self.data_dir, self.dataset_name, "data", self.split
        )

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Load metadata first (grid_type, n_spatial_dims, coord_keys, etc.)
        self._load_metadata(split_dir)

        # Find HDF5 files - try multiple patterns
        chunk_files = []

        # Pattern 1: {dataset_name}_chunk_*.hdf5 (e.g., acoustic_scattering_*)
        pattern = os.path.join(split_dir, f"{self.dataset_name}_chunk_*.hdf5")
        chunk_files = sorted(glob(pattern))

        # Pattern 2: Any .hdf5 files (e.g., active_matter, MHD, etc.)
        if not chunk_files:
            pattern = os.path.join(split_dir, "*.hdf5")
            chunk_files = sorted(glob(pattern))

        # Pattern 3: Try .h5 extension
        if not chunk_files:
            pattern = os.path.join(split_dir, "*.h5")
            chunk_files = sorted(glob(pattern))

        if not chunk_files:
            raise FileNotFoundError(f"No HDF5 files found in {split_dir}")

        logging.info(f"Found {len(chunk_files)} chunk files")

        # Determine which chunk contains the trajectory
        trajectories_per_chunk = self._get_trajectories_per_chunk(chunk_files[0])
        chunk_idx = self.trajectory_idx // trajectories_per_chunk
        local_idx = self.trajectory_idx % trajectories_per_chunk

        if chunk_idx >= len(chunk_files):
            raise ValueError(
                f"Trajectory {self.trajectory_idx} not found. "
                f"Max trajectory: {len(chunk_files) * trajectories_per_chunk - 1}"
            )

        h5_path = chunk_files[chunk_idx]
        logging.info(
            f"Loading trajectory {self.trajectory_idx} from "
            f"{os.path.basename(h5_path)} (local index {local_idx})"
        )

        # Load and process data
        coords, values = self._load_trajectory(h5_path, local_idx)

        return CoordinateDataset(
            coords=coords,
            values=values,
            num_samples=self.num_samples,
            shuffle=self.shuffle,
            seed=self.shuffle_seed
        )

    def _get_trajectories_per_chunk(self, h5_path: str) -> int:
        """Get the number of trajectories in a chunk file."""
        with h5py.File(h5_path, 'r') as f:
            # Check t0_fields for trajectory count
            if 't0_fields' in f:
                for field_name in f['t0_fields'].keys():
                    shape = f[f't0_fields/{field_name}'].shape
                    return shape[0]
            # Check t1_fields
            if 't1_fields' in f:
                for field_name in f['t1_fields'].keys():
                    shape = f[f't1_fields/{field_name}'].shape
                    return shape[0]
        return 100  # Default fallback

    def _load_trajectory(
        self, h5_path: str, local_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single trajectory and convert to coordinate-value pairs.

        Returns:
            coords: [N, input_dim] array of (t, spatial_coords...) normalized coordinates
            values: [N, output_dim] array of field values
        """
        with h5py.File(h5_path, 'r') as f:
            # Get time dimension
            T = f['dimensions/time'].shape[0]

            # Get spatial dimensions dynamically based on detected coordinate keys
            spatial_dims = []
            for key in self.coord_keys:
                dim_size = f[f'dimensions/{key}'].shape[0]
                spatial_dims.append(dim_size)

            logging.info(f"Grid type: {self.grid_type}")
            logging.info(f"Coordinate keys: {self.coord_keys}")
            logging.info(f"Time steps: T={T}")
            logging.info(f"Spatial dimensions: {dict(zip(self.coord_keys, spatial_dims))}")

            # Set input_dim: time + spatial dimensions
            self.input_dim = 1 + len(spatial_dims)
            logging.info(f"Input dimension: {self.input_dim}")

            # Detect and load fields (pass spatial_dims as tuple)
            self._detect_and_load_fields(f, local_idx, T, tuple(spatial_dims))

        # Log field info
        logging.info(f"Static fields: {list(self.static_fields.keys())}")
        logging.info(f"Dynamic fields: {list(self.dynamic_fields.keys())}")

        # Create coordinate-value pairs from dynamic fields
        if self.dynamic_fields_only and self.dynamic_fields:
            return self._create_dynamic_dataset(T, tuple(spatial_dims))
        else:
            return self._create_combined_dataset(T, tuple(spatial_dims))

    def _detect_and_load_fields(
        self, f: h5py.File, local_idx: int, T: int, spatial_dims: Tuple[int, ...]
    ):
        """
        Detect static vs dynamic fields and load them.

        Args:
            f: Open HDF5 file handle
            local_idx: Index of trajectory within the file
            T: Number of time steps
            spatial_dims: Tuple of spatial dimension sizes, e.g., (256, 128, 256) for 3D
        """
        self.static_fields = {}
        self.dynamic_fields = {}
        n_spatial = len(spatial_dims)

        # Load t0_fields (scalar fields)
        if 't0_fields' in f:
            for field_name in f['t0_fields'].keys():
                data = f[f't0_fields/{field_name}'][local_idx]

                if len(data.shape) == n_spatial:
                    # Shape matches spatial dims only - truly static
                    self.static_fields[field_name] = data.astype(np.float32)
                    logging.info(f"  [STATIC]  {field_name}: shape {data.shape}")
                elif len(data.shape) == n_spatial + 1:
                    # Shape is (T, *spatial_dims) - dynamic
                    self.dynamic_fields[field_name] = data.astype(np.float32)
                    logging.info(f"  [DYNAMIC] {field_name}: shape {data.shape}")
                else:
                    logging.warning(f"  [SKIPPED] {field_name}: unexpected shape {data.shape}")

        # Load t1_fields (vector fields)
        if 't1_fields' in f:
            for field_name in f['t1_fields'].keys():
                data = f[f't1_fields/{field_name}'][local_idx]

                if len(data.shape) == n_spatial + 2:
                    # Shape is (T, *spatial_dims, C) - dynamic with components
                    n_components = data.shape[-1]
                    for c in range(n_components):
                        component_name = f"{field_name}_{c}"
                        self.dynamic_fields[component_name] = data[..., c].astype(np.float32)
                        logging.info(f"  [DYNAMIC] {component_name}: shape {data[..., c].shape}")
                elif len(data.shape) == n_spatial + 1:
                    # Shape is (*spatial_dims, C) - static with components
                    n_components = data.shape[-1]
                    for c in range(n_components):
                        component_name = f"{field_name}_{c}"
                        self.static_fields[component_name] = data[..., c].astype(np.float32)
                        logging.info(f"  [STATIC]  {component_name}: shape {data[..., c].shape}")
                else:
                    logging.warning(f"  [SKIPPED] {field_name}: unexpected shape {data.shape}")

    def _create_dynamic_dataset(
        self, T: int, spatial_dims: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate-value pairs from dynamic fields only.

        Args:
            T: Number of time steps
            spatial_dims: Tuple of spatial dimension sizes

        Returns:
            coords: [N, input_dim] array of normalized coordinates (t, spatial...)
            values: [N, output_dim] array of field values
        """
        n_dynamic = len(self.dynamic_fields)
        if n_dynamic == 0:
            raise ValueError("No dynamic fields found!")

        # Limit timesteps if specified
        if self.num_timesteps is not None:
            T = min(T, self.num_timesteps)

        # Stack dynamic fields into (T, *spatial_dims, n_fields)
        field_names = sorted(self.dynamic_fields.keys())
        dynamic_data = np.stack(
            [self.dynamic_fields[name][:T] for name in field_names],
            axis=-1
        )

        # Set output_dim
        self.output_dim = n_dynamic

        # Apply z-score normalization if requested
        if self.normalize_fields:
            self.normalization_stats = {}
            for i, name in enumerate(field_names):
                mean = dynamic_data[..., i].mean()
                std = dynamic_data[..., i].std()
                std = max(std, 1e-8)  # Avoid division by zero
                dynamic_data[..., i] = (dynamic_data[..., i] - mean) / std
                self.normalization_stats[name] = {'mean': float(mean), 'std': float(std)}
                logging.info(f"  {name}: mean={mean:.6f}, std={std:.6f}")

        # Create coordinate grids (normalized to [0, 1])
        # Time is always first
        t_coords = np.linspace(0, 1, T)

        # Spatial coordinates based on n_spatial_dims
        spatial_coord_arrays = [np.linspace(0, 1, dim) for dim in spatial_dims]

        # Create meshgrid: (t, spatial_0, spatial_1, ..., spatial_n)
        all_coord_arrays = [t_coords] + spatial_coord_arrays
        mesh = np.meshgrid(*all_coord_arrays, indexing='ij')

        # Flatten and stack to (N, input_dim) coordinates
        coords = np.stack([m.flatten() for m in mesh], axis=-1)

        # Flatten values to (N, n_fields)
        values = dynamic_data.reshape(-1, n_dynamic)

        logging.info(f"Created dataset with {len(coords)} samples, {n_dynamic} fields")
        logging.info(f"  Input dim: {self.input_dim}, Output dim: {self.output_dim}")
        logging.info(f"  Coords shape: {coords.shape}")
        logging.info(f"  Values shape: {values.shape}")

        return coords.astype(np.float32), values.astype(np.float32)

    def _create_combined_dataset(
        self, T: int, spatial_dims: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate-value pairs from all fields (static + dynamic).

        Args:
            T: Number of time steps
            spatial_dims: Tuple of spatial dimension sizes

        Returns:
            coords: [N, input_dim] array of normalized coordinates
            values: [N, output_dim] array of field values
        """
        # For static fields, expand to match dynamic fields' temporal dimension
        if self.num_timesteps is not None:
            T = min(T, self.num_timesteps)

        all_fields = []
        field_names = []

        # Add dynamic fields
        for name in sorted(self.dynamic_fields.keys()):
            data = self.dynamic_fields[name][:T]  # (T, *spatial_dims)
            all_fields.append(data)
            field_names.append(name)

        # Add static fields (expanded to T timesteps)
        for name in sorted(self.static_fields.keys()):
            data = self.static_fields[name]  # (*spatial_dims)
            # Expand to (T, *spatial_dims)
            target_shape = (T,) + spatial_dims
            data_expanded = np.broadcast_to(data[np.newaxis, ...], target_shape)
            all_fields.append(data_expanded)
            field_names.append(name)

        if not all_fields:
            raise ValueError("No fields found!")

        # Stack all fields
        combined_data = np.stack(all_fields, axis=-1)  # (T, *spatial_dims, n_fields)
        n_fields = combined_data.shape[-1]

        # Set output_dim
        self.output_dim = n_fields

        # Apply normalization
        if self.normalize_fields:
            self.normalization_stats = {}
            for i, name in enumerate(field_names):
                mean = combined_data[..., i].mean()
                std = combined_data[..., i].std()
                std = max(std, 1e-8)
                combined_data[..., i] = (combined_data[..., i] - mean) / std
                self.normalization_stats[name] = {'mean': float(mean), 'std': float(std)}

        # Create coordinate grids (normalized to [0, 1])
        t_coords = np.linspace(0, 1, T)
        spatial_coord_arrays = [np.linspace(0, 1, dim) for dim in spatial_dims]

        # Create meshgrid
        all_coord_arrays = [t_coords] + spatial_coord_arrays
        mesh = np.meshgrid(*all_coord_arrays, indexing='ij')

        # Flatten and stack
        coords = np.stack([m.flatten() for m in mesh], axis=-1)
        values = combined_data.reshape(-1, n_fields)

        logging.info(f"Created combined dataset with {len(coords)} samples, {n_fields} fields")
        logging.info(f"  Input dim: {self.input_dim}, Output dim: {self.output_dim}")

        return coords.astype(np.float32), values.astype(np.float32)

    def get_static_fields_compressed(self) -> Dict[str, bytes]:
        """
        Get static fields compressed as float16.

        Returns dict mapping field name to compressed bytes.
        This should be called after create_dataloader() to get the static fields
        that were separated from the SINCS training data.
        """
        compressed = {}
        for name, data in self.static_fields.items():
            # Convert to float16 and serialize
            data_f16 = data.astype(np.float16)
            compressed[name] = data_f16.tobytes()
        return compressed

    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics for denormalization during inference."""
        return self.normalization_stats

    def get_input_dim(self) -> int:
        """
        Get the input dimension for the model.

        Returns n_spatial_dims + 1 (for time).
        Must be called after create_dataloader().
        """
        if self.input_dim is None:
            raise RuntimeError("input_dim not set. Call create_dataloader() first.")
        return self.input_dim

    def get_output_dim(self) -> int:
        """
        Get the output dimension for the model.

        Returns the number of fields being predicted.
        Must be called after create_dataloader().
        """
        if self.output_dim is None:
            raise RuntimeError("output_dim not set. Call create_dataloader() first.")
        return self.output_dim

    def get_metadata(self) -> Dict:
        """
        Get dataset metadata including coordinate system info.

        Returns dict with grid_type, n_spatial_dims, coord_keys, etc.
        """
        return {
            'grid_type': self.grid_type,
            'n_spatial_dims': self.n_spatial_dims,
            'spatial_resolution': self.spatial_resolution,
            'coord_keys': self.coord_keys,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        }
