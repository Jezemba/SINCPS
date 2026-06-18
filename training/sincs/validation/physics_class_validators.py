#!/usr/bin/env python3
"""
Physics Class-Based Validators for SINCS Compression Evaluation

This module implements physics-grounded metrics for evaluating SINCS compressed
models across The Well datasets. Each physics class has specific validation tests
based on the underlying governing equations.

Reference: physics_metrics_for_sincs.md
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
from scipy import fft
from scipy.ndimage import laplace

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset to Physics Class Mapping
# =============================================================================

DATASET_TO_CLASS = {
    # Acoustic Scattering (Wave Equation)
    'acoustic_scattering_discontinuous': 'acoustic_scattering',
    'acoustic_scattering_inclusions': 'acoustic_scattering',
    'acoustic_scattering_maze': 'acoustic_scattering',

    # Compressible Euler (Gas Dynamics)
    'euler_multi_quadrants': 'euler',

    # Reaction-Diffusion
    'gray_scott': 'gray_scott',

    # Magnetohydrodynamics
    'mhd_64': 'mhd',
    'mhd_256': 'mhd',

    # Convection
    'rayleigh_benard': 'rayleigh_benard',
    'rayleigh_benard_uniform': 'rayleigh_benard',

    # Instabilities
    'rayleigh_taylor_instability': 'rayleigh_taylor',

    # Shear/Mixing
    'shear_flow': 'shear_flow',

    # Shallow Water / Geophysical
    'planetswe': 'shallow_water',

    # Astrophysical
    'supernova_explosion_64': 'supernova',
    'supernova_explosion_128': 'supernova',
    'convective_envelope_rsg': 'convective_envelope',
    'post_neutron_star_merger': 'neutron_star_merger',

    # Active Matter
    'active_matter': 'active_matter',

    # Wave Equations (Helmholtz)
    'helmholtz_staircase': 'helmholtz',

    # Turbulent/Radiative
    'turbulent_radiative_layer_2d': 'turbulent_radiative',
    'turbulent_radiative_layer_3d': 'turbulent_radiative',
    'turbulence_gravity_cooling': 'turbulence_gravity',

    # Viscoelastic
    'viscoelastic_instability': 'viscoelastic',
}


def get_physics_class(dataset_name: str) -> str:
    """Get the physics class for a dataset name."""
    # Normalize name (lowercase, handle variations)
    name_lower = dataset_name.lower().replace('-', '_')

    if name_lower in DATASET_TO_CLASS:
        return DATASET_TO_CLASS[name_lower]

    # Try partial matching
    for key, value in DATASET_TO_CLASS.items():
        if key in name_lower or name_lower in key:
            return value

    return 'unknown'


# =============================================================================
# Helper Functions
# =============================================================================

def laplacian_2d(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Compute 2D Laplacian using finite differences."""
    return laplace(field) / (dx ** 2)


def laplacian_3d(field: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """Compute 3D Laplacian."""
    d2x = np.gradient(np.gradient(field, dx, axis=-3), dx, axis=-3)
    d2y = np.gradient(np.gradient(field, dy, axis=-2), dy, axis=-2)
    d2z = np.gradient(np.gradient(field, dz, axis=-1), dz, axis=-1)
    return d2x + d2y + d2z


def divergence_2d(vx: np.ndarray, vy: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute 2D divergence."""
    dvx_dx = np.gradient(vx, dx, axis=-1)
    dvy_dy = np.gradient(vy, dy, axis=-2)
    return dvx_dx + dvy_dy


def divergence_3d(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                  dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """Compute 3D divergence."""
    dvx_dx = np.gradient(vx, dx, axis=-3)
    dvy_dy = np.gradient(vy, dy, axis=-2)
    dvz_dz = np.gradient(vz, dz, axis=-1)
    return dvx_dx + dvy_dy + dvz_dz


def curl_2d(vx: np.ndarray, vy: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute 2D vorticity (z-component of curl)."""
    dvy_dx = np.gradient(vy, dx, axis=-1)
    dvx_dy = np.gradient(vx, dy, axis=-2)
    return dvy_dx - dvx_dy


def curl_3d(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
            dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D curl."""
    dvz_dy = np.gradient(vz, dy, axis=-2)
    dvy_dz = np.gradient(vy, dz, axis=-1)
    dvx_dz = np.gradient(vx, dz, axis=-1)
    dvz_dx = np.gradient(vz, dx, axis=-3)
    dvy_dx = np.gradient(vy, dx, axis=-3)
    dvx_dy = np.gradient(vx, dy, axis=-2)

    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy

    return omega_x, omega_y, omega_z


def spectral_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute absolute spectral error (L2 norm of FFT difference).

    Note: Returns absolute error, not relative, to avoid division issues
    with z-normalized or sparse data.
    """
    fft_pred = np.fft.fftn(pred)
    fft_true = np.fft.fftn(true)
    return float(np.linalg.norm(fft_pred - fft_true))


def compute_conservation_metrics(raw_total: np.ndarray, recon_total: np.ndarray) -> Dict[str, float]:
    """
    Compute conservation error metrics (absolute and relative percentage).

    Args:
        raw_total: Array of summed values per timestep (ground truth)
        recon_total: Array of summed values per timestep (reconstructed)

    Returns:
        Dictionary with:
        - abs_error: Absolute L2 error of totals
        - error_pct: Relative percentage error (||diff|| / ||raw|| * 100)

    Note: Uses L2-norm-based relative error which is more stable for
    z-normalized data than range-based normalization. For conservation
    metrics, this represents "error magnitude relative to total magnitude
    of the conserved quantity across all timesteps".
    """
    # Absolute error (L2 norm of difference)
    abs_error = float(np.linalg.norm(raw_total - recon_total))

    # Relative percentage error: ||error|| / ||raw|| * 100
    # This is the standard relative L2 error, stable for conservation time series
    raw_norm = float(np.linalg.norm(raw_total))
    if raw_norm > 1e-10:
        error_pct = float(100.0 * abs_error / raw_norm)
    else:
        # If raw totals are near zero, try recon norm
        recon_norm = float(np.linalg.norm(recon_total))
        if recon_norm > 1e-10:
            error_pct = float(100.0 * abs_error / recon_norm)
        else:
            error_pct = 0.0 if abs_error < 1e-10 else float('inf')

    return {'abs_error': abs_error, 'error_pct': error_pct}


def compute_field_metrics(raw_field: np.ndarray, recon_field: np.ndarray) -> Dict[str, float]:
    """
    Compute field comparison metrics (absolute and percentage).

    Args:
        raw_field: Ground truth field array
        recon_field: Reconstructed field array

    Returns:
        Dictionary with:
        - abs_error: Absolute L2 error
        - error_pct: Percentage error normalized by range (NRMSE)
    """
    # Absolute error (L2 norm of difference)
    abs_error = float(np.linalg.norm(raw_field - recon_field))

    # Percentage error normalized by range
    data_range = float(np.max(raw_field) - np.min(raw_field))
    if data_range > 1e-10:
        error_pct = float(100.0 * abs_error / (data_range * np.sqrt(raw_field.size)))
    else:
        error_pct = 0.0 if abs_error < 1e-10 else float('inf')

    return {'abs_error': abs_error, 'error_pct': error_pct}


def power_spectrum_1d(field: np.ndarray, dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D radially-averaged power spectrum."""
    ndim = field.ndim
    fft_field = np.fft.fftn(field)
    power = np.abs(fft_field) ** 2

    # Get frequency magnitudes
    freq_arrays = [np.fft.fftfreq(s, d=dx) for s in field.shape]
    freq_grids = np.meshgrid(*freq_arrays, indexing='ij')
    freq_mag = np.sqrt(sum(f**2 for f in freq_grids))

    # Bin by frequency magnitude
    max_freq = np.max(freq_mag)
    n_bins = min(field.shape) // 2
    bins = np.linspace(0, max_freq, n_bins + 1)

    spectrum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (freq_mag >= bins[i]) & (freq_mag < bins[i+1])
        if np.sum(mask) > 0:
            spectrum[i] = np.mean(power[mask])

    k = (bins[:-1] + bins[1:]) / 2
    return k, spectrum


def gradient_error(pred: np.ndarray, true: np.ndarray, dx: float = 1.0) -> float:
    """Compute relative gradient error."""
    grad_pred = np.gradient(pred, dx)
    grad_true = np.gradient(true, dx)

    if isinstance(grad_pred, list):
        grad_pred = np.array(grad_pred)
        grad_true = np.array(grad_true)

    return float(np.linalg.norm(grad_pred - grad_true) / (np.linalg.norm(grad_true) + 1e-10))


def temporal_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute mean temporal correlation across spatial points."""
    correlations = []
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_true = true.reshape(true.shape[0], -1)

    for i in range(flat_pred.shape[1]):
        r = np.corrcoef(flat_pred[:, i], flat_true[:, i])[0, 1]
        if not np.isnan(r):
            correlations.append(r)

    return float(np.mean(correlations)) if correlations else 0.0


# =============================================================================
# Base Validator Class
# =============================================================================

class PhysicsClassValidator(ABC):
    """Base class for physics-based validation."""

    def __init__(self, raw_data: Dict[str, np.ndarray],
                 reconstructed_data: Dict[str, np.ndarray],
                 metadata: Dict[str, Any] = None):
        """
        Args:
            raw_data: Dictionary of raw field arrays
            reconstructed_data: Dictionary of reconstructed field arrays
            metadata: Optional metadata (grid spacing, physical constants, etc.)
        """
        self.raw = raw_data
        self.recon = reconstructed_data
        self.metadata = metadata or {}

        # Default grid spacing
        self.dx = self.metadata.get('dx', 1.0)
        self.dy = self.metadata.get('dy', 1.0)
        self.dz = self.metadata.get('dz', 1.0)
        self.dt = self.metadata.get('dt', 1.0)

    def _get_field(self, data: Dict, *names) -> Optional[np.ndarray]:
        """Get field from data dict, trying multiple possible names.

        This avoids the numpy 'or' ambiguity issue when checking multiple field names.
        """
        for name in names:
            val = data.get(name)
            if val is not None:
                return val
        return None

    @abstractmethod
    def get_primary_metrics(self) -> Dict[str, Any]:
        """Compute primary physics metrics (conservation laws)."""
        pass

    @abstractmethod
    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Compute secondary metrics (constitutive relations)."""
        pass

    def get_universal_metrics(self) -> Dict[str, Any]:
        """Compute universal metrics applicable to all physics classes.

        Reports only:
        - PSNR (Peak Signal-to-Noise Ratio)
        - Absolute L2 Error (no division, avoids issues with z-normalized/sparse data)
        - Absolute Spectral Error (no division)
        """
        results = {}

        # Find a representative DYNAMIC field (ndim >= 3) to compute universal metrics
        # This avoids picking static fields that are copied unchanged
        field_name = None
        for name in self.raw.keys():
            if self.raw[name] is not None and self.raw[name].ndim >= 3:
                field_name = name
                break

        # Fallback to any field if no dynamic field found
        if field_name is None:
            for name in self.raw.keys():
                if self.raw[name] is not None and self.raw[name].ndim >= 2:
                    field_name = name
                    break

        if field_name is None:
            return {'error': 'No suitable field found for universal metrics'}

        raw_field = self.raw[field_name]
        recon_field = self.recon.get(field_name)

        if recon_field is None:
            return {'error': f'Reconstructed field {field_name} not found'}

        # Ensure shapes match
        min_shape = tuple(min(r, c) for r, c in zip(raw_field.shape, recon_field.shape))
        raw_field = raw_field[tuple(slice(0, s) for s in min_shape)]
        recon_field = recon_field[tuple(slice(0, s) for s in min_shape)]

        results['field_used'] = field_name

        # Absolute L2 Error (no division - avoids issues with z-normalized/sparse data)
        abs_l2 = float(np.linalg.norm(raw_field - recon_field))
        results['abs_l2_error'] = abs_l2

        # Range-normalized L2 Error (percentage) - robust alternative to relative L2
        # Divides by (max - min) instead of ||true||, which is stable for z-normalized data
        data_range = float(np.max(raw_field) - np.min(raw_field))
        if data_range > 1e-10:
            # Normalize by range and convert to percentage
            results['l2_error_pct'] = float(100.0 * abs_l2 / (data_range * np.sqrt(raw_field.size)))
        else:
            results['l2_error_pct'] = 0.0 if abs_l2 < 1e-10 else float('inf')

        # Absolute Spectral Error (no division)
        abs_spectral = spectral_error(recon_field, raw_field)
        results['spectral_error'] = abs_spectral

        # Spectral Error Percentage (range-normalized in frequency domain)
        fft_raw = np.fft.fftn(raw_field)
        fft_range = float(np.max(np.abs(fft_raw)) - np.min(np.abs(fft_raw)))
        if fft_range > 1e-10:
            results['spectral_error_pct'] = float(100.0 * abs_spectral / (fft_range * np.sqrt(raw_field.size)))
        else:
            results['spectral_error_pct'] = 0.0 if abs_spectral < 1e-10 else float('inf')

        # Peak Signal-to-Noise Ratio
        # Uses max value for normalization which is more robust than norm-based approaches
        max_val = np.max(np.abs(raw_field))
        mse = np.mean((raw_field - recon_field) ** 2)
        if mse > 0 and max_val > 0:
            results['psnr'] = float(20 * np.log10(max_val / (np.sqrt(mse) + 1e-10)))
        else:
            results['psnr'] = float('inf') if mse == 0 else 0.0

        return results

    def run_validation(self) -> Dict[str, Any]:
        """Run validation and return results.

        Computes universal metrics (PSNR, absolute L2 error, spectral error) plus
        physics-specific metrics (conservation laws, etc.) using absolute errors
        and range-normalized percentages to avoid division issues with z-normalized
        or sparse data.
        """
        results = {
            'physics_class': self.__class__.__name__,
            'primary_metrics': {},
            'secondary_metrics': {},
            'universal_metrics': {},
            'summary': {},
        }

        # Compute universal metrics: PSNR, absolute L2 error, spectral error
        try:
            results['universal_metrics'] = self.get_universal_metrics()
        except Exception as e:
            logger.error(f"Universal metrics failed: {e}")
            results['universal_metrics'] = {'error': str(e)}

        # Compute primary metrics (conservation laws)
        try:
            results['primary_metrics'] = self.get_primary_metrics()
        except Exception as e:
            logger.error(f"Primary metrics failed: {e}")
            results['primary_metrics'] = {'error': str(e)}

        # Compute secondary metrics (physics-specific)
        try:
            results['secondary_metrics'] = self.get_secondary_metrics()
        except Exception as e:
            logger.error(f"Secondary metrics failed: {e}")
            results['secondary_metrics'] = {'error': str(e)}

        # Generate summary
        results['summary'] = self._generate_summary(results)

        return results

    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary based on all computed metrics.

        Includes universal metrics (PSNR, absolute L2 error, spectral error) and
        physics-specific metrics from primary and secondary categories.
        """
        summary = {
            'metrics_computed': ['psnr', 'abs_l2_error', 'l2_error_pct', 'spectral_error', 'spectral_error_pct'],
            'notes': [],
        }

        universal = results.get('universal_metrics', {})
        primary = results.get('primary_metrics', {})
        secondary = results.get('secondary_metrics', {})

        has_error = False

        if 'error' in universal:
            has_error = True
            summary['notes'].append(f"Universal metrics error: {universal['error']}")
        else:
            # Report universal metrics
            if 'psnr' in universal:
                summary['psnr'] = universal['psnr']
            if 'abs_l2_error' in universal:
                summary['abs_l2_error'] = universal['abs_l2_error']
            if 'l2_error_pct' in universal:
                summary['l2_error_pct'] = universal['l2_error_pct']
            if 'spectral_error' in universal:
                summary['spectral_error'] = universal['spectral_error']
            if 'spectral_error_pct' in universal:
                summary['spectral_error_pct'] = universal['spectral_error_pct']
            if 'field_used' in universal:
                summary['field_used'] = universal['field_used']

        if 'error' in primary:
            summary['notes'].append(f"Primary metrics error: {primary['error']}")
        else:
            # Add primary metric names to the list
            for key in primary.keys():
                if key not in summary['metrics_computed']:
                    summary['metrics_computed'].append(key)

        if 'error' in secondary:
            summary['notes'].append(f"Secondary metrics error: {secondary['error']}")
        else:
            # Add secondary metric names to the list
            for key in secondary.keys():
                if key not in summary['metrics_computed']:
                    summary['metrics_computed'].append(key)

        summary['status'] = 'error' if has_error else 'completed'

        return summary


# =============================================================================
# Physics Class Validators
# =============================================================================

class AcousticScatteringValidator(PhysicsClassValidator):
    """
    Validator for Acoustic Scattering datasets.

    Governing equations:
        dp/dt + K(x,y)(du/dx + dv/dy) = 0
        du/dt + (1/rho) dp/dx = 0
        dv/dt + (1/rho) dp/dy = 0
    """

    def __init__(self, raw_data, reconstructed_data, metadata=None):
        super().__init__(raw_data, reconstructed_data, metadata)
        self.K = self.metadata.get('bulk_modulus', 4.0)

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Energy conservation and wave speed (absolute + percentage errors)."""
        results = {}

        # Get fields (use helper to avoid numpy 'or' issues)
        raw_p = self._get_field(self.raw, 'pressure')
        raw_vx = self._get_field(self.raw, 'velocity_x', 'velocity_0')
        raw_vy = self._get_field(self.raw, 'velocity_y', 'velocity_1')
        raw_rho = self._get_field(self.raw, 'density')

        recon_p = self._get_field(self.recon, 'pressure')
        recon_vx = self._get_field(self.recon, 'velocity_x', 'velocity_0')
        recon_vy = self._get_field(self.recon, 'velocity_y', 'velocity_1')
        recon_rho = self._get_field(self.recon, 'density')

        if raw_p is None or raw_rho is None:
            return {'error': 'Missing pressure or density fields'}

        # Wave speed consistency: c = sqrt(K/rho)
        if 'speed_of_sound' in self.raw:
            raw_c = self.raw['speed_of_sound']
            recon_c = self.recon.get('speed_of_sound', raw_c)
        else:
            raw_c = np.sqrt(self.K / (raw_rho + 1e-10))
            recon_c = np.sqrt(self.K / (recon_rho + 1e-10))

        # Wave speed error (absolute + percentage)
        wave_metrics = compute_field_metrics(raw_c, recon_c)
        results['wave_speed_abs_error'] = wave_metrics['abs_error']
        results['wave_speed_error_pct'] = wave_metrics['error_pct']

        # Energy conservation
        if raw_vx is not None and raw_vy is not None:
            # Handle time dimension
            if raw_p.ndim == 3:  # (T, H, W)
                T = raw_p.shape[0]
                raw_energies = []
                recon_energies = []

                for t in range(T):
                    # Kinetic: 0.5 * rho * (u^2 + v^2)
                    raw_ke = 0.5 * raw_rho * (raw_vx[t]**2 + raw_vy[t]**2)
                    # Potential: 0.5 * p^2 / K
                    raw_pe = 0.5 * raw_p[t]**2 / self.K
                    raw_energies.append(np.sum(raw_ke + raw_pe))

                    if recon_p is not None and recon_vx is not None:
                        recon_ke = 0.5 * recon_rho * (recon_vx[t]**2 + recon_vy[t]**2)
                        recon_pe = 0.5 * recon_p[t]**2 / self.K
                        recon_energies.append(np.sum(recon_ke + recon_pe))

                raw_energies = np.array(raw_energies)
                recon_energies = np.array(recon_energies)

                # Energy conservation error (absolute + percentage)
                energy_metrics = compute_conservation_metrics(raw_energies, recon_energies)
                results['energy_conservation_abs_error'] = energy_metrics['abs_error']
                results['energy_conservation_error_pct'] = energy_metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Impedance and frequency spectrum."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        # Acoustic impedance: Z = rho * c
        if 'speed_of_sound' in self.raw:
            raw_c = self.raw['speed_of_sound']
            recon_c = self.recon.get('speed_of_sound', raw_c)
        else:
            raw_c = np.sqrt(self.K / (raw_rho + 1e-10))
            recon_c = np.sqrt(self.K / (recon_rho + 1e-10))

        raw_Z = raw_rho * raw_c
        recon_Z = recon_rho * recon_c

        results['impedance_relative_error'] = float(
            np.mean(np.abs(raw_Z - recon_Z)) / (np.mean(np.abs(raw_Z)) + 1e-10)
        )

        # Frequency spectrum (pressure field)
        raw_p = self.raw.get('pressure')
        recon_p = self.recon.get('pressure')

        if raw_p is not None and recon_p is not None:
            # Use middle timestep for spectrum comparison
            if raw_p.ndim == 3:
                t_mid = raw_p.shape[0] // 2
                raw_p_mid = raw_p[t_mid]
                recon_p_mid = recon_p[t_mid]
            else:
                raw_p_mid = raw_p
                recon_p_mid = recon_p

            results['spectral_pressure_error'] = spectral_error(recon_p_mid, raw_p_mid)

        return results


class EulerValidator(PhysicsClassValidator):
    """
    Validator for Euler Multi-Quadrants (Compressible Gas Dynamics).

    Conservation laws:
        Mass: d(rho)/dt + div(rho*v) = 0
        Momentum: d(rho*v)/dt + div(rho*v*v + p*I) = 0
        Energy: dE/dt + div((E+p)*v) = 0
    """

    def __init__(self, raw_data, reconstructed_data, metadata=None):
        super().__init__(raw_data, reconstructed_data, metadata)
        self.gamma = self.metadata.get('gamma', 1.4)  # Specific heat ratio

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass and energy conservation (absolute + percentage errors)."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        # Mass conservation
        if raw_rho.ndim >= 3:  # Has time dimension
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            mass_metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = mass_metrics['abs_error']
            results['mass_conservation_error_pct'] = mass_metrics['error_pct']

        # Energy conservation
        raw_E = self._get_field(self.raw, 'energy', 'total_energy')
        recon_E = self._get_field(self.recon, 'energy', 'total_energy')

        if raw_E is not None and raw_E.ndim >= 3:
            raw_total_E = np.sum(raw_E, axis=tuple(range(1, raw_E.ndim)))
            recon_total_E = np.sum(recon_E, axis=tuple(range(1, recon_E.ndim)))

            energy_metrics = compute_conservation_metrics(raw_total_E, recon_total_E)
            results['energy_conservation_abs_error'] = energy_metrics['abs_error']
            results['energy_conservation_error_pct'] = energy_metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Entropy and shock structure."""
        results = {}

        raw_p = self.raw.get('pressure')
        raw_rho = self.raw.get('density')
        recon_p = self.recon.get('pressure')
        recon_rho = self.recon.get('density')

        if raw_p is not None and raw_rho is not None:
            # Entropy: S = p / rho^gamma
            raw_S = raw_p / (raw_rho ** self.gamma + 1e-10)
            recon_S = recon_p / (recon_rho ** self.gamma + 1e-10)

            results['entropy_relative_error'] = float(
                np.mean(np.abs(raw_S - recon_S)) / (np.mean(np.abs(raw_S)) + 1e-10)
            )

            # Entropy should increase at shocks
            if raw_S.ndim >= 3:
                dS_dt_raw = np.gradient(raw_S, self.dt, axis=0)
                dS_dt_recon = np.gradient(recon_S, self.dt, axis=0)

                # Fraction of domain where entropy increases
                results['raw_entropy_increase_fraction'] = float(np.mean(dS_dt_raw >= -1e-6))
                results['recon_entropy_increase_fraction'] = float(np.mean(dS_dt_recon >= -1e-6))

        # Shock sharpness (gradient magnitude of density)
        if raw_rho is not None and raw_rho.ndim >= 3:
            t_mid = raw_rho.shape[0] // 2
            grad_raw = np.gradient(raw_rho[t_mid])
            grad_recon = np.gradient(recon_rho[t_mid])

            if isinstance(grad_raw, list):
                grad_mag_raw = np.sqrt(sum(g**2 for g in grad_raw))
                grad_mag_recon = np.sqrt(sum(g**2 for g in grad_recon))
            else:
                grad_mag_raw = np.abs(grad_raw)
                grad_mag_recon = np.abs(grad_recon)

            results['shock_sharpness_error'] = float(
                np.mean(np.abs(grad_mag_raw - grad_mag_recon)) / (np.mean(grad_mag_raw) + 1e-10)
            )

        return results


class GrayScottValidator(PhysicsClassValidator):
    """
    Validator for Gray-Scott Reaction-Diffusion.

    Governing equations:
        dA/dt = D_A * laplacian(A) - A*B^2 + F*(1-A)
        dB/dt = D_B * laplacian(B) + A*B^2 - (F+k)*B
    """

    def __init__(self, raw_data, reconstructed_data, metadata=None):
        super().__init__(raw_data, reconstructed_data, metadata)
        self.D_A = self.metadata.get('D_A', 2e-5)
        self.D_B = self.metadata.get('D_B', 1e-5)
        self.F = self.metadata.get('feed_rate', 0.04)
        self.k = self.metadata.get('kill_rate', 0.06)

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Pattern spectral error (absolute + percentage)."""
        results = {}

        raw_B = self._get_field(self.raw, 'B', 'concentration_B')
        recon_B = self._get_field(self.recon, 'B', 'concentration_B')

        if raw_B is None:
            return {'error': 'Missing B concentration field'}

        # Pattern wavelength from FFT
        if raw_B.ndim >= 3:
            t_mid = raw_B.shape[0] // 2
            raw_B_mid = raw_B[t_mid]
            recon_B_mid = recon_B[t_mid]
        else:
            raw_B_mid = raw_B
            recon_B_mid = recon_B

        # Pattern spectral error (absolute)
        results['spectral_pattern_abs_error'] = spectral_error(recon_B_mid, raw_B_mid)

        # Spectral pattern percentage error
        fft_true = np.fft.fftn(raw_B_mid)
        fft_range = float(np.max(np.abs(fft_true)) - np.min(np.abs(fft_true)))
        if fft_range > 1e-10:
            results['spectral_pattern_error_pct'] = float(
                100.0 * results['spectral_pattern_abs_error'] / (fft_range * np.sqrt(raw_B_mid.size))
            )
        else:
            results['spectral_pattern_error_pct'] = 0.0

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Pattern characteristics."""
        return {}


class MHDValidator(PhysicsClassValidator):
    """
    Validator for Magnetohydrodynamics.

    Key constraint: div(B) = 0 (solenoidal)
    Conservation: mass, momentum, magnetic helicity
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Magnetic energy conservation (absolute + percentage)."""
        results = {}

        # Check for 3D magnetic field components
        raw_Bx = self._get_field(self.raw, 'magnetic_field_x', 'magnetic_field_0')
        raw_By = self._get_field(self.raw, 'magnetic_field_y', 'magnetic_field_1')
        raw_Bz = self._get_field(self.raw, 'magnetic_field_z', 'magnetic_field_2')

        recon_Bx = self._get_field(self.recon, 'magnetic_field_x', 'magnetic_field_0')
        recon_By = self._get_field(self.recon, 'magnetic_field_y', 'magnetic_field_1')
        recon_Bz = self._get_field(self.recon, 'magnetic_field_z', 'magnetic_field_2')

        if raw_Bx is None:
            return {'error': 'Missing magnetic field components'}

        # Magnetic energy
        B_sq_raw = raw_Bx**2 + raw_By**2 + (raw_Bz**2 if raw_Bz is not None else 0)
        B_sq_recon = recon_Bx**2 + recon_By**2 + (recon_Bz**2 if recon_Bz is not None else 0)

        if B_sq_raw.ndim >= 3:
            raw_mag_energy = np.sum(B_sq_raw / 2, axis=tuple(range(1, B_sq_raw.ndim)))
            recon_mag_energy = np.sum(B_sq_recon / 2, axis=tuple(range(1, B_sq_recon.ndim)))

            metrics = compute_conservation_metrics(raw_mag_energy, recon_mag_energy)
            results['magnetic_energy_abs_error'] = metrics['abs_error']
            results['magnetic_energy_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary MHD metrics."""
        return {}


class RayleighBenardValidator(PhysicsClassValidator):
    """
    Validator for Rayleigh-Benard Convection.

    Key quantities: Nusselt number, incompressibility, vertical heat flux
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Incompressibility and energy."""
        results = {}

        raw_vx = self._get_field(self.raw, 'velocity_x', 'velocity_0')
        raw_vy = self._get_field(self.raw, 'velocity_y', 'velocity_1')

        if raw_vx is None:
            return {'error': 'Missing velocity fields'}

        recon_vx = self._get_field(self.recon, 'velocity_x', 'velocity_0')
        recon_vy = self._get_field(self.recon, 'velocity_y', 'velocity_1')

        # Incompressibility: div(v) = 0
        if raw_vx.ndim >= 3:
            t_mid = raw_vx.shape[0] // 2
            div_v_raw = divergence_2d(raw_vx[t_mid], raw_vy[t_mid], self.dx, self.dy)
            div_v_recon = divergence_2d(recon_vx[t_mid], recon_vy[t_mid], self.dx, self.dy)
        else:
            div_v_raw = divergence_2d(raw_vx, raw_vy, self.dx, self.dy)
            div_v_recon = divergence_2d(recon_vx, recon_vy, self.dx, self.dy)

        # Incompressibility error (absolute + percentage)
        metrics = compute_field_metrics(div_v_raw, div_v_recon)
        results['incompressibility_abs_error'] = metrics['abs_error']
        results['incompressibility_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}


class RayleighTaylorValidator(PhysicsClassValidator):
    """
    Validator for Rayleigh-Taylor Instability.

    Key quantities: Mixing width, growth rate, Atwood number
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass conservation (absolute + percentage)."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        # Mass conservation
        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}

        rho_max_recon = np.max(recon_rho)
        rho_min_recon = np.min(recon_rho)
        At_recon = (rho_max_recon - rho_min_recon) / (rho_max_recon + rho_min_recon + 1e-10)

        results['raw_atwood_number'] = float(At_raw)
        results['recon_atwood_number'] = float(At_recon)
        results['atwood_number_error'] = float(abs(At_raw - At_recon))

        # Density contrast
        results['density_contrast_raw'] = float(rho_max_raw / (rho_min_raw + 1e-10))
        results['density_contrast_recon'] = float(rho_max_recon / (rho_min_recon + 1e-10))

        return results


class ShallowWaterValidator(PhysicsClassValidator):
    """
    Validator for Shallow Water Equations (PlanetSWE).

    Key quantities: Potential vorticity, total energy
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Energy and potential vorticity."""
        results = {}

        raw_u = self._get_field(self.raw, 'velocity_x', 'velocity_0')
        raw_v = self._get_field(self.raw, 'velocity_y', 'velocity_1')
        raw_h = self._get_field(self.raw, 'height', 'eta')

        if raw_u is None or raw_h is None:
            return {'error': 'Missing velocity or height fields'}

        recon_u = self._get_field(self.recon, 'velocity_x', 'velocity_0')
        recon_v = self._get_field(self.recon, 'velocity_y', 'velocity_1')
        recon_h = self._get_field(self.recon, 'height', 'eta')

        g = self.metadata.get('gravity', 9.81)

        # Total energy: E = 0.5 * h * (u^2 + v^2) + 0.5 * g * h^2
        if raw_h.ndim >= 3:
            raw_KE = 0.5 * raw_h * (raw_u**2 + raw_v**2)
            raw_PE = 0.5 * g * raw_h**2
            raw_E = np.sum(raw_KE + raw_PE, axis=tuple(range(1, raw_h.ndim)))

            recon_KE = 0.5 * recon_h * (recon_u**2 + recon_v**2)
            recon_PE = 0.5 * g * recon_h**2
            recon_E = np.sum(recon_KE + recon_PE, axis=tuple(range(1, recon_h.ndim)))

            metrics = compute_conservation_metrics(raw_E, recon_E)
            results['energy_conservation_abs_error'] = metrics['abs_error']
            results['energy_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}


class SupernovaValidator(PhysicsClassValidator):
    """
    Validator for Supernova Explosion simulations.

    Key quantities: Mass, total energy, expansion rate
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass conservation (absolute + percentage)."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        # Mass conservation
        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}


class ShearFlowValidator(PhysicsClassValidator):
    """
    Validator for Shear Flow (incompressible Navier-Stokes with tracer).

    Key quantities: Enstrophy, tracer variance, energy spectrum
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Enstrophy error (absolute + percentage)."""
        results = {}

        raw_u = self._get_field(self.raw, 'velocity_x', 'velocity_0')
        raw_v = self._get_field(self.raw, 'velocity_y', 'velocity_1')

        if raw_u is None:
            return {'error': 'Missing velocity fields'}

        recon_u = self._get_field(self.recon, 'velocity_x', 'velocity_0')
        recon_v = self._get_field(self.recon, 'velocity_y', 'velocity_1')

        # Enstrophy: integral of vorticity squared
        if raw_u.ndim >= 3:
            t_mid = raw_u.shape[0] // 2
            omega_raw = curl_2d(raw_u[t_mid], raw_v[t_mid], self.dx, self.dy)
            omega_recon = curl_2d(recon_u[t_mid], recon_v[t_mid], self.dx, self.dy)
        else:
            omega_raw = curl_2d(raw_u, raw_v, self.dx, self.dy)
            omega_recon = curl_2d(recon_u, recon_v, self.dx, self.dy)

        # Enstrophy error (absolute + percentage)
        enstrophy_raw = omega_raw**2
        enstrophy_recon = omega_recon**2
        metrics = compute_field_metrics(enstrophy_raw, enstrophy_recon)
        results['enstrophy_abs_error'] = metrics['abs_error']
        results['enstrophy_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}


class TurbulentRadiativeValidator(PhysicsClassValidator):
    """
    Validator for Turbulent Radiative Layer.

    Key quantities: Cooling time, mixing layer width
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass conservation (absolute + percentage)."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        # Mass conservation
        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Secondary metrics."""
        return {}


class ViscoelasticValidator(PhysicsClassValidator):
    """
    Validator for Viscoelastic Instability (Oldroyd-B model).

    Key quantities: Polymer stress, Weissenberg number
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass and momentum."""
        results = {}

        raw_u = self._get_field(self.raw, 'velocity_x', 'velocity_0')
        raw_v = self._get_field(self.raw, 'velocity_y', 'velocity_1')

        if raw_u is None:
            return {'error': 'Missing velocity fields'}

        recon_u = self._get_field(self.recon, 'velocity_x', 'velocity_0')
        recon_v = self._get_field(self.recon, 'velocity_y', 'velocity_1')

        # Incompressibility
        if raw_u.ndim >= 3:
            t_mid = raw_u.shape[0] // 2
            div_raw = divergence_2d(raw_u[t_mid], raw_v[t_mid], self.dx, self.dy)
            div_recon = divergence_2d(recon_u[t_mid], recon_v[t_mid], self.dx, self.dy)
        else:
            div_raw = divergence_2d(raw_u, raw_v, self.dx, self.dy)
            div_recon = divergence_2d(recon_u, recon_v, self.dx, self.dy)

        results['raw_max_divergence'] = float(np.max(np.abs(div_raw)))
        results['recon_max_divergence'] = float(np.max(np.abs(div_recon)))

        # Incompressibility error (absolute + percentage)
        metrics = compute_field_metrics(div_raw, div_recon)
        results['incompressibility_abs_error'] = metrics['abs_error']
        results['incompressibility_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Polymer stress."""
        results = {}

        raw_tau = self._get_field(self.raw, 'polymer_stress', 'tau_p')

        if raw_tau is not None:
            recon_tau = self._get_field(self.recon, 'polymer_stress', 'tau_p')

            # Stress trace (elastic energy)
            if raw_tau.ndim >= 3:
                raw_trace = np.trace(raw_tau.reshape(-1, 2, 2), axis1=-2, axis2=-1).reshape(raw_tau.shape[:-2])
                recon_trace = np.trace(recon_tau.reshape(-1, 2, 2), axis1=-2, axis2=-1).reshape(recon_tau.shape[:-2])
            else:
                raw_trace = raw_tau
                recon_trace = recon_tau

            results['stress_trace_error'] = float(
                np.mean(np.abs(raw_trace - recon_trace)) / (np.mean(np.abs(raw_trace)) + 1e-10)
            )

        return results


class ConvectiveEnvelopeValidator(PhysicsClassValidator):
    """
    Validator for Convective Envelope (Red Supergiant).

    Key quantities: Mass, luminosity, convective velocity
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass conservation."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Convective velocity and Mach number."""
        results = {}

        raw_vr = self._get_field(self.raw, 'velocity_r', 'velocity_0')

        if raw_vr is not None:
            recon_vr = self._get_field(self.recon, 'velocity_r', 'velocity_0')

            # Convective velocity (RMS of radial velocity fluctuations)
            if raw_vr.ndim >= 3:
                raw_v_conv = np.sqrt(np.mean(raw_vr**2, axis=tuple(range(1, raw_vr.ndim))))
                recon_v_conv = np.sqrt(np.mean(recon_vr**2, axis=tuple(range(1, recon_vr.ndim))))

                results['convective_velocity_error'] = float(
                    np.mean(np.abs(raw_v_conv - recon_v_conv)) / (np.abs(np.mean(raw_v_conv)) + 1e-10)
                )

        return results


class NeutronStarMergerValidator(PhysicsClassValidator):
    """
    Validator for Post Neutron Star Merger.

    Key quantities: Electron fraction, mass ejection
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass conservation."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Electron fraction."""
        results = {}

        raw_Ye = self._get_field(self.raw, 'electron_fraction', 'Y_e')

        if raw_Ye is not None:
            recon_Ye = self._get_field(self.recon, 'electron_fraction', 'Y_e')

            results['electron_fraction_error'] = float(
                np.mean(np.abs(raw_Ye - recon_Ye)) / (np.abs(np.mean(raw_Ye)) + 1e-10)
            )

        # Magnetic energy
        raw_Bx = self._get_field(self.raw, 'magnetic_field_x', 'magnetic_field_0')
        if raw_Bx is not None:
            raw_By = self._get_field(self.raw, 'magnetic_field_y', 'magnetic_field_1')
            raw_Bz = self._get_field(self.raw, 'magnetic_field_z', 'magnetic_field_2')

            recon_Bx = self._get_field(self.recon, 'magnetic_field_x', 'magnetic_field_0')
            recon_By = self._get_field(self.recon, 'magnetic_field_y', 'magnetic_field_1')
            recon_Bz = self._get_field(self.recon, 'magnetic_field_z', 'magnetic_field_2')

            B_sq_raw = raw_Bx**2 + raw_By**2 + (raw_Bz**2 if raw_Bz is not None else 0)
            B_sq_recon = recon_Bx**2 + recon_By**2 + (recon_Bz**2 if recon_Bz is not None else 0)

            results['magnetic_energy_error'] = float(
                np.mean(np.abs(B_sq_raw - B_sq_recon)) / (np.mean(B_sq_raw) + 1e-10)
            )

        return results


class ActiveMatterValidator(PhysicsClassValidator):
    """
    Validator for Active Matter simulations.

    Key quantities: Concentration conservation, nematic order
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Concentration conservation (absolute + percentage errors)."""
        results = {}

        raw_c = self._get_field(self.raw, 'concentration', 'c')
        recon_c = self._get_field(self.recon, 'concentration', 'c')

        if raw_c is None:
            return {'error': 'Missing concentration field'}

        if raw_c.ndim >= 3:
            raw_total = np.sum(raw_c, axis=tuple(range(1, raw_c.ndim)))
            recon_total = np.sum(recon_c, axis=tuple(range(1, recon_c.ndim)))

            metrics = compute_conservation_metrics(raw_total, recon_total)
            results['concentration_conservation_abs_error'] = metrics['abs_error']
            results['concentration_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Nematic order parameter."""
        return {}


class HelmholtzValidator(PhysicsClassValidator):
    """
    Validator for Helmholtz Staircase (wave scattering).

    Key quantities: Frequency preservation, boundary conditions

    Dataset fields: pressure_re, pressure_im (real and imaginary parts of pressure)
    """

    def _get_pressure_field(self, data):
        """Get pressure field, handling real/imaginary components."""
        # Try direct pressure field first
        p = self._get_field(data, 'pressure', 'u', 'field')
        if p is not None:
            return p

        # Try real/imaginary components
        p_re = self._get_field(data, 'pressure_re')
        p_im = self._get_field(data, 'pressure_im')

        if p_re is not None and p_im is not None:
            # Combine as complex field magnitude
            return np.sqrt(p_re**2 + p_im**2)
        elif p_re is not None:
            return p_re
        elif p_im is not None:
            return p_im

        return None

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Energy flux."""
        results = {}

        raw_u = self._get_pressure_field(self.raw)
        recon_u = self._get_pressure_field(self.recon)

        if raw_u is None:
            # List available fields for debugging
            available = list(self.raw.keys()) if hasattr(self.raw, 'keys') else 'unknown'
            return {'error': f'Missing pressure field. Available: {available}'}

        # Field energy (absolute + percentage)
        raw_energy = np.sum(np.abs(raw_u)**2)
        recon_energy = np.sum(np.abs(recon_u)**2)

        abs_error = float(abs(raw_energy - recon_energy))
        # For percentage: use range-based normalization
        data_range = float(raw_energy)  # Energy is always positive
        if data_range > 1e-10:
            error_pct = float(100.0 * abs_error / data_range)
        else:
            error_pct = 0.0 if abs_error < 1e-10 else float('inf')

        results['energy_conservation_abs_error'] = abs_error
        results['energy_conservation_error_pct'] = error_pct

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Boundary derivative."""
        results = {}

        raw_u = self._get_pressure_field(self.raw)
        recon_u = self._get_pressure_field(self.recon)

        if raw_u is None:
            return {'error': 'Missing pressure field'}

        # Neumann BC: du/dn = 0 at walls
        # Check boundary normal derivatives
        if raw_u.ndim >= 2:
            # Gradient at boundaries
            raw_grad_x = np.gradient(raw_u, self.dx, axis=-1)
            recon_grad_x = np.gradient(recon_u, self.dx, axis=-1)

            # At x=0 and x=L boundaries
            results['boundary_derivative_x0_error'] = float(
                np.mean(np.abs(raw_grad_x[..., 0] - recon_grad_x[..., 0]))
            )

        return results


class TurbulenceGravityCoolingValidator(PhysicsClassValidator):
    """
    Validator for Turbulence with Gravity and Cooling.

    Similar to turbulent radiative layer but with gravitational effects.
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Mass and energy."""
        results = {}

        raw_rho = self.raw.get('density')
        recon_rho = self.recon.get('density')

        if raw_rho is None:
            return {'error': 'Missing density field'}

        if raw_rho.ndim >= 3:
            raw_mass = np.sum(raw_rho, axis=tuple(range(1, raw_rho.ndim)))
            recon_mass = np.sum(recon_rho, axis=tuple(range(1, recon_rho.ndim)))

            metrics = compute_conservation_metrics(raw_mass, recon_mass)
            results['mass_conservation_abs_error'] = metrics['abs_error']
            results['mass_conservation_error_pct'] = metrics['error_pct']

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """Temperature and velocity statistics."""
        results = {}

        raw_T = self.raw.get('temperature')

        if raw_T is not None:
            recon_T = self.recon.get('temperature')

            results['temperature_mean_error'] = float(
                abs(np.mean(raw_T) - np.mean(recon_T)) / (np.abs(np.mean(raw_T)) + 1e-10)
            )
            results['temperature_std_error'] = float(
                abs(np.std(raw_T) - np.std(recon_T)) / (np.std(raw_T) + 1e-10)
            )

        raw_v = self.raw.get('velocity_0')
        if raw_v is not None:
            recon_v = self.recon.get('velocity_0')

            results['velocity_rms_error'] = float(
                abs(np.sqrt(np.mean(raw_v**2)) - np.sqrt(np.mean(recon_v**2))) /
                (np.sqrt(np.mean(raw_v**2)) + 1e-10)
            )

        return results


class GenericValidator(PhysicsClassValidator):
    """
    Generic validator for unknown physics classes.
    Uses only universal metrics.
    """

    def get_primary_metrics(self) -> Dict[str, Any]:
        """Generic field comparison."""
        results = {}

        for field_name in self.raw.keys():
            if field_name in self.recon:
                raw_field = self.raw[field_name]
                recon_field = self.recon[field_name]

                if raw_field is not None and recon_field is not None:
                    results[f'{field_name}_rel_error'] = float(
                        np.linalg.norm(raw_field - recon_field) /
                        (np.linalg.norm(raw_field) + 1e-10)
                    )

        return results

    def get_secondary_metrics(self) -> Dict[str, Any]:
        """No specific secondary metrics."""
        return {}


# =============================================================================
# Validator Factory
# =============================================================================

VALIDATOR_MAP = {
    'acoustic_scattering': AcousticScatteringValidator,
    'euler': EulerValidator,
    'gray_scott': GrayScottValidator,
    'mhd': MHDValidator,
    'rayleigh_benard': RayleighBenardValidator,
    'rayleigh_taylor': RayleighTaylorValidator,
    'shallow_water': ShallowWaterValidator,
    'supernova': SupernovaValidator,
    'shear_flow': ShearFlowValidator,
    'turbulent_radiative': TurbulentRadiativeValidator,
    'viscoelastic': ViscoelasticValidator,
    'convective_envelope': ConvectiveEnvelopeValidator,
    'neutron_star_merger': NeutronStarMergerValidator,
    'active_matter': ActiveMatterValidator,
    'helmholtz': HelmholtzValidator,
    'turbulence_gravity': TurbulenceGravityCoolingValidator,
    'unknown': GenericValidator,
}


def get_validator(dataset_name: str,
                  raw_data: Dict[str, np.ndarray],
                  reconstructed_data: Dict[str, np.ndarray],
                  metadata: Dict[str, Any] = None) -> PhysicsClassValidator:
    """
    Factory function to get the appropriate validator for a dataset.

    Args:
        dataset_name: Name of the dataset
        raw_data: Dictionary of raw field arrays
        reconstructed_data: Dictionary of reconstructed field arrays
        metadata: Optional metadata

    Returns:
        Appropriate PhysicsClassValidator instance
    """
    physics_class = get_physics_class(dataset_name)
    validator_class = VALIDATOR_MAP.get(physics_class, GenericValidator)

    logger.info(f"Using {validator_class.__name__} for dataset '{dataset_name}' (class: {physics_class})")

    return validator_class(raw_data, reconstructed_data, metadata)


def run_physics_validation(dataset_name: str,
                           raw_data: Dict[str, np.ndarray],
                           reconstructed_data: Dict[str, np.ndarray],
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to run physics validation for a dataset.

    Args:
        dataset_name: Name of the dataset
        raw_data: Dictionary of raw field arrays
        reconstructed_data: Dictionary of reconstructed field arrays
        metadata: Optional metadata

    Returns:
        Validation results dictionary
    """
    validator = get_validator(dataset_name, raw_data, reconstructed_data, metadata)
    return validator.run_validation()
