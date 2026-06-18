# SINCPS: Semantic-aware Implicit Neural Compression for Physics Simulations

Machine learning surrogates and data-driven scientific discovery require efficient access to simulation data, yet physics simulations generate terabyte-scale datasets. Traditional compression either achieves insufficient ratios or corrupts physics-critical features like conservation laws.

**SINCPS** leverages wafer-scale computing to train implicit neural representations in 2-3 hours each. Across 22 datasets from [The Well benchmark](https://github.com/PolymathicAI/the_well), we achieve **150x to 25,000x compression**, evaluated with standard reconstruction metrics (PSNR and L2 error). Physics-aware validation of domain-specific conservation laws is ongoing.

## Key Results

Per-domain aggregates across the 22 datasets:

| Physics Domain | Compression Ratio | PSNR (dB) | L2 Error (%) |
|---------------|-------------------|-----------|--------------|
| Compressible Flows | 25,348x | 14.7 | 16.2 |
| Turbulence | 2,394x | 13.7 | 13.6 |
| Astrophysics | 1,767x | 32.3 | 3.4 |
| Wave Phenomena | 1,750x | 21.5 | 8.9 |
| Fluid Dynamics | 1,483x | 21.4 | 5.8 |
| Magnetohydrodynamics | 220x | 14.9 | 9.7 |
| Geophysical | 182x | 14.1 | 13.7 |
| Reaction-Diffusion | 152x | 23.9 | 5.8 |

All 22 models compress to **37.6 MB each** from original sizes of 5.4 GB to 1.9 TB.

## Installation

```bash
pip install torch numpy h5py pyyaml
```

## Quick Start

### Decompress a Checkpoint

```python
from sincps import SINCPSDecompressor

# Load a trained model
decompressor = SINCPSDecompressor(
    checkpoint_path='checkpoints/shear_flow/checkpoint_50000.mdl',
    config={
        'input_dim': 3,      # time + 2D spatial
        'output_dim': 4,     # number of fields
        'hidden_dim': 1024,
        'num_hidden_layers': 4,
    }
)

# Reconstruct specific timesteps
data = decompressor.reconstruct(
    spatial_shape=(256, 256),
    num_timesteps=100,
    timestep_indices=[0, 50, 99],  # reconstruct 3 timesteps
)
# Returns shape: (3, 256, 256, 4)
```

### Low-Level API

```python
from sincps import load_model, reconstruct

# Load model
model = load_model(
    'checkpoints/mhd_64/checkpoint_50000.mdl',
    config={'input_dim': 4, 'output_dim': 6, 'hidden_dim': 1024}
)

# Reconstruct full 3D volume at middle timestep
data = reconstruct(
    model,
    spatial_shape=(64, 64, 64),
    num_timesteps=100,
    timestep_indices=[50],
)
```

## Model Architecture

SINCPS uses a **SIREN network** with **Fourier positional encoding**:

1. **Input**: Normalized spatiotemporal coordinates [0, 1]
2. **Fourier Encoding**: Multi-scale frequencies from π to 512π
3. **SIREN Layers**: 4 hidden layers × 1024 units with sinusoidal activations
4. **Output**: Z-score normalized field values

```
Coordinates → Fourier Encoding → SIREN Layers → Field Values
   (t,x,y)        (63-dim)        (4×1024)        (n_fields)
```

## Available Checkpoints

Trained models for 22 physics datasets from The Well:

| Dataset | Dimensions | Fields | Checkpoint |
|---------|------------|--------|------------|
| shear_flow | 2D+T | 4 | `checkpoints/shear_flow/` |
| rayleigh_taylor_instability | 2D+T | 4 | `checkpoints/rayleigh_taylor_instability/` |
| mhd_64 | 3D+T | 6 | `checkpoints/mhd_64/` |
| supernova_explosion_64 | 3D+T | 6 | `checkpoints/supernova_explosion_64/` |
| acoustic_scattering_maze | 2D+T | 2 | `checkpoints/acoustic_scattering_maze/` |
| gray_scott_reaction_diffusion | 2D+T | 2 | `checkpoints/gray_scott_reaction_diffusion/` |
| ... | ... | ... | ... |

See `checkpoints/` for the complete list and `configs/` for model configurations.

## Training

Models were trained on the **Cerebras CS-3** wafer-scale engine:

- **Training time**: 2-3 hours per dataset (vs 6-11 hours on GPU/CPU)
- **Batch size**: 16,384
- **Steps**: 50,000
- **Loss**: MSE
- **Optimizer**: Adam

Per-dataset training configurations are in the `configs/` directory. Models were trained with the Cerebras ModelZoo trainer on the CS-3; the training framework itself is external to this repository.

## Citation

```bibtex
@inproceedings{ezemba2026sincps,
  title     = {Semantic-aware Implicit Neural Compression for Physics Simulations},
  author    = {Ezemba, Jessica and Afful, James and Wang, Mei-Yu},
  booktitle = {Platform for Advanced Scientific Computing (PASC),
               ACM Student Research Competition},
  year      = {2026}
}
```

## Acknowledgments

This work was made possible by:
- **ByteBoost Cybertraining Program** (NSF Awards: 2320990, 2320991, 2320992)
- **Neocortex Project** (NSF Award: 2005597)
- **ACES Platform** (NSF Award: 2112356)
- **Ookami Cluster** (NSF Award: 1927880)
- Cerebras Systems for CS-3 access and technical support

## License

MIT License
