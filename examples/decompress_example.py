#!/usr/bin/env python3
"""
Example: Decompress a SINCPS checkpoint and visualize the reconstruction.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from sincps import SINCPSDecompressor, load_model, reconstruct


def example_shear_flow():
    """Decompress shear_flow dataset."""
    print("=" * 60)
    print("Example: Decompressing shear_flow")
    print("=" * 60)

    # Model configuration for shear_flow
    config = {
        'input_dim': 3,           # time + 2D spatial
        'output_dim': 4,          # pressure, density, velocity_x, velocity_y
        'hidden_dim': 1024,
        'num_hidden_layers': 4,
        'omega_0': 30.0,
        'omega_hidden': 30.0,
        'use_fourier_encoding': True,
        'encoding_levels': 10,
    }

    # Load model
    checkpoint_path = '../checkpoints/shear_flow/checkpoint_50000.mdl'
    decompressor = SINCPSDecompressor(checkpoint_path, config)

    # Reconstruct middle timestep
    print("Reconstructing timestep 50 of 100...")
    data = decompressor.reconstruct(
        spatial_shape=(256, 256),
        num_timesteps=100,
        timestep_indices=[50],
    )

    print(f"Reconstructed shape: {data.shape}")
    # Shape: (1, 256, 256, 4)

    # Visualize first field (pressure)
    plt.figure(figsize=(8, 8))
    plt.imshow(data[0, :, :, 0], cmap='coolwarm', origin='lower')
    plt.colorbar(label='Pressure (normalized)')
    plt.title('Shear Flow - Pressure Field (t=50)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('shear_flow_reconstruction.png', dpi=150)
    plt.close()
    print("Saved: shear_flow_reconstruction.png")


def example_3d_mhd():
    """Decompress 3D MHD dataset."""
    print("\n" + "=" * 60)
    print("Example: Decompressing mhd_64 (3D)")
    print("=" * 60)

    config = {
        'input_dim': 4,           # time + 3D spatial
        'output_dim': 6,          # density, pressure, velocity (3), magnetic (3)
        'hidden_dim': 1024,
        'num_hidden_layers': 4,
    }

    checkpoint_path = '../checkpoints/mhd_64/checkpoint_50000.mdl'
    model = load_model(checkpoint_path, config)

    # Reconstruct single timestep
    print("Reconstructing timestep 25 of 50...")
    data = reconstruct(
        model,
        spatial_shape=(64, 64, 64),
        num_timesteps=50,
        timestep_indices=[25],
    )

    print(f"Reconstructed shape: {data.shape}")
    # Shape: (1, 64, 64, 64, 6)

    # Visualize middle z-slice of density
    mid_z = 32
    plt.figure(figsize=(8, 8))
    plt.imshow(data[0, :, :, mid_z, 0], cmap='inferno', origin='lower')
    plt.colorbar(label='Density (normalized)')
    plt.title(f'MHD 3D - Density Field (t=25, z={mid_z})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('mhd_64_reconstruction.png', dpi=150)
    plt.close()
    print("Saved: mhd_64_reconstruction.png")


def example_batch_reconstruction():
    """Reconstruct multiple timesteps at once."""
    print("\n" + "=" * 60)
    print("Example: Batch reconstruction of multiple timesteps")
    print("=" * 60)

    config = {
        'input_dim': 3,
        'output_dim': 2,
        'hidden_dim': 1024,
    }

    checkpoint_path = '../checkpoints/gray_scott_reaction_diffusion/checkpoint_50000.mdl'
    decompressor = SINCPSDecompressor(checkpoint_path, config)

    # Reconstruct 5 evenly-spaced timesteps
    timesteps = [0, 25, 50, 75, 99]
    print(f"Reconstructing timesteps: {timesteps}")

    data = decompressor.reconstruct(
        spatial_shape=(128, 128),
        num_timesteps=100,
        timestep_indices=timesteps,
    )

    print(f"Reconstructed shape: {data.shape}")
    # Shape: (5, 128, 128, 2)

    # Plot time evolution
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, t) in enumerate(zip(axes, timesteps)):
        ax.imshow(data[i, :, :, 0], cmap='viridis', origin='lower')
        ax.set_title(f't={t}')
        ax.axis('off')
    plt.suptitle('Gray-Scott Reaction-Diffusion - Time Evolution')
    plt.tight_layout()
    plt.savefig('gray_scott_evolution.png', dpi=150)
    plt.close()
    print("Saved: gray_scott_evolution.png")


if __name__ == '__main__':
    # Uncomment examples to run:
    # example_shear_flow()
    # example_3d_mhd()
    # example_batch_reconstruction()

    print("\nTo run examples, edit this file and uncomment the desired functions.")
    print("Make sure checkpoint files are in ../checkpoints/")
