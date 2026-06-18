#!/usr/bin/env python3
"""
Evaluate trained SINCS model on test data.

Computes metrics:
- MSE (Mean Squared Error)
- PSNR (Peak Signal-to-Noise Ratio)
- Relative L2 Error
- Per-field metrics

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.mdl --config path/to/config.yaml
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import yaml
import h5py

# Add modelzoo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from model import SINCS, SINCSConfig
from data import SINCSDataProcessor, SINCSDataProcessorConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_hdf5_checkpoint(checkpoint_path: str) -> dict:
    """Load Cerebras HDF5 checkpoint into state dict."""
    state_dict = {}
    with h5py.File(checkpoint_path, 'r') as f:
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Convert to tensor
                data = torch.from_numpy(np.array(obj))
                state_dict[name] = data
        f.visititems(visit_fn)
    return state_dict


def load_model(checkpoint_path: str, config: dict) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    model_config = config['trainer']['init']['model']

    # Create model config
    sincs_config = SINCSConfig(
        input_dim=model_config.get('input_dim', 3),
        output_dim=model_config.get('output_dim', 1),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_hidden_layers=model_config.get('num_hidden_layers', 4),
        omega_0=model_config.get('omega_0', 30.0),
        omega_hidden=model_config.get('omega_hidden', 30.0),
        use_fourier_encoding=model_config.get('use_fourier_encoding', True),
        encoding_levels=model_config.get('encoding_levels', 10),
    )

    # Create SINCS model directly (not the wrapper)
    model = SINCS(sincs_config)

    # Load checkpoint - Cerebras uses HDF5 format
    state_dict = load_hdf5_checkpoint(checkpoint_path)

    # The checkpoint contains model, optimizer, and scheduler state
    # Filter to only get model weights
    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip optimizer and scheduler keys
        if k.startswith('optimizer.') or k.startswith('schedulers.'):
            continue
        # Remove 'model.model.' or 'model.' prefix
        new_key = k
        if new_key.startswith('model.model.'):
            new_key = new_key[12:]
        elif new_key.startswith('model.'):
            new_key = new_key[6:]
        new_state_dict[new_key] = v

    logging.info(f"  Loading {len(new_state_dict)} weight tensors")
    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error."""
    return np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    num_batches: int = None,
) -> dict:
    """Evaluate model on dataloader."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total_mse = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break

            coords, targets = batch
            coords = coords.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(coords)

            # Compute MSE
            mse = torch.mean((preds - targets) ** 2).item()
            total_mse += mse * coords.shape[0]
            num_samples += coords.shape[0]

            # Store for detailed analysis
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            if (i + 1) % 10 == 0:
                logging.info(f"  Batch {i+1}: MSE={mse:.6f}")

    # Aggregate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_mse = total_mse / num_samples
    psnr = compute_psnr(avg_mse)
    rel_l2 = compute_relative_l2(all_preds, all_targets)

    # Per-field metrics
    num_fields = all_targets.shape[1]
    field_metrics = {}
    for f in range(num_fields):
        field_mse = np.mean((all_preds[:, f] - all_targets[:, f]) ** 2)
        field_psnr = compute_psnr(field_mse)
        field_rel_l2 = compute_relative_l2(all_preds[:, f], all_targets[:, f])
        field_metrics[f"field_{f}"] = {
            'mse': field_mse,
            'psnr': field_psnr,
            'rel_l2': field_rel_l2,
        }

    return {
        'mse': avg_mse,
        'psnr': psnr,
        'rel_l2': rel_l2,
        'num_samples': num_samples,
        'field_metrics': field_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SINCS model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', '-p', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--split', '-s', type=str, default='train',
                        help='Data split to evaluate (train/test)')
    parser.add_argument('--num_batches', '-n', type=int, default=100,
                        help='Number of batches to evaluate (default: 100)')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("SINCS Model Evaluation")
    logging.info("=" * 60)

    # Load config
    logging.info(f"\nLoading config from {args.config}")
    config = load_config(args.config)

    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")

    # Create dataloader
    logging.info(f"\nCreating dataloader for split: {args.split}")
    dl_config = config['trainer']['fit']['train_dataloader']

    processor_config = SINCSDataProcessorConfig(
        data_dir=dl_config['data_dir'],
        dataset_name=dl_config['dataset_name'],
        split=args.split,
        trajectory_idx=dl_config.get('trajectory_idx', 0),
        num_timesteps=dl_config.get('num_timesteps', 50),
        batch_size=dl_config.get('batch_size', 16384),
        num_samples_per_epoch=dl_config.get('num_samples_per_epoch', 500000),
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=0,
        drop_last=False,
        use_fake_data=False,
        split_static_dynamic=dl_config.get('split_static_dynamic', True),
        dynamic_fields_only=dl_config.get('dynamic_fields_only', True),
        normalize_fields=dl_config.get('normalize_fields', True),
    )

    processor = SINCSDataProcessor(processor_config)
    dataloader = processor.create_dataloader()

    # Evaluate
    logging.info(f"\nEvaluating on {args.num_batches} batches...")
    metrics = evaluate_model(
        model,
        dataloader,
        device=args.device,
        num_batches=args.num_batches,
    )

    # Print results
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"\nDataset: {dl_config['dataset_name']}")
    logging.info(f"Split: {args.split}")
    logging.info(f"Samples evaluated: {metrics['num_samples']:,}")
    logging.info(f"\nOverall Metrics:")
    logging.info(f"  MSE:          {metrics['mse']:.6f}")
    logging.info(f"  PSNR:         {metrics['psnr']:.2f} dB")
    logging.info(f"  Relative L2:  {metrics['rel_l2']:.4f} ({metrics['rel_l2']*100:.2f}%)")

    logging.info(f"\nPer-Field Metrics:")
    field_names = ['pressure', 'velocity_0', 'velocity_1']  # For acoustic dataset
    for i, (field_key, field_data) in enumerate(metrics['field_metrics'].items()):
        name = field_names[i] if i < len(field_names) else field_key
        logging.info(f"  {name}:")
        logging.info(f"    MSE:         {field_data['mse']:.6f}")
        logging.info(f"    PSNR:        {field_data['psnr']:.2f} dB")
        logging.info(f"    Relative L2: {field_data['rel_l2']:.4f} ({field_data['rel_l2']*100:.2f}%)")

    logging.info("\n" + "=" * 60)

    # Summary table
    logging.info("\nSUMMARY TABLE")
    logging.info("-" * 50)
    logging.info(f"{'Metric':<20} {'Value':<15} {'Unit':<10}")
    logging.info("-" * 50)
    logging.info(f"{'MSE':<20} {metrics['mse']:<15.6f} {'':<10}")
    logging.info(f"{'PSNR':<20} {metrics['psnr']:<15.2f} {'dB':<10}")
    logging.info(f"{'Relative L2 Error':<20} {metrics['rel_l2']*100:<15.2f} {'%':<10}")
    logging.info("-" * 50)


if __name__ == '__main__':
    main()
