#!/usr/bin/env python3
"""
SINCS Validation Runner

Runs both Tier 1 (Physics Validation) and Tier 2 (FNO Comparison)
and generates a comprehensive report.

Usage:
    python run_validation.py \
        --dataset acoustic_scattering_inclusions \
        --checkpoint path/to/checkpoint.mdl \
        --config path/to/config.yaml \
        --output_dir ./validation_results

    # Quick test mode
    python run_validation.py \
        --dataset acoustic_scattering_inclusions \
        --checkpoint path/to/checkpoint.mdl \
        --config path/to/config.yaml \
        --quick
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_validation(
    dataset_name: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    train_data_dir: str = "/cra-1272/PhysicsAlchemists/datasets",
    val_data_dir: str = "/cra-1272/PhysicsAlchemists/datasets_valid",
    device: str = 'cpu',
    quick: bool = False,
    skip_tier1: bool = False,
    skip_tier2: bool = False,
) -> Dict:
    """
    Run full validation framework.

    Args:
        dataset_name: Name of the dataset (e.g., 'acoustic_scattering_inclusions')
        checkpoint_path: Path to SINCS model checkpoint
        config_path: Path to SINCS config YAML
        output_dir: Directory to save results
        train_data_dir: Base directory for training data
        val_data_dir: Base directory for validation data
        device: Device to use
        quick: Quick mode for testing
        skip_tier1: Skip physics validation
        skip_tier2: Skip FNO comparison

    Returns:
        Combined results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'metadata': {
            'dataset': dataset_name,
            'checkpoint': checkpoint_path,
            'config': config_path,
            'timestamp': datetime.now().isoformat(),
            'quick_mode': quick,
        }
    }

    # Paths
    raw_train_path = os.path.join(train_data_dir, dataset_name)
    raw_val_path = os.path.join(val_data_dir, dataset_name)

    # -------------------------------------------------------------------------
    # Tier 1: Physics Validation
    # -------------------------------------------------------------------------
    if not skip_tier1:
        logger.info("\n" + "=" * 70)
        logger.info("TIER 1: PHYSICS-PRESERVING VALIDATION")
        logger.info("=" * 70)

        try:
            from physics_validation import (
                PhysicsValidator, load_raw_data, load_reconstructed_data
            )

            # Load raw data
            logger.info(f"Loading raw data from {raw_train_path}")
            raw_data, bc = load_raw_data(raw_train_path, trajectory_idx=0)

            # Quick mode: use fewer timesteps
            if quick:
                if 'pressure' in raw_data:
                    raw_data['pressure'] = raw_data['pressure'][:20]
                if 'velocity' in raw_data:
                    raw_data['velocity'] = raw_data['velocity'][:20]

            # Load reconstructed data
            # load_reconstructed_data returns (normalized_raw, recon_data) for fair comparison
            # since SINCS outputs z-score normalized values
            logger.info("Reconstructing data with SINCS model...")
            normalized_raw, recon_data = load_reconstructed_data(
                checkpoint_path, config_path, raw_data, device
            )

            if quick:
                if 'pressure' in normalized_raw:
                    normalized_raw['pressure'] = normalized_raw['pressure'][:20]
                if 'velocity' in normalized_raw:
                    normalized_raw['velocity'] = normalized_raw['velocity'][:20]
                if 'pressure' in recon_data:
                    recon_data['pressure'] = recon_data['pressure'][:20]
                if 'velocity' in recon_data:
                    recon_data['velocity'] = recon_data['velocity'][:20]

            # Run validations
            # Use normalized_raw instead of raw_data for fair comparison with SINCS output
            logger.info("Running physics validations...")
            validator = PhysicsValidator(normalized_raw, recon_data, bc)
            tier1_results = validator.run_all_validations()

            results['tier1_physics'] = tier1_results

            # Save Tier 1 results
            tier1_path = os.path.join(output_dir, 'tier1_physics_results.json')
            with open(tier1_path, 'w') as f:
                json.dump(tier1_results, f, indent=2)
            logger.info(f"Tier 1 results saved to {tier1_path}")

        except Exception as e:
            logger.error(f"Tier 1 validation failed: {e}")
            results['tier1_physics'] = {'error': str(e)}

    # -------------------------------------------------------------------------
    # Tier 2: FNO Surrogate Comparison
    # -------------------------------------------------------------------------
    if not skip_tier2:
        logger.info("\n" + "=" * 70)
        logger.info("TIER 2: FNO SURROGATE MODEL COMPARISON")
        logger.info("=" * 70)

        try:
            from fno_surrogate import run_comparison

            tier2_output = os.path.join(output_dir, 'tier2_fno_results.json')

            tier2_results = run_comparison(
                raw_train_path=raw_train_path,
                raw_val_path=raw_val_path,
                sincs_model_path=checkpoint_path,
                sincs_config_path=config_path,
                output_path=tier2_output,
                num_train_traj=10 if not quick else 2,
                num_val_traj=5 if not quick else 1,
                epochs=50 if not quick else 10,
                device=device,
                quick=quick,
            )

            results['tier2_fno'] = tier2_results
            logger.info(f"Tier 2 results saved to {tier2_output}")

        except Exception as e:
            logger.error(f"Tier 2 validation failed: {e}")
            results['tier2_fno'] = {'error': str(e)}

    # -------------------------------------------------------------------------
    # Generate Summary Report
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY REPORT")
    logger.info("=" * 70)

    summary = generate_summary(results)
    results['summary'] = summary

    # Save combined results
    combined_path = os.path.join(output_dir, 'validation_results.json')
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nCombined results saved to {combined_path}")

    # Print summary
    print_summary(summary)

    return results


def generate_summary(results: Dict) -> Dict:
    """Generate summary from validation results."""
    summary = {
        'tier1_pass': False,
        'tier2_pass': False,
        'overall_pass': False,
        'key_metrics': {},
        'recommendations': [],
    }

    # Tier 1 Summary
    if 'tier1_physics' in results and 'error' not in results['tier1_physics']:
        t1 = results['tier1_physics']

        if 'summary' in t1:
            summary['tier1_pass'] = t1['summary'].get('all_pass', False)

        summary['key_metrics']['wave_speed_relative_error'] = t1.get('wave_speed', {}).get('relative_error', 'N/A')
        summary['key_metrics']['energy_deviation'] = t1.get('energy_conservation', {}).get('max_energy_deviation', 'N/A')
        summary['key_metrics']['high_freq_error'] = t1.get('frequency_spectrum', {}).get('high_freq_error', 'N/A')

        # Recommendations
        if t1.get('wave_speed', {}).get('relative_error', 1) > 0.05:
            summary['recommendations'].append("Wave speed error > 5%: Consider training SINCS longer or with higher capacity")

        if t1.get('frequency_spectrum', {}).get('high_freq_error', 1) > 0.20:
            summary['recommendations'].append("High-frequency error > 20%: May affect fine-scale features")

    # Tier 2 Summary
    if 'tier2_fno' in results and 'error' not in results['tier2_fno']:
        t2 = results['tier2_fno']

        if 'degradation' in t2:
            summary['tier2_pass'] = t2['degradation'].get('acceptable', False)
            summary['key_metrics']['fno_degradation'] = t2['degradation'].get('one_step_degradation_percent', 'N/A')

        if 'raw_training' in t2 and 'compressed_training' in t2:
            summary['key_metrics']['raw_fno_rel_l2'] = t2['raw_training'].get('one_step_rel_l2', 'N/A')
            summary['key_metrics']['compressed_fno_rel_l2'] = t2['compressed_training'].get('one_step_rel_l2', 'N/A')

        # Recommendations
        if t2.get('degradation', {}).get('one_step_degradation_percent', 100) > 10:
            summary['recommendations'].append("FNO degradation > 10%: Compression may impact downstream ML tasks")

    # Overall
    summary['overall_pass'] = summary['tier1_pass'] and summary['tier2_pass']

    if not summary['recommendations']:
        summary['recommendations'].append("All metrics within acceptable ranges - compression is suitable for this dataset")

    return summary


def print_summary(summary: Dict):
    """Print summary report."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\nTier 1 (Physics Validation): {'PASS' if summary['tier1_pass'] else 'FAIL'}")
    print(f"Tier 2 (FNO Comparison):      {'PASS' if summary['tier2_pass'] else 'FAIL'}")
    print(f"Overall:                      {'PASS' if summary['overall_pass'] else 'FAIL'}")

    print("\n--- Key Metrics ---")
    for key, value in summary['key_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Recommendations ---")
    for rec in summary['recommendations']:
        print(f"  • {rec}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='SINCS Validation Runner')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name (e.g., acoustic_scattering_inclusions)')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to SINCS model checkpoint')
    parser.add_argument('--config', '-p', type=str, required=True,
                        help='Path to SINCS config YAML')
    parser.add_argument('--output_dir', '-o', type=str, default='./validation_results',
                        help='Output directory for results')
    parser.add_argument('--train_data_dir', type=str,
                        default='/cra-1272/PhysicsAlchemists/datasets',
                        help='Base directory for training data')
    parser.add_argument('--val_data_dir', type=str,
                        default='/cra-1272/PhysicsAlchemists/datasets_valid',
                        help='Base directory for validation data')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode for testing')
    parser.add_argument('--skip_tier1', action='store_true',
                        help='Skip Tier 1 physics validation')
    parser.add_argument('--skip_tier2', action='store_true',
                        help='Skip Tier 2 FNO comparison')

    args = parser.parse_args()

    results = run_full_validation(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        device=args.device,
        quick=args.quick,
        skip_tier1=args.skip_tier1,
        skip_tier2=args.skip_tier2,
    )


if __name__ == '__main__':
    main()
