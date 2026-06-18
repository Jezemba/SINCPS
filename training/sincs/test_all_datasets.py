#!/usr/bin/env python3
"""
Test data loading for all 23 PhysicsAlchemists datasets.

This script validates that the SINCS data processor can:
1. Find and load HDF5 chunk files
2. Detect static vs dynamic fields correctly
3. Create coordinate-value pairs
4. Apply normalization

Run: python test_all_datasets.py
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# All 23 datasets
ALL_DATASETS = [
    "acoustic_scattering_discontinuous",
    "acoustic_scattering_inclusions",
    "acoustic_scattering_maze",
    "active_matter",
    "convective_envelope_rsg",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "gray_scott_reaction_diffusion",
    "helmholtz_staircase",
    "MHD_256",
    "MHD_64",
    "planetswe",
    "post_neutron_star_merger",
    "rayleigh_benard",
    "rayleigh_benard_uniform",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_128",
    "supernova_explosion_64",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability",
]

DATA_PATH = "/cra-1272/PhysicsAlchemists/datasets"


@dataclass
class DatasetTestResult:
    """Result of testing a single dataset."""
    name: str
    success: bool
    error: str = ""
    n_chunk_files: int = 0
    grid_shape: Tuple[int, ...] = ()
    static_fields: List[str] = None
    dynamic_fields: List[str] = None
    n_samples: int = 0
    coords_shape: Tuple[int, ...] = ()
    values_shape: Tuple[int, ...] = ()
    load_time_ms: float = 0.0

    def __post_init__(self):
        if self.static_fields is None:
            self.static_fields = []
        if self.dynamic_fields is None:
            self.dynamic_fields = []


def test_dataset(dataset_name: str) -> DatasetTestResult:
    """Test loading a single dataset."""
    from data import SINCSDataProcessorConfig, SINCSDataProcessor

    result = DatasetTestResult(name=dataset_name, success=False)

    try:
        start_time = time.time()

        # Create config
        config = SINCSDataProcessorConfig(
            data_dir=DATA_PATH,
            dataset_name=dataset_name,
            split="train",
            trajectory_idx=0,
            num_timesteps=10,  # Limit for faster testing
            batch_size=1024,
            num_samples_per_epoch=10000,
            shuffle=True,
            shuffle_seed=42,
            num_workers=0,
            drop_last=True,
            use_fake_data=False,
            split_static_dynamic=True,
            dynamic_fields_only=True,
            normalize_fields=True,
        )

        # Create processor
        processor = SINCSDataProcessor(config)

        # Create dataloader (this triggers data loading)
        dataloader = processor.create_dataloader()

        # Get a batch to verify
        batch = next(iter(dataloader))
        coords, values = batch

        end_time = time.time()

        # Populate results
        result.success = True
        result.static_fields = list(processor.static_fields.keys())
        result.dynamic_fields = list(processor.dynamic_fields.keys())
        result.n_samples = len(coords)
        result.coords_shape = tuple(coords.shape)
        result.values_shape = tuple(values.shape)
        result.load_time_ms = (end_time - start_time) * 1000

        # Try to get grid shape from first dynamic field
        if processor.dynamic_fields:
            first_field = list(processor.dynamic_fields.values())[0]
            result.grid_shape = first_field.shape
        elif processor.static_fields:
            first_field = list(processor.static_fields.values())[0]
            result.grid_shape = first_field.shape

    except Exception as e:
        import traceback
        result.error = f"{type(e).__name__}: {str(e)}"
        # Uncomment for detailed traceback:
        # result.error += f"\n{traceback.format_exc()}"

    return result


def print_results_table(results: List[DatasetTestResult]):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print(f"{'Dataset':<40} {'Status':<8} {'Static':<15} {'Dynamic':<20} {'Shape':<20} {'Time(ms)':<10}")
    print("=" * 120)

    passed = 0
    failed = 0

    for r in results:
        status = "✓ PASS" if r.success else "✗ FAIL"

        static_str = ", ".join(r.static_fields[:3])
        if len(r.static_fields) > 3:
            static_str += f"... (+{len(r.static_fields)-3})"
        if not static_str:
            static_str = "none"

        dynamic_str = ", ".join(r.dynamic_fields[:3])
        if len(r.dynamic_fields) > 3:
            dynamic_str += f"... (+{len(r.dynamic_fields)-3})"
        if not dynamic_str:
            dynamic_str = "none"

        shape_str = str(r.grid_shape) if r.grid_shape else "?"

        # Truncate long strings
        if len(static_str) > 13:
            static_str = static_str[:10] + "..."
        if len(dynamic_str) > 18:
            dynamic_str = dynamic_str[:15] + "..."
        if len(shape_str) > 18:
            shape_str = shape_str[:15] + "..."

        print(f"{r.name:<40} {status:<8} {static_str:<15} {dynamic_str:<20} {shape_str:<20} {r.load_time_ms:>8.1f}")

        if r.success:
            passed += 1
        else:
            failed += 1

    print("=" * 120)
    print(f"Total: {len(results)} datasets | Passed: {passed} | Failed: {failed}")
    print("=" * 120)

    # Print details for failed tests
    if failed > 0:
        print("\n" + "=" * 80)
        print("FAILED DATASET DETAILS:")
        print("=" * 80)
        for r in results:
            if not r.success:
                print(f"\n{r.name}:")
                print(f"  Error: {r.error[:500]}")


def print_dataset_details(results: List[DatasetTestResult]):
    """Print detailed field information for each dataset."""
    print("\n" + "=" * 100)
    print("DATASET FIELD DETAILS")
    print("=" * 100)

    for r in results:
        if not r.success:
            continue

        print(f"\n{r.name}:")
        print(f"  Grid shape: {r.grid_shape}")
        print(f"  Static fields ({len(r.static_fields)}): {r.static_fields}")
        print(f"  Dynamic fields ({len(r.dynamic_fields)}): {r.dynamic_fields}")
        print(f"  Batch: coords={r.coords_shape}, values={r.values_shape}")


def main():
    """Run tests for all datasets."""
    import argparse
    parser = argparse.ArgumentParser(description="Test SINCS data loading for all datasets")
    parser.add_argument("--dataset", "-d", type=str, help="Test only specific dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less output")
    args = parser.parse_args()

    print("=" * 80)
    print("SINCS DATA PROCESSOR - COMPREHENSIVE DATASET TEST")
    print(f"Testing {len(ALL_DATASETS)} datasets from {DATA_PATH}")
    print("=" * 80)

    if args.dataset:
        # Test single dataset
        datasets = [args.dataset]
    else:
        datasets = ALL_DATASETS

    results = []

    for i, dataset_name in enumerate(datasets):
        if not args.quiet:
            print(f"\n[{i+1}/{len(datasets)}] Testing {dataset_name}...", end=" ", flush=True)

        # Suppress logging during test unless verbose
        if not args.verbose:
            logging.getLogger().setLevel(logging.WARNING)

        result = test_dataset(dataset_name)
        results.append(result)

        # Restore logging
        logging.getLogger().setLevel(logging.INFO)

        if not args.quiet:
            if result.success:
                print(f"✓ ({result.load_time_ms:.1f}ms)")
                if args.verbose:
                    print(f"    Static: {result.static_fields}")
                    print(f"    Dynamic: {result.dynamic_fields}")
            else:
                print(f"✗ FAILED")
                print(f"    Error: {result.error[:200]}")

    # Print summary table
    print_results_table(results)

    # Print details
    if args.verbose:
        print_dataset_details(results)

    # Exit with error if any failed
    failed = sum(1 for r in results if not r.success)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
