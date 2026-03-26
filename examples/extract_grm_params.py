#!/usr/bin/env python
"""Extract GRM parameters from fitted bayesianquilts models.

Loads a fitted GRModel from disk, samples from the variational
surrogate to get posterior means (handling bijectors correctly),
and saves slope/calibration arrays as .npz for use by the CAT
simulation.

Usage:
    uv run python extract_grm_params.py \
        --model-dir ../bayesianquilts/notebooks/irt/wpi/grm_baseline \
        --dataset wpi \
        --output-dir params/

Memory: Loads one model at a time. ~2-4 GB depending on instrument size.
"""

import argparse
import gc
import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np


DATASET_CONFIGS = {
    'eqsq': {
        'module': 'bayesianquilts.data.eqsq',
        'model_dir': None,  # set via --model-dir
    },
    'wpi': {
        'module': 'bayesianquilts.data.wpi',
        'model_dir': None,
    },
    'tma': {
        'module': 'bayesianquilts.data.tma',
        'model_dir': None,
    },
    'npi': {
        'module': 'bayesianquilts.data.npi',
        'model_dir': None,
    },
}

# Default model locations (relative to bayesianquilts notebooks)
DEFAULT_MODEL_DIRS = {
    'eqsq': os.path.expanduser(
        '~/workspace/bayesianquilts/notebooks/irt/synthetic/results/eqsq/grm_baseline'),
    'wpi': os.path.expanduser(
        '~/workspace/bayesianquilts/notebooks/irt/wpi/grm_baseline'),
    'tma': os.path.expanduser(
        '~/workspace/bayesianquilts/notebooks/irt/tma/grm_baseline'),
    'npi': os.path.expanduser(
        '~/workspace/bayesianquilts/notebooks/irt/npi/grm_baseline'),
}


def extract_params(model_dir, dataset_name, n_samples=1000, seed=42):
    """Load a fitted GRM and extract slope/calibration via surrogate sampling.

    Returns:
        dict with keys: slope, calibration, item_keys, response_cardinality
    """
    import importlib
    import jax
    import jax.numpy as jnp
    from bayesianquilts.irt.grm import GRModel

    # Load dataset metadata
    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    print(f"Loading model from {model_dir}")
    model = GRModel.load_from_disk(model_dir)

    # Sample from surrogate posterior
    print(f"Sampling {n_samples} draws from surrogate posterior...")
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)

    # Take mean across samples — bijectors already applied by surrogate
    # discriminations: shape (n_samples, 1, dim, n_items, 1) -> mean -> squeeze
    disc_samples = np.array(samples['discriminations'])
    slope = np.mean(disc_samples, axis=0).squeeze()  # (n_items,)

    # difficulties0: shape (n_samples, 1, dim, n_items, 1) -> mean -> squeeze
    diff0_samples = np.array(samples['difficulties0'])
    diff0 = np.mean(diff0_samples, axis=0).squeeze()  # (n_items,)

    if response_cardinality > 2 and 'ddifficulties' in samples:
        # ddifficulties: shape (n_samples, 1, dim, n_items, K-2) -> mean -> squeeze
        ddiff_samples = np.array(samples['ddifficulties'])
        ddiff = np.mean(ddiff_samples, axis=0).squeeze()  # (n_items, K-2)

        # Reconstruct calibration thresholds via cumulative sum
        # calibration[:, 0] = diff0
        # calibration[:, j] = diff0 + sum(ddiff[:, :j]) for j >= 1
        calibration = np.zeros((len(item_keys), response_cardinality - 1))
        calibration[:, 0] = diff0
        for j in range(1, response_cardinality - 1):
            calibration[:, j] = calibration[:, j - 1] + ddiff[:, j - 1]
    else:
        # Binary (K=2): single threshold per item
        calibration = diff0[:, np.newaxis]  # (n_items, 1)

    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  Slope range: [{slope.min():.3f}, {slope.max():.3f}]")
    print(f"  Calibration range: [{calibration.min():.3f}, {calibration.max():.3f}]")

    result = {
        'slope': slope,
        'calibration': calibration,
        'item_keys': np.array(item_keys),
        'response_cardinality': response_cardinality,
    }

    # Free JAX memory
    del model, surrogate, samples, disc_samples, diff0_samples
    if response_cardinality > 2:
        del ddiff_samples
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract GRM params from fitted bayesianquilts models')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--model-dir', default=None,
                        help='Path to grm_baseline/ directory (default: auto)')
    parser.add_argument('--output-dir', default='params',
                        help='Output directory for .npz files')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of surrogate samples for posterior mean')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir or DEFAULT_MODEL_DIRS[args.dataset]

    if not os.path.exists(os.path.join(model_dir, 'params.h5')):
        print(f"ERROR: No params.h5 found in {model_dir}")
        print(f"Fit the model first with: uv run python run_single_notebook.py --dataset {args.dataset}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    result = extract_params(model_dir, args.dataset,
                            n_samples=args.n_samples, seed=args.seed)

    out_path = os.path.join(args.output_dir, f'{args.dataset}_grm_params.npz')
    np.savez(out_path, **result)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
