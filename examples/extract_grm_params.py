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
    'grit': {
        'module': 'bayesianquilts.data.grit',
        'model_dir': None,
    },
    'rwa': {
        'module': 'bayesianquilts.data.rwa',
        'model_dir': None,
    },
    'eqsq': {
        'module': 'bayesianquilts.data.eqsq',
        'model_dir': None,
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
    'gcbs': {
        'module': 'bayesianquilts.data.gcbs',
        'model_dir': None,
    },
    'scs': {
        'module': 'bayesianquilts.data.scs',
        'model_dir': None,
    },
}

# Default model locations (relative to bayesianquilts notebooks)
DEFAULT_MODEL_DIRS = {k: os.path.expanduser(
    f'~/workspace/bayesianquilts/notebooks/irt/{k}/grm_baseline')
    for k in DATASET_CONFIGS}


def extract_params(model_dir, dataset_name, n_samples=1000, seed=42):
    """Load a fitted GRM and extract slope/calibration via IS-corrected surrogate.

    Draws samples from the ADVI surrogate, computes importance weights
    using the unnormalized log posterior vs the surrogate log density,
    applies PSIS smoothing, then takes weighted posterior means.

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

    # Load data for computing the log-likelihood
    get_data_kwargs = {'polars_out': True}
    import inspect
    if 'reorient' in inspect.signature(mod.get_data).parameters:
        get_data_kwargs['reorient'] = True
    df, num_people = mod.get_data(**get_data_kwargs)
    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy().astype(np.float32)
    data['person'] = np.arange(num_people, dtype=np.float32)

    # Sample from surrogate posterior
    print(f"Sampling {n_samples} draws from surrogate posterior...")
    surrogate = model.surrogate_distribution_generator(model.params)
    key = jax.random.PRNGKey(seed)
    samples = surrogate.sample(n_samples, seed=key)
    model.surrogate_sample = samples

    # Compute importance weights: log w_s = log p(data, theta_s) - log q(theta_s)
    print("  Computing importance weights...")

    # Compute log joint for all samples at once (batched)
    log_joints = np.array(model.unormalized_log_prob(data, **samples)).squeeze()

    # Surrogate log prob for all samples
    surrogate_dist = model.surrogate_distribution_generator(model.params)
    log_surrogates = np.array(surrogate_dist.log_prob(samples)).squeeze()

    log_weights = log_joints - log_surrogates

    # PSIS smoothing
    from bayesianquilts.metrics.nppsis import psislw
    log_weights_smoothed, khat = psislw(log_weights)
    print(f"  PSIS k-hat: {khat:.3f}")
    if khat > 0.7:
        print(f"  WARNING: k-hat {khat:.3f} > 0.7, IS correction may be unreliable")
        print(f"  Falling back to unweighted surrogate mean")
        weights = np.ones(n_samples) / n_samples
    else:
        weights = np.exp(log_weights_smoothed)
        weights /= weights.sum()

    n_eff = 1.0 / np.sum(weights ** 2)
    print(f"  Effective sample size: {n_eff:.0f} / {n_samples}")

    # Standardize so theta ~ N(0,1) in the calibration population
    # Use unweighted standardize (the IS weights correct parameter means,
    # not the ability scale)
    print("  Standardizing abilities...")
    stats = model.standardize_abilities()
    samples = model.surrogate_sample  # now rescaled in-place
    print(f"  mu={np.array(stats['mu']).squeeze():.4f}, "
          f"sigma={np.array(stats['sigma']).squeeze():.4f}")

    # IS-weighted mean across samples
    disc_samples = np.array(samples['discriminations'])
    slope = np.tensordot(weights, disc_samples, axes=([0], [0])).squeeze()

    diff0_samples = np.array(samples['difficulties0'])
    diff0 = np.tensordot(weights, diff0_samples, axes=([0], [0])).squeeze()

    if response_cardinality > 2 and 'ddifficulties' in samples:
        ddiff_samples = np.array(samples['ddifficulties'])
        ddiff = np.tensordot(weights, ddiff_samples, axes=([0], [0])).squeeze()

        calibration = np.zeros((len(item_keys), response_cardinality - 1))
        calibration[:, 0] = diff0
        for j in range(1, response_cardinality - 1):
            calibration[:, j] = calibration[:, j - 1] + ddiff[:, j - 1]
    else:
        calibration = diff0[:, np.newaxis]

    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"  Slope range: [{slope.min():.3f}, {slope.max():.3f}]")
    print(f"  Calibration range: [{calibration.min():.3f}, {calibration.max():.3f}]")

    result = {
        'slope': slope,
        'calibration': calibration,
        'item_keys': np.array(item_keys),
        'response_cardinality': response_cardinality,
        'khat': khat,
        'n_eff': n_eff,
    }

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
