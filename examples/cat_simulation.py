#!/usr/bin/env python3
"""Run CAT simulations on public psychometric instruments.

Loads extracted GRM parameters (.npz) and runs simulations comparing
6 item selection methods, matching the protocol from the paper.

Usage:
    uv run python cat_simulation.py --dataset eqsq --params-dir params/ --output-dir results/
    uv run python cat_simulation.py --dataset wpi --params-dir params/ --output-dir results/

Memory: Pure numpy, ~1-2 GB per instrument.
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

# Add parent so libfabulouscatpy is importable when running from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.irt.prediction.grm import GradedResponseModel, MultivariateGRM
from libfabulouscatpy.irt.scoring.bayesian import BayesianScoring, gaussian_dens
from libfabulouscatpy.cat.session import CatSession, CatSessionTracker
from libfabulouscatpy.cat.itemselectors.fisher import FisherItemSelector
from libfabulouscatpy.cat.itemselectors.bayesianfisher import BayesianFisherItemSelector
from libfabulouscatpy.cat.itemselectors.globalinfo import GlobalInfoSelector
from libfabulouscatpy.cat.itemselectors.variance import VarianceItemSelector
from libfabulouscatpy.cat.itemselectors.crossentropy import (
    CrossEntropyItemSelector,
    StochasticCrossEntropyItemSelector,
)


# Grid for Bayesian scoring
INTERPOLATION_PTS = np.arange(-6.0, 6.0, step=0.03)

# Simulation protocol (matching the paper)
TRUE_ABILITIES = np.arange(-3, 3.5, 0.5)  # {-3, -2.5, ..., 2.5, 3}
N_REPLICATES = 500
TEST_LENGTHS = [5, 10, 20, 30, 40]

# Exposure experiment
EXPOSURE_SESSIONS = [8, 16, 32, 64, 128]

SELECTOR_CONFIGS = {
    'fisher': {
        'class': FisherItemSelector,
        'kwargs': {'deterministic': True},
    },
    'bayesian_fisher': {
        'class': BayesianFisherItemSelector,
        'kwargs': {},
    },
    'global_info': {
        'class': GlobalInfoSelector,
        'kwargs': {},
    },
    'bayesian_variance': {
        'class': VarianceItemSelector,
        'kwargs': {},
    },
    'cross_entropy': {
        'class': CrossEntropyItemSelector,
        'kwargs': {'deterministic': True},
    },
    'stochastic_ce': {
        'class': StochasticCrossEntropyItemSelector,
        'kwargs': {},
    },
}


def build_model_from_params(npz_path, scale_name):
    """Build a MultivariateGRM from extracted .npz parameters.

    Since these are single-scale instruments, we build item dicts
    with one scale each, matching the format MultivariateGRM expects.
    """
    data = np.load(npz_path, allow_pickle=True)
    slope = data['slope']
    calibration = data['calibration']
    item_keys = list(data['item_keys'])
    response_cardinality = int(data['response_cardinality'])

    n_items = len(item_keys)

    # Build item dicts in the format expected by MultivariateGRM
    items = []
    for i in range(n_items):
        item_dict = {
            'item': item_keys[i],
            'scales': {
                scale_name: {
                    'discrimination': float(slope[i]),
                    'difficulties': calibration[i].tolist(),
                }
            }
        }
        items.append(item_dict)

    # Build a fake ItemDatabase and ScaleDatabase
    itemdb = _FakeItemDatabase(items)
    scaledb = _FakeScaleDatabase({scale_name: {'description': scale_name}})

    model = MultivariateGRM(
        itemdb=itemdb,
        scaledb=scaledb,
        interpolation_pts=INTERPOLATION_PTS,
    )
    return model, items, item_keys, response_cardinality


class _FakeItemDatabase:
    """Minimal ItemDatabase compatible with MultivariateGRM."""
    def __init__(self, items):
        self.items = items


class _FakeScaleDatabase:
    """Minimal ScaleDatabase compatible with MultivariateGRM."""
    def __init__(self, scales):
        self.scales = scales
        self.scale_keys = list(scales.keys())


def make_selector_kwargs(items, scales, model, scoring):
    """Build the common kwargs needed by all ItemSelector constructors."""
    return {
        'items': items,
        'scales': scales,
        'model': model,
        'temperature': 0.01,
        'randomize_items': True,
        'randomize_scales': False,
        'precision_limit': 0.33333,
        'min_responses': 5,
        'max_responses': 200,  # don't stop early via max_responses
    }


def run_accuracy_experiment(model, items, scale_name, scales,
                            selector_name, selector_config,
                            max_items, seed=42):
    """Run the accuracy experiment for one selector.

    For each true ability and replicate, simulate responses and run CAT.
    Collect KL discrepancy, absolute error, and posterior SD at each step.
    """
    n_abilities = len(TRUE_ABILITIES)
    kl = np.full((n_abilities, N_REPLICATES, max_items), np.nan)
    l2 = np.full((n_abilities, N_REPLICATES, max_items), np.nan)
    se = np.full((n_abilities, N_REPLICATES, max_items), np.nan)

    log_prior_fn = {scale_name: gaussian_dens(1.0)}

    for ai, theta_val in enumerate(TRUE_ABILITIES):
        theta = {scale_name: np.atleast_1d(theta_val)}
        t0 = time.time()

        for rep in range(N_REPLICATES):
            np.random.seed(seed + ai * N_REPLICATES + rep)

            # Simulate true responses
            true_responses = model.sample(theta)
            true_responses_scoring = {k: v + 1 for k, v in true_responses.items()}

            # Score with all responses to get ground truth
            scoring_truth = BayesianScoring(
                model=model, log_prior_fn=log_prior_fn)
            true_scores = scoring_truth.score_responses(true_responses_scoring)

            # Fresh scorer + selector for CAT
            scoring = BayesianScoring(
                model=model, log_prior_fn=log_prior_fn)
            common_kwargs = make_selector_kwargs(items, scales, model, scoring)
            sel = selector_config['class'](
                scoring=scoring,
                **selector_config['kwargs'],
                **common_kwargs,
            )
            tracker = CatSessionTracker(
                session=CatSession(), scales=[scale_name])

            for step in range(max_items):
                item_dict = sel.next_item(tracker, scale=scale_name)
                if item_dict is None or item_dict == {}:
                    break

                item_id = item_dict['item']
                response = true_responses_scoring[item_id]
                tracker.responses[item_id] = response

                step_scores = scoring.score_responses(tracker.responses)
                for s in step_scores:
                    tracker.scores[s] = step_scores[s].score
                    tracker.errors[s] = step_scores[s].error

                if scale_name in step_scores and scale_name in true_scores:
                    ts = true_scores[scale_name]
                    cs = step_scores[scale_name]
                    l2[ai, rep, step] = abs(ts.score - cs.score)
                    se[ai, rep, step] = cs.error
                    # KL(true || current)
                    p = ts.density
                    q = cs.density
                    eps = 1e-20
                    integrand = p * np.log((p + eps) / (q + eps))
                    kl[ai, rep, step] = float(
                        _trapz(y=integrand, x=ts.interpolation_pts))

        dt = time.time() - t0
        print(f"    theta={theta_val:+.1f}: {dt:.1f}s")

    return {'kl': kl, 'l2': l2, 'se': se}


def run_exposure_experiment(model, items, scale_name, scales,
                            selector_name, selector_config,
                            items_per_session, seed=42):
    """Run the exposure experiment for one selector.

    Count unique items exposed across increasing numbers of sessions.
    """
    n_session_counts = len(EXPOSURE_SESSIONS)
    max_sessions = max(EXPOSURE_SESSIONS)
    n_exposure_replicates = 100
    log_prior_fn = {scale_name: gaussian_dens(1.0)}

    unique_items = np.full((n_session_counts, n_exposure_replicates), np.nan)

    for rep in range(n_exposure_replicates):
        np.random.seed(seed + 10000 + rep)
        seen_items = set()

        for sess in range(max_sessions):
            theta_val = np.random.randn()
            theta = {scale_name: np.atleast_1d(theta_val)}

            scoring = BayesianScoring(
                model=model, log_prior_fn=log_prior_fn)
            common_kwargs = make_selector_kwargs(items, scales, model, scoring)
            sel = selector_config['class'](
                scoring=scoring,
                **selector_config['kwargs'],
                **common_kwargs,
            )
            tracker = CatSessionTracker(
                session=CatSession(), scales=[scale_name])

            true_responses = model.sample(theta)
            true_responses_scoring = {k: v + 1 for k, v in true_responses.items()}

            for step in range(items_per_session):
                item_dict = sel.next_item(tracker, scale=scale_name)
                if item_dict is None or item_dict == {}:
                    break
                item_id = item_dict['item']
                response = true_responses_scoring[item_id]
                tracker.responses[item_id] = response

                scores = scoring.score_responses(tracker.responses)
                for s in scores:
                    tracker.scores[s] = scores[s].score
                    tracker.errors[s] = scores[s].error

                seen_items.add(item_id)

            # Record at each checkpoint
            for si, n_sess in enumerate(EXPOSURE_SESSIONS):
                if sess + 1 == n_sess:
                    unique_items[si, rep] = len(seen_items)

    return {'unique_items': unique_items}


def main():
    parser = argparse.ArgumentParser(description='Run CAT simulations')
    parser.add_argument('--dataset', required=True,
                        choices=['eqsq', 'wpi', 'tma', 'npi'])
    parser.add_argument('--params-dir', default='params',
                        help='Directory containing .npz param files')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for results')
    parser.add_argument('--selectors', nargs='+',
                        default=list(SELECTOR_CONFIGS.keys()),
                        help='Which selectors to run')
    parser.add_argument('--skip-accuracy', action='store_true',
                        help='Skip accuracy experiment')
    parser.add_argument('--skip-exposure', action='store_true',
                        help='Skip exposure experiment')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    npz_path = os.path.join(args.params_dir, f'{args.dataset}_grm_params.npz')
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run extract_grm_params.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    scale_name = args.dataset.upper()

    print(f"Loading model for {args.dataset}...")
    model, items, item_keys, response_cardinality = build_model_from_params(
        npz_path, scale_name)
    n_items = len(item_keys)
    scales = {scale_name: {'description': scale_name}}

    max_items = min(n_items, max(TEST_LENGTHS))
    items_per_session = min(12, n_items // 4)

    print(f"  Items: {n_items}, K: {response_cardinality}")
    print(f"  Max CAT length: {max_items}, Items/session (exposure): {items_per_session}")
    print(f"  Selectors: {args.selectors}")

    for sel_name in args.selectors:
        if sel_name not in SELECTOR_CONFIGS:
            print(f"WARNING: Unknown selector {sel_name}, skipping")
            continue

        sel_config = SELECTOR_CONFIGS[sel_name]
        print(f"\n{'='*60}")
        print(f"Selector: {sel_name}")
        print(f"{'='*60}")

        if not args.skip_accuracy:
            print("  Running accuracy experiment...")
            acc = run_accuracy_experiment(
                model, items, scale_name, scales,
                sel_name, sel_config, max_items, seed=args.seed)

            out_path = os.path.join(
                args.output_dir, f'{args.dataset}_accuracy_{sel_name}.npz')
            np.savez_compressed(out_path,
                                true_abilities=TRUE_ABILITIES,
                                test_lengths=np.array(TEST_LENGTHS),
                                **acc)
            print(f"  Saved accuracy: {out_path}")

        if not args.skip_exposure:
            print("  Running exposure experiment...")
            exp = run_exposure_experiment(
                model, items, scale_name, scales,
                sel_name, sel_config, items_per_session, seed=args.seed)

            out_path = os.path.join(
                args.output_dir, f'{args.dataset}_exposure_{sel_name}.npz')
            np.savez_compressed(out_path,
                                exposure_sessions=np.array(EXPOSURE_SESSIONS),
                                n_items=n_items,
                                **exp)
            print(f"  Saved exposure: {out_path}")

        gc.collect()

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == '__main__':
    main()
