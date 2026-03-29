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
from libfabulouscatpy.cat.itemselectors.entropy import (
    EntropyItemSelector,
    StochasticEntropyItemSelector,
)
from libfabulouscatpy.imputation.irt_pairwise import pairwise_imputation_from_grm


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
    'entropy': {
        'class': EntropyItemSelector,
        'kwargs': {'deterministic': True},
    },
    'stochastic_entropy': {
        'class': StochasticEntropyItemSelector,
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


def sample_from_ensemble(model, scale_name, theta, imputation_model,
                         n_categories, mixing_weight=0.5):
    """Sample responses from the imputation-adjusted ensemble.

    Instead of sampling purely from the IRT model P(x_i|θ), we generate
    responses sequentially.  For each item, the sampling distribution is:

        p_gen(x_i=k) = (1-w) · P_irt(x_i=k|θ)  +  w · π*(x_i=k|x_observed)

    where x_observed are the items already sampled.  This creates pairwise
    dependencies beyond what the IRT model captures, simulating the M-open
    regime where the scoring model is misspecified.

    Returns:
        dict mapping item_id -> 0-indexed response
    """
    grm = model.models[scale_name]
    item_labels = list(grm.item_labels)

    # IRT probabilities for all items at this theta: (n_items, K)
    log_p = grm.log_likelihood(
        theta=np.atleast_1d(theta), observed_only=False)  # (1, n_items, K)
    p_irt = np.exp(log_p[0])
    p_irt = p_irt / p_irt.sum(axis=-1, keepdims=True)

    # Random item ordering so the sequential blending doesn't favour
    # any particular direction
    order = np.random.permutation(len(item_labels))
    responses = {}

    for idx in order:
        item = item_labels[idx]
        irt_pmf = p_irt[idx]

        if responses and imputation_model is not None:
            # Imputation model prediction given already-sampled items
            observed_dict = {k: float(v) for k, v in responses.items()}
            try:
                imp_pmf = imputation_model.predict_pmf(
                    items=observed_dict, target=item,
                    n_categories=n_categories)
                imp_pmf = np.asarray(imp_pmf, dtype=float)
                imp_pmf = np.maximum(imp_pmf, 1e-20)
                imp_pmf /= imp_pmf.sum()
                blended = (1 - mixing_weight) * irt_pmf + mixing_weight * imp_pmf
            except (KeyError, ValueError):
                blended = irt_pmf
        else:
            blended = irt_pmf

        blended = np.maximum(blended, 1e-20)
        blended /= blended.sum()
        responses[item] = int(np.random.choice(len(blended), p=blended))

    return responses


def make_selector_kwargs(items, scales, model, scoring):
    """Build the common kwargs needed by all ItemSelector constructors."""
    return {
        'items': items,
        'scales': scales,
        'model': model,
        'temperature': 0.01,
        'randomize_items': True,
        'randomize_scales': False,
        'precision_limit': 0.0,   # no early stopping by precision
        'min_responses': 0,
        'max_responses': 9999,    # uncapped
    }


def run_accuracy_experiment(model, items, scale_name, scales,
                            selector_name, selector_config,
                            max_items, seed=42,
                            imputation_model=None,
                            response_cardinality=None):
    """Run the accuracy experiment for one selector.

    Responses are generated from the imputation-adjusted ensemble
    (M-open regime), not the bare IRT model.  The ground truth is
    the full-data posterior (scoring all items).
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

            # Generate responses from the imputation-adjusted ensemble
            true_responses = sample_from_ensemble(
                model, scale_name, theta_val, imputation_model,
                n_categories=response_cardinality)
            # Convert to 1-indexed for scoring (GRM expects 1..K)
            true_responses_scoring = {k: v + 1 for k, v in true_responses.items()}

            # Score with all responses to get full-data posterior
            scoring_truth = BayesianScoring(
                model=model, log_prior_fn=log_prior_fn,
                imputation_model=imputation_model)
            true_scores = scoring_truth.score_responses(true_responses_scoring)

            # Fresh scorer + selector for CAT
            scoring = BayesianScoring(
                model=model, log_prior_fn=log_prior_fn,
                imputation_model=imputation_model)
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
                            test_lengths, seed=42,
                            imputation_model=None,
                            response_cardinality=None):
    """Run the exposure experiment for one selector at multiple test lengths.

    For each test length t, runs EXPOSURE_SESSIONS sessions of t items each
    and counts the total unique items exposed.
    """
    n_session_counts = len(EXPOSURE_SESSIONS)
    max_sessions = max(EXPOSURE_SESSIONS)
    n_exposure_replicates = 100
    log_prior_fn = {scale_name: gaussian_dens(1.0)}

    # Results: one set per test length
    results = {}
    for tl in test_lengths:
        unique_items = np.full((n_session_counts, n_exposure_replicates), np.nan)

        for rep in range(n_exposure_replicates):
            np.random.seed(seed + 10000 + tl * 1000 + rep)
            seen_items = set()

            for sess in range(max_sessions):
                theta_val = np.random.randn()

                scoring = BayesianScoring(
                    model=model, log_prior_fn=log_prior_fn,
                    imputation_model=imputation_model)
                common_kwargs = make_selector_kwargs(items, scales, model, scoring)
                sel = selector_config['class'](
                    scoring=scoring,
                    **selector_config['kwargs'],
                    **common_kwargs,
                )
                tracker = CatSessionTracker(
                    session=CatSession(), scales=[scale_name])

                true_responses = sample_from_ensemble(
                    model, scale_name, theta_val, imputation_model,
                    n_categories=response_cardinality)
                true_responses_scoring = {k: v + 1 for k, v in true_responses.items()}

                for step in range(tl):
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

                for si, n_sess in enumerate(EXPOSURE_SESSIONS):
                    if sess + 1 == n_sess:
                        unique_items[si, rep] = len(seen_items)

        results[tl] = unique_items
        print(f"    t={tl}: mean unique at 32 sessions = "
              f"{np.nanmean(unique_items[2]):.1f}")

    return results


def print_accuracy_summary(results_dir, dataset, selectors, test_lengths):
    """Print accuracy results broken down by ability range."""
    # Ability ranges: low (<= -1.5), mid (-1 to 1), high (>= 1.5)
    ranges = {
        'low (theta <= -1.5)': TRUE_ABILITIES <= -1.5,
        'mid (-1 <= theta <= 1)': (TRUE_ABILITIES >= -1.0) & (TRUE_ABILITIES <= 1.0),
        'high (theta >= 1.5)': TRUE_ABILITIES >= 1.5,
        'all': np.ones(len(TRUE_ABILITIES), dtype=bool),
    }

    for metric_key, metric_label in [('l2', 'Mean |score_true - score_CAT|'),
                                      ('kl', 'Mean KL(true || CAT)')]:
        print(f"\n  Metric: {metric_label}")

        for tl in test_lengths:
            step_idx = tl - 1
            print(f"\n{'='*72}")
            print(f"  Test length = {tl}")
            print(f"{'='*72}")
            header = f"  {'Range':<25s}"
            for sel in selectors:
                header += f" {sel:>14s}"
            print(header)
            print("  " + "-" * (25 + 15 * len(selectors)))

            for range_name, mask in ranges.items():
                row = f"  {range_name:<25s}"
                for sel in selectors:
                    path = os.path.join(
                        results_dir, f'{dataset}_accuracy_{sel}.npz')
                    if not os.path.exists(path):
                        row += f" {'N/A':>14s}"
                        continue
                    data = np.load(path)
                    arr = data[metric_key]
                    subset = arr[mask, :, step_idx]
                    val = np.nanmean(subset)
                    row += f" {val:>14.4f}"
                print(row)


def main():
    parser = argparse.ArgumentParser(description='Run CAT simulations')
    parser.add_argument('--dataset', required=True,
                        choices=['grit', 'rwa', 'eqsq', 'wpi', 'tma', 'npi', 'gcbs', 'scs'])
    parser.add_argument('--params-dir', default=None,
                        help='Directory containing .npz param files '
                             '(default: examples/<dataset>/)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results '
                             '(default: examples/<dataset>/results/)')
    parser.add_argument('--selectors', nargs='+',
                        default=list(SELECTOR_CONFIGS.keys()),
                        help='Which selectors to run')
    parser.add_argument('--skip-accuracy', action='store_true',
                        help='Skip accuracy experiment')
    parser.add_argument('--skip-exposure', action='store_true',
                        help='Skip exposure experiment')
    parser.add_argument('--imputation-model', default=None,
                        help='Path to imputation model JSON '
                             '(default: examples/<dataset>/imputation_model.json)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, args.dataset)
    params_dir = args.params_dir or dataset_dir
    output_dir = args.output_dir or os.path.join(dataset_dir, 'results')

    npz_path = os.path.join(params_dir, f'{args.dataset}_grm_params.npz')
    if not os.path.exists(npz_path):
        # Fall back to bundled package data
        try:
            from libfabulouscatpy.data import get_grm_params_path
            npz_path = str(get_grm_params_path(args.dataset))
            print(f"Using bundled params: {npz_path}")
        except (ImportError, FileNotFoundError):
            print(f"ERROR: {npz_path} not found and no bundled params available.")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    scale_name = args.dataset.upper()

    print(f"Loading model for {args.dataset}...")
    model, items, item_keys, response_cardinality = build_model_from_params(
        npz_path, scale_name)
    n_items = len(item_keys)
    scales = {scale_name: {'description': scale_name}}

    # Load imputation model: explicit path > dataset dir > IRT-derived fallback
    imp_path = args.imputation_model
    if imp_path is None:
        imp_path = os.path.join(dataset_dir, 'imputation_model.json')
    if os.path.exists(imp_path):
        from libfabulouscatpy.imputation.pairwise import PairwiseImputationModel
        imputation_model = PairwiseImputationModel.from_json(imp_path)
        print(f"Loaded imputation model from {imp_path}")
    else:
        imputation_model = pairwise_imputation_from_grm(model, scale_name)
        print(f"No pretrained imputation model found; "
              f"derived from IRT ({n_items} items)")

    max_items = max(TEST_LENGTHS)
    exposure_test_lengths = [tl for tl in TEST_LENGTHS if tl <= n_items]

    print(f"  Items: {n_items}, K: {response_cardinality}")
    print(f"  Max CAT length: {max_items}")
    print(f"  Exposure test lengths: {exposure_test_lengths}")
    print(f"  Selectors: {args.selectors}")
    print(f"  Imputation: ENABLED (adjusted posterior scoring)")

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
                sel_name, sel_config, max_items, seed=args.seed,
                imputation_model=imputation_model,
                response_cardinality=response_cardinality)

            out_path = os.path.join(
                output_dir, f'{args.dataset}_accuracy_{sel_name}.npz')
            np.savez_compressed(out_path,
                                true_abilities=TRUE_ABILITIES,
                                test_lengths=np.array(TEST_LENGTHS),
                                **acc)
            print(f"  Saved accuracy: {out_path}")

        if not args.skip_exposure:
            print("  Running exposure experiment...")
            exp_results = run_exposure_experiment(
                model, items, scale_name, scales,
                sel_name, sel_config, exposure_test_lengths, seed=args.seed,
                imputation_model=imputation_model,
                response_cardinality=response_cardinality)

            for tl, unique_items in exp_results.items():
                out_path = os.path.join(
                    output_dir,
                    f'{args.dataset}_exposure_{sel_name}_t{tl}.npz')
                np.savez_compressed(out_path,
                                    exposure_sessions=np.array(EXPOSURE_SESSIONS),
                                    n_items=n_items,
                                    test_length=tl,
                                    unique_items=unique_items)
            # Also save combined for backward compat (using largest test length)
            largest_tl = max(exposure_test_lengths)
            out_path = os.path.join(
                output_dir, f'{args.dataset}_exposure_{sel_name}.npz')
            np.savez_compressed(out_path,
                                exposure_sessions=np.array(EXPOSURE_SESSIONS),
                                n_items=n_items,
                                unique_items=exp_results[largest_tl])
            print(f"  Saved exposure: {output_dir}/")

        gc.collect()

    # Print summary table broken down by ability range
    if not args.skip_accuracy:
        print(f"\n\n{'#'*72}")
        print(f"  ACCURACY SUMMARY: {args.dataset.upper()}")
        print(f"  Mean |score_true - score_CAT| by ability range")
        print(f"{'#'*72}")
        print_accuracy_summary(
            output_dir, args.dataset, args.selectors,
            [tl for tl in TEST_LENGTHS if tl <= max_items])

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()
