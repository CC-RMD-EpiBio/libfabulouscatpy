#!/usr/bin/env python
"""BCM trained on the stratified-population (population-oversampling) regime.

Companion to ``fit_bcm_end_to_end.py``. The earlier example trained a
BCM on *real* respondents, with the gold target being the full-bank EAP
score under the IRT model. This example trains a BCM on *simulated*
respondents drawn from the same stratified-ability protocol that
``cat_simulation.py`` uses (``TRUE_ABILITIES = {-3, -2.5, ..., +3}``,
``N_REPLICATES`` per stratum). The gold target is the **true latent
ability** that generated the responses, not the IRT EAP.

The point of this regime:

* The stratified ability grid heavily *oversamples the tails* of the
  ability distribution relative to a ``N(0, 1)`` population. This is
  precisely where naive subset scoring is most biased (regression to
  the prior + asymmetric Fisher information). A BCM trained here learns
  the correction at the extremes that a real-respondent BCM cannot
  observe (because real respondents cluster near ``theta = 0``).
* The gold target is the simulated true ``theta``, so we are measuring
  *true* bias rather than the imputation-corrected EAP gap.

Pipeline:

  1. Load pre-fitted GRM parameters (GCBS used here for tutorial speed).
  2. For each stratum ``theta_val`` in TRUE_ABILITIES, simulate
     ``N_REPLICATES`` respondents from the IRT model.
  3. For each simulated respondent, run a Bayesian-Fisher CAT to length
     ``TEST_LENGTH`` and record (subset_score, item_indicators, true_theta).
  4. Train BCMConditional with 5-fold CV.
  5. Report L2(naive) vs L2(BCM) overall and stratified into
     low / mid / high ability buckets.

Usage:
    python examples/fit_bcm_stratified_population.py
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import sys
from pathlib import Path

import numpy as np

REPO_LIBFAB = Path('/home/josh/workspace/libfabulouscatpy')
REPO_BAYESIAN = Path('/home/josh/workspace/bayesianquilts/python')
sys.path.insert(0, str(REPO_LIBFAB))
sys.path.insert(0, str(REPO_BAYESIAN))

from libfabulouscatpy.irt.prediction.grm import MultivariateGRM
from libfabulouscatpy.irt.scoring.bayesian import BayesianScoring, gaussian_dens
from libfabulouscatpy.cat.session import CatSession, CatSessionTracker
from libfabulouscatpy.cat.itemselectors.bayesianfisher import (
    BayesianFisherItemSelector,
)
from libfabulouscatpy.biascorrection import BCMConditional


# ---------------------------------------------------------------------------
# Protocol parameters (match cat_simulation.py at a smaller scale)
# ---------------------------------------------------------------------------

GCBS_DIR = REPO_LIBFAB / 'examples' / 'gcbs'
SCALE_NAME = 'gcbs'
INTERPOLATION_PTS = np.arange(-6.0, 6.0, step=0.05)

# Stratified-population sampling -- oversamples tails relative to N(0,1).
TRUE_ABILITIES = np.arange(-3.0, 3.5, 0.5)  # 13 strata
N_REPLICATES = 25                            # respondents per stratum (use 100 for paper)
TEST_LENGTH = 10                             # CAT length per respondent
PRIOR_SIGMA = 2.0                            # matches cat_simulation.py
RNG_SEED = 42


class _FakeItemDatabase:
    def __init__(self, items):
        self.items = items


class _FakeScaleDatabase:
    def __init__(self, scales):
        self.scales = scales
        self.scale_keys = list(scales.keys())


def load_grm_model():
    data = np.load(GCBS_DIR / 'gcbs_grm_params.npz')
    slope = data['slope']
    calibration = data['calibration']
    raw_keys = data['item_keys']
    item_keys = [k.decode() if isinstance(k, bytes) else str(k)
                 for k in raw_keys]
    K = int(data['response_cardinality'])
    items = [{
        'item': item_keys[i],
        'scales': {SCALE_NAME: {
            'discrimination': float(slope[i]),
            'difficulties': calibration[i].tolist(),
        }},
    } for i in range(len(item_keys))]
    scales = {SCALE_NAME: {'description': SCALE_NAME}}
    model = MultivariateGRM(
        itemdb=_FakeItemDatabase(items),
        scaledb=_FakeScaleDatabase(scales),
        interpolation_pts=INTERPOLATION_PTS,
    )
    return model, items, scales, item_keys, K


def sample_responses_at_theta(model, theta_val, item_keys, rng):
    """Sample one synthetic respondent's full response vector from the GRM.

    Uses the IRT model directly (no imputation blending) so that the
    'true' ability of the respondent is exactly ``theta_val``.
    """
    grm = model.models[SCALE_NAME]
    log_p = grm.log_likelihood(theta=np.atleast_1d(theta_val), observed_only=False)
    p_irt = np.exp(log_p[0])
    p_irt = p_irt / p_irt.sum(axis=-1, keepdims=True)
    item_labels = list(grm.item_labels)
    responses = {}
    for idx, item in enumerate(item_labels):
        # 1-indexed for the GRM scoring convention
        responses[item] = int(rng.choice(len(p_irt[idx]), p=p_irt[idx])) + 1
    return responses


def run_cat_session(model, items, scales, true_responses, test_length):
    """Run one Bayesian-Fisher CAT session of length ``test_length``."""
    scoring = BayesianScoring(
        model=model,
        log_prior_fn={SCALE_NAME: gaussian_dens(PRIOR_SIGMA)},
        imputation_model=None,
    )
    sel = BayesianFisherItemSelector(
        scoring=scoring,
        items=items,
        scales=scales,
        model=model,
        temperature=1.0,
        randomize_items=True,
        randomize_scales=False,
        precision_limit=0.0,
        min_responses=0,
        max_responses=test_length,
    )
    tracker = CatSessionTracker(session=CatSession(), scales=[SCALE_NAME])
    administered = []
    for _ in range(test_length):
        item_dict = sel.next_item(tracker, scale=SCALE_NAME)
        if item_dict is None or item_dict == {}:
            break
        item_id = item_dict['item']
        administered.append(item_id)
        tracker.responses[item_id] = true_responses[item_id]
    final = scoring.score_responses(tracker.responses)
    if SCALE_NAME not in final:
        return None, administered
    return float(final[SCALE_NAME].score), administered


def main():
    rng = np.random.default_rng(RNG_SEED)

    print("[1/4] Loading GRM model")
    model, items, scales, item_keys, K = load_grm_model()
    print(f"      I={len(item_keys)}, K={K}, strata={len(TRUE_ABILITIES)}, "
          f"reps={N_REPLICATES}, test_length={TEST_LENGTH}")

    key_to_idx = {k: i for i, k in enumerate(item_keys)}
    I = len(item_keys)

    print("[2/4] Simulating respondents and running Bayesian-Fisher CAT")
    subset_scores, indicator_rows, true_thetas = [], [], []
    for ai, theta_val in enumerate(TRUE_ABILITIES):
        for rep in range(N_REPLICATES):
            full_responses = sample_responses_at_theta(
                model, theta_val, item_keys, rng)
            score, administered = run_cat_session(
                model, items, scales, full_responses, TEST_LENGTH)
            if score is None:
                continue
            indicator = np.zeros(I, dtype=np.float32)
            for item_id in administered:
                if item_id in key_to_idx:
                    indicator[key_to_idx[item_id]] = 1.0
            subset_scores.append(score)
            indicator_rows.append(indicator)
            true_thetas.append(theta_val)
        print(f"    theta={theta_val:+.1f}: {N_REPLICATES} sessions complete")

    subset_scores = np.asarray(subset_scores, dtype=float)
    indicator_mat = np.stack(indicator_rows, axis=0).astype(float)
    true_thetas = np.asarray(true_thetas, dtype=float)
    print(f"    n_triples = {subset_scores.size}")

    print("[3/4] Fitting BCMConditional (gold = true theta)")
    bcm = BCMConditional.fit(
        subset_scores, indicator_mat, true_thetas,
        item_keys=item_keys,
        scale_name=SCALE_NAME,
        n_folds=5, seed=RNG_SEED,
        max_iter=200, learning_rate=0.05, max_depth=4,
    )
    out_path = Path('bcm_gcbs_stratified.joblib')
    bcm.save(out_path)
    print(f"      saved -> {out_path}")

    # ---- Per-stratum evaluation -----------------------------------------
    naive_bias = subset_scores - true_thetas
    bcm_bias = bcm.oof_predictions - true_thetas

    print("\n[4/4] L2(bias) by ability stratum (5-fold held-out)")
    print(f"      {'stratum':<22s} {'n':>5s} {'L2 naive':>10s} {'L2 BCM':>10s}  "
          f"{'reduction':>10s}")
    print(f"      {'-'*72}")
    masks = [
        ('low (theta <= -1.5)', true_thetas <= -1.5),
        ('mid (-1 <= theta <= 1)', (true_thetas >= -1.0) & (true_thetas <= 1.0)),
        ('high (theta >= 1.5)', true_thetas >= 1.5),
        ('all', np.ones_like(true_thetas, dtype=bool)),
    ]
    for label, mask in masks:
        if mask.sum() == 0:
            continue
        l2_n = float(np.sqrt(np.mean(naive_bias[mask] ** 2)))
        l2_b = float(np.sqrt(np.mean(bcm_bias[mask] ** 2)))
        red = 100.0 * (l2_n - l2_b) / l2_n if l2_n > 0 else float('nan')
        print(f"      {label:<22s} {int(mask.sum()):>5d} "
              f"{l2_n:>10.4f} {l2_b:>10.4f}  {red:>9.1f}%")


if __name__ == '__main__':
    main()
