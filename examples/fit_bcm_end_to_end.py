#!/usr/bin/env python
"""End-to-end example: fit a subset-specific BCM bias-correction map.

Bias-Correction Map (BCM) is a *scoring-time* post-hoc correction that
maps a naive subset-IRT score to the score the same respondent would have
received from the full item bank, given which items happened to be
administered. BCM is fit at calibration time using only the original
training data:

  1. For each training respondent, compute the gold-standard EAP score
     using their *full* response vector.
  2. Sample many random subsets of items at fixed sizes (here J=5 and
     J=10).
  3. For each (respondent, subset) pair, compute the naive subset EAP
     score (no imputation -- just the EAP of theta given the observed
     responses on items in the subset).
  4. Train BCMConditional on the matrix
        [ subset_score,  z_1, z_2, ..., z_I ]    ->   gold_score
     where z_i in {0,1} is 1 iff item i was in the subset. The score
     feature carries a monotone non-decreasing constraint so that
     higher raw scores always map to higher corrected scores; the item
     indicators are unconstrained, letting the regressor learn a
     subset-specific bias-adjustment surface. 5-fold cross-validation
     is used to obtain held-out predictions.

At scoring time (last block below), the same BCMConditional is applied
to any new (subset_score, item_indicators) pair to produce a
bias-corrected score. No imputation is needed at scoring time -- the
correction is fit on the (subset score, items administered) pair alone.

We run the example on the GCBS dataset (15 items, 2495 respondents, 5
response categories, ~0.3% missingness) because it is small enough for
the full pipeline to complete in a few minutes on a single CPU.

Usage:

    python examples/fit_bcm_end_to_end.py
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
from libfabulouscatpy.biascorrection import BCMConditional


# ---------------------------------------------------------------------------
# 1.  Load IRT model and training data
# ---------------------------------------------------------------------------

GCBS_DIR = REPO_LIBFAB / 'examples' / 'gcbs'
SCALE_NAME = 'gcbs'
INTERPOLATION_PTS = np.arange(-6.0, 6.0, step=0.05)  # theta grid
PRIOR_SIGMA = 10.0           # near-flat prior so prior shrinkage is small
N_SUBSETS_PER_SIZE = 200     # subset draws per size (use 500 for production)
SUBSET_SIZES = [5, 10]
MAX_RESPONDENTS = 200        # stratify+downsample for tutorial speed
RNG_SEED = 42


class _FakeItemDatabase:
    def __init__(self, items):
        self.items = items


class _FakeScaleDatabase:
    def __init__(self, scales):
        self.scales = scales
        self.scale_keys = list(scales.keys())


def load_grm_model():
    """Build a MultivariateGRM from the pre-fitted GCBS parameters."""
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
    model = MultivariateGRM(
        itemdb=_FakeItemDatabase(items),
        scaledb=_FakeScaleDatabase({SCALE_NAME: {'description': SCALE_NAME}}),
        interpolation_pts=INTERPOLATION_PTS,
    )
    return model, item_keys, K


def load_responses(item_keys, K):
    """Load real GCBS responses; encode missing/invalid as -1."""
    import inspect
    from bayesianquilts.data.gcbs import get_data
    kwargs = {'polars_out': True}
    if 'reorient' in inspect.signature(get_data).parameters:
        kwargs['reorient'] = True
    df, n = get_data(**kwargs)
    all_responses = {}
    for key in item_keys:
        vals = df[key].to_numpy().astype(float)
        valid = ~(np.isnan(vals) | (vals < 0) | (vals >= K))
        all_responses[key] = np.where(valid, vals.astype(int) + 1, -1)
    return all_responses, n


def score(model, person_responses, prior_sigma=PRIOR_SIGMA):
    """Naive EAP score for one person; no imputation."""
    if len(person_responses) < 2:
        return None
    scorer = BayesianScoring(
        model=model,
        log_prior_fn={SCALE_NAME: gaussian_dens(prior_sigma)},
        imputation_model=None,
    )
    scores = scorer.score_responses(person_responses)
    if SCALE_NAME not in scores:
        return None
    return float(scores[SCALE_NAME].score)


# ---------------------------------------------------------------------------
# 2.  Build (subset_score, item_indicators, gold_score) training triples
# ---------------------------------------------------------------------------

def build_training_triples(model, item_keys, all_responses, num_people, rng):
    """Return (subset_score, item_indicator_matrix, gold_score) arrays."""
    I = len(item_keys)
    key_to_idx = {k: i for i, k in enumerate(item_keys)}

    # Eligible respondents: enough observed items to define a gold score
    observed = np.zeros(num_people, dtype=int)
    for k in item_keys:
        observed += (all_responses[k] > 0).astype(int)
    eligible = np.where(observed >= I)[0]

    # Quantile-stratify down to MAX_RESPONDENTS for tutorial speed
    if len(eligible) > MAX_RESPONDENTS:
        sum_score = np.zeros(len(eligible))
        for k in item_keys:
            sum_score += np.maximum(all_responses[k][eligible].astype(float), 0)
        sum_score += rng.normal(0, 1e-9, sum_score.shape)  # tie-break
        sort_order = np.argsort(sum_score)
        pick = np.linspace(0, len(eligible) - 1, MAX_RESPONDENTS).astype(int)
        chosen = eligible[sort_order[pick]]
    else:
        chosen = eligible
    N = len(chosen)
    print(f"  Using {N} respondents")

    # Full-response dicts and gold-standard scores (one EAP per person)
    print("  Computing gold-standard EAP scores...")
    full_dicts, gold = [], np.full(N, np.nan)
    for pi, idx in enumerate(chosen):
        person = {k: int(all_responses[k][idx]) for k in item_keys
                  if all_responses[k][idx] > 0}
        full_dicts.append(person)
        gold[pi] = score(model, person) if len(person) >= 2 else np.nan

    # Sample random subsets and score each person on each subset
    print("  Scoring naive EAP on random subsets...")
    scores, indicators, golds = [], [], []
    for size in SUBSET_SIZES:
        n_total = min(N_SUBSETS_PER_SIZE, _binom_cap(I, size))
        print(f"    J={size}: {n_total} subset draws")
        for d in range(n_total):
            subset = list(rng.choice(item_keys, size=size, replace=False))
            indicator = np.zeros(I, dtype=np.float32)
            for k in subset:
                indicator[key_to_idx[k]] = 1.0
            for pi in range(N):
                if np.isnan(gold[pi]):
                    continue
                resp = {k: full_dicts[pi][k] for k in subset
                        if k in full_dicts[pi]}
                if len(resp) < 2:
                    continue
                s = score(model, resp)
                if s is None:
                    continue
                scores.append(s)
                indicators.append(indicator.copy())
                golds.append(gold[pi])

    return (np.asarray(scores, dtype=float),
            np.stack(indicators, axis=0).astype(float),
            np.asarray(golds, dtype=float),
            item_keys)


def _binom_cap(n, k):
    """min(500, C(n,k)); avoids overflow for small banks."""
    from math import comb
    return min(N_SUBSETS_PER_SIZE, comb(n, k))


# ---------------------------------------------------------------------------
# 3.  Fit the BCM
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    print("[1/4] Loading GRM model and responses")
    model, item_keys, K = load_grm_model()
    all_responses, num_people = load_responses(item_keys, K)
    print(f"      I={len(item_keys)}, K={K}, N={num_people}")

    print("[2/4] Building training triples")
    subset_scores, indicators, gold_scores, item_keys = build_training_triples(
        model, item_keys, all_responses, num_people, rng)
    print(f"      n_triples = {subset_scores.size}")

    print("[3/4] Fitting BCMConditional (5-fold CV, HistGradientBoosting)")
    bcm = BCMConditional.fit(
        subset_scores, indicators, gold_scores,
        item_keys=item_keys,
        scale_name=SCALE_NAME,
        n_folds=5,
        seed=RNG_SEED,
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
    )
    out_path = Path('bcm_gcbs.joblib')
    bcm.save(out_path)
    print(f"      saved -> {out_path}")

    # ---- Evaluation ------------------------------------------------------
    naive_bias = subset_scores - gold_scores
    corrected = bcm.oof_predictions          # held-out predictions
    bcm_bias = corrected - gold_scores

    print("\n[4/4] Bias summary (5-fold held-out)")
    print(f"      L2(naive)   = {np.sqrt(np.mean(naive_bias**2)):.4f}")
    print(f"      L2(BCM)     = {np.sqrt(np.mean(bcm_bias**2)):.4f}")
    print(f"      mean naive bias = {naive_bias.mean():+.4f}  "
          f"sd = {naive_bias.std():.4f}")
    print(f"      mean BCM bias   = {bcm_bias.mean():+.4f}  "
          f"sd = {bcm_bias.std():.4f}")

    # ---- Scoring-time use of the fitted BCM ------------------------------
    # At scoring time, pass any new (subset score, item-indicator vector)
    # through bcm.apply(); the corrected scores are bias-adjusted relative
    # to the gold-standard full-bank score.
    sample_idx = np.random.default_rng(0).choice(len(subset_scores), 5)
    sample_scores = subset_scores[sample_idx]
    sample_indicators = indicators[sample_idx]
    sample_gold = gold_scores[sample_idx]
    sample_corrected = bcm.apply(sample_scores, sample_indicators)
    print("\n[demo] Scoring-time application on 5 random rows")
    print(f"      {'naive':>9s} {'BCM':>9s} {'gold':>9s}")
    for s, c, g in zip(sample_scores, sample_corrected, sample_gold):
        print(f"      {s:>9.3f} {c:>9.3f} {g:>9.3f}")


if __name__ == '__main__':
    main()
