"""Newton-step bias correction for CAT subset scoring.

Implements the one-step Newton aggregation from eq. (subset_bias_newton)
of Chang et al. (2026):

    hat_theta_Newton = hat_theta_B + Newton_step

where

    Newton_step = -I_S(theta)^{-1} * sum_{i not in S} grad_theta delta_i(theta)

    delta_i(theta) = d/d_theta log a_i(theta)
                   = sum_k pi*(z_i=k) * dp(z_i=k|theta)/d_theta
                     / sum_k pi*(z_i=k) * p(z_i=k|theta)

    a_i(theta)  = sum_k pi*(z_i=k) p(z_i=k|theta)   [blended likelihood weight]

    I_S(theta)  = sum_{i in S} I_i(theta)             [Fisher info of subset]

The per-item score delta_i(theta) is the gradient of log(a_i(theta)), i.e.,
the score equation contribution from unobserved item i under the blended
likelihood.  When pi* = p_IRT, delta_i reduces to the IRT score function,
and the Newton step reduces to zero for the fully-observed case.

The second-order quantity grad_theta delta_i(theta) = d^2/d_theta^2 log a_i
is computed numerically on a dense theta grid and interpolated at the
person's naive MAP score.

The marginal imputation prediction pi*(y_i) used here is extracted from the
pairwise PMF tables in the threeway_imputation.json, averaged over all
conditioning item/response combinations.  This is the same approximation
used in compute_item_bias_leverage_flat.py.

Usage
-----
    from libfabulouscatpy.biascorrection.newton import NewtonCorrector

    corrector = NewtonCorrector.from_threeway_json(path_to_threeway_json)
    # naive_scores: (N,) array of per-person naive MAP/MLE scores
    # subset_items: list of item keys that were administered
    corrected = corrector.apply(naive_scores, subset_items)
    # step: (N,) Newton correction (add to naive_score)
    step = corrector.step(naive_scores, subset_items)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# GRM helpers
# ---------------------------------------------------------------------------

def _grm_pmf_grid(theta_grid: np.ndarray,
                  a: float,
                  thresholds: np.ndarray) -> np.ndarray:
    """GRM PMF for a single item over a theta grid.

    Parameters
    ----------
    theta_grid : (T,) array
    a : scalar discrimination
    thresholds : (K-1,) cumulative difficulty thresholds

    Returns
    -------
    pmf : (T, K) array, rows sum to 1
    """
    T = len(theta_grid)
    eta = a * (theta_grid[:, np.newaxis] - thresholds[np.newaxis, :])  # (T, K-1)
    cum_p = 1.0 / (1.0 + np.exp(-eta))  # (T, K-1)
    cdf = np.concatenate([
        np.ones((T, 1)),
        cum_p,
        np.zeros((T, 1))
    ], axis=1)  # (T, K+1)
    pmf = cdf[:, :-1] - cdf[:, 1:]  # (T, K)
    return np.maximum(pmf, 1e-20)


def _grm_score_grid(theta_grid: np.ndarray,
                    a: float,
                    thresholds: np.ndarray) -> np.ndarray:
    """GRM score function dp(y=k|theta)/d_theta for each category.

    Returns
    -------
    dpmf : (T, K) array of derivatives of p(y=k|theta) w.r.t. theta
    """
    T = len(theta_grid)
    K = len(thresholds) + 1
    eta = a * (theta_grid[:, np.newaxis] - thresholds[np.newaxis, :])  # (T, K-1)
    cum_p = 1.0 / (1.0 + np.exp(-eta))  # (T, K-1)
    # d sigmoid(eta) / d_theta = a * sigmoid * (1 - sigmoid)
    dcum_p = a * cum_p * (1.0 - cum_p)  # (T, K-1)
    # dP(Y=k)/d_theta = dCDF(k) - dCDF(k+1)
    # with dCDF(0)=0 (P(Y>=0)=1, constant) and dCDF(K)=0
    zeros = np.zeros((T, 1))
    dpmf = (np.concatenate([dcum_p, zeros], axis=1)
            - np.concatenate([zeros, dcum_p], axis=1))  # (T, K)
    return dpmf


def _fisher_info_grid(theta_grid: np.ndarray,
                      a: float,
                      thresholds: np.ndarray) -> np.ndarray:
    """Fisher information I_i(theta) = sum_k [dp_k/d_theta]^2 / p_k.

    Parameters
    ----------
    theta_grid : (T,)
    a, thresholds : GRM parameters

    Returns
    -------
    info : (T,) array >= 0
    """
    pmf = _grm_pmf_grid(theta_grid, a, thresholds)   # (T, K)
    dpmf = _grm_score_grid(theta_grid, a, thresholds)  # (T, K)
    info = np.sum(dpmf ** 2 / np.maximum(pmf, 1e-20), axis=1)  # (T,)
    return np.maximum(info, 0.0)


def _delta_i_grid(theta_grid: np.ndarray,
                  a: float,
                  thresholds: np.ndarray,
                  marginal_pmf: np.ndarray) -> np.ndarray:
    """Per-item score contribution delta_i(theta) = d/d_theta log a_i(theta).

    a_i(theta) = sum_k pi*(k) * p(k | theta)

    delta_i(theta) = [sum_k pi*(k) * dp(k|theta)/dtheta]
                     / [sum_k pi*(k) * p(k|theta)]

    When pi* = p_IRT, this equals the IRT score function d/dtheta log p(y|theta),
    which integrates to zero under the IRT model (as expected).

    Parameters
    ----------
    theta_grid : (T,)
    a, thresholds : GRM parameters for item i
    marginal_pmf : (K,) pi*(y_i) from the imputation model (marginal)

    Returns
    -------
    delta : (T,) array (signed, can be negative)
    """
    pmf = _grm_pmf_grid(theta_grid, a, thresholds)    # (T, K)
    dpmf = _grm_score_grid(theta_grid, a, thresholds)  # (T, K)

    marg = np.maximum(marginal_pmf, 1e-20)
    marg = marg / marg.sum()
    K = len(marg)

    # a_i(theta) = sum_k pi*(k) p(k|theta)
    a_i = np.sum(marg[np.newaxis, :K] * pmf[:, :K], axis=1)  # (T,)
    # numerator = sum_k pi*(k) dp(k|theta)/d_theta
    num = np.sum(marg[np.newaxis, :K] * dpmf[:, :K], axis=1)  # (T,)

    a_i = np.maximum(a_i, 1e-20)
    delta = num / a_i  # (T,)
    return delta


def _grad_delta_grid(theta_grid: np.ndarray,
                     delta: np.ndarray) -> np.ndarray:
    """Numerical gradient d delta_i / d theta = d^2 log a_i / d theta^2.

    Uses numpy.gradient (central differences interior, one-sided boundaries).

    Parameters
    ----------
    theta_grid : (T,) strictly increasing
    delta : (T,) delta_i values

    Returns
    -------
    grad : (T,) numerical gradient
    """
    return np.gradient(delta, theta_grid)


# ---------------------------------------------------------------------------
# Marginal PMF extraction from threeway JSON
# ---------------------------------------------------------------------------

def _pairwise_marginal_pmf(pairwise_pmfs_for_item: dict,
                           K: int) -> np.ndarray:
    """Compute marginal pi*(y_i) from pairwise conditional tables.

    Averages all available conditioning-item/response combinations with
    equal weight to get the unconditional marginal prediction.

    Parameters
    ----------
    pairwise_pmfs_for_item : dict[predictor -> dict[resp_val -> list[float]]]
        Pairwise PMF tables for a single target item.
    K : int
        Number of response categories.

    Returns
    -------
    marginal : (K,) array summing to 1
    """
    accum = np.zeros(K)
    count = 0
    for predictor, resp_table in pairwise_pmfs_for_item.items():
        for resp_val, pmf_list in resp_table.items():
            arr = np.asarray(pmf_list, dtype=float)
            if arr.shape[0] < K:
                arr = np.pad(arr, (0, K - arr.shape[0]))
            elif arr.shape[0] > K:
                arr = arr[:K]
            arr = np.maximum(arr, 1e-20)
            arr /= arr.sum()
            accum += arr
            count += 1
    if count == 0:
        return np.ones(K) / K
    marginal = accum / count
    marginal = np.maximum(marginal, 1e-20)
    marginal /= marginal.sum()
    return marginal


# ---------------------------------------------------------------------------
# NewtonCorrector class
# ---------------------------------------------------------------------------

class NewtonCorrector:
    """Closed-form per-item Newton bias corrector for CAT subset scoring.

    The corrector pre-computes per-item delta_i(theta), grad_delta_i(theta),
    and Fisher info I_i(theta) curves on a fine theta grid.  At correction
    time it interpolates at the person's naive MAP score and applies the
    aggregated Newton step:

        step = -I_S(theta_hat)^{-1} * sum_{i not in S} grad_delta_i(theta_hat)

    where I_S = sum_{i in S} I_i (Fisher info of administered subset) and
    grad_delta_i = d^2/d_theta^2 log a_i (second derivative of blended log-lik).

    Attributes
    ----------
    item_keys : list[str]
    theta_grid : (T,) ndarray
    delta_grid : (n_items, T) ndarray -- d/d_theta log a_i(theta)
    grad_delta_grid : (n_items, T) ndarray -- d^2/d_theta^2 log a_i(theta)
    fisher_grid : (n_items, T) ndarray -- I_i(theta) for each item
    """

    def __init__(
        self,
        item_keys: List[str],
        slope: np.ndarray,
        calibration: np.ndarray,
        marginal_pmfs: Dict[str, np.ndarray],
        n_categories: int,
        theta_lo: float = -6.0,
        theta_hi: float = 6.0,
        n_theta: int = 601,
    ):
        self.item_keys = list(item_keys)
        self.item_idx = {k: i for i, k in enumerate(item_keys)}
        self.n_items = len(item_keys)
        self.n_categories = n_categories

        self.theta_grid = np.linspace(theta_lo, theta_hi, n_theta)
        T = n_theta

        self.delta_grid = np.zeros((self.n_items, T))
        self.grad_delta_grid = np.zeros((self.n_items, T))
        self.fisher_grid = np.zeros((self.n_items, T))

        for i, key in enumerate(item_keys):
            a = float(slope[i])
            thresh = np.asarray(calibration[i], dtype=float)
            marg = marginal_pmfs.get(key, np.ones(n_categories) / n_categories)
            # delta_i(theta): score contribution from unobserved item i
            delta = _delta_i_grid(self.theta_grid, a, thresh, marg)
            self.delta_grid[i] = delta
            # grad_delta_i(theta): Newton curvature term
            self.grad_delta_grid[i] = _grad_delta_grid(self.theta_grid, delta)
            # Fisher info
            self.fisher_grid[i] = _fisher_info_grid(self.theta_grid, a, thresh)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def step(self,
             naive_scores: np.ndarray,
             subset_items: Sequence[str],
             regularize: bool = True,
             fisher_min_frac: float = 0.02,
             theta_ref: Optional[float] = 0.0) -> np.ndarray:
        """Compute the Newton correction step.

        Evaluates the bias approximation:
            Newton_step = -I_S(theta_ref)^{-1} * sum_{i not in S} grad_delta_i(theta_ref)

        where grad_delta_i = d^2/d_theta^2 log a_i is the Newton curvature term
        from the bias identity (eq. subset_bias_newton in the manuscript), and
        a_i(theta) = sum_k pi*(k) p(k|theta) is the blended item likelihood.

        By default (theta_ref=0.0) the step is evaluated at typical ability
        (theta=0), consistent with the manuscript's presentation of the formula.
        This avoids the 1/I_S blow-up at extreme thetas where the subset Fisher
        information is negligible.

        Parameters
        ----------
        naive_scores : (N,) naive MAP/MLE ability estimates
        subset_items : iterable of item keys that were administered
        regularize : bool
            If True (default), add a floor to the Fisher denominator.
        fisher_min_frac : float
            Regularization fraction relative to full-bank Fisher at theta=0.
        theta_ref : float or None
            Reference theta for the Newton step.  Default 0.0 (typical ability).
            Set to None to evaluate per-person at each person's naive score
            (may be numerically unstable in the tails for small subsets).

        Returns
        -------
        correction : (N,) Newton step.  When theta_ref is not None, this is a
            constant shift applied identically to all N persons.
        """
        naive = np.asarray(naive_scores, dtype=float)
        N = naive.shape[0]

        subset_set = set(subset_items)
        omitted_keys = [k for k in self.item_keys if k not in subset_set]
        admin_keys = [k for k in self.item_keys if k in subset_set]

        if not omitted_keys:
            return np.zeros(N)

        omit_idx = np.array([self.item_idx[k] for k in omitted_keys], dtype=int)
        admin_idx = np.array([self.item_idx[k] for k in admin_keys], dtype=int)

        # Reference point for evaluation
        theta0_idx = int(np.searchsorted(self.theta_grid, 0.0))
        fisher_full_0 = float(self.fisher_grid[:, theta0_idx].sum())
        reg_floor = fisher_min_frac * fisher_full_0 if regularize else 0.0

        if theta_ref is not None:
            # Evaluate at fixed reference theta -- constant shift across persons.
            # Newton step = -bias(S, theta_ref) where bias = -(sum_grad / I_S),
            # so step = sum_grad / I_S  [corrects in the direction of sum_grad].
            sum_grad = float(sum(
                np.interp(theta_ref, self.theta_grid, self.grad_delta_grid[i])
                for i in omit_idx))
            fisher_S_val = float(sum(
                np.interp(theta_ref, self.theta_grid, self.fisher_grid[i])
                for i in admin_idx))
            denom = max(fisher_S_val + reg_floor, 1e-10)
            step_scalar = sum_grad / denom
            return np.full(N, step_scalar)

        else:
            # Evaluate per-person at their naive score.
            # step = sum_grad / I_S (note: no leading minus, per manuscript)
            grad_delta_at = np.zeros((len(omitted_keys), N))
            for j, i in enumerate(omit_idx):
                grad_delta_at[j] = np.interp(
                    naive, self.theta_grid, self.grad_delta_grid[i])

            fisher_at = np.zeros((len(admin_keys), N))
            for j, i in enumerate(admin_idx):
                fisher_at[j] = np.interp(
                    naive, self.theta_grid, self.fisher_grid[i])
            fisher_S = fisher_at.sum(axis=0)

            sum_grad = grad_delta_at.sum(axis=0)
            denom = np.maximum(fisher_S + reg_floor, 1e-10)
            return sum_grad / denom

    def apply(self,
              naive_scores: np.ndarray,
              subset_items: Sequence[str],
              **kw) -> np.ndarray:
        """Return Newton-corrected scores.

        corrected[n] = naive_scores[n] + step[n]
        """
        return np.asarray(naive_scores, dtype=float) + self.step(
            naive_scores, subset_items, **kw)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_threeway_json(
        cls,
        json_path: Union[str, Path],
        theta_lo: float = -6.0,
        theta_hi: float = 6.0,
        n_theta: int = 601,
    ) -> 'NewtonCorrector':
        """Build a NewtonCorrector from a threeway_imputation.json file.

        The marginal pi*(y_i) is extracted from the pairwise PMF tables by
        averaging all conditioning-item/response combinations, consistent with
        the B_i computation in compute_item_bias_leverage_flat.py.
        """
        with open(json_path) as fh:
            d = json.load(fh)

        item_keys = list(d['item_keys'])
        n_categories = int(d['n_categories'])
        slope = np.asarray(d['irt_slope'], dtype=float)
        calibration = np.asarray(d['irt_calibration'], dtype=float)
        pairwise_pmfs = d.get('pairwise_pmfs', {})

        marginal_pmfs = {}
        for key in item_keys:
            if key in pairwise_pmfs and isinstance(pairwise_pmfs[key], dict):
                marginal_pmfs[key] = _pairwise_marginal_pmf(
                    pairwise_pmfs[key], n_categories)
            else:
                marginal_pmfs[key] = np.ones(n_categories) / n_categories

        return cls(
            item_keys=item_keys,
            slope=slope,
            calibration=calibration,
            marginal_pmfs=marginal_pmfs,
            n_categories=n_categories,
            theta_lo=theta_lo,
            theta_hi=theta_hi,
            n_theta=n_theta,
        )

    @classmethod
    def from_grm_npz(
        cls,
        npz_path: Union[str, Path],
        marginal_pmfs: Optional[Dict[str, np.ndarray]] = None,
        n_categories: Optional[int] = None,
        theta_lo: float = -6.0,
        theta_hi: float = 6.0,
        n_theta: int = 601,
    ) -> 'NewtonCorrector':
        """Build from a GRM params NPZ file (slope, calibration, item_keys).

        Parameters
        ----------
        npz_path : path to <dataset>_grm_params.npz
        marginal_pmfs : optional dict item_key -> (K,) array; if None,
            uniform marginals are used.
        n_categories : override; if None inferred from calibration shape + 1
        """
        data = np.load(npz_path)
        item_keys_raw = data['item_keys']
        item_keys = [k.decode() if isinstance(k, bytes) else str(k)
                     for k in item_keys_raw]
        slope = data['slope'].astype(float)
        calibration = data['calibration'].astype(float)
        K = n_categories if n_categories is not None else int(
            data.get('response_cardinality', calibration.shape[1] + 1))

        marg = marginal_pmfs if marginal_pmfs is not None else {}
        for k in item_keys:
            if k not in marg:
                marg[k] = np.ones(K) / K

        return cls(
            item_keys=item_keys,
            slope=slope,
            calibration=calibration,
            marginal_pmfs=marg,
            n_categories=K,
            theta_lo=theta_lo,
            theta_hi=theta_hi,
            n_theta=n_theta,
        )
