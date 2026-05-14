"""Three-way Yao stacking imputation for libfabulouscatpy CAT scoring.

Blends three component predictions for each unobserved item:
  1. MICE (pairwise stacking PMF tables)
  2. IRT baseline (per-item discriminations GRM at current theta estimate)
  3. Shared-disc GRM at current theta estimate (optional; falls back to 2-way)

Per-item weights (w_mice, w_irt, w_shared) are the Yao stacking weights
fitted in bayesianquilts and serialised by
notebooks/irt/serialize_threeway_for_libfab.py.

The IRT and shared-disc PMFs are evaluated at a point theta_estimate
(updated externally by BayesianScoring as scoring proceeds), exactly like
MixedImputationModel -- this is simpler than full grid integration and
matches the existing pipeline.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np


def _grm_pmf(slope: np.ndarray, calibration: np.ndarray,
             theta: float, n_categories: int) -> np.ndarray:
    """Compute per-item PMF vectors for a GRM at a scalar theta.

    Parameters
    ----------
    slope : (I,) array of item discriminations (a-parameters).
    calibration : (I, K-1) array of cumulative difficulty thresholds.
    theta : scalar ability estimate.
    n_categories : K.

    Returns
    -------
    pmf : (I, K) array; rows sum to 1.
    """
    # P(X >= k | theta) = sigmoid(a * (theta - b_k))
    # P(X = k | theta) = P(X >= k) - P(X >= k+1), with P(X >= 0) = 1, P(X >= K) = 0
    I, Km1 = calibration.shape
    assert Km1 == n_categories - 1, (
        f"calibration has {Km1} columns but n_categories={n_categories}")
    # (I, K-1) cumulative probs
    eta = slope[:, np.newaxis] * (theta - calibration)   # (I, K-1)
    cum_p = 1.0 / (1.0 + np.exp(-eta))                   # (I, K-1)
    # boundary columns
    ones = np.ones((I, 1), dtype=np.float64)
    zeros = np.zeros((I, 1), dtype=np.float64)
    cum_p_full = np.concatenate([ones, cum_p, zeros], axis=1)  # (I, K+1)
    pmf = cum_p_full[:, :-1] - cum_p_full[:, 1:]               # (I, K)
    pmf = np.maximum(pmf, 1e-20)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


class ThreeWayImputationModel:
    """Scoring-time three-way Yao ensemble for libfabulouscatpy.

    Parameters
    ----------
    pairwise_pmfs : dict[target -> predictor -> resp_val -> list[float]]
        Pre-computed pairwise PMF tables (same structure as
        PairwiseImputationModel).
    pairwise_stacking_weights : dict[target -> predictor -> float]
        Per-predictor stacking weights for the MICE component.
    yao_weights : dict[item_key -> list[float, float, float]]
        Per-item (w_mice, w_irt, w_shared) Yao weights.
    irt_slope : (I,) array or list
        Posterior mean discriminations from the baseline GRM.
    irt_calibration : (I, K-1) array or list
        Posterior mean cumulative difficulties (diff0 + cumsum(ddiff)).
    item_keys : list[str]
        Ordered item keys matching irt_slope / irt_calibration rows.
    scale_name : str
    shared_slope : (I,) array or list or None
        Shared-disc GRM discriminations (broadcast across items). None
        triggers a 2-component (MICE + IRT) fallback.
    shared_calibration : (I, K-1) array or list or None
    n_categories : int
    theta_estimate : float
        Current point estimate of ability. Updated externally by
        BayesianScoring before imputed log-likelihood is computed.
    """

    def __init__(
        self,
        pairwise_pmfs: dict,
        pairwise_stacking_weights: dict,
        yao_weights: dict,
        irt_slope: np.ndarray,
        irt_calibration: np.ndarray,
        item_keys: list,
        scale_name: str,
        shared_slope=None,
        shared_calibration=None,
        n_categories: int = 5,
        theta_estimate: float = 0.0,
    ):
        self.pairwise_pmfs = pairwise_pmfs
        self.pairwise_stacking_weights = pairwise_stacking_weights
        self.yao_weights = {k: np.asarray(v, dtype=np.float64)
                            for k, v in yao_weights.items()}
        self.irt_slope = np.asarray(irt_slope, dtype=np.float64)
        self.irt_calibration = np.asarray(irt_calibration, dtype=np.float64)
        self.item_keys = list(item_keys)
        self.item_idx = {k: i for i, k in enumerate(item_keys)}
        self.scale_name = scale_name
        self.shared_slope = (np.asarray(shared_slope, dtype=np.float64)
                             if shared_slope is not None else None)
        self.shared_calibration = (np.asarray(shared_calibration, dtype=np.float64)
                                   if shared_calibration is not None else None)
        self.n_categories = n_categories
        self.theta_estimate = float(theta_estimate)

    # ------------------------------------------------------------------
    # MICE lookup (identical to PairwiseImputationModel.predict_pmf)
    # ------------------------------------------------------------------

    def _mice_pmf(self, items: dict, target: str) -> np.ndarray:
        K = self.n_categories
        if target not in self.pairwise_pmfs:
            return np.ones(K) / K

        target_predictors = self.pairwise_pmfs[target]
        target_weights = self.pairwise_stacking_weights.get(target, {})

        pmf = np.zeros(K)
        total_weight = 0.0

        for predictor_item, response_val in items.items():
            if predictor_item not in target_predictors:
                continue
            w = target_weights.get(predictor_item, 1.0)
            resp_key = int(response_val)
            predictor_pmfs = target_predictors[predictor_item]
            table = (predictor_pmfs.get(resp_key)
                     or predictor_pmfs.get(str(resp_key)))
            if table is not None:
                pmf += w * np.asarray(table, dtype=float)
                total_weight += w

        if total_weight > 0:
            pmf /= total_weight
        else:
            pmf = np.ones(K) / K
        pmf = np.maximum(pmf, 1e-20)
        pmf /= pmf.sum()
        return pmf

    # ------------------------------------------------------------------
    # IRT PMF at current theta
    # ------------------------------------------------------------------

    def _irt_pmf_for_item(self, item: str) -> np.ndarray:
        idx = self.item_idx.get(item)
        if idx is None:
            return np.ones(self.n_categories) / self.n_categories
        pmf_matrix = _grm_pmf(
            self.irt_slope, self.irt_calibration,
            self.theta_estimate, self.n_categories)
        return pmf_matrix[idx]

    def _shared_pmf_for_item(self, item: str) -> np.ndarray:
        if self.shared_slope is None or self.shared_calibration is None:
            return np.ones(self.n_categories) / self.n_categories
        idx = self.item_idx.get(item)
        if idx is None:
            return np.ones(self.n_categories) / self.n_categories
        pmf_matrix = _grm_pmf(
            self.shared_slope, self.shared_calibration,
            self.theta_estimate, self.n_categories)
        return pmf_matrix[idx]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_pmf(
        self,
        items: dict,
        target: str,
        n_categories: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict response PMF for target given observed items.

        Parameters
        ----------
        items : dict[item_key -> response_value]
            Observed responses so far.
        target : str
            Item key to predict.
        n_categories : int or None
            Response cardinality. Defaults to self.n_categories.

        Returns
        -------
        pmf : ndarray of shape (n_categories,) summing to ~1.
        """
        K = n_categories if n_categories is not None else self.n_categories

        default_w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        w = self.yao_weights.get(target, default_w)
        w_m, w_i, w_s = float(w[0]), float(w[1]), float(w[2])

        mice = self._mice_pmf(items, target)[:K]
        if mice.size < K:
            mice = np.pad(mice, (0, K - mice.size))
        mice = np.maximum(mice, 1e-20)
        mice /= mice.sum()

        irt = self._irt_pmf_for_item(target)[:K]
        if irt.size < K:
            irt = np.pad(irt, (0, K - irt.size))
        irt = np.maximum(irt, 1e-20)
        irt /= irt.sum()

        shared = self._shared_pmf_for_item(target)[:K]
        if shared.size < K:
            shared = np.pad(shared, (0, K - shared.size))
        shared = np.maximum(shared, 1e-20)
        shared /= shared.sum()

        blended = w_m * mice + w_i * irt + w_s * shared
        blended = np.maximum(blended, 1e-20)
        blended /= blended.sum()
        return blended

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "threeway_imputation_v1",
            "n_categories": self.n_categories,
            "scale_name": self.scale_name,
            "item_keys": self.item_keys,
            "pairwise_pmfs": self.pairwise_pmfs,
            "pairwise_stacking_weights": self.pairwise_stacking_weights,
            "yao_weights": {k: v.tolist() for k, v in self.yao_weights.items()},
            "irt_slope": self.irt_slope.tolist(),
            "irt_calibration": self.irt_calibration.tolist(),
            "shared_slope": (self.shared_slope.tolist()
                             if self.shared_slope is not None else None),
            "shared_calibration": (self.shared_calibration.tolist()
                                   if self.shared_calibration is not None else None),
        }

    @classmethod
    def from_json(cls, path: str) -> "ThreeWayImputationModel":
        """Load from a JSON file written by serialize_threeway_for_libfab.py."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            pairwise_pmfs=data["pairwise_pmfs"],
            pairwise_stacking_weights=data["pairwise_stacking_weights"],
            yao_weights=data["yao_weights"],
            irt_slope=data["irt_slope"],
            irt_calibration=data["irt_calibration"],
            item_keys=data["item_keys"],
            scale_name=data["scale_name"],
            shared_slope=data.get("shared_slope"),
            shared_calibration=data.get("shared_calibration"),
            n_categories=data["n_categories"],
            theta_estimate=0.0,
        )
