"""Bias-Correction Map (BCM): isotonic post-hoc correction for CAT scoring.

A BCM is a fitted monotone piecewise-linear map from a (biased) subset
score to a corrected score, produced by isotonic regression of the
gold-standard full-bank score on the subset score. The fit happens here
(in Python via sklearn); the resulting JSON can be consumed either by
this module or by the Go runtime in
``gofluttercat/backend-golang/pkg/biascorrection``.

JSON shape (single BCM)::

    {
      "scale":        "scs",
      "subset_size":  5,
      "x_thresholds": [...],
      "y_thresholds": [...]
    }

JSON shape (a Set of BCMs, one per subset size for the same scale)::

    {
      "scale": "scs",
      "maps": {
        "5":  {"subset_size": 5,  "x_thresholds": [...], "y_thresholds": [...]},
        "10": {"subset_size": 10, "x_thresholds": [...], "y_thresholds": [...]}
      }
    }

BCMConditional
--------------
A bivariate extension that conditions on the Fisher information of the
administered subset at theta=0 in addition to the subset score. The
corrector is backed by an sklearn HistGradientBoostingRegressor with a
monotonic constraint on the score feature (feature 0 must be
non-decreasing in the prediction) and an unconstrained smooth
dimension for the Fisher information feature (feature 1). A 5-fold
cross-validated held-out prediction is produced at fit time and stored
in ``oof_predictions``. Persistence is via joblib dump (HistGBR is not
trivially JSON-serialisable).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


@dataclass
class BCM:
    """A single isotonic bias-correction map.

    ``x_thresholds`` and ``y_thresholds`` must have the same length and at
    least two entries, with both monotone non-decreasing. The mapping is
    piecewise-linear interpolation between adjacent (x, y), with clipping
    to the end y-values outside the x-range — matching sklearn
    ``IsotonicRegression(out_of_bounds='clip')``.
    """

    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    scale: str = ""
    subset_size: int = 0

    def __post_init__(self) -> None:
        self.x_thresholds = np.asarray(self.x_thresholds, dtype=float)
        self.y_thresholds = np.asarray(self.y_thresholds, dtype=float)
        self.validate()

    def validate(self) -> None:
        if self.x_thresholds.shape != self.y_thresholds.shape:
            raise ValueError(
                f"BCM threshold shape mismatch: x={self.x_thresholds.shape} "
                f"y={self.y_thresholds.shape}"
            )
        if self.x_thresholds.size < 2:
            raise ValueError(
                f"BCM requires at least 2 threshold points, got {self.x_thresholds.size}"
            )
        if np.any(np.diff(self.x_thresholds) < 0):
            raise ValueError("BCM x_thresholds must be non-decreasing")
        if np.any(np.diff(self.y_thresholds) < 0):
            raise ValueError("BCM y_thresholds must be non-decreasing")

    def apply(self, raw_score):
        """Map a raw subset score to the corrected score.

        Accepts a scalar or an array; returns the same shape.
        """
        return np.interp(
            raw_score,
            self.x_thresholds,
            self.y_thresholds,
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )

    def to_dict(self) -> dict:
        d = {
            "x_thresholds": self.x_thresholds.tolist(),
            "y_thresholds": self.y_thresholds.tolist(),
        }
        if self.scale:
            d["scale"] = self.scale
        if self.subset_size:
            d["subset_size"] = int(self.subset_size)
        return d

    @classmethod
    def from_dict(cls, d: Mapping, default_scale: str = "") -> "BCM":
        return cls(
            x_thresholds=np.asarray(d["x_thresholds"], dtype=float),
            y_thresholds=np.asarray(d["y_thresholds"], dtype=float),
            scale=str(d.get("scale", default_scale)),
            subset_size=int(d.get("subset_size", 0)),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "BCM":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class BCMSet:
    """A collection of BCMs keyed by integer subset size for one scale."""

    scale: str = ""
    maps: Dict[int, BCM] = field(default_factory=dict)

    def for_subset(self, administered: int) -> Optional[BCM]:
        """Return the BCM for ``administered`` items, falling back to the
        closest available subset size. Returns ``None`` if empty."""
        if not self.maps:
            return None
        if administered in self.maps:
            return self.maps[administered]
        return self.maps[
            min(self.maps.keys(), key=lambda k: abs(k - administered))
        ]

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "maps": {
                str(k): {**v.to_dict(), "subset_size": int(k)}
                for k, v in sorted(self.maps.items())
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "BCMSet":
        scale = str(d.get("scale", ""))
        maps_raw = d.get("maps", {})
        maps: Dict[int, BCM] = {}
        for k, v in maps_raw.items():
            j = int(k)
            bcm = BCM.from_dict(v, default_scale=scale)
            if not bcm.subset_size:
                bcm.subset_size = j
            maps[j] = bcm
        return cls(scale=scale, maps=maps)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "BCMSet":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_bcm(
    subset_scores: np.ndarray,
    gold_scores: np.ndarray,
    *,
    scale: str = "",
    subset_size: int = 0,
) -> BCM:
    """Fit an isotonic BCM on ``(subset, gold)`` pairs and wrap as a
    ``BCM``. Drops any non-finite rows. Requires ``IsotonicRegression``
    from scikit-learn.
    """
    from sklearn.isotonic import IsotonicRegression  # lazy import

    s = np.asarray(subset_scores, dtype=float)
    g = np.asarray(gold_scores, dtype=float)
    if s.shape != g.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {g.shape}")
    valid = np.isfinite(s) & np.isfinite(g)
    if int(valid.sum()) < 2:
        raise ValueError("need at least 2 finite (subset, gold) pairs to fit a BCM")
    iso = IsotonicRegression(out_of_bounds="clip").fit(s[valid], g[valid])
    return BCM(
        x_thresholds=np.asarray(iso.X_thresholds_, dtype=float),
        y_thresholds=np.asarray(iso.y_thresholds_, dtype=float),
        scale=scale,
        subset_size=subset_size,
    )


def fit_bcm_set(
    cells: Mapping[int, Tuple[Iterable[float], Iterable[float]]],
    *,
    scale: str = "",
) -> BCMSet:
    """Fit one BCM per subset-size cell and bundle as a ``BCMSet``.

    ``cells`` maps J -> (subset_scores, gold_scores).
    """
    out = BCMSet(scale=scale)
    for j, (s, g) in cells.items():
        out.maps[int(j)] = fit_bcm(
            np.asarray(list(s), dtype=float),
            np.asarray(list(g), dtype=float),
            scale=scale,
            subset_size=int(j),
        )
    return out


class BCMConditional:
    """Bivariate BCM: monotone in subset score, smooth in subset Fisher info.

    Backed by ``sklearn.ensemble.HistGradientBoostingRegressor`` with
    ``monotonic_cst=[1, 0]`` (monotone non-decreasing in feature 0 = score,
    unconstrained in feature 1 = Fisher information of the administered
    subset at theta=0). 5-fold cross-validated held-out predictions are
    stored in ``oof_predictions`` after ``fit()``.

    Parameters
    ----------
    model : HistGradientBoostingRegressor
        Already-fitted sklearn model.
    scale_name : str
        Psychometric scale this corrector applies to.
    feature_names : tuple of str
        Names of the two input features, default ``("score", "info")``.
    oof_predictions : np.ndarray or None
        Out-of-fold predictions from 5-fold CV (same length as training data).
    """

    def __init__(
        self,
        model,
        scale_name: str,
        feature_names: Tuple[str, str] = ("score", "info"),
        oof_predictions: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.scale_name = scale_name
        self.feature_names = feature_names
        self.oof_predictions = oof_predictions

    def apply(self, score, info) -> np.ndarray:
        """Return the conditional-BCM-corrected score.

        Parameters
        ----------
        score : array-like, shape (n,)
            Raw (imputed) subset scores.
        info : array-like, shape (n,)
            Subset Fisher information at theta=0, one scalar per row.

        Returns
        -------
        np.ndarray, shape (n,)
            Bias-corrected scores.
        """
        score = np.asarray(score, dtype=float).ravel()
        info = np.asarray(info, dtype=float).ravel()
        if score.shape != info.shape:
            raise ValueError(
                f"score and info must have the same length; "
                f"got {score.shape} vs {info.shape}"
            )
        X = np.column_stack([score, info])
        return self.model.predict(X)

    @classmethod
    def fit(
        cls,
        score: np.ndarray,
        info: np.ndarray,
        gold: np.ndarray,
        scale_name: str = "",
        n_folds: int = 5,
        seed: int = 0,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 4,
    ) -> "BCMConditional":
        """Fit a BCMConditional via k-fold cross-validation.

        Trains one ``HistGradientBoostingRegressor`` on the full data and
        additionally collects out-of-fold (OOF) predictions via k-fold CV
        so that held-out bias can be computed without re-fitting.

        Parameters
        ----------
        score : array-like, shape (n,)
            Imputed subset scores (biased).
        info : array-like, shape (n,)
            Subset Fisher information at theta=0.
        gold : array-like, shape (n,)
            Gold-standard full-bank scores (target).
        scale_name : str
            Name of the psychometric scale.
        n_folds : int
            Number of cross-validation folds (default 5).
        seed : int
            Random state for fold shuffle.
        max_iter : int
            Maximum number of boosting iterations.
        learning_rate : float
            Learning rate for HistGBR.
        max_depth : int
            Maximum tree depth.

        Returns
        -------
        BCMConditional
            Instance with the full-data fitted model and OOF predictions.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        score = np.asarray(score, dtype=float).ravel()
        info = np.asarray(info, dtype=float).ravel()
        gold = np.asarray(gold, dtype=float).ravel()

        if not (score.shape == info.shape == gold.shape):
            raise ValueError(
                "score, info, and gold must all have the same length; "
                f"got {score.shape}, {info.shape}, {gold.shape}"
            )

        valid = np.isfinite(score) & np.isfinite(info) & np.isfinite(gold)
        if valid.sum() < 2 * n_folds:
            raise ValueError(
                f"BCMConditional.fit requires at least {2 * n_folds} finite "
                f"training rows; got {int(valid.sum())}"
            )

        X = np.column_stack([score, info])

        # monotonic_cst: +1 means non-decreasing for that feature.
        # Feature 0 (score) is constrained monotone; feature 1 (info) is free.
        full_model = HistGradientBoostingRegressor(
            monotonic_cst=[1, 0],
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
        )
        full_model.fit(X[valid], gold[valid])

        # Collect OOF predictions via k-fold CV.
        oof = np.full(len(score), np.nan, dtype=float)
        rng = np.random.default_rng(seed)
        idx = np.where(valid)[0]
        rng.shuffle(idx)
        folds = np.array_split(idx, n_folds)
        for fk in range(n_folds):
            test_idx = folds[fk]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fk])
            if len(train_idx) < 2 or len(test_idx) == 0:
                oof[test_idx] = score[test_idx]
                continue
            fold_model = HistGradientBoostingRegressor(
                monotonic_cst=[1, 0],
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=seed,
            )
            fold_model.fit(X[train_idx], gold[train_idx])
            oof[test_idx] = fold_model.predict(X[test_idx])

        # For any invalid rows fall back to raw score.
        nan_mask = ~np.isfinite(oof)
        oof[nan_mask] = score[nan_mask]

        return cls(
            model=full_model,
            scale_name=scale_name,
            oof_predictions=oof,
        )

    def save(self, path) -> None:
        """Persist the fitted model to ``path`` using joblib."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> "BCMConditional":
        """Load a persisted BCMConditional from ``path``."""
        import joblib
        return joblib.load(path)
