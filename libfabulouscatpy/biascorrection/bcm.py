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
