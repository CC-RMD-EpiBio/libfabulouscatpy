"""Loader for the bayesianquilts/gofluttercat converged GRM+imputation artifact.

The artifact layout is the one produced by
``bayesianquilts.io.converged.export_artifact`` / ``export_multi_scale_artifact``:

    <root>/
        items/<item_key>.json   -- one per item, with ``scales`` payload
        scales.json             -- per-scale metadata
        imputation/             -- (optional) gofluttercat imputation bundle
        manifest.yaml           -- provenance

This module loads the IRT half into :class:`libfabulouscatpy.irt.prediction.grm.GradedResponseModel`
instances (one per scale) and exposes the manifest plus the path to the
imputation directory. libfab does not consume the imputation bundle itself
(it is for the Go runtime), but the path is returned so callers can hand it
to the runtime they need.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from libfabulouscatpy.irt.item import ItemDatabase, ScaleDatabase
from libfabulouscatpy.irt.prediction.grm import GradedResponseModel


@dataclass
class ConvergedArtifact:
    """In-memory view of a converged artifact bundle."""

    root: Path
    items: List[Dict[str, Any]]
    scales: Dict[str, Dict[str, Any]]
    models: Dict[str, GradedResponseModel]
    """One GradedResponseModel per scale, indexed by scale name."""
    manifest: Dict[str, Any] = field(default_factory=dict)
    imputation_path: Optional[Path] = None
    """Path to ``<root>/imputation`` if it exists, else ``None``."""

    @property
    def scale_names(self) -> List[str]:
        return list(self.scales.keys())

    def model(self, scale: str) -> GradedResponseModel:
        return self.models[scale]


def _read_manifest(root: Path) -> Dict[str, Any]:
    path = root / "manifest.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_artifact(root) -> ConvergedArtifact:
    """Load the bundle at ``root`` into a :class:`ConvergedArtifact`."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"converged artifact root not found: {root}")

    items_dir = root / "items"
    if not items_dir.is_dir():
        raise FileNotFoundError(f"missing items/ under {root}")

    itemdb = ItemDatabase(items_dir)

    scales_path = root / "scales.json"
    if scales_path.exists():
        scales = json.loads(scales_path.read_text())
    else:
        scales = {}
        for item in itemdb.items:
            for sc in item.get("scales", {}):
                scales.setdefault(sc, {"description": sc})

    # Build one GradedResponseModel per scale.
    models: Dict[str, GradedResponseModel] = {}
    for scale_name in scales.keys():
        slopes: List[float] = []
        diffs: List[List[float]] = []
        labels: List[str] = []
        for item in itemdb.items:
            payload = item.get("scales", {}).get(scale_name)
            if payload is None:
                continue
            slopes.append(float(payload["discrimination"]))
            diffs.append([float(x) for x in payload["difficulties"]])
            labels.append(str(item["item"]))
        if not slopes:
            continue
        models[scale_name] = GradedResponseModel(
            slope=slopes, calibration=diffs, item_labels=labels,
        )

    imp_dir = root / "imputation"
    imp_path = imp_dir if imp_dir.is_dir() else None

    return ConvergedArtifact(
        root=root,
        items=list(itemdb.items),
        scales=dict(scales),
        models=models,
        manifest=_read_manifest(root),
        imputation_path=imp_path,
    )
