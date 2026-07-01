"""Post-hoc bias-correction maps for CAT scoring.

See ``BCM`` and ``BCMSet`` in :mod:`bcm` for details. The on-disk JSON
format is shared with the gofluttercat Go runtime (package
``backend-golang/pkg/biascorrection``); both implementations agree on
piecewise-linear interpolation between sklearn's ``X_thresholds_`` /
``y_thresholds_`` knots, with clipping outside the training range.
"""

from .bcm import (
    BCM,
    BCMConditional,
    BCMConditionalInfo,
    BCMSet,
    fit_bcm,
    fit_bcm_set,
)
from .newton import NewtonCorrector
from .triples import (
    attach_imputation_pmfs,
    build_bcm_triples,
    extract_item_params,
    extract_item_params_from_mcmc,
    score_subset,
    stratify_respondents,
    subsample_data,
)

__all__ = [
    "BCM",
    "BCMConditional",
    "BCMConditionalInfo",
    "BCMSet",
    "fit_bcm",
    "fit_bcm_set",
    "NewtonCorrector",
    "attach_imputation_pmfs",
    "build_bcm_triples",
    "extract_item_params",
    "extract_item_params_from_mcmc",
    "score_subset",
    "stratify_respondents",
    "subsample_data",
]
