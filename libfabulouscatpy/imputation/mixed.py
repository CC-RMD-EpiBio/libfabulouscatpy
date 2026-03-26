"""IRT Mixed Imputation Model (deployment version).

Blends predictions from a MICE imputation model with a uniform (ignorable)
distribution using precomputed per-item weights. The weights are derived
offline during training via softmax over comparable ELPD scores (MICE LOO-ELPD
vs IRT per-item WAIC).

For each missing item, the resulting PMF is:

    q_mixed(k) = w_mice * q_mice(k) + (1 - w_mice) * (1/K)

This class satisfies the same ``predict_pmf`` interface that
``BayesianScoring`` expects from an ``imputation_model``.
"""

from typing import Dict, Optional

import numpy as np


class IrtMixedImputationModel:
    """Blends MICE and uniform imputation PMFs with precomputed per-item weights.

    Parameters
    ----------
    mice_model
        A fitted imputation model with a ``predict_pmf`` method.
    weights : dict
        Mapping from item name to MICE mixing weight (0 to 1).
        Higher values mean more trust in the MICE model's predictions.
    default_weight : float
        Weight used for items not in the weights dict.
    """

    def __init__(
        self,
        mice_model,
        weights: Dict[str, float],
        default_weight: float = 0.5,
    ):
        self.mice_model = mice_model
        self.weights = dict(weights)
        self.default_weight = default_weight

    def predict_pmf(
        self,
        items: Dict[str, float],
        target: str,
        n_categories: int,
        uncertainty_penalty: float = 1.0,
    ) -> np.ndarray:
        """Return a blended PMF for a missing cell.

        Parameters
        ----------
        items : dict
            Observed variable name -> value for this person.
        target : str
            Item to impute.
        n_categories : int
            Number of response categories.
        uncertainty_penalty : float
            Passed through to the MICE model's predict_pmf.

        Returns
        -------
        np.ndarray of shape (n_categories,) summing to 1.
        """
        w_mice = self.weights.get(target, self.default_weight)

        try:
            mice_pmf = self.mice_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
            mice_pmf = np.asarray(mice_pmf, dtype=float)
        except (ValueError, KeyError, AttributeError):
            mice_pmf = np.ones(n_categories) / n_categories

        uniform_pmf = np.ones(n_categories) / n_categories

        blended = w_mice * mice_pmf + (1.0 - w_mice) * uniform_pmf

        total = blended.sum()
        if total > 0:
            blended /= total
        else:
            blended = uniform_pmf

        return blended
