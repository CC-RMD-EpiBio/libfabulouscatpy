"""IRT Mixed Imputation Model (deployment version).

Blends predictions from a pairwise imputation model with a uniform (ignorable)
distribution using precomputed per-item weights. The weights are derived
offline during training via softmax over comparable ELPD scores (pairwise
stacking LOO-ELPD vs IRT per-item WAIC).

For each missing item, the resulting PMF is:

    q_mixed(k) = w_pairwise * q_pairwise(k) + (1 - w_pairwise) * (1/K)

This class satisfies the same ``predict_pmf`` interface that
``BayesianScoring`` expects from an ``imputation_model``.
"""

from typing import Dict, Optional

import numpy as np


class IrtMixedImputationModel:
    """Blends pairwise stacking and uniform imputation PMFs with precomputed
    per-item weights.

    Parameters
    ----------
    pairwise_model
        A fitted imputation model with a ``predict_pmf`` method.
    weights : dict
        Mapping from item name to pairwise model mixing weight (0 to 1).
        Higher values mean more trust in the pairwise model's predictions.
    default_weight : float
        Weight used for items not in the weights dict.
    """

    def __init__(
        self,
        pairwise_model,
        weights: Dict[str, float],
        default_weight: float = 0.5,
    ):
        self.pairwise_model = pairwise_model
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
            Passed through to the pairwise model's predict_pmf.

        Returns
        -------
        np.ndarray of shape (n_categories,) summing to 1.
        """
        w_pairwise = self.weights.get(target, self.default_weight)

        try:
            pairwise_pmf = self.pairwise_model.predict_pmf(
                items, target, n_categories,
                uncertainty_penalty=uncertainty_penalty,
            )
            pairwise_pmf = np.asarray(pairwise_pmf, dtype=float)
        except (ValueError, KeyError, AttributeError):
            pairwise_pmf = np.ones(n_categories) / n_categories

        uniform_pmf = np.ones(n_categories) / n_categories

        blended = w_pairwise * pairwise_pmf + (1.0 - w_pairwise) * uniform_pmf

        total = blended.sum()
        if total > 0:
            blended /= total
        else:
            blended = uniform_pmf

        return blended
