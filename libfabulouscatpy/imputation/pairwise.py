"""Pairwise imputation model for CAT scoring.

Implements the stacked univariate imputation ensemble π* described in
the paper.  For each unobserved item i, the imputation distribution is
a weighted mixture of pairwise ordinal regressions:

    π*(z_i = k | x) = Σ_{l ∈ O} w_{i|l} · P(z_i = k | x_l)

where the weights w_{i|l} are proportional to the predictive
performance of each pairwise model, measured by leave-one-out ELPD.

This module provides two implementations:
- ``PairwiseImputationModel``: loads pre-computed pairwise PMF tables
- ``MixedImputationModel``: blends pairwise predictions with the IRT
  model's own predictions via per-item stacking weights
"""

import json

import numpy as np


class PairwiseImputationModel:
    """Stacked univariate (MICE) imputation model.

    Each target item has a set of pairwise predictors, one per observed
    predictor item.  At prediction time, the model averages the pairwise
    PMFs using pre-computed stacking weights.

    Parameters
    ----------
    pairwise_pmfs : dict
        Mapping ``target_item -> predictor_item -> response_value -> ndarray``
        where each ndarray is a PMF of shape (n_categories,).
    stacking_weights : dict
        Mapping ``target_item -> predictor_item -> float``.
    n_categories : int
        Number of response categories (K).
    """

    def __init__(self, pairwise_pmfs, stacking_weights, n_categories):
        self.pairwise_pmfs = pairwise_pmfs
        self.stacking_weights = stacking_weights
        self.n_categories = n_categories

    def predict_pmf(self, items, target, n_categories, **kwargs):
        """Predict response PMF for target given observed items.

        Args:
            items: Dict mapping observed item_id -> response value.
            target: Item ID to predict.
            n_categories: Number of response categories.

        Returns:
            ndarray of shape (n_categories,) with the imputed PMF.
        """
        if target not in self.pairwise_pmfs:
            return np.ones(n_categories) / n_categories

        target_predictors = self.pairwise_pmfs[target]
        target_weights = self.stacking_weights.get(target, {})

        pmf = np.zeros(n_categories)
        total_weight = 0.0

        for predictor_item, response_val in items.items():
            if predictor_item not in target_predictors:
                continue
            w = target_weights.get(predictor_item, 1.0)
            # Look up the PMF for this predictor's observed response
            resp_key = int(response_val)
            predictor_pmfs = target_predictors[predictor_item]
            if resp_key in predictor_pmfs:
                pmf += w * np.asarray(predictor_pmfs[resp_key], dtype=float)
                total_weight += w
            elif str(resp_key) in predictor_pmfs:
                pmf += w * np.asarray(predictor_pmfs[str(resp_key)], dtype=float)
                total_weight += w

        if total_weight > 0:
            pmf /= total_weight
        else:
            pmf = np.ones(n_categories) / n_categories

        # Ensure valid PMF
        pmf = np.maximum(pmf, 1e-20)
        pmf /= pmf.sum()
        return pmf

    @classmethod
    def from_json(cls, path):
        """Load from a JSON file.

        Expected format::

            {
                "n_categories": 4,
                "pairwise_pmfs": {
                    "item_0": {
                        "item_1": {"0": [0.1, 0.2, 0.3, 0.4], ...},
                        ...
                    },
                    ...
                },
                "stacking_weights": {
                    "item_0": {"item_1": 0.3, "item_2": 0.7, ...},
                    ...
                }
            }
        """
        with open(path) as f:
            data = json.load(f)
        return cls(
            pairwise_pmfs=data['pairwise_pmfs'],
            stacking_weights=data['stacking_weights'],
            n_categories=data['n_categories'],
        )


class MixedImputationModel:
    """Blends pairwise imputation with IRT model predictions.

    For each unobserved item i:

        π*(z_i = k | x) = w_i · π_mice(z_i = k | x)
                         + (1 - w_i) · π_irt(z_i = k | θ_hat)

    where w_i is a per-item stacking weight determined by
    cross-validated predictive performance.

    Parameters
    ----------
    mice_model : PairwiseImputationModel
        The pairwise (MICE) imputation model.
    irt_model : object
        An IRT model with a ``log_likelihood(theta, observed_only=False)``
        method returning shape (n_theta, n_items, K).
    mixing_weights : dict
        Mapping ``item_id -> float`` giving the weight on the MICE model.
        Items not in the dict default to 0.5.
    theta_estimate : float
        Current point estimate of ability for the IRT prediction.
        Updated externally as CAT progresses.
    """

    def __init__(self, mice_model, irt_model, mixing_weights,
                 theta_estimate=0.0, scale_name=None):
        self.mice_model = mice_model
        self.irt_model = irt_model
        self.mixing_weights = mixing_weights
        self.theta_estimate = theta_estimate
        self.scale_name = scale_name

    def predict_pmf(self, items, target, n_categories, **kwargs):
        """Predict response PMF blending MICE and IRT predictions."""
        w_mice = self.mixing_weights.get(target, 0.5)

        # MICE prediction
        pmf_mice = self.mice_model.predict_pmf(items, target, n_categories)

        # IRT prediction at current theta estimate
        grm = self.irt_model.models[self.scale_name]
        theta = np.atleast_1d(self.theta_estimate)
        log_p = grm.log_likelihood(theta=theta, observed_only=False)
        # log_p shape: (1, n_items, K)
        item_idx = grm.item_labels.index(target)
        pmf_irt = np.exp(log_p[0, item_idx, :n_categories])
        pmf_irt = np.maximum(pmf_irt, 1e-20)
        pmf_irt /= pmf_irt.sum()

        # Blend
        pmf = w_mice * pmf_mice + (1 - w_mice) * pmf_irt
        pmf = np.maximum(pmf, 1e-20)
        pmf /= pmf.sum()
        return pmf
