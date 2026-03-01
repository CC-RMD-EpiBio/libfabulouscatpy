
from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from libfabulouscatpy import constants as const
from libfabulouscatpy._compat import trapz as _trapz, cumtrapz as _cumtrapz
from libfabulouscatpy.irt.prediction.irt import IRTModel
from libfabulouscatpy.irt.scoring.scoring import (ScoreBase, ScoringBase,
                                                  sample_from_cdf)


class BayesianScore(ScoreBase):
    def __init__(
        self,
        scale: str,
        description: str,
        density: npt.ArrayLike,
        interpolation_pts: npt.ArrayLike,
        offset_factor:float=0,
        scaling_factor:float=1,
        symmetric_errors:bool=True
    ):
        self.interpolation_pts = interpolation_pts
        z = _trapz(y=density, x=interpolation_pts)
        score = _trapz(y=density * interpolation_pts, x=interpolation_pts)/z
        cdf = _cumtrapz(density, interpolation_pts, initial=0)/z
        variance = _trapz(y=density * interpolation_pts**2, x=interpolation_pts)/z - score**2
        median = np.interp(0.5, cdf, interpolation_pts)
        
        error = np.sqrt(variance)
        if not symmetric_errors:
            "Give width of the 1sigma interval"
            
            lower = np.interp(0.158655, cdf, interpolation_pts)
            upper = np.interp(0.8413447, cdf, interpolation_pts)
            if median > 0:
                error = median - lower
            else:
                error = upper - median
        super().__init__(
            scale, description, score, error, offset_factor, scaling_factor
        )
        self.density = density
        self.cdf = cdf
        self.median = median
        
    def sample(self, shape: Union[int, Tuple[int, ...]] = 1) -> Optional[npt.ArrayLike]:
        return sample_from_cdf(self.interpolation_pts, self.cdf, shape)
        
def gaussian_dens(sigma):
    def _gaussian_dens(x):
        return  -0.5 * (x / sigma) ** 2

    return _gaussian_dens

class BayesianScoring(ScoringBase):

    def __init__(
        self,
        model: IRTModel | None = None,
        log_prior_fn: dict[str, Callable] | None = None,
        skipped_response: int | None = None,
        imputation_model=None,
    ) -> None:
        super().__init__(model)
        self.log_like = {}
        self.log_prior_fn =log_prior_fn
        self.interpolation_pts = {}
        self.log_prior = {}
        self.skipped_response = skipped_response if skipped_response is not None else const.SKIPPED_RESPONSE
        self.n_scored = defaultdict(int)
        self.imputation_model = imputation_model

        for scale in self.model.models.keys():
            self.log_like[scale] = model.interpolation_pts * 0
            if log_prior_fn is not None:
                self.log_prior[scale] = self.log_prior_fn[scale](model.interpolation_pts)
            else:
                self.log_prior[scale] = -model.interpolation_pts**2/(1*2)
            self.interpolation_pts[scale] = model.interpolation_pts
        self.score_responses({})
        

    def remove_responses(self, responses: dict) -> None:
        to_compute = {
            k: [x for x in v if x in responses.keys()]
            for k, v in self.model.item_labels.items()
        }
        for scale, i in to_compute.items():
            if len(i) == 0:
                continue
            log_l = self.model.models[scale].log_likelihood(
                theta=self.interpolation_pts[scale],
                responses={ii: responses[ii] for ii in i},
            )
            self.log_like[scale] -= log_l
            for ii in i:
                self.scored_responses[ii] = responses[ii]


    def add_responses(self, responses: dict) -> None:
        to_compute = {
            k: [x for x in v if x in responses.keys()]
            for k, v in self.model.item_labels.items()
        }
        for scale, i in to_compute.items():
            if len(i) == 0:
                continue
            log_l = self.model.models[scale].log_likelihood(
                theta=self.interpolation_pts[scale],
                responses={ii: responses[ii] for ii in i},
            )
            self.log_like[scale] += log_l
            for ii in i:
                self.scored_responses[ii] = responses[ii]
            self.n_scored[scale] += len(i)

    def _compute_imputed_log_likelihood(self, responses, scale):
        """Compute marginal log-likelihood from unobserved items via imputation.

        For each unobserved item on the scale, uses the imputation model to
        predict a response PMF conditioned on the observed responses, then
        marginalizes over possible responses:
            log P_imp(X_j | theta) = log sum_k P_imp(X_j=k) * P(X_j=k | theta)

        Args:
            responses: Full dict of current observed responses {item_id: value}.
            scale: Scale name to compute imputed contributions for.

        Returns:
            Array of shape (n_theta,) with the marginal log-likelihood
            contribution, or 0.0 if there is nothing to impute.
        """
        all_items = self.model.item_labels[scale]
        observed_items = {
            k for k, v in responses.items() if v != self.skipped_response
        }
        unobserved_items = [
            item for item in all_items if item not in observed_items
        ]

        if not unobserved_items:
            return 0.0

        grm = self.model.models[scale]
        # log P(response=k | theta) for all items: shape (n_theta, n_items, n_categories)
        log_p = grm.log_likelihood(
            theta=self.interpolation_pts[scale],
            observed_only=False,
        )

        observed_dict = {
            k: float(v) for k, v in responses.items()
            if v != self.skipped_response
        }
        n_categories = log_p.shape[-1]
        marginal_ll = np.zeros_like(self.interpolation_pts[scale])

        for item in unobserved_items:
            item_idx = grm.item_labels.index(item)
            try:
                pmf = self.imputation_model.predict_pmf(
                    items=observed_dict,
                    target=item,
                    n_categories=n_categories,
                )
            except (KeyError, ValueError):
                continue
            pmf = np.asarray(pmf, dtype=float)
            log_pmf = np.log(pmf + 1e-20)
            # log(sum_k P_imp(k) * P(k|theta)) via log-sum-exp for stability
            weighted = log_p[:, item_idx, :] + log_pmf[np.newaxis, :]
            max_w = np.max(weighted, axis=-1, keepdims=True)
            marginal_ll += (
                np.log(np.sum(np.exp(weighted - max_w), axis=-1))
                + max_w.squeeze(axis=-1)
            )

        return marginal_ll

    def score_responses(
        self, responses: dict, scales: list[str] | None = None, **kwargs
    ) -> dict[str:BayesianScore]:
        """Bayesian update for self.log_like and compute resulting density

        Returns:
            _type_: _description_
        """
        # removed skipped first


        to_add = {
            k: v
            for k, v in responses.items()
            if (v != self.scored_responses.get(k, None) and v!=self.skipped_response)
        }
        to_delete = {
            k: v
            for k, v in self.scored_responses.items()
            if (v != responses.get(k, None) and v!=self.skipped_response)
        }
        self.add_responses(to_add)
        self.remove_responses(to_delete)
        densities = {}
        if scales is None or len(scales) == 0:
            scales = list(self.log_like.keys())
        for scale in scales:
            log_prob = self.log_prior[scale] + self.log_like[scale]
            if self.imputation_model is not None:
                log_prob = log_prob + self._compute_imputed_log_likelihood(
                    responses, scale
                )
            densities[scale] = np.exp(log_prob - np.max(log_prob))
            densities[scale] /= _trapz(y=densities[scale], x=self.interpolation_pts[scale])

        scores = {
            k: BayesianScore(k, k, v, self.interpolation_pts[k])
            for k, v in densities.items()
        }
        self.scores = scores

        return scores
