
from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate

from libfabulouscatpy import constants as const
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
        z = np.trapz(y=density, x=interpolation_pts)
        score = np.trapz(y=density * interpolation_pts, x=interpolation_pts)/z
        cdf = integrate.cumtrapz(density, interpolation_pts, initial=0)/z
        variance = np.trapz(y=density * interpolation_pts**2, x=interpolation_pts)/z - score**2
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
    ) -> None:
        super().__init__(model)
        self.log_like = {}
        self.log_prior_fn =log_prior_fn
        self.interpolation_pts = {}
        self.log_prior = {}
        self.skipped_response = skipped_response if skipped_response is not None else const.SKIPPED_RESPONSE
        self.n_scored = defaultdict(int)

        for scale in self.model.models.keys():
            self.log_like[scale] = model.interpolation_pts * 0
            if log_prior_fn is not None:
                self.log_prior[scale] = self.log_prior_fn[scale](model.interpolation_pts)
            else:
                self.log_prior[scale] = 0 
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
            densities[scale] = np.exp(log_prob - np.max(log_prob))
            densities[scale] /= np.trapz(y=densities[scale], x=self.interpolation_pts[scale])

        scores = {
            k: BayesianScore(k, k, v, self.interpolation_pts[k])
            for k, v in densities.items()
        }
        self.scores = scores
        
        return scores
