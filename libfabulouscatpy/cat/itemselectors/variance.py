from typing import Any

import numpy as np

from libfabulouscatpy.cat.itemselection import ItemSelector
from libfabulouscatpy.cat.session import CatSessionTracker
from libfabulouscatpy.irt.scoring import BayesianScoring


class VarianceItemSelector(ItemSelector):

    description = """Bayesian variance selector"""

    def __init__(self, scoring, **kwargs):
        super(VarianceItemSelector, self).__init__(**kwargs)
        self.scoring = scoring

    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:

        unresponded = [i for i in items if "scales" in i.keys()]
        in_scale = [i for i in unresponded if scale in i["scales"].keys()]

        if len(in_scale) == 0:
            return None

        unresponded_ndx = [
            self.model.item_labels[scale].index(j["item"]) for j in unresponded
        ]

        #####
        # current
        ######
        log_ell = self.model.models[scale].log_likelihood(
            theta=self.model.interpolation_pts, observed_only=False
        )[
            :, unresponded_ndx, :
        ]  # ll for unobserved items

        # previously observed
        energy = self.scoring.log_like[scale] + self.scoring.log_prior[scale]
        pi_now = scoring.scores[scale].density

        ###

        #######
        # Future

        lp_infty = log_ell + energy[:, np.newaxis, np.newaxis]
        p_now = np.exp(log_ell)
        p_now = np.sum(pi_now[:, np.newaxis, np.newaxis]*p_now, axis=0)
        pi_infty = np.exp(lp_infty - np.max(lp_infty, axis=0, keepdims=True))
        pi_infty /= np.trapz(
            y=pi_infty, x=self.scoring.interpolation_pts[scale], axis=0
        )
        ##########
        mean = np.trapz(
            y=pi_infty
            * self.scoring.interpolation_pts[scale][:, np.newaxis, np.newaxis],
            x=self.scoring.interpolation_pts[scale],
            axis=0,
        )
        second = np.trapz(
            y=pi_infty
            * self.scoring.interpolation_pts[scale][:, np.newaxis, np.newaxis] ** 2,
            x=self.scoring.interpolation_pts[scale],
            axis=0,
        )
        mean = np.sum(mean*p_now, axis=-1)
        second = np.sum(second*p_now, axis=-1)
        variance = second - mean**2
        
        criterion = dict(zip([x['item'] for x in items], variance))
        return criterion

    def _next_scored_item(
        
        self, tracker: CatSessionTracker, scale=None
        ) -> dict[str : dict[str:Any]]:
        
        scale = self.next_scale(tracker)
        un_items = self.un_items(tracker, scale)

        if un_items is None:
            # Not sure if this can happen under normal testing, but included as
            # a safety feature.
            return None

        trait = tracker.scores[scale]
        trait = 0.0 if trait is None else trait
        error = tracker.errors[scale]
        error = 100.0 if error is None else error
        
        criterion = self.criterion(scoring=self.scoring, items = un_items, scale=scale)
        if criterion is None:
            return None
        variance = list(criterion.values())
        
        variance /= np.max(variance)
        probs = np.exp(-variance / self.temperature)
        probs /= np.sum(probs)

        if self.deterministic:
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(un_items)), p=probs)
        result = un_items[ndx]
        return result


class StochasticVarianceItemSelector(VarianceItemSelector):
    description = "Stochastic variance selector"

    def __init__(self, scoring, **kwargs):
        super(StochasticVarianceItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
