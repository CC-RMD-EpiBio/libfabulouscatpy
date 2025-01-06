from typing import Any

import numpy as np

from libfabulouscatpy.cat.itemselection import ItemSelector
from libfabulouscatpy.cat.session import CatSessionTracker


class VarianceItemSelector(ItemSelector):

    description = """Bayesian variance selector"""

    def __init__(self, scoring, **kwargs):
        super(VarianceItemSelector, self).__init__(**kwargs)
        self.scoring = scoring

    def _next_scored_item(
        self, tracker: CatSessionTracker, scale=None
    ) -> dict[str : dict[str:Any]]:
        """
        Parameters: session: instance of CatSession
        Returns:    item dictionary entry or None
        """
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

        # We now have a value for the trait, this allows the calculation of the
        # Fisher Information, which is used to select the new item.

        unresponded = [i for i in un_items if "scales" in i.keys()]
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

        ###

        #######
        # Future

        lp_infty = log_ell + energy[:, np.newaxis, np.newaxis]
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
        variance = second - mean**2
        variance = np.sum(variance * np.exp(log_ell) * pi_infty, axis=-1)
        variance = np.sum(variance, axis=0)
        variance /= np.max(variance)
        probs = np.exp(-variance) ** (1 / self.temperature)
        probs /= np.sum(probs)

        if self.deterministic:
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(in_scale)), p=probs)
        result = in_scale[ndx]
        return result


class StochasticVarianceItemSelector(VarianceItemSelector):
    description = "Stochastic variance selector"

    def __init__(self, scoring, **kwargs):
        super(StochasticVarianceItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
