###############################################################################
#
#                           COPYRIGHT NOTICE
#                  Mark O. Hatfield Clinical Research Center
#                       National Institutes of Health
#            United States Department of Health and Human Services
#
# This software was developed and is owned by the National Institutes of
# Health Clinical Center (NIHCC), an agency of the United States Department
# of Health and Human Services, which is making the software available to the
# public for any commercial or non-commercial purpose under the following
# open-source BSD license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# (1) Redistributions of source code must retain this copyright
# notice, this list of conditions and the following disclaimer.
#
# (2) Redistributions in binary form must reproduce this copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# (3) Neither the names of the National Institutes of Health Clinical
# Center, the National Institutes of Health, the U.S. Department of
# Health and Human Services, nor the names of any of the software
# developers may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# (4) Please acknowledge NIHCC as the source of this software by including
# the phrase "Courtesy of the U.S. National Institutes of Health Clinical
# Center"or "Source: U.S. National Institutes of Health Clinical Center."
#
# THIS SOFTWARE IS PROVIDED BY THE U.S. GOVERNMENT AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# You are under no obligation whatsoever to provide any bug fixes,
# patches, or upgrades to the features, functionality or performance of
# the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to
# the National Institutes of Health Clinical Center, without imposing a
# separate written license agreement for such Enhancements, then you hereby
# grant the following license: a non-exclusive, royalty-free perpetual license
# to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such Enhancements or
# derivative works thereof, in binary and source code form.
#
###############################################################################
# D(pi(\theta|x_t) || pi(\theta|x_t+1))

from typing import Any

import numpy as np
from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.cat.itemselection import ItemSelector
from libfabulouscatpy.cat.session import CatSessionTracker
from libfabulouscatpy.irt.scoring import BayesianScoring


class KLItemSelector(ItemSelector):

    description = """Greedy KL selector"""

    def __init__(self, scoring, deterministic=True, hybrid=False, em_iters=10, **kwargs):
        super(KLItemSelector, self).__init__(**kwargs)
        self.scoring = scoring
        self.hybrid = hybrid
        self.em_iters = em_iters
        self.deterministic = deterministic

    def criterion(
        self, scoring: BayesianScoring, items: list[dict], scale=None
    ) -> dict[str:Any]:
        """
        Parameters: session: instance of CatSession
        Returns:    item dictionary entry or None
        """

        unresponded = [i for i in items if "scales" in i.keys()]
        in_scale = [i for i in unresponded if scale in i["scales"].keys()]

        if len(in_scale) == 0:
            return {}

        unresponded_ndx = [
            self.model.item_labels[scale].index(j["item"]) for j in unresponded
        ]

        #####
        # current
        ######

        #######
        # Future
        lp_itemized = self.model.models[scale].log_likelihood(
            theta=self.model.interpolation_pts, observed_only=False
        )[
            :, unresponded_ndx, :
        ]  # ll for unobserved items as a function of the theta grid
        # N_grid x N_item x K

        pi_density_t = self.scoring.scores[scale].density
        lpi_density_t = self.scoring.log_like[scale] + self.scoring.log_prior[scale]

        # q_z[j, k] = π*(x_j=k | x_t): predictive probability for response k
        imputation_model = getattr(self.scoring, 'imputation_model', None)
        if imputation_model is not None:
            observed_dict = {
                k: float(v) for k, v in self.scoring.scored_responses.items()
                if v != self.scoring.skipped_response
            }
            n_categories = lp_itemized.shape[-1]
            q_z = np.zeros((len(unresponded), n_categories))
            for j_idx, item_dict in enumerate(unresponded):
                try:
                    q_z[j_idx] = imputation_model.predict_pmf(
                        items=observed_dict,
                        target=item_dict["item"],
                        n_categories=n_categories,
                    )
                except (KeyError, ValueError):
                    q_z[j_idx] = _trapz(
                        np.exp(lp_itemized[:, j_idx, :]) * pi_density_t[:, np.newaxis],
                        self.model.interpolation_pts,
                        axis=0,
                    )
                    q_z[j_idx] /= np.sum(q_z[j_idx])
        else:
            q_z = _trapz(
                np.exp(lp_itemized) * pi_density_t[:, np.newaxis, np.newaxis],
                self.model.interpolation_pts,
                axis=0,
            )
            q_z /= np.sum(q_z, axis=-1, keepdims=True)

        lpi_next = lpi_density_t[:, np.newaxis, np.newaxis] + lp_itemized
        KL = pi_density_t[:, np.newaxis, np.newaxis] * (lpi_density_t[:, np.newaxis, np.newaxis] - lpi_next)
        KL = _trapz(KL, self.model.interpolation_pts, axis=0)
        KL = np.sum(KL * q_z, axis=-1)
        criterion = dict(zip([x["item"] for x in items], KL))
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

        criterion = self.criterion(scoring=self.scoring, items=un_items, scale=scale)
        valid_items = [x["item"] for x in un_items]
        items = []
        Delta = []
        for k, v in criterion.items():
            if k in valid_items:
                items += [k]
                Delta += [v]
        if len(items) == 0:
            return {}
        Delta -= np.min(Delta)
        probs = np.exp(-Delta / self.temperature)
        probs /= np.sum(probs)

        if self.deterministic or (self.hybrid and ((self.scoring.n_scored[scale] > 3))):
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(criterion.keys())), p=probs)
        result = list(criterion.keys())[ndx]
        for i in un_items:
            if i["item"] == result:
                return i
        return {}


class StochasticKLItemSelector(KLItemSelector):
    description = "Stochastic KL selector"

    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(StochasticKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )


class HybridStochasticKLItemSelector(KLItemSelector):
    description = "Hybrid Stochastic KL selector"

    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(HybridStochasticKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, hybrid=True, **kwargs
        )
