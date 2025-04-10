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

from typing import Any

import numpy as np

from libfabulouscatpy.cat.itemselection import ItemSelector
from libfabulouscatpy.cat.session import CatSessionTracker
from libfabulouscatpy.irt.scoring import BayesianScoring


class KLItemSelector(ItemSelector):

    description = """Greedy plugin KL selector"""

    def __init__(self, scoring, deterministic=True, hybrid=False, **kwargs):
        super(KLItemSelector, self).__init__(**kwargs)
        self.scoring = scoring
        self.hybrid = hybrid
        self.deterministic = deterministic

    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:

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
        energy = self.scoring.log_like[scale] + self.scoring.log_prior[scale]

        ###

        #######
        # Future
        lp_itemized = self.model.models[scale].log_likelihood(
            theta=self.model.interpolation_pts, observed_only=False
        )[
            :, unresponded_ndx, :
        ]  # ll for unobserved items
        # N_grid x N_item x K
        p_itemized = np.exp(lp_itemized)
        pi_density = scoring.scores[scale].density
        
        # lp_infty = lp_itemized + energy[:, np.newaxis, np.newaxis]
        lp_infty = 0.5*lp_itemized + energy[:, np.newaxis, np.newaxis] * (lp_itemized.shape[1] - 1)/2
        # N_grid x N_item x K
        expected_lp_infty = np.sum(lp_infty*p_itemized, axis=-1, keepdims=True)
        expected_lp_infty = np.sum(expected_lp_infty, axis=-2, keepdims=True)
        # shift it so that it is closer to the running estimate
        
        
        # N_grid x 1 x 1
        pi_infty = np.exp(expected_lp_infty - np.max(expected_lp_infty, axis=0, keepdims=True))
        pi_infty /= np.trapz(
            y=pi_infty, x=scoring.interpolation_pts[scale], axis=0
        ) # N_grid x 1 x 1
        ##########
        # $\pi_\infty$ is computed
        ########

        expected_p_itemized = np.trapz(
            y=pi_density[:, np.newaxis, np.newaxis] * p_itemized,
            x=self.scoring.interpolation_pts[scale],
            axis=0,
        )  # p_{ik}^{\alpha_t}
        A = np.trapz(
            y=pi_infty * lp_itemized, # pi_infty
            x=scoring.interpolation_pts[scale],
            axis=0,
        )  # This is an integral over \theta, A will have shape I x K
        A = np.sum(A * expected_p_itemized, axis=-1) # Now A has shape I
        B = np.log(
            np.trapz(
                y=pi_density[:, np.newaxis, np.newaxis] * p_itemized,
                x=scoring.interpolation_pts[scale],
                axis=0,
            )
        ) # this is an integral over theta, B has shape I x K
        B = np.sum(B * expected_p_itemized, axis=-1) # now B has shape I
        
        Delta = -A + B
        criterion = dict(zip([x['item'] for x in items], Delta))
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
        valid_items = [x['item'] for x in un_items]
        items = []
        Delta = []
        for k, v in criterion.items():
            if k in valid_items:
                items += [k]
                Delta += [v]
        if len(items) == 0:
            return {}
        Delta -= np.min(Delta)
        probs = np.exp(-Delta/self.temperature)
        probs /= np.sum(probs)
    
        if self.deterministic or (self.hybrid  and  ((self.scoring.n_scored[scale] > 3))):
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(criterion.keys())), p=probs)
        result = list(criterion.keys())[ndx]
        for i in un_items:
            if i['item'] == result:
                return i
        return {}


class InfinityKLItemSelector(KLItemSelector):
    description = """Greedy ∞ plugin KL selector """

    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:

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
        energy = self.scoring.log_like[scale] + self.scoring.log_prior[scale]

        ###

        #######
        # Future
        lp_itemized = self.model.models[scale].log_likelihood(
            theta=self.model.interpolation_pts, observed_only=False
        )[
            :, unresponded_ndx, :
        ]  # ll for unobserved items
        # N_grid x N_item x K
        p_itemized = np.exp(lp_itemized)
        pi_density = scoring.scores[scale].density
        
        # lp_infty = lp_itemized + energy[:, np.newaxis, np.newaxis]
        lp_infty = 0.5*lp_itemized + energy[:, np.newaxis, np.newaxis] * (lp_itemized.shape[1] - 1)/2

        # N_grid x N_item x K
        expected_lp_infty = np.sum(lp_infty*p_itemized, axis=-1, keepdims=True)
        # N_grid x N_item x 1
        expected_lp_infty = np.sum(expected_lp_infty, axis=-2, keepdims=True)
        pi_infty = np.exp(expected_lp_infty - np.max(expected_lp_infty, axis=0, keepdims=True))
        pi_infty /= np.trapz(
            y=pi_infty, x=scoring.interpolation_pts[scale], axis=0
        ) # N_grid x 1 x 1
        ##########
        # $\pi_\infty$ is computed
        ########

        expected_p_itemized = np.trapz(
            y=pi_infty * p_itemized,
            x=self.scoring.interpolation_pts[scale],
            axis=0,
        )  # p_{ik}^{\alpha_t}
        A = np.trapz(
            y=pi_infty * lp_itemized, # pi_infty
            x=scoring.interpolation_pts[scale],
            axis=0,
        )  # This is an integral over \theta, A will have shape I x K
        A = np.sum(A * expected_p_itemized, axis=-1) # Now A has shape I
        B = np.log(
            np.trapz(
                y=pi_density[:, np.newaxis, np.newaxis] * p_itemized,
                x=scoring.interpolation_pts[scale],
                axis=0,
            )
        ) # this is an integral over theta, B has shape I x K
        B = np.sum(B * expected_p_itemized, axis=-1) # now B has shape I
        
        Delta = -A + B
        criterion = dict(zip([x['item'] for x in items], Delta))
        return criterion
    
class StochasticKLItemSelector(KLItemSelector):
    description = "Stochastic KL selector"

    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(StochasticKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
        
class StochasticInfinityKLItemSelector(InfinityKLItemSelector):
    description = "Stochastic ∞ KL selector"

    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(StochasticInfinityKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )

class HybridStochasticKLItemSelector(KLItemSelector):
    description = "Hybrid Stochastic KL selector"
    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(HybridStochasticKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, hybrid=True, **kwargs
        )

class McKlItemSelector(KLItemSelector):
    description = "Greedy Monte-Carlo KL selector"
    def __init__(self, scoring, n=16, deterministic=True, hybrid=False, **kwargs):
        super(McKlItemSelector, self).__init__(scoring, **kwargs)
        self.hybrid = hybrid
        self.deterministic = deterministic
        self.n = n
        
    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:
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
        current_score = scoring.scores[scale]
        ability_samples = current_score.sample(self.n)
        model = self.model.models[scale]
        observed_energy = self.scoring.log_like[scale] + self.scoring.log_prior[scale]
        observed_density = self.scoring.scores[scale].density
        # future
        lp_itemized = self.model.models[scale].log_likelihood(
            theta=self.model.interpolation_pts, observed_only=False
        )[
            :, unresponded_ndx, :
        ]  # ll for unobserved items
        p_itemized = np.exp(lp_itemized)
        item_labels = [x['item'] for x in in_scale]
        
        crit = []
        
        for theta in ability_samples:
            x = model.sample(theta)
            x_ = {k['item']: x[k['item']] for k in in_scale}
            ll_over_grid = model.log_likelihood(scoring.interpolation_pts[scale], responses=x_)
            ndx = np.array(list(x_.values()))
            lpi_infty = observed_energy + ll_over_grid
            p_itemized_ = np.take_along_axis(p_itemized, ndx[np.newaxis, :, np.newaxis], axis=-1)[..., 0]
            lp_itemized_ = np.take_along_axis(lp_itemized, ndx[np.newaxis, :, np.newaxis], axis=-1)[..., 0]

            lpi_infty -= np.max(lpi_infty)
            pi_infty = np.exp(lpi_infty)
            z = np.trapz(pi_infty, scoring.interpolation_pts[scale])
            pi_infty /= z
            B = np.log(np.trapz(p_itemized_ * observed_density[:, np.newaxis], scoring.interpolation_pts[scale], axis=0))
            A = np.trapz(lp_itemized_ * pi_infty[:, np.newaxis], scoring.interpolation_pts[scale], axis=0)
            crit += [B-A]
        crit = np.mean(crit, axis=0)
        return dict(zip(item_labels, crit))
      

class StochasticMcKlItemSelector(McKlItemSelector):
    description = "Stochastic MC-KL selector"

    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(StochasticKLItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
          
class HybridStochasticMcKlItemSelector(McKlItemSelector):
    description = "Hybrid Stochastic MC-KL selector"
    def __init__(self, scoring, **kwargs):
        self.deterministic = False
        super(HybridStochasticMcKlItemSelector, self).__init__(
            scoring=scoring, deterministic=False, hybrid=True, **kwargs
        )