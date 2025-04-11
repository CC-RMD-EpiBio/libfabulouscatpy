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


class FisherItemSelector(ItemSelector):
    
    description = """Fisher information selector"""
    def __init__(self, scoring, deterministic=True, **kwargs):
        super(FisherItemSelector, self).__init__(**kwargs)
        self.scoring = scoring
        self.deterministic = deterministic
        
    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:

        means = {}
        for k, v in scoring.scores.items():
            means[k] = v.score
        # We now have a value for the trait, this allows the calculation of the
        # Fisher Information, which is used to select the new item.

        scored = [i for i in items if "scales" in i.keys()]
        in_scale = [i for i in scored if scale in i["scales"].keys()]

        if len(in_scale) == 0:
            return None
        
        item_info = self.model.item_information(
            items=[x['item'] for x in in_scale],
            abilities=means
        )
        fish_scored = {i["item"]: item_info[i['item']][0] for i in in_scale}

        return fish_scored
    
class StochasticFisherItemSelector(FisherItemSelector):
    
    description = """Selection based on Fisher information"""

    def _next_scored_item(self, tracker: CatSessionTracker, scale=None) -> dict[str: dict[str:Any]]:
        """
        Parameters: session: instance of CatSessionTracker
        Returns:    item dictionary entry or None
        """
        if scale is None:
            scale = self.next_scale(tracker)
        un_items = self.un_items(tracker, scale)

        if un_items is None:
            # Not sure if this can happen under normal testing, but included as
            # a safety feature.
            return {}
        
        fish_scored = self.criterion(tracker, scale)
        probs = np.array(fish_scored) ** (1/self.temperature)
        probs /= np.sum(probs)

        if self.deterministic:
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(probs)), p=probs)
        result = fish_scored.keys()[ndx]
        return result


class StochasticFisherItemSelector(FisherItemSelector):
    description = "Stochastic Fisher selector"
    def __init__(self, scoring, **kwargs):
        super(StochasticFisherItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
