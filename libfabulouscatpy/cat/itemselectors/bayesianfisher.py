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
from libfabulouscatpy.irt.scoring import BayesianScoring


class BayesianFisherItemSelector(ItemSelector):
    description = """Selection based on Bayesian average Fisher information"""

    def __init__(self, scoring, **kwargs):
        super(BayesianFisherItemSelector, self).__init__(**kwargs)
        self.scoring = scoring

    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:

        """
        Parameters: session: instance of CatSessionTracker
        Returns:    item dictionary entry or None
        """

        scored = [i for i in items if "scales" in i.keys()]
        in_scale = [i for i in scored if scale in i["scales"].keys()]

        if len(in_scale) == 0:
            return None

        item_info = self.model.item_information(
            items=[x["item"] for x in in_scale],
            abilities=scoring.interpolation_pts,
        )
        fish_scored = [
            np.trapz(
                y=item_info[i["item"]] * scoring.scores[scale].density,
                x=scoring.interpolation_pts[scale],
            )
            / np.trapz(
                y=scoring.scores[scale].density,
                x=scoring.interpolation_pts[scale],
            )
            for i in in_scale
        ]
        criterion = dict(zip([x['item'] for x in items], fish_scored))

        return criterion


class StochasticBayesianFisherItemSelector(BayesianFisherItemSelector):
    description = "Stochastic Bayesian Fisher selector"
    def __init__(self, scoring, **kwargs):
        super(StochasticBayesianFisherItemSelector, self).__init__(
            scoring=scoring, deterministic=False, **kwargs
        )
