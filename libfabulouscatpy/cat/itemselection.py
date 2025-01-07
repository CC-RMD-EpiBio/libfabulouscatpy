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

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from libfabulouscatpy import constants as const
from libfabulouscatpy.cat.session import CatSessionTracker
from libfabulouscatpy.irt.prediction import IRTModel
from libfabulouscatpy.irt.scoring import BayesianScoring


class ItemSelector(ABC):
    """Multiscale item selector class
    An object of this type gets embedded into each CatDataBase object.
    This class gives out items stochastically, according to a temperature
    set at initialization. When T\to\infty, selection approaches uniform.
    On the other end, as T\to 0^+, selection is more-deterministic.

    Args:
        object ([type]): [description]
    Returns:
        [type]: [description]
    """

    description: str

    def __init__(
        self,
        items: list[dict[str:Any]],
        scales: dict[str : dict[str:Any]],
        model: IRTModel,
        temperature: float = 0.01,
        randomize_items: bool = True,
        randomize_scales: bool = True,
        unscored_freq: float = 0.33,
        precision_limit: float = 0.3873,
        min_responses: int = 5,
        max_responses: int = 12,
        inadmissable_scales: list[str] | None = None,
        deterministic: bool = True,
        **kwargs,
    ) -> None:
        """Instantiate item selector class
        Args:
            items (list): List of items, imported from .json files or ItemDatabase
            scales (list): Dict of scales
            temperature (float, optional): [description]. Defaults to 0.01.
            randomize_items (bool, optional): [description]. Defaults to True.
            randomize_scales (bool, optional): [description]. Defaults to True.
            unscored_freq (float, optional): [description]. Defaults to 0.33.
        """
        self.model = model
        self.log_add = ""
        self.max_responses = max_responses
        self.PRECISION_LIMIT = precision_limit
        self.PRECISION_RESPONSE_MINIMUM = min_responses
        self.randomize_items = randomize_items
        self.randomize_scales = randomize_scales
        self.unscored_freq = unscored_freq
        self.temperature = temperature
        self.items = items
        self.scales = scales
        self.itemdict = {x["item"]: x for x in items}
        self.deterministic = deterministic
        self.unscored_count = 0
        self.inadmissable_scales = (
            [] if inadmissable_scales is None else inadmissable_scales
        )



    def un_items(
        self, tracker: CatSessionTracker, scale: str | None = None
    ) -> list[dict[str:Any]] | None:
        """Get list of unanswered items in given session, in a given scale
        Args:
            session (CatTracker): [Tracking object for CAT session]
            scale (string, optional): [description]. Defaults to None.
        Returns:
            [type]: [description]
        """
        if len(self.open_scales_for_session(tracker)) == 0:
            resp = []
            return None

        answered = tracker.responses.keys()
        un = [i for i in self.items if i["item"] not in answered]
        if scale is not None:
            un = [i for i in un if "scales" in i.keys()]
            un = [i for i in un if scale in i["scales"].keys()]
        subject = tracker.subject

        def dif_admissable(item):
            if "diff" not in item.keys():
                return True
            matched = True
            if "required" in item["diff"].keys():
                for k, v in item["diff"]["required"].items():
                    if hasattr(subject, k):
                        if v != getattr(subject, k):
                            matched = False
            if "excluded" in item["diff"].keys():
                for k, v in item["diff"]["excluded"].items():
                    if hasattr(subject, k):
                        if v == getattr(subject, k):
                            matched = False
            return matched

        un_items = list(filter(dif_admissable, un))

        resp = self.responses_for_session(tracker, scale=scale).values()
        if len(resp) > 0:
            unskipped = sum(1 for x in resp if x != const.SKIPPED_RESPONSE)
        else:
            unskipped = 0

        if scale is None:
            return None

        elif (
            tracker.errors[scale] is not None
            and (tracker.errors[scale] < self.PRECISION_LIMIT)
            and unskipped >= self.PRECISION_RESPONSE_MINIMUM
        ):
            # If we have not reached the maximum number of responses, but
            # the precision is satisfied, finish this scale and select from
            # those unused.
            tracker.close_scale(scale)

        else:
            # All that remains is to choose an item for this scale. We find
            # all items not yet responded to.

            if un_items == {} or un_items is None:
                # If no items remain for response, finish this scale and
                # selct from those unused.
                tracker.close_scale(scale)

        return un_items

    def open_scales_for_session(self, tracker: CatSessionTracker):
        """Get list of scales that are still open in a session
        Args:
            session ([type]): [description]
        Returns:
            [type]: [description]
        """
        allowed_scales = list(self.scales.keys())
        allowed_scales = [
            s for s in allowed_scales if s not in self.inadmissable_scales
        ]
        open_scales = [s for s in allowed_scales if s not in tracker.closed_scales]

        # cull out any unadmissable open scales
        return open_scales


    def responses_for_session(
        self, tracker: CatSessionTracker, scale=None, unscored=False
    ):
        """
        Parameters: responses: dict
                        responses[num] = Response(num, ...)
        Returns:   w: dict
                        Instances of Response() for the current scale.
                        w[num] = Response(num, ...)
        """
        responses = tracker.responses
        if scale is None:
            return responses
        with_scales = [
            k for k in responses.keys() if "scales" in self.itemdict[k].keys()
        ]

        in_scale = {
            k: tracker.responses[k]
            for k in with_scales
            if scale in self.itemdict[k]["scales"].keys()
        }

        return in_scale

    def select_scale(self, session):
        """Chooses a scale given a session
        Args:
            session ([type]): [description]
        Returns:
            [type]: [description]
        """
        available_scales = self.open_scales_for_session(session)
        if self.randomize_scales:
            scale = random.choice(available_scales)

        else:
            scale = available_scales[0]
        return scale

    def next_item(self, tracker: CatSessionTracker, scale=None) -> str | None:
        """Retrieve the next item
        Args:
            session ([type]): [description]
            scale ([type], optional): [description]. Defaults to None.
        Returns:
            [type]: [description]
        """
        if (
            random.random() < self.unscored_freq
        ) and self.unscored_count < const.MAX_UNSCORED:
            candidate = self._next_unscored_item(tracker)
            if candidate is not None:
                self.unscored_count += 1
                return candidate
        return self._next_scored_item(tracker, scale)
    
    @abstractmethod
    def criterion(self, scoring: BayesianScoring, items: list[dict], scale=None) -> dict[str: Any]:
        return {}
    
    def _next_scored_item(
        
        self, tracker: CatSessionTracker, scale=None
        ) -> dict[str : dict[str:Any]]:
        
        scale = self.next_scale(tracker)
        un_items = self.un_items(tracker, scale)

        if un_items is None or len(un_items) == 0:
            # Not sure if this can happen under normal testing, but included as
            # a safety feature.
            return None

        trait = tracker.scores[scale]
        trait = 0.0 if trait is None else trait
        error = tracker.errors[scale]
        error = 100.0 if error is None else error
        
        criterion = self.criterion(scoring=self.scoring, items = un_items, scale=scale)

        variance = list(criterion.values())
        
        variance /= np.max(variance)
        probs = np.exp(-variance) ** (1 / self.temperature)
        probs /= np.sum(probs)

        if self.deterministic:
            ndx = np.argmax(probs)
        else:
            ndx = np.random.choice(np.arange(len(un_items)), p=probs)
        result = un_items[ndx]
        return result

    def _next_unscored_item(self, tracker: CatSessionTracker) -> dict[str: Any]:
        """Retrieve the next unscored item
        Args:
            session ([type]): [description]
        Returns:
            [type]: [description]
        """
        answered = [self.itemdict[k] for k in tracker.responses.keys()]
        un_unscored = [i for i in self.items if "scales" not in i.keys()]
        un_unscored = [i for i in un_unscored if i["item"] not in answered]
        if len(un_unscored) == 0:
            return None
        result = random.choice(un_unscored)
        return result

    def init_score(self, subject, scale):
        """
        Parameters: subject: instance of Subject()
                    scale: instance of Scales()
        Returns:    w: 0.0
        """

        return 0.0

    def eligible_scales(self):
        """
        Returns:    w: dict
                        All eligible scales for this session.
        """
        return list(self.scales)

    def next_scale(self, session: CatSessionTracker, scale: str | None = None):
        cons = const.SKIPPED_RESPONSE

        # if scale is None:
        successful_scale = False
        # Loop until we get a scale that has open items
        while (not successful_scale) and (
            len(self.open_scales_for_session(session)) > 0
        ):
            if scale is None:
                proposed_scale = self.select_scale(session)
            else:
                proposed_scale = scale

            resp = self.responses_for_session(session, scale=proposed_scale).values()
            if len(resp) > 0:
                unskipped = sum(1 for x in resp if x != cons)
            else:
                unskipped = 0
            # print(f"{proposed_scale} unskipped: {unskipped}", file=sys.stderr)
            if (
                session.errors[proposed_scale] is not None
                and (session.errors[proposed_scale] < self.PRECISION_LIMIT)
                and unskipped >= self.PRECISION_RESPONSE_MINIMUM
            ):
                # print(f"closing scale: {proposed_scale} with {unskipped} responses", file=sys.stderr)

                session.close_scale(proposed_scale)

            resp = self.responses_for_session(session, scale=proposed_scale).values()
            un_items = self.un_items(session, proposed_scale)
            if un_items is None:
                un_items = []
            if len(un_items) > 0:
                successful_scale = True
                scale = proposed_scale
            else:
                session.close_scale(proposed_scale)
        if unskipped >= self.max_responses:
            # If we have more responses that needed, finish this scale and
            # select from those unused.
            # print(f"exceeded max responses, closing scale: {scale}", file=sys.stderr)

            session.close_scale(scale)
            allowed_scales = list(self.scales.keys())
            open_scales = self.open_scales_for_session(session)

            if (open_scales is None) or (len(open_scales) == 0):
                return None

            if self.randomize_scales:
                next_scale = np.random.choice(open_scales)
                rand_text = "randomized"
            else:
                next_scale = open_scales[0]

            scale = next_scale
            un_items = self.un_items(session, scale)
        return scale