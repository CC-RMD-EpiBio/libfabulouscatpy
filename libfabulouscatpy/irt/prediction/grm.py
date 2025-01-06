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

from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from numpy import random

from libfabulouscatpy.irt.item import ItemDatabase, ScaleDatabase
from libfabulouscatpy.irt.prediction.irt import FactorizedIRTModel, IRTModel


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GradedResponseModel(IRTModel):
    description = "Factorized multidimensional GRM"

    def __init__(
        self,
        slope: list[float],
        calibration: list[float],
        item_labels: list[str],
        responses=None,
    ) -> None:
        """
        Parameters: slope: float list of length N
                        This indicates how steeply the probability of correct
                        response changes as the trait increases.
                    calibration: 2D float list of length N
                        If for a given item x and score y, if b_{x,y} = t, then
                        50% of examinees with a trait score of t are expected
                        to have scores of y or higher.
                    answers: int list of length N
                        The value of answers[x] is the numeric answer given for
                        item x.
        Returns:    None

        """

        # self.N is the number of items.
        self.N = len(calibration)
        self.item_labels = item_labels
        self.slope = np.array(slope)
        # Since the calibration rows vary in length, we pad with zeros.
        self.calibration = np.atleast_1d(calibration)

        return None


    def survival(self, trait: float = 0.0, order: int = 0) -> npt.NDArray[Any]:
        """
        Parameters: trait: float
                        The respective trait of the individual
                    m: int
                        Calculate (t^m_ij) as defined in the BU score
                        estimation paper.
        Returns:    If k = (k_{x,y}) is the probability that an individual
                    response to item x is y (given the trait value), then:
                        m == 0 --> t0 = k
                        m == 1 --> t1 = k^2 - k
                        m == 2 --> t2 = 2k^3 - 3k^2 + k
                        m == 3 --> t3 = 6k^4 - 12k^3 + 7k^2 - k
        """

        p = 1.0 / (1.0 + (np.exp((self.calibration - trait).T * self.slope).T))
        p = np.append(np.ones((self.N, 1)), p, axis=1)
        p = np.append(p, np.zeros((self.N, 1)), axis=1)

        if order == 0:
            return p

        elif order == 1:
            return p**2.0 - p

        elif order == 2:
            return 2.0 * (p**3.0) - 3.0 * (p**2.0) + p

        elif order == 3:
            return 6.0 * (p**4.0) - 12.0 * (p**3.0) + 7.0 * (p**2.0) - p

        else:
            return None

    def gather_params(self, indices: list[int]) -> dict[str : npt.NDArray]:
        slopes = self.slope[indices]
        return {"discrimination": slopes, "difficulties": self.calibration[indices]}

    def log_likelihood(
        self,
        theta: npt.NDArray[Any],
        observed_only: bool = True,
        responses: dict[str, int] = None,
    ) -> npt.NDArray[Any]:
        """
        Vectorized form of the log-likelihood, taking in a vector-valued trait
        and evaluating the log-likelihood everywhere
        :param theta: trait vector
        :return: log-likelihood vector of same shape as trait
        """
        # N = number of questions
        # self.calibration is N x 4
        # self.responses in N x 5

        augmented_calibration = np.empty(
            (self.calibration.shape[0], self.calibration.shape[1] + 2)
        )
        augmented_calibration[:] = np.nan
        augmented_calibration[:, 1:-1] = self.calibration
        slopes = self.slope[:, np.newaxis]
        if responses is not None:
            answered_items = [self.item_labels.index(k) for k in responses.keys()]
            augmented_calibration = augmented_calibration[answered_items]
            slopes = slopes[answered_items]

        # selected is of shape (N, )
        # self.slope is of shape (N, )
        # augmented_calibration is of shape (N, K) K choices
        # theta is of shape (M, ) M number of grid points

        p = slopes * (augmented_calibration - theta[:, np.newaxis, np.newaxis])
        p = 1 / (1 + np.exp(-p))
        p[..., 0] = 0
        p[..., -1] = 1
        np.seterr(divide="warn")
        p = p[..., 1:] - p[..., :-1]

        if not observed_only:
            return np.log(p + 1e-20)
        selected = np.array(list(responses.values()))
        p_observed = p.T[selected-1].T

        p_observed = np.sum(np.log(p_observed + 1e-20), axis=-1)
        p_observed = np.sum(p_observed, axis=-1)

        #

        return p_observed

    def sample(self, theta: npt.NDArray[Any]) -> dict[str:int]:
        augmented_calibration = np.empty(
            (self.calibration.shape[0], self.calibration.shape[1] + 2)
        )
        augmented_calibration[:] = np.nan
        augmented_calibration[:, 1:-1] = self.calibration
        slopes = self.slope[:, np.newaxis]
        p = slopes * (
            augmented_calibration - np.atleast_1d(theta)[:, np.newaxis, np.newaxis]
        )
        p = 1 / (1 + np.exp(-p))
        p[..., 0] = 0
        p[..., -1] = 1

        p = p[..., 1:] - p[..., :-1]
        p = (p / np.sum(p, axis=-1, keepdims=True))[0]
        resp = {
            k: random.choice(a=np.arange(len(v)), p=v)
            for k, v in zip(self.item_labels, p)
        }
        return resp

    def sample_retest(
        self, theta: npt.NDArray[Any], responses: dict[str:int]
    ) -> dict[str:int]:
        augmented_calibration = np.empty(
            (self.calibration.shape[0], self.calibration.shape[1] + 2)
        )
        augmented_calibration[:] = np.nan
        augmented_calibration[:, 1:-1] = self.calibration
        slopes = self.slope[:, np.newaxis]
        p = slopes * (
            augmented_calibration - np.atleast_1d(theta)[:, np.newaxis, np.newaxis]
        )
        p = 1 / (1 + np.exp(-p))
        p[..., 0] = 0
        p[..., -1] = 1

        p = p[..., 1:] - p[..., :-1]
        p = (p / np.sum(p, axis=-1, keepdims=True))[0]

        retest = {}
        for k, v in zip(self.item_labels, p):
            choice = responses[k]
            p_retest = np.zeros_like(p)
            p_retest[choice] = p[choice]
            if choice == 0:
                p_retest[choice + 1] = p[choice + 1]
            elif choice == (len(p) - 1):
                p_retest[choice - 1] = p[choice - 1]
            else:
                p_retest[choice + 1] = p[choice + 1]
                p_retest[choice - 1] = p[choice - 1]
            p_retest = p_retest / np.sum(p_retest)
            choice_retest = random.choice(a=np.arange(len(p)), p=p_retest)
            retest[k] = choice_retest
        return retest


class MultivariateGRM(FactorizedIRTModel):
    description = (
        "Multidimensional graded response model with all the calibrations pre-loaded"
    )

    def __init__(
        self,
        itemdb: ItemDatabase,
        scaledb: ScaleDatabase,
        interpolation_pts: npt.NDArray[Any] | None = None,
    ) -> None:
        models = {}
        probs = {}
        discriminations = defaultdict(list)
        difficulties = defaultdict(list)
        item_labels = defaultdict(list)
        for item in itemdb.items:

            try:
                scales = item["scales"]
                for scale, vals in scales.items():
                    discriminations[scale] += [vals["discrimination"]]
                    difficulties[scale] += [vals["difficulties"]]
                    item_labels[scale] += [item["item"]]
            except KeyError:
                pass
        for scale in discriminations.keys():
            models[scale] = GradedResponseModel(
                discriminations[scale],
                difficulties[scale],
                item_labels=item_labels[scale],
            )
            models[scale].description = f"GRM for {scale}"
        self.models = models
        self.item_labels = item_labels
        if interpolation_pts is None:
            interpolation_pts = np.arange(-6.0, 6.0, step=0.05)
        self.interpolation_pts = interpolation_pts
        for scale, model in self.models.items():
            probs[scale] = model.log_likelihood(interpolation_pts, observed_only=False)
            probs[scale] = xr.DataArray(
                probs[scale],
                dims=["theta", "item", "choice"],
                coords={
                    "theta": self.interpolation_pts,
                    "item": item_labels[scale],
                    "choice": np.arange(probs[scale].shape[-1]),
                },
            )
        return

    def log_likelihood(
        self, theta: npt.ArrayLike | None, responses: dict[str:int] = None
    ):
        return

    def item_information(
        self, items: list[str], abilities: dict[str, float]
    ) -> dict[str:float]:
        fisher = {}
        to_compute = {
            k: [x for x in v if x in items] for k, v in self.item_labels.items()
        }
        for scale, i in to_compute.items():
            if len(i) == 0:
                continue
            score = abilities[scale]
            if score is None:
                score = 0
            score = np.atleast_1d(score)
            item_ndx = [self.item_labels[scale].index(j) for j in i]
            _p = self.models[scale].gather_params(item_ndx)
            sigma = sigmoid(
                (_p["difficulties"][np.newaxis, ...] - score[:, np.newaxis, np.newaxis])
                * _p["discrimination"][np.newaxis, :, np.newaxis]
            )
            sigma = np.append(np.zeros(list(sigma.shape[:-1]) + [1]), sigma, axis=-1)
            sigma = np.append(sigma, np.ones(list(sigma.shape[:-1]) + [1]), axis=-1)
            p = sigma[..., 1:] - sigma[..., :-1]
            _fish = (
                sigma[..., 1:] * (1 - sigma[..., 1:])
                + sigma[..., :-1] * (1 - sigma[..., :-1])
            ) * _p["discrimination"][:, np.newaxis] ** 2
            expected = np.sum(p * _fish, axis=-1)
            fisher = {**fisher, **{k: v.T for k, v in zip(i, expected.T)}}
        return fisher

    def sample(self, theta: dict[str : npt.NDArray[Any]]) -> dict[str:int]:
        responses = {}
        for scale, model in self.models.items():
            responses = {**responses, **model.sample(theta[scale])}
        return responses

    def sample_test_retest(self, theta: dict[str : npt.NDArray[Any]]) -> dict[str:int]:
        responses = {}
        retest = {}
        for scale, model in self.models.items():
            scale_responses = model.sample(theta[scale])
            responses = {**responses, **scale_responses}
            retest = {**retest, **model.sample_retest(theta[scale], scale_responses)}
        return responses, retest
