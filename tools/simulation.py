#!/usr/bin/env python3
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.cat.itemselection import ItemSelector
from libfabulouscatpy.cat.session import CatSession, CatSessionTracker
from libfabulouscatpy.irt.prediction.grm import MultivariateGRM
from libfabulouscatpy.irt.scoring.bayesian import BayesianScore, BayesianScoring


@dataclass
class StepRecord:
    """One CAT step."""
    step: int
    item_id: str
    response: int
    scale: str
    scores: dict[str, BayesianScore]


@dataclass
class ReplicateResult:
    """One complete CAT run."""
    replicate_id: int
    true_responses: dict[str, int]
    true_scores: dict[str, BayesianScore]
    steps: list[StepRecord]
    n_items: int


@dataclass
class SimulationSummary:
    """Aggregated results across replicates."""
    n_replicates: int
    max_items: int
    scales: list[str]
    # Per-scale arrays indexed by step
    mean_l2: dict[str, np.ndarray] = field(default_factory=dict)
    std_l2: dict[str, np.ndarray] = field(default_factory=dict)
    mean_kl: dict[str, np.ndarray] = field(default_factory=dict)
    std_kl: dict[str, np.ndarray] = field(default_factory=dict)
    mean_se: dict[str, np.ndarray] = field(default_factory=dict)
    std_se: dict[str, np.ndarray] = field(default_factory=dict)
    # Raw matrices: shape (S, max_steps), NaN-padded
    l2_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    kl_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    replicates: list[ReplicateResult] = field(default_factory=list)


class CATSimulator:
    """General-purpose CAT session simulator.

    Given a true ability level and IRT model, runs adaptive item selection
    and tracks how the posterior ability estimate converges to the ground
    truth (the posterior from scoring ALL items).
    """

    def __init__(
        self,
        model: MultivariateGRM,
        selector_class: type[ItemSelector],
        selector_kwargs: dict[str, Any],
        log_prior_fn: dict[str, Callable] | None = None,
        imputation_model: Any | None = None,
        max_items: int | None = None,
        seed: int | None = None,
    ):
        self.model = model
        self.selector_class = selector_class
        self.selector_kwargs = selector_kwargs
        self.log_prior_fn = log_prior_fn
        self.imputation_model = imputation_model
        self.max_items = max_items
        self.seed = seed

    def _make_scorer(self) -> BayesianScoring:
        return BayesianScoring(
            model=self.model,
            log_prior_fn=self.log_prior_fn,
            imputation_model=self.imputation_model,
        )

    def _make_selector(self, scorer: BayesianScoring) -> ItemSelector:
        return self.selector_class(scoring=scorer, **self.selector_kwargs)

    def _make_tracker(self) -> CatSessionTracker:
        scales = list(self.model.models.keys())
        return CatSessionTracker(session=CatSession(), scales=scales)

    def compute_true_scores(
        self, responses: dict[str, int]
    ) -> dict[str, BayesianScore]:
        scorer = self._make_scorer()
        return scorer.score_responses(responses)

    def run_single(
        self,
        theta: dict[str, float],
        responses: dict[str, int] | None = None,
    ) -> ReplicateResult:
        if responses is None:
            true_responses = self.model.sample(theta)
        else:
            true_responses = dict(responses)

        # Shift responses from 0-indexed to 1-indexed for scoring
        true_responses_scoring = {k: v + 1 for k, v in true_responses.items()}
        true_scores = self.compute_true_scores(true_responses_scoring)

        scorer = self._make_scorer()
        selector = self._make_selector(scorer)
        tracker = self._make_tracker()

        steps: list[StepRecord] = []
        max_possible = self.max_items if self.max_items is not None else len(true_responses)

        for step_idx in range(max_possible):
            item_dict = selector.next_item(tracker)
            if item_dict is None:
                break
            item_id = item_dict["item"]
            response = true_responses_scoring[item_id]
            tracker.responses[item_id] = response

            scores = scorer.score_responses(tracker.responses)
            scale = item_dict.get("scales", {})
            scale_name = list(scale.keys())[0] if scale else list(scores.keys())[0]
            for s in scores:
                tracker.scores[s] = scores[s].score
                tracker.errors[s] = scores[s].error

            steps.append(
                StepRecord(
                    step=step_idx,
                    item_id=item_id,
                    response=response,
                    scale=scale_name,
                    scores=dict(scores),
                )
            )

        return ReplicateResult(
            replicate_id=0,
            true_responses=true_responses_scoring,
            true_scores=true_scores,
            steps=steps,
            n_items=len(steps),
        )

    def simulate(
        self,
        theta: dict[str, float],
        n_replicates: int,
        responses: dict[str, int] | None = None,
    ) -> SimulationSummary:
        replicates: list[ReplicateResult] = []

        for r in range(n_replicates):
            if self.seed is not None:
                np.random.seed(self.seed + r)
            result = self.run_single(theta, responses)
            result.replicate_id = r
            replicates.append(result)

        scales = list(self.model.models.keys())
        max_steps = max(rep.n_items for rep in replicates) if replicates else 0

        l2_matrices: dict[str, np.ndarray] = {}
        kl_matrices: dict[str, np.ndarray] = {}

        for scale in scales:
            l2_mat = np.full((n_replicates, max_steps), np.nan)
            kl_mat = np.full((n_replicates, max_steps), np.nan)

            for r, rep in enumerate(replicates):
                for step in rep.steps:
                    if scale in step.scores and scale in rep.true_scores:
                        idx = step.step
                        l2_mat[r, idx] = self.l2_mean_discrepancy(
                            rep.true_scores[scale], step.scores[scale]
                        )
                        kl_mat[r, idx] = self.kl_divergence(
                            rep.true_scores[scale].density,
                            step.scores[scale].density,
                            rep.true_scores[scale].interpolation_pts,
                        )

            l2_matrices[scale] = l2_mat
            kl_matrices[scale] = kl_mat

        summary = SimulationSummary(
            n_replicates=n_replicates,
            max_items=max_steps,
            scales=scales,
            l2_matrix=l2_matrices,
            kl_matrix=kl_matrices,
            replicates=replicates,
        )

        for scale in scales:
            summary.mean_l2[scale] = np.nanmean(l2_matrices[scale], axis=0)
            summary.std_l2[scale] = np.nanstd(l2_matrices[scale], axis=0)
            summary.mean_kl[scale] = np.nanmean(kl_matrices[scale], axis=0)
            summary.std_kl[scale] = np.nanstd(kl_matrices[scale], axis=0)

            se_mat = np.full((n_replicates, max_steps), np.nan)
            for r, rep in enumerate(replicates):
                for step in rep.steps:
                    if scale in step.scores:
                        se_mat[r, step.step] = step.scores[scale].error
            summary.mean_se[scale] = np.nanmean(se_mat, axis=0)
            summary.std_se[scale] = np.nanstd(se_mat, axis=0)

        return summary

    @staticmethod
    def kl_divergence(
        p_density: np.ndarray,
        q_density: np.ndarray,
        grid: np.ndarray,
        eps: float = 1e-20,
    ) -> float:
        """KL(p || q) via trapezoidal integration."""
        p = np.asarray(p_density, dtype=float)
        q = np.asarray(q_density, dtype=float)
        integrand = p * np.log((p + eps) / (q + eps))
        return float(_trapz(y=integrand, x=grid))

    @staticmethod
    def l2_mean_discrepancy(
        true_score: BayesianScore, current_score: BayesianScore
    ) -> float:
        """|E[true] - E[current]|"""
        return float(abs(true_score.score - current_score.score))
