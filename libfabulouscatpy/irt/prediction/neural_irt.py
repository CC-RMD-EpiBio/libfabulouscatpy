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

"""Neural IRT Model implementation.

This module provides a Neural Item Response Theory model that uses neural networks
as item response functions. The model loads pre-fitted weights from CmdStan MCMC
output and supports prediction (log-likelihood computation and response sampling).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from libfabulouscatpy.irt.prediction.irt import IRTModel


def _softmax(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Numerically stable softmax function.

    Parameters
    ----------
    x : ndarray
        Input logits, shape (..., K)

    Returns
    -------
    ndarray
        Probabilities summing to 1 along last axis, same shape as input
    """
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class NeuralIRTModel(IRTModel):
    """Neural IRT model using neural networks as item response functions.

    This model uses a single hidden layer neural network for each item to compute
    response probabilities. The architecture follows:
        hidden = tanh(W1 * theta + b1)
        logits = W2 @ hidden + b2
        probs = softmax(logits)

    Parameters
    ----------
    theta : ndarray of shape (J,)
        Posterior mean ability estimates for J individuals
    W1 : ndarray of shape (I, H)
        Input-to-hidden weights for I items with H hidden units
    b1 : ndarray of shape (I,)
        Hidden layer biases for I items
    W2 : ndarray of shape (I, K, H)
        Hidden-to-output weights for I items, K categories, H hidden units
    b2 : ndarray of shape (I, K)
        Output layer biases for I items, K categories
    item_labels : list of str
        Labels for I items
    individual_labels : list of str
        Labels for J individuals
    K : int
        Number of ordinal response categories
    H : int
        Number of hidden units in the neural network

    Attributes
    ----------
    description : str
        Human-readable description of the model
    theta : ndarray
        Ability estimates
    item_labels : list
        Item label strings
    individual_labels : list
        Individual label strings
    K : int
        Number of response categories
    H : int
        Number of hidden units
    n_items : int
        Number of items
    """

    description = "Neural IRT model with single hidden layer"

    def __init__(
        self,
        theta: npt.NDArray[Any],
        W1: npt.NDArray[Any],
        b1: npt.NDArray[Any],
        W2: npt.NDArray[Any],
        b2: npt.NDArray[Any],
        item_labels: list[str],
        individual_labels: list[str],
        K: int,
        H: int,
    ) -> None:
        self.theta = np.asarray(theta)
        self.W1 = np.asarray(W1)
        self.b1 = np.asarray(b1)
        self.W2 = np.asarray(W2)
        self.b2 = np.asarray(b2)
        self.item_labels = list(item_labels)
        self.individual_labels = list(individual_labels)
        self.K = K
        self.H = H

        # Validate shapes
        n_items = len(item_labels)
        n_individuals = len(individual_labels)

        if self.theta.shape != (n_individuals,):
            raise ValueError(
                f"theta shape {self.theta.shape} doesn't match "
                f"n_individuals={n_individuals}"
            )
        if self.W1.shape != (n_items, H):
            raise ValueError(
                f"W1 shape {self.W1.shape} doesn't match (n_items={n_items}, H={H})"
            )
        if self.b1.shape != (n_items,):
            raise ValueError(
                f"b1 shape {self.b1.shape} doesn't match (n_items={n_items},)"
            )
        if self.W2.shape != (n_items, K, H):
            raise ValueError(
                f"W2 shape {self.W2.shape} doesn't match "
                f"(n_items={n_items}, K={K}, H={H})"
            )
        if self.b2.shape != (n_items, K):
            raise ValueError(
                f"b2 shape {self.b2.shape} doesn't match (n_items={n_items}, K={K})"
            )

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self.item_labels)

    @classmethod
    def from_cmdstan_output(
        cls,
        output_dir: str | Path,
        abilities_file: str = "model1_complete_case_abilities.csv",
        samples_dir: str = "model1_complete_case_samples",
    ) -> NeuralIRTModel:
        """Load a NeuralIRTModel from CmdStan MCMC output.

        Parameters
        ----------
        output_dir : str or Path
            Directory containing CmdStan output files
        abilities_file : str
            Name of CSV file with ability posterior means
        samples_dir : str
            Name of subdirectory containing MCMC sample CSV files

        Returns
        -------
        NeuralIRTModel
            Model loaded with posterior mean parameters
        """
        output_dir = Path(output_dir)

        # Load ID mappings
        with open(output_dir / "id_mappings.json") as f:
            id_mappings = json.load(f)

        # Sort by ID to get ordered lists
        climber_items = sorted(id_mappings["climber_ids"].items(), key=lambda x: x[1])
        boulder_items = sorted(id_mappings["boulder_ids"].items(), key=lambda x: x[1])

        individual_labels = [name for name, _ in climber_items]
        item_labels = [name for name, _ in boulder_items]

        n_individuals = len(individual_labels)
        n_items = len(item_labels)

        # Load abilities from summary CSV
        abilities_path = output_dir / abilities_file
        theta = np.zeros(n_individuals)

        with open(abilities_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["climber_id"]) - 1  # 1-indexed to 0-indexed
                theta[idx] = float(row["theta_mean"])

        # Find and load MCMC sample files
        samples_path = output_dir / samples_dir
        sample_files = sorted(samples_path.glob("*.csv"))

        if not sample_files:
            raise FileNotFoundError(f"No sample CSV files found in {samples_path}")

        # Parse first file to get column structure
        columns = None
        with open(sample_files[0]) as f:
            for line in f:
                # Skip comment lines (start with #)
                if line.startswith("#"):
                    continue
                # First non-comment line is the header
                columns = line.strip().split(",")
                break

        if columns is None:
            raise ValueError("Could not find header line in sample file")

        # Determine H and K from column names
        # W1.i.h -> H = max(h)
        # W2.i.k.h -> K = max(k)
        H = 0
        K = 0
        for col in columns:
            if col.startswith("W1."):
                parts = col.split(".")
                if len(parts) == 3:
                    H = max(H, int(parts[2]))
            elif col.startswith("W2."):
                parts = col.split(".")
                if len(parts) == 4:
                    K = max(K, int(parts[2]))

        # Build column index maps
        col_indices = {col: idx for idx, col in enumerate(columns)}

        # Initialize arrays for accumulating samples
        W1_sum = np.zeros((n_items, H))
        b1_sum = np.zeros(n_items)
        W2_sum = np.zeros((n_items, K, H))
        b2_sum = np.zeros((n_items, K))
        n_samples = 0

        # Load all sample files
        for sample_file in sample_files:
            # Read file and skip comments/header manually
            with open(sample_file) as f:
                lines = f.readlines()

            # Find data lines (skip comments and header)
            data_lines = []
            header_found = False
            for line in lines:
                if line.startswith("#"):
                    continue
                if not header_found:
                    # This is the header line, skip it
                    header_found = True
                    continue
                # This is a data line
                data_lines.append(line.strip())

            # Parse data lines
            for line in data_lines:
                row = line.split(",")
                n_samples += 1

                # Extract W1
                for i in range(n_items):
                    for h in range(H):
                        col_name = f"W1.{i + 1}.{h + 1}"
                        if col_name in col_indices:
                            W1_sum[i, h] += float(row[col_indices[col_name]])

                # Extract b1
                for i in range(n_items):
                    col_name = f"b1.{i + 1}"
                    if col_name in col_indices:
                        b1_sum[i] += float(row[col_indices[col_name]])

                # Extract W2
                for i in range(n_items):
                    for k in range(K):
                        for h in range(H):
                            col_name = f"W2.{i + 1}.{k + 1}.{h + 1}"
                            if col_name in col_indices:
                                W2_sum[i, k, h] += float(row[col_indices[col_name]])

                # Extract b2
                for i in range(n_items):
                    for k in range(K):
                        col_name = f"b2.{i + 1}.{k + 1}"
                        if col_name in col_indices:
                            b2_sum[i, k] += float(row[col_indices[col_name]])

        # Compute posterior means
        W1 = W1_sum / n_samples
        b1 = b1_sum / n_samples
        W2 = W2_sum / n_samples
        b2 = b2_sum / n_samples

        return cls(
            theta=theta,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            item_labels=item_labels,
            individual_labels=individual_labels,
            K=K,
            H=H,
        )

    def _nn_item_response_vectorized(
        self,
        theta: npt.NDArray[Any],
        W1: npt.NDArray[Any],
        b1: float,
        W2: npt.NDArray[Any],
        b2: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Compute response probabilities using neural network (always returns 2D).

        Parameters
        ----------
        theta : ndarray
            Ability values, shape (M,)
        W1 : ndarray of shape (H,)
            Input-to-hidden weights
        b1 : float
            Hidden layer bias
        W2 : ndarray of shape (K, H)
            Hidden-to-output weights
        b2 : ndarray of shape (K,)
            Output layer biases

        Returns
        -------
        ndarray of shape (M, K)
            Response probabilities
        """
        theta = np.atleast_1d(theta)

        # Hidden layer: h = tanh(W1 * theta + b1)
        # theta: (M,), W1: (H,) -> need (M, H)
        hidden = np.tanh(np.outer(theta, W1) + b1)  # (M, H)

        # Output layer: logits = W2 @ h + b2
        # hidden: (M, H), W2: (K, H), b2: (K,)
        logits = hidden @ W2.T + b2  # (M, K)

        # Softmax to get probabilities
        return _softmax(logits)  # (M, K)

    def _nn_item_response(
        self,
        theta: npt.NDArray[Any],
        W1: npt.NDArray[Any],
        b1: float,
        W2: npt.NDArray[Any],
        b2: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Compute response probabilities using neural network.

        Parameters
        ----------
        theta : ndarray
            Ability values, scalar or shape (M,)
        W1 : ndarray of shape (H,)
            Input-to-hidden weights
        b1 : float
            Hidden layer bias
        W2 : ndarray of shape (K, H)
            Hidden-to-output weights
        b2 : ndarray of shape (K,)
            Output layer biases

        Returns
        -------
        ndarray
            Response probabilities, shape (K,) for scalar theta or (M, K) for
            vector theta
        """
        theta = np.atleast_1d(theta)
        scalar_input = theta.shape == (1,)

        probs = self._nn_item_response_vectorized(theta, W1, b1, W2, b2)

        if scalar_input:
            return probs[0]  # (K,)
        return probs  # (M, K)

    def item_probabilities(
        self,
        theta: npt.NDArray[Any],
        item_idx: int | None = None,
        item_label: str | None = None,
    ) -> npt.NDArray[Any]:
        """Compute response probabilities for an item.

        Parameters
        ----------
        theta : ndarray
            Ability values, scalar or shape (M,)
        item_idx : int, optional
            Index of the item (0-based)
        item_label : str, optional
            Label of the item (alternative to item_idx)

        Returns
        -------
        ndarray
            Response probabilities, shape (K,) for scalar theta or (M, K) for
            vector theta
        """
        if item_idx is None and item_label is None:
            raise ValueError("Must specify either item_idx or item_label")

        if item_label is not None:
            item_idx = self.item_labels.index(item_label)

        return self._nn_item_response(
            theta,
            self.W1[item_idx],
            self.b1[item_idx],
            self.W2[item_idx],
            self.b2[item_idx],
        )

    def log_likelihood(
        self,
        theta: npt.NDArray[Any],
        observed_only: bool = True,
        responses: dict[str, int] | None = None,
    ) -> npt.NDArray[Any]:
        """Compute log-likelihood of responses given ability.

        Parameters
        ----------
        theta : ndarray of shape (M,)
            Ability values (grid points)
        observed_only : bool
            If True, only use items in responses dict
        responses : dict
            Mapping of item_label -> observed_category (0 to K-1)

        Returns
        -------
        ndarray of shape (M,)
            Log-likelihood at each theta value
        """
        theta = np.atleast_1d(theta)
        M = len(theta)

        if responses is None:
            raise ValueError("responses dict is required")

        log_lik = np.zeros(M)

        for item_label, response in responses.items():
            if item_label not in self.item_labels:
                continue

            item_idx = self.item_labels.index(item_label)
            # Always get (M, K) shape by calling _nn_item_response directly
            # and not collapsing for scalar input
            probs = self._nn_item_response_vectorized(
                theta,
                self.W1[item_idx],
                self.b1[item_idx],
                self.W2[item_idx],
                self.b2[item_idx],
            )  # (M, K)
            # Select probability of observed response
            p_observed = probs[:, response]  # (M,)
            log_lik += np.log(p_observed + 1e-20)

        return log_lik

    def sample(self, theta: npt.NDArray[Any]) -> dict[str, int]:
        """Sample responses for all items given ability.

        Parameters
        ----------
        theta : float or ndarray
            Ability value (scalar or 1-element array)

        Returns
        -------
        dict
            Mapping of item_label -> sampled_category (0 to K-1)
        """
        theta = np.atleast_1d(theta)
        if len(theta) != 1:
            raise ValueError("sample() expects a single theta value")

        theta_val = theta[0]
        responses = {}

        for i, item_label in enumerate(self.item_labels):
            probs = self._nn_item_response(
                theta_val,
                self.W1[i],
                self.b1[i],
                self.W2[i],
                self.b2[i],
            )
            sampled = np.random.choice(self.K, p=probs)
            responses[item_label] = int(sampled)

        return responses
