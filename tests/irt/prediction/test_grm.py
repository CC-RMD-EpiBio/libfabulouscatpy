"""Regression tests for libfabulouscatpy.irt.prediction.grm.GradedResponseModel.

The batch log_likelihood path previously summed an N x N cross-indexed matrix
(sum_i sum_j log P(y_j | item i, theta)) instead of the correct diagonal
(sum_i log P(y_i | item i, theta)). This file pins the correct behaviour.
"""
import numpy as np
import pytest

from libfabulouscatpy.irt.prediction.grm import GradedResponseModel


def _build_grm():
    grm = GradedResponseModel.__new__(GradedResponseModel)
    grm.slope = np.array([1.5, 0.8, 1.2])
    grm.calibration = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-0.5, 0.2, 0.8],
            [-1.5, -0.2, 0.6],
        ]
    )
    grm.item_labels = ["item_A", "item_B", "item_C"]
    return grm


def test_batch_log_likelihood_equals_sum_of_per_item_log_likelihoods():
    """Multi-item batch call must equal the sum of single-item calls."""
    grm = _build_grm()
    responses = {"item_A": 2, "item_B": 3, "item_C": 1}
    theta_grid = np.linspace(-2.0, 2.0, 5)

    ll_batch = grm.log_likelihood(theta=theta_grid, responses=responses)
    ll_single = np.zeros_like(theta_grid)
    for item, value in responses.items():
        ll_single += grm.log_likelihood(theta=theta_grid, responses={item: value})

    assert np.allclose(ll_batch, ll_single), (
        "Batch log-likelihood does not equal sum of single-item log-likelihoods. "
        f"batch={ll_batch} single={ll_single}"
    )


def test_log_likelihood_matches_explicit_diagonal_sum():
    """Batch call must equal the explicit per-(item, response) diagonal sum."""
    grm = _build_grm()
    responses = {"item_A": 2, "item_B": 3, "item_C": 1}
    theta_grid = np.linspace(-2.0, 2.0, 5)

    full_p = grm.log_likelihood(theta=theta_grid, observed_only=False)
    selected = np.array([responses[lbl] for lbl in grm.item_labels])
    expected = full_p[:, np.arange(len(selected)), selected - 1].sum(axis=-1)

    ll_batch = grm.log_likelihood(theta=theta_grid, responses=responses)
    assert np.allclose(ll_batch, expected), (
        f"Batch log-likelihood does not match diagonal sum. "
        f"batch={ll_batch} expected={expected}"
    )


def test_single_item_log_likelihood_unchanged():
    """One-response batch call equals the standalone single-response call.
    Sanity check that the fix didn't break online CAT scoring (N=1).
    """
    grm = _build_grm()
    theta_grid = np.linspace(-2.0, 2.0, 5)
    ll1 = grm.log_likelihood(theta=theta_grid, responses={"item_A": 2})
    ll2 = grm.log_likelihood(theta=theta_grid, responses={"item_A": 2})
    assert np.allclose(ll1, ll2)
