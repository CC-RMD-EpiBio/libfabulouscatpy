"""Tests for CATSimulator."""

import numpy as np
import pytest

from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.cat.itemselectors.bayesianfisher import (
    BayesianFisherItemSelector,
)
from libfabulouscatpy.irt.prediction.grm import MultivariateGRM
from tools.simulation import CATSimulator, ReplicateResult, SimulationSummary


def _make_grm(n_items=8, n_categories=5, scale_name="test_scale"):
    """Build a small MultivariateGRM for testing."""
    np.random.seed(42)
    items = []
    for i in range(n_items):
        difficulties = sorted(np.random.randn(n_categories - 1).tolist())
        items.append(
            {
                "item": f"item_{i}",
                "scales": {
                    scale_name: {
                        "discrimination": float(np.random.uniform(0.5, 2.0)),
                        "difficulties": difficulties,
                    }
                },
            }
        )

    class FakeItemDB:
        def __init__(self, item_list):
            self.items = item_list

    class FakeScaleDB:
        def __init__(self):
            self.scales = {scale_name: {}}

    return MultivariateGRM(
        itemdb=FakeItemDB(items),
        scaledb=FakeScaleDB(),
        interpolation_pts=np.arange(-4.0, 4.0, step=0.1),
    ), items


class TestCATSimulator:
    @pytest.fixture
    def simulator(self):
        model, items = _make_grm()
        scales = {"test_scale": {}}
        return CATSimulator(
            model=model,
            selector_class=BayesianFisherItemSelector,
            selector_kwargs={
                "items": items,
                "scales": scales,
                "model": model,
                "temperature": 0.01,
                "max_responses": 8,
                "min_responses": 1,
                "precision_limit": 0.3,
                "randomize_scales": False,
            },
            max_items=6,
            seed=123,
        )

    def test_run_single_returns_replicate_result(self, simulator):
        theta = {"test_scale": np.array([0.5])}
        result = simulator.run_single(theta)
        assert isinstance(result, ReplicateResult)
        assert result.n_items > 0
        assert len(result.steps) == result.n_items

    def test_run_single_true_scores_are_finite(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        result = simulator.run_single(theta)
        for scale, score in result.true_scores.items():
            assert np.isfinite(score.score)
            assert np.isfinite(score.error)
            assert score.error > 0

    def test_run_single_step_scores_are_finite(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        result = simulator.run_single(theta)
        for step in result.steps:
            for scale, score in step.scores.items():
                assert np.isfinite(score.score), f"Non-finite score at step {step.step}"
                assert np.isfinite(score.error), f"Non-finite error at step {step.step}"

    def test_simulate_returns_summary(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        summary = simulator.simulate(theta, n_replicates=3)
        assert isinstance(summary, SimulationSummary)
        assert summary.n_replicates == 3
        assert len(summary.replicates) == 3
        assert "test_scale" in summary.scales

    def test_simulate_summary_shapes(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        summary = simulator.simulate(theta, n_replicates=3)
        max_steps = summary.max_items
        for scale in summary.scales:
            assert summary.mean_l2[scale].shape == (max_steps,)
            assert summary.std_l2[scale].shape == (max_steps,)
            assert summary.mean_kl[scale].shape == (max_steps,)
            assert summary.std_kl[scale].shape == (max_steps,)
            assert summary.mean_se[scale].shape == (max_steps,)
            assert summary.std_se[scale].shape == (max_steps,)
            assert summary.l2_matrix[scale].shape == (3, max_steps)
            assert summary.kl_matrix[scale].shape == (3, max_steps)

    def test_l2_is_nonnegative(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        summary = simulator.simulate(theta, n_replicates=3)
        for scale in summary.scales:
            finite_vals = summary.l2_matrix[scale][
                np.isfinite(summary.l2_matrix[scale])
            ]
            assert np.all(finite_vals >= 0)

    def test_kl_is_nonnegative(self, simulator):
        theta = {"test_scale": np.array([0.0])}
        summary = simulator.simulate(theta, n_replicates=3)
        for scale in summary.scales:
            finite_vals = summary.kl_matrix[scale][
                np.isfinite(summary.kl_matrix[scale])
            ]
            assert np.all(finite_vals >= -1e-10)

    def test_all_replicates_have_finite_scores(self, simulator):
        theta = {"test_scale": np.array([0.5])}
        summary = simulator.simulate(theta, n_replicates=5)
        for rep in summary.replicates:
            for scale, score in rep.true_scores.items():
                assert np.isfinite(score.score)
                assert np.isfinite(score.error)
            for step in rep.steps:
                for scale, score in step.scores.items():
                    assert np.isfinite(score.score)
                    assert np.isfinite(score.error)

    def test_with_provided_responses(self, simulator):
        """Test that providing responses directly works."""
        theta = {"test_scale": np.array([0.0])}
        # Sample once to get valid responses
        np.random.seed(42)
        responses = simulator.model.sample(theta)
        result = simulator.run_single(theta, responses=responses)
        assert isinstance(result, ReplicateResult)
        assert result.n_items > 0


class TestDiscrepancyMetrics:
    def test_kl_divergence_identical_distributions(self):
        grid = np.linspace(-3, 3, 100)
        p = np.exp(-0.5 * grid**2)
        p /= _trapz(p, grid)
        kl = CATSimulator.kl_divergence(p, p, grid)
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_kl_divergence_different_distributions(self):
        grid = np.linspace(-5, 5, 200)
        p = np.exp(-0.5 * grid**2)
        p /= _trapz(p, grid)
        q = np.exp(-0.5 * (grid - 1)**2)
        q /= _trapz(q, grid)
        kl = CATSimulator.kl_divergence(p, q, grid)
        assert kl > 0

    def test_l2_mean_discrepancy_identical(self):
        grid = np.linspace(-3, 3, 100)
        density = np.exp(-0.5 * grid**2)
        density /= _trapz(density, grid)
        from libfabulouscatpy.irt.scoring.bayesian import BayesianScore
        s1 = BayesianScore("s", "s", density, grid)
        s2 = BayesianScore("s", "s", density, grid)
        assert np.isclose(CATSimulator.l2_mean_discrepancy(s1, s2), 0.0, atol=1e-10)

    def test_l2_mean_discrepancy_shifted(self):
        grid = np.linspace(-5, 5, 200)
        d1 = np.exp(-0.5 * grid**2)
        d1 /= _trapz(d1, grid)
        d2 = np.exp(-0.5 * (grid - 1)**2)
        d2 /= _trapz(d2, grid)
        from libfabulouscatpy.irt.scoring.bayesian import BayesianScore
        s1 = BayesianScore("s", "s", d1, grid)
        s2 = BayesianScore("s", "s", d2, grid)
        l2 = CATSimulator.l2_mean_discrepancy(s1, s2)
        assert l2 > 0
        assert np.isclose(l2, 1.0, atol=0.1)
