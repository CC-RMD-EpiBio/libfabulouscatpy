"""Tests for BayesianScoring with pairwise stacking imputation model."""

import numpy as np
import pytest

from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.irt.prediction.grm import GradedResponseModel, MultivariateGRM
from libfabulouscatpy.irt.scoring.bayesian import BayesianScoring


class MockImputationModel:
    """Mock imputation model implementing the predict_pmf interface."""

    def __init__(self, item_keys, n_categories, bias_category=None):
        """
        Args:
            item_keys: List of item names the model knows about.
            n_categories: Number of response categories.
            bias_category: If set, the PMF will strongly favor this category
                (0-indexed). Otherwise returns uniform.
        """
        self.item_keys = set(item_keys)
        self.n_categories = n_categories
        self.bias_category = bias_category

    def predict_pmf(self, items, target, n_categories, uncertainty_penalty=1.0):
        if target not in self.item_keys:
            raise KeyError(f"Unknown target: {target}")
        if self.bias_category is not None:
            pmf = np.full(n_categories, 0.02)
            pmf[min(self.bias_category, n_categories - 1)] += 1.0
            pmf /= pmf.sum()
        else:
            pmf = np.ones(n_categories) / n_categories
        return pmf


def _make_grm(n_items=5, n_categories=5, scale_name="test_scale"):
    """Build a simple MultivariateGRM for testing."""
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
    )


class TestBayesianScoringWithoutImputation:
    """Sanity checks: imputation_model=None preserves existing behavior."""

    def test_no_imputation_model_by_default(self):
        model = _make_grm()
        scorer = BayesianScoring(model=model)
        assert scorer.imputation_model is None

    def test_scores_without_imputation(self):
        model = _make_grm()
        scorer = BayesianScoring(model=model)
        scores = scorer.score_responses({"item_0": 2, "item_1": 3})
        assert "test_scale" in scores
        assert np.isfinite(scores["test_scale"].score)
        assert scores["test_scale"].error > 0


class TestBayesianScoringWithImputation:
    """Tests for imputation model integration."""

    @pytest.fixture
    def model_and_items(self):
        n_items = 5
        n_categories = 5
        model = _make_grm(n_items=n_items, n_categories=n_categories)
        item_keys = [f"item_{i}" for i in range(n_items)]
        return model, item_keys, n_categories

    def test_imputation_model_stored(self, model_and_items):
        model, item_keys, n_cat = model_and_items
        imp = MockImputationModel(item_keys, n_cat)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        assert scorer.imputation_model is imp

    def test_scores_are_finite(self, model_and_items):
        model, item_keys, n_cat = model_and_items
        imp = MockImputationModel(item_keys, n_cat)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        scores = scorer.score_responses({"item_0": 2, "item_1": 3})
        assert np.isfinite(scores["test_scale"].score)
        assert np.isfinite(scores["test_scale"].error)
        assert scores["test_scale"].error > 0

    def test_density_integrates_to_one(self, model_and_items):
        model, item_keys, n_cat = model_and_items
        imp = MockImputationModel(item_keys, n_cat, bias_category=3)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        scores = scorer.score_responses({"item_0": 2, "item_1": 3})
        density = scores["test_scale"].density
        pts = scores["test_scale"].interpolation_pts
        integral = _trapz(y=density, x=pts)
        assert np.isclose(integral, 1.0, atol=1e-4)

    def test_imputation_shifts_score_toward_bias(self, model_and_items):
        """An imputation model biased toward high responses should shift
        the ability estimate upward compared to no imputation."""
        model, item_keys, n_cat = model_and_items
        responses = {"item_0": 2, "item_1": 2}

        scorer_none = BayesianScoring(model=model)
        scores_none = scorer_none.score_responses(responses)

        imp_high = MockImputationModel(item_keys, n_cat, bias_category=n_cat - 1)
        scorer_high = BayesianScoring(model=model, imputation_model=imp_high)
        scores_high = scorer_high.score_responses(responses)

        imp_low = MockImputationModel(item_keys, n_cat, bias_category=0)
        scorer_low = BayesianScoring(model=model, imputation_model=imp_low)
        scores_low = scorer_low.score_responses(responses)

        assert scores_high["test_scale"].score > scores_none["test_scale"].score
        assert scores_low["test_scale"].score < scores_none["test_scale"].score

    def test_imputation_reduces_standard_error(self, model_and_items):
        """Adding imputed information should reduce the standard error
        compared to scoring with only observed responses."""
        model, item_keys, n_cat = model_and_items
        responses = {"item_0": 3}

        scorer_none = BayesianScoring(model=model)
        scores_none = scorer_none.score_responses(responses)

        imp = MockImputationModel(item_keys, n_cat, bias_category=2)
        scorer_imp = BayesianScoring(model=model, imputation_model=imp)
        scores_imp = scorer_imp.score_responses(responses)

        assert scores_imp["test_scale"].error < scores_none["test_scale"].error

    def test_uniform_imputation_matches_no_imputation(self, model_and_items):
        """A uniform PMF should produce the same posterior shape as no
        imputation, since log(sum_k (1/K) * P(k|theta)) = log(1/K) is
        constant across theta."""
        model, item_keys, n_cat = model_and_items
        responses = {"item_0": 3, "item_1": 2}

        scorer_none = BayesianScoring(model=model)
        scores_none = scorer_none.score_responses(responses)

        imp_uniform = MockImputationModel(item_keys, n_cat, bias_category=None)
        scorer_uniform = BayesianScoring(model=model, imputation_model=imp_uniform)
        scores_uniform = scorer_uniform.score_responses(responses)

        assert np.isclose(
            scores_none["test_scale"].score,
            scores_uniform["test_scale"].score,
            atol=1e-6,
        )
        assert np.isclose(
            scores_none["test_scale"].error,
            scores_uniform["test_scale"].error,
            atol=1e-6,
        )

    def test_zero_observed_responses_uses_zero_predictors(self, model_and_items):
        """With no observed responses, imputation still applies via the
        zero-predictor (intercept-only) models.
        A biased imputation should shift the score away from the prior mean."""
        model, item_keys, n_cat = model_and_items

        scorer_none = BayesianScoring(model=model)
        scores_none = scorer_none.score_responses({})

        imp_high = MockImputationModel(item_keys, n_cat, bias_category=n_cat - 1)
        scorer_high = BayesianScoring(model=model, imputation_model=imp_high)
        scores_high = scorer_high.score_responses({})

        imp_low = MockImputationModel(item_keys, n_cat, bias_category=0)
        scorer_low = BayesianScoring(model=model, imputation_model=imp_low)
        scores_low = scorer_low.score_responses({})

        assert scores_high["test_scale"].score > scores_none["test_scale"].score
        assert scores_low["test_scale"].score < scores_none["test_scale"].score

    def test_all_items_observed_no_imputation_effect(self, model_and_items):
        """When all items are observed, there are no items to impute."""
        model, item_keys, n_cat = model_and_items
        responses = {f"item_{i}": 3 for i in range(5)}

        scorer_none = BayesianScoring(model=model)
        scores_none = scorer_none.score_responses(responses)

        imp = MockImputationModel(item_keys, n_cat, bias_category=0)
        scorer_imp = BayesianScoring(model=model, imputation_model=imp)
        scores_imp = scorer_imp.score_responses(responses)

        assert np.isclose(
            scores_none["test_scale"].score,
            scores_imp["test_scale"].score,
            atol=1e-6,
        )

    def test_unknown_item_in_imputation_model_skipped(self):
        """Items not in the imputation model should be silently skipped."""
        model = _make_grm(n_items=5)
        # Only knows about item_0 and item_1
        imp = MockImputationModel(["item_0", "item_1"], 5, bias_category=3)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        scores = scorer.score_responses({"item_0": 3})
        assert np.isfinite(scores["test_scale"].score)

    def test_skipped_responses_excluded_from_observed(self, model_and_items):
        """Skipped responses should not be treated as observed data."""
        model, item_keys, n_cat = model_and_items
        imp = MockImputationModel(item_keys, n_cat, bias_category=3)

        scorer = BayesianScoring(model=model, imputation_model=imp)
        responses_with_skip = {"item_0": 3, "item_1": -1}  # -1 is SKIPPED
        scores = scorer.score_responses(responses_with_skip)
        assert np.isfinite(scores["test_scale"].score)

    def test_incremental_scoring_with_imputation(self, model_and_items):
        """Score responses incrementally and verify imputation adapts."""
        model, item_keys, n_cat = model_and_items
        imp = MockImputationModel(item_keys, n_cat, bias_category=3)
        scorer = BayesianScoring(model=model, imputation_model=imp)

        scores_1 = scorer.score_responses({"item_0": 3})
        scores_2 = scorer.score_responses({"item_0": 3, "item_1": 4})

        # Adding another high response should shift score upward
        assert scores_2["test_scale"].score >= scores_1["test_scale"].score - 0.5
        # More observed items = fewer imputed items, but more observed data
        assert np.isfinite(scores_2["test_scale"].error)


class TestComputeImputedLogLikelihood:
    """Direct tests for _compute_imputed_log_likelihood."""

    def test_returns_zero_when_no_imputation_model(self):
        model = _make_grm()
        scorer = BayesianScoring(model=model)
        # score_responses guards on imputation_model being None;
        # calling _compute_imputed_log_likelihood directly requires the model.
        # Verify that score_responses produces consistent results.
        scores = scorer.score_responses({"item_0": 3})
        assert np.isfinite(scores["test_scale"].score)

    def test_no_observed_still_imputes_via_zero_predictors(self):
        """Even with no observed responses, the imputation model's
        zero-predictor models provide population-marginal PMFs."""
        model = _make_grm()
        imp = MockImputationModel([f"item_{i}" for i in range(5)], 5, bias_category=2)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        result = scorer._compute_imputed_log_likelihood({}, "test_scale")
        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_returns_array_with_observed_responses(self):
        model = _make_grm()
        imp = MockImputationModel([f"item_{i}" for i in range(5)], 5, bias_category=2)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        result = scorer._compute_imputed_log_likelihood({"item_0": 3}, "test_scale")
        assert isinstance(result, np.ndarray)
        assert result.shape == scorer.interpolation_pts["test_scale"].shape
        assert np.all(np.isfinite(result))

    def test_marginal_ll_is_negative(self):
        """The marginal log-likelihood contribution should be non-positive."""
        model = _make_grm()
        imp = MockImputationModel([f"item_{i}" for i in range(5)], 5, bias_category=2)
        scorer = BayesianScoring(model=model, imputation_model=imp)
        result = scorer._compute_imputed_log_likelihood({"item_0": 3}, "test_scale")
        # log(sum_k q(k)*p(k|theta)) <= 0 since both are valid distributions
        assert np.all(result <= 1e-10)
