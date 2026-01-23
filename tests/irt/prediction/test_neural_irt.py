"""Tests for NeuralIRTModel."""

import numpy as np
import pytest
from pathlib import Path

from libfabulouscatpy.irt.prediction.neural_irt import NeuralIRTModel, _softmax


class TestSoftmax:
    """Tests for the softmax helper function."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _softmax(x)
        assert np.isclose(result.sum(), 1.0)

    def test_softmax_non_negative(self):
        """Softmax output should be non-negative."""
        x = np.array([-5.0, -1.0, 0.0, 1.0])
        result = _softmax(x)
        assert np.all(result >= 0)

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values without overflow."""
        x = np.array([1000.0, 1001.0, 1002.0, 1003.0])
        result = _softmax(x)
        assert np.isclose(result.sum(), 1.0)
        assert np.all(np.isfinite(result))

    def test_softmax_2d(self):
        """Softmax should work on 2D arrays along last axis."""
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = _softmax(x)
        assert result.shape == (2, 3)
        assert np.allclose(result.sum(axis=1), [1.0, 1.0])


class TestNeuralIRTModelInit:
    """Tests for NeuralIRTModel initialization."""

    @pytest.fixture
    def synthetic_model_params(self):
        """Create synthetic model parameters for testing."""
        n_items = 3
        n_individuals = 5
        K = 4
        H = 2

        np.random.seed(42)
        theta = np.random.randn(n_individuals)
        W1 = np.random.randn(n_items, H) * 0.1
        b1 = np.random.randn(n_items) * 0.1
        W2 = np.random.randn(n_items, K, H) * 0.1
        b2 = np.random.randn(n_items, K) * 0.1
        item_labels = ["item_1", "item_2", "item_3"]
        individual_labels = ["person_1", "person_2", "person_3", "person_4", "person_5"]

        return {
            "theta": theta,
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "item_labels": item_labels,
            "individual_labels": individual_labels,
            "K": K,
            "H": H,
        }

    def test_init_stores_parameters(self, synthetic_model_params):
        """Model should store all parameters."""
        model = NeuralIRTModel(**synthetic_model_params)

        assert np.array_equal(model.theta, synthetic_model_params["theta"])
        assert np.array_equal(model.W1, synthetic_model_params["W1"])
        assert np.array_equal(model.b1, synthetic_model_params["b1"])
        assert np.array_equal(model.W2, synthetic_model_params["W2"])
        assert np.array_equal(model.b2, synthetic_model_params["b2"])
        assert model.item_labels == synthetic_model_params["item_labels"]
        assert model.individual_labels == synthetic_model_params["individual_labels"]
        assert model.K == synthetic_model_params["K"]
        assert model.H == synthetic_model_params["H"]

    def test_n_items_property(self, synthetic_model_params):
        """n_items should return correct count."""
        model = NeuralIRTModel(**synthetic_model_params)
        assert model.n_items == 3

    def test_description_attribute(self, synthetic_model_params):
        """Model should have description attribute."""
        model = NeuralIRTModel(**synthetic_model_params)
        assert isinstance(model.description, str)
        assert len(model.description) > 0

    def test_init_validates_theta_shape(self, synthetic_model_params):
        """Init should raise error for wrong theta shape."""
        params = synthetic_model_params.copy()
        params["theta"] = np.random.randn(10)  # Wrong size
        with pytest.raises(ValueError, match="theta shape"):
            NeuralIRTModel(**params)

    def test_init_validates_W1_shape(self, synthetic_model_params):
        """Init should raise error for wrong W1 shape."""
        params = synthetic_model_params.copy()
        params["W1"] = np.random.randn(10, 2)  # Wrong size
        with pytest.raises(ValueError, match="W1 shape"):
            NeuralIRTModel(**params)


class TestNNItemResponse:
    """Tests for the neural network item response function."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        n_items = 2
        n_individuals = 3
        K = 4
        H = 2

        np.random.seed(123)
        return NeuralIRTModel(
            theta=np.random.randn(n_individuals),
            W1=np.random.randn(n_items, H) * 0.5,
            b1=np.random.randn(n_items) * 0.5,
            W2=np.random.randn(n_items, K, H) * 0.5,
            b2=np.random.randn(n_items, K) * 0.5,
            item_labels=["item_a", "item_b"],
            individual_labels=["p1", "p2", "p3"],
            K=K,
            H=H,
        )

    def test_scalar_theta_returns_K_probs(self, model):
        """Single theta value should return K probabilities."""
        probs = model._nn_item_response(
            0.0, model.W1[0], model.b1[0], model.W2[0], model.b2[0]
        )
        assert probs.shape == (model.K,)

    def test_vector_theta_returns_M_by_K(self, model):
        """Multiple theta values should return (M, K) probabilities."""
        theta = np.array([-1.0, 0.0, 1.0, 2.0])
        probs = model._nn_item_response(
            theta, model.W1[0], model.b1[0], model.W2[0], model.b2[0]
        )
        assert probs.shape == (4, model.K)

    def test_probs_sum_to_one(self, model):
        """Probabilities should sum to 1."""
        theta = np.linspace(-3, 3, 10)
        probs = model._nn_item_response(
            theta, model.W1[0], model.b1[0], model.W2[0], model.b2[0]
        )
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_probs_non_negative(self, model):
        """All probabilities should be non-negative."""
        theta = np.linspace(-5, 5, 20)
        probs = model._nn_item_response(
            theta, model.W1[0], model.b1[0], model.W2[0], model.b2[0]
        )
        assert np.all(probs >= 0)

    def test_known_weights_produce_expected_output(self):
        """Test with known weights for deterministic verification."""
        # Simple case: identity-like behavior
        K, H = 2, 1
        W1 = np.array([1.0])  # (H,)
        b1 = 0.0
        W2 = np.array([[1.0], [-1.0]])  # (K, H)
        b2 = np.array([0.0, 0.0])  # (K,)

        model = NeuralIRTModel(
            theta=np.array([0.0]),
            W1=np.array([[1.0]]),
            b1=np.array([0.0]),
            W2=np.array([[[1.0], [-1.0]]]),
            b2=np.array([[0.0, 0.0]]),
            item_labels=["test_item"],
            individual_labels=["test_person"],
            K=K,
            H=H,
        )

        # At theta=0: hidden = tanh(0) = 0
        # logits = [0, 0], probs = [0.5, 0.5]
        probs = model._nn_item_response(0.0, W1, b1, W2, b2)
        assert np.allclose(probs, [0.5, 0.5])

        # At theta=2: hidden = tanh(2) ≈ 0.964
        # logits ≈ [0.964, -0.964]
        # probs ≈ softmax([0.964, -0.964])
        probs = model._nn_item_response(2.0, W1, b1, W2, b2)
        assert probs[0] > probs[1]  # First category more likely


class TestLogLikelihood:
    """Tests for log-likelihood computation."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        n_items = 3
        n_individuals = 2
        K = 4
        H = 3

        np.random.seed(456)
        return NeuralIRTModel(
            theta=np.random.randn(n_individuals),
            W1=np.random.randn(n_items, H) * 0.3,
            b1=np.random.randn(n_items) * 0.3,
            W2=np.random.randn(n_items, K, H) * 0.3,
            b2=np.random.randn(n_items, K) * 0.3,
            item_labels=["item_x", "item_y", "item_z"],
            individual_labels=["alice", "bob"],
            K=K,
            H=H,
        )

    def test_log_likelihood_shape(self, model):
        """Log-likelihood should have shape (M,) for M grid points."""
        theta = np.linspace(-3, 3, 50)
        responses = {"item_x": 1, "item_y": 2}
        ll = model.log_likelihood(theta, responses=responses)
        assert ll.shape == (50,)

    def test_log_likelihood_non_positive(self, model):
        """Log-likelihood should be <= 0."""
        theta = np.linspace(-3, 3, 20)
        responses = {"item_x": 0, "item_y": 1, "item_z": 2}
        ll = model.log_likelihood(theta, responses=responses)
        assert np.all(ll <= 0)

    def test_log_likelihood_requires_responses(self, model):
        """Log-likelihood should raise error without responses."""
        theta = np.array([0.0])
        with pytest.raises(ValueError, match="responses"):
            model.log_likelihood(theta)

    def test_log_likelihood_ignores_unknown_items(self, model):
        """Unknown item labels should be ignored."""
        theta = np.array([0.0])
        responses = {"item_x": 1, "unknown_item": 2}
        # Should not raise error
        ll = model.log_likelihood(theta, responses=responses)
        assert ll.shape == (1,)

    def test_log_likelihood_more_items_lower(self, model):
        """More observed items should generally give lower log-likelihood."""
        theta = np.array([0.0])
        responses_1 = {"item_x": 1}
        responses_2 = {"item_x": 1, "item_y": 1, "item_z": 1}

        ll_1 = model.log_likelihood(theta, responses=responses_1)
        ll_2 = model.log_likelihood(theta, responses=responses_2)

        # More items = more terms in sum = lower (more negative) log-likelihood
        assert ll_2[0] <= ll_1[0]


class TestSample:
    """Tests for response sampling."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        n_items = 5
        n_individuals = 3
        K = 4
        H = 2

        np.random.seed(789)
        return NeuralIRTModel(
            theta=np.random.randn(n_individuals),
            W1=np.random.randn(n_items, H) * 0.3,
            b1=np.random.randn(n_items) * 0.3,
            W2=np.random.randn(n_items, K, H) * 0.3,
            b2=np.random.randn(n_items, K) * 0.3,
            item_labels=["q1", "q2", "q3", "q4", "q5"],
            individual_labels=["x", "y", "z"],
            K=K,
            H=H,
        )

    def test_sample_returns_all_items(self, model):
        """Sample should return dict with all item labels."""
        responses = model.sample(0.0)
        assert set(responses.keys()) == set(model.item_labels)

    def test_sample_values_in_range(self, model):
        """Sampled values should be in [0, K-1]."""
        for _ in range(10):
            responses = model.sample(np.random.randn())
            for val in responses.values():
                assert 0 <= val < model.K

    def test_sample_returns_int_values(self, model):
        """Sampled values should be integers."""
        responses = model.sample(0.0)
        for val in responses.values():
            assert isinstance(val, int)

    def test_sample_requires_single_theta(self, model):
        """Sample should raise error for multiple theta values."""
        with pytest.raises(ValueError, match="single theta"):
            model.sample(np.array([0.0, 1.0]))

    def test_sample_statistical_consistency(self, model):
        """Sampled frequencies should approximate probabilities."""
        np.random.seed(42)
        theta = 0.5
        n_samples = 1000

        # Sample many times
        counts = {label: np.zeros(model.K) for label in model.item_labels}
        for _ in range(n_samples):
            responses = model.sample(theta)
            for label, response in responses.items():
                counts[label][response] += 1

        # Check first item's distribution
        probs = model.item_probabilities(theta, item_label=model.item_labels[0])
        empirical_probs = counts[model.item_labels[0]] / n_samples

        # Empirical should be close to theoretical (within statistical tolerance)
        assert np.allclose(empirical_probs, probs, atol=0.1)


class TestImports:
    """Tests for module imports."""

    def test_import_from_prediction_package(self):
        """Should be able to import from prediction subpackage."""
        from libfabulouscatpy.irt.prediction import NeuralIRTModel

        assert NeuralIRTModel is not None

    def test_import_from_neural_irt_module(self):
        """Should be able to import from neural_irt module."""
        from libfabulouscatpy.irt.prediction.neural_irt import NeuralIRTModel

        assert NeuralIRTModel is not None


class TestCmdStanLoader:
    """Tests for loading from CmdStan output."""

    @pytest.fixture
    def cmdstan_output_dir(self):
        """Path to real CmdStan output (if available)."""
        path = Path("/home/josh/workspace/bouldering/neural_irt_results/men")
        if path.exists():
            return path
        pytest.skip("CmdStan output directory not available")

    def test_load_from_cmdstan_output(self, cmdstan_output_dir):
        """Should successfully load model from CmdStan output."""
        model = NeuralIRTModel.from_cmdstan_output(cmdstan_output_dir)

        assert model.n_items > 0
        assert len(model.individual_labels) > 0
        assert model.K > 0
        assert model.H > 0

    def test_loaded_theta_matches_abilities_csv(self, cmdstan_output_dir):
        """Loaded theta should match values in abilities CSV."""
        import csv

        model = NeuralIRTModel.from_cmdstan_output(cmdstan_output_dir)

        # Read abilities CSV directly
        abilities_path = cmdstan_output_dir / "model1_complete_case_abilities.csv"
        with open(abilities_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["climber_id"]) - 1
                expected = float(row["theta_mean"])
                assert np.isclose(model.theta[idx], expected, rtol=1e-5)

    def test_loaded_model_can_compute_probs(self, cmdstan_output_dir):
        """Loaded model should compute valid probabilities."""
        model = NeuralIRTModel.from_cmdstan_output(cmdstan_output_dir)

        theta = np.linspace(-3, 3, 10)
        probs = model.item_probabilities(theta, item_idx=0)

        assert probs.shape == (10, model.K)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0)

    def test_loaded_model_can_sample(self, cmdstan_output_dir):
        """Loaded model should be able to sample responses."""
        model = NeuralIRTModel.from_cmdstan_output(cmdstan_output_dir)

        responses = model.sample(0.0)
        assert len(responses) == model.n_items
        for val in responses.values():
            assert 0 <= val < model.K
