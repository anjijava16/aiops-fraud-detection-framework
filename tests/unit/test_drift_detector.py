"""Unit tests for DriftDetector."""

import logging

import numpy as np
import pytest

from src.monitoring.drift_detector import DriftDetector


@pytest.fixture
def reference_data():
    """Create reference data with known distribution (standard normal)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 3))


@pytest.fixture
def feature_names():
    """Feature names for test data."""
    return ["V1", "V2", "V3"]


@pytest.fixture
def detector(reference_data, feature_names):
    """Create a DriftDetector with default settings."""
    return DriftDetector(
        reference_data=reference_data,
        feature_names=feature_names,
        psi_threshold=0.2,
        ks_p_value_threshold=0.05,
        psi_bins=10,
        window_size=10_000,
        min_samples=100,
    )


class TestDriftDetectorInit:
    """Test DriftDetector initialization."""

    def test_valid_initialization(self, reference_data, feature_names):
        detector = DriftDetector(reference_data, feature_names)
        assert detector.psi_threshold == 0.2
        assert detector.ks_p_value_threshold == 0.05
        assert detector.psi_bins == 10
        assert detector.window_size == 10_000
        assert detector.min_samples == 100
        assert detector.current_window_size == 0

    def test_custom_parameters(self, reference_data, feature_names):
        detector = DriftDetector(
            reference_data,
            feature_names,
            psi_threshold=0.3,
            ks_p_value_threshold=0.01,
            psi_bins=20,
            window_size=5000,
            min_samples=50,
        )
        assert detector.psi_threshold == 0.3
        assert detector.ks_p_value_threshold == 0.01
        assert detector.psi_bins == 20
        assert detector.window_size == 5000
        assert detector.min_samples == 50

    def test_invalid_reference_1d(self, feature_names):
        with pytest.raises(ValueError, match="2-dimensional"):
            DriftDetector(np.array([1, 2, 3]), feature_names)

    def test_mismatched_features(self, reference_data):
        with pytest.raises(ValueError, match="feature names provided"):
            DriftDetector(reference_data, ["V1", "V2"])


class TestAddPrediction:
    """Test adding predictions to the sliding window."""

    def test_add_single_prediction(self, detector):
        features = np.array([0.5, -0.3, 1.2])
        detector.add_prediction(features)
        assert detector.current_window_size == 1

    def test_add_multiple_predictions(self, detector):
        for _ in range(10):
            detector.add_prediction(np.array([0.5, -0.3, 1.2]))
        assert detector.current_window_size == 10

    def test_add_prediction_wrong_dimensions(self, detector):
        with pytest.raises(ValueError, match="1-dimensional"):
            detector.add_prediction(np.array([[0.5, -0.3, 1.2]]))

    def test_add_prediction_wrong_feature_count(self, detector):
        with pytest.raises(ValueError, match="Expected 3 features"):
            detector.add_prediction(np.array([0.5, -0.3]))

    def test_window_size_limit(self, reference_data, feature_names):
        detector = DriftDetector(
            reference_data, feature_names, window_size=5, min_samples=2
        )
        for i in range(10):
            detector.add_prediction(np.array([float(i), float(i), float(i)]))
        assert detector.current_window_size == 5


class TestAddBatch:
    """Test batch addition to the sliding window."""

    def test_add_batch(self, detector):
        batch = np.random.default_rng(0).standard_normal((50, 3))
        detector.add_batch(batch)
        assert detector.current_window_size == 50

    def test_add_batch_wrong_dimensions(self, detector):
        with pytest.raises(ValueError, match="2-dimensional"):
            detector.add_batch(np.array([1.0, 2.0, 3.0]))

    def test_add_batch_wrong_feature_count(self, detector):
        with pytest.raises(ValueError, match="Expected 3 features"):
            detector.add_batch(np.random.default_rng(0).standard_normal((10, 2)))


class TestDetect:
    """Test drift detection."""

    def test_no_drift_same_distribution(self, reference_data, feature_names):
        """No drift when current data is a large sample from the same distribution."""
        # Use a large sample from the same distribution seed to minimize variance
        rng = np.random.default_rng(42)
        detector = DriftDetector(
            reference_data=rng.standard_normal((2000, 3)),
            feature_names=feature_names,
            min_samples=100,
        )
        batch = rng.standard_normal((1000, 3))
        detector.add_batch(batch)

        result = detector.detect()
        assert result.is_drifted is False
        assert result.drifted_features == []
        assert len(result.psi_values) == 3
        assert len(result.ks_results) == 3
        assert result.detection_timestamp != ""

    def test_drift_detected_shifted_distribution(self, detector):
        """Drift detected when distribution is significantly shifted."""
        rng = np.random.default_rng(456)
        # Shift mean by 5 standard deviations — guaranteed drift
        batch = rng.standard_normal((200, 3)) + 5.0
        detector.add_batch(batch)

        result = detector.detect()
        assert result.is_drifted is True
        assert len(result.drifted_features) > 0
        # All features should be drifted with a 5-sigma shift
        assert set(result.drifted_features) == {"V1", "V2", "V3"}

    def test_skip_detection_below_min_samples(self, detector, caplog):
        """Detection is skipped with a warning when window < min_samples."""
        detector.add_prediction(np.array([0.5, -0.3, 1.2]))

        with caplog.at_level(logging.WARNING):
            result = detector.detect()

        assert result.is_drifted is False
        assert result.psi_values == {}
        assert result.ks_results == {}
        assert "skipped" in caplog.text.lower()

    def test_psi_values_non_negative(self, detector):
        """PSI values should always be non-negative."""
        rng = np.random.default_rng(789)
        batch = rng.standard_normal((200, 3)) + 2.0
        detector.add_batch(batch)

        result = detector.detect()
        for psi_val in result.psi_values.values():
            assert psi_val >= 0.0

    def test_ks_results_structure(self, detector):
        """KS results contain (statistic, p-value) tuples."""
        rng = np.random.default_rng(101)
        batch = rng.standard_normal((150, 3))
        detector.add_batch(batch)

        result = detector.detect()
        for feature_name, (ks_stat, p_value) in result.ks_results.items():
            assert 0.0 <= ks_stat <= 1.0
            assert 0.0 <= p_value <= 1.0

    def test_partial_drift(self, reference_data, feature_names):
        """Only shifted features should be flagged."""
        detector = DriftDetector(
            reference_data, feature_names, min_samples=50
        )
        rng = np.random.default_rng(202)
        # Only shift the first feature
        batch = rng.standard_normal((150, 3))
        batch[:, 0] += 5.0  # Shift V1 only

        detector.add_batch(batch)
        result = detector.detect()

        assert result.is_drifted is True
        assert "V1" in result.drifted_features


class TestResetWindow:
    """Test window reset."""

    def test_reset_clears_window(self, detector):
        rng = np.random.default_rng(0)
        detector.add_batch(rng.standard_normal((50, 3)))
        assert detector.current_window_size == 50

        detector.reset_window()
        assert detector.current_window_size == 0


class TestPSIComputation:
    """Test PSI computation edge cases."""

    def test_identical_distributions_near_zero_psi(self, feature_names):
        """Same-distribution samples should have low PSI."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((5000, 3))
        detector = DriftDetector(reference, feature_names, min_samples=50)
        # Use a large current sample from same distribution
        current = rng.standard_normal((2000, 3))
        detector.add_batch(current)

        result = detector.detect()
        for psi_val in result.psi_values.values():
            assert psi_val < 0.2  # Below drift threshold

    def test_constant_values_zero_psi(self, feature_names):
        """Constant values in both distributions produce PSI = 0."""
        constant_ref = np.ones((100, 3))
        detector = DriftDetector(constant_ref, feature_names, min_samples=50)
        detector.add_batch(np.ones((100, 3)))

        result = detector.detect()
        for psi_val in result.psi_values.values():
            assert psi_val == 0.0
