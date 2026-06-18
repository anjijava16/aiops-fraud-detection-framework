"""Unit tests for MetricComparator."""

from __future__ import annotations

import pytest

from src.mlflow_manager.metric_comparator import ComparisonResult, MetricComparator
from src.models.errors import MLflowError


class TestMetricComparator:
    """Tests for MetricComparator class."""

    def test_default_thresholds(self):
        """Test default promotion thresholds."""
        comparator = MetricComparator()
        assert comparator.thresholds["f1"] == 0.85
        assert comparator.thresholds["auc_roc"] == 0.90
        assert comparator.thresholds["precision"] == 0.80
        assert comparator.thresholds["recall"] == 0.75

    def test_custom_thresholds(self):
        """Test custom promotion thresholds."""
        comparator = MetricComparator(thresholds={"f1": 0.90, "auc_roc": 0.95})
        assert comparator.thresholds["f1"] == 0.90
        assert comparator.thresholds["auc_roc"] == 0.95

    def test_primary_metric_default(self):
        """Test default primary metric is f1."""
        comparator = MetricComparator()
        assert comparator.primary_metric == "f1"

    def test_rank_runs_descending_by_f1(self):
        """Test that runs are ranked in descending order of primary metric."""
        comparator = MetricComparator(primary_metric="f1")

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.80, "auc_roc": 0.85}},
            {"run_id": "run-2", "metrics": {"f1": 0.95, "auc_roc": 0.90}},
            {"run_id": "run-3", "metrics": {"f1": 0.88, "auc_roc": 0.92}},
        ]

        result = comparator.compare_runs(runs)

        assert result.rankings[0]["run_id"] == "run-2"
        assert result.rankings[1]["run_id"] == "run-3"
        assert result.rankings[2]["run_id"] == "run-1"
        assert result.rankings[0]["rank"] == 1
        assert result.rankings[1]["rank"] == 2
        assert result.rankings[2]["rank"] == 3

    def test_rank_by_custom_primary_metric(self):
        """Test ranking by a custom primary metric."""
        comparator = MetricComparator(primary_metric="auc_roc")

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.95, "auc_roc": 0.80}},
            {"run_id": "run-2", "metrics": {"f1": 0.80, "auc_roc": 0.98}},
        ]

        result = comparator.compare_runs(runs)

        assert result.rankings[0]["run_id"] == "run-2"
        assert result.rankings[1]["run_id"] == "run-1"

    def test_compute_deltas_vs_production(self):
        """Test computing signed deltas against production metrics."""
        comparator = MetricComparator()

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.85}},
        ]
        production_metrics = {"f1": 0.85, "auc_roc": 0.90, "precision": 0.80, "recall": 0.78}

        result = comparator.compare_runs(runs, production_metrics)

        assert abs(result.deltas["f1"] - 0.07) < 1e-10
        assert abs(result.deltas["auc_roc"] - 0.05) < 1e-10
        assert abs(result.deltas["precision"] - 0.08) < 1e-10
        assert abs(result.deltas["recall"] - 0.07) < 1e-10

    def test_compute_deltas_without_production(self):
        """Test deltas when no production model exists."""
        comparator = MetricComparator()

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.85}},
        ]

        result = comparator.compare_runs(runs, production_metrics=None)

        assert result.deltas["f1"] == 0.92
        assert result.deltas["auc_roc"] == 0.95

    def test_promotion_eligible_when_thresholds_met(self):
        """Test promotion eligibility when all thresholds are met."""
        comparator = MetricComparator()

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.85}},
        ]

        result = comparator.compare_runs(runs)

        assert result.promotion_eligible is True
        assert result.failed_thresholds == []

    def test_promotion_blocked_when_thresholds_not_met(self):
        """Test promotion is blocked when thresholds are not met."""
        comparator = MetricComparator()

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.70, "auc_roc": 0.80, "precision": 0.60, "recall": 0.50}},
        ]

        result = comparator.compare_runs(runs)

        assert result.promotion_eligible is False
        assert len(result.failed_thresholds) == 4

        failed_metrics = [t["metric"] for t in result.failed_thresholds]
        assert "f1" in failed_metrics
        assert "auc_roc" in failed_metrics
        assert "precision" in failed_metrics
        assert "recall" in failed_metrics

    def test_partial_threshold_failure(self):
        """Test when only some thresholds fail."""
        comparator = MetricComparator()

        runs = [
            {"run_id": "run-1", "metrics": {"f1": 0.92, "auc_roc": 0.95, "precision": 0.88, "recall": 0.60}},
        ]

        result = comparator.compare_runs(runs)

        assert result.promotion_eligible is False
        assert len(result.failed_thresholds) == 1
        assert result.failed_thresholds[0]["metric"] == "recall"
        assert result.failed_thresholds[0]["actual"] == 0.60
        assert result.failed_thresholds[0]["required"] == 0.75

    def test_empty_runs_raises_error(self):
        """Test that comparing empty runs list raises MLflowError."""
        comparator = MetricComparator()

        with pytest.raises(MLflowError, match="Cannot compare empty list"):
            comparator.compare_runs([])

    def test_runs_missing_primary_metric_raises_error(self):
        """Test that runs without the primary metric raise an error."""
        comparator = MetricComparator(primary_metric="f1")

        runs = [
            {"run_id": "run-1", "metrics": {"auc_roc": 0.95}},
        ]

        with pytest.raises(MLflowError, match="No runs contain the primary metric"):
            comparator.compare_runs(runs)

    def test_comparison_result_structure(self):
        """Test ComparisonResult dataclass structure."""
        result = ComparisonResult()
        assert result.rankings == []
        assert result.deltas == {}
        assert result.promotion_eligible is False
        assert result.failed_thresholds == []
