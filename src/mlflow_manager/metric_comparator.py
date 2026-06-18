"""MLflow metric comparison - rank runs and compute deltas vs production.

Provides MetricComparator class that ranks experiment runs by a configurable
primary metric, computes signed deltas against the production model, and
enforces promotion thresholds.

Validates: Requirements 19.1, 19.2, 19.4, 19.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.models.errors import MLflowError

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing multiple experiment runs.

    Attributes:
        rankings: Runs sorted by primary metric descending. Each entry is a dict
                  with run_id, metrics, and rank.
        deltas: Signed differences (candidate - production) for each metric.
        promotion_eligible: True if the top-ranked run meets all thresholds.
        failed_thresholds: List of dicts with metric, actual, and required values.
    """

    rankings: list[dict[str, Any]] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)
    promotion_eligible: bool = False
    failed_thresholds: list[dict[str, Any]] = field(default_factory=list)


class MetricComparator:
    """Compares experiment runs by metrics and determines promotion eligibility.

    Ranks runs by a configurable primary metric (default: F1) in descending order,
    computes signed deltas vs the current production model, and blocks automatic
    promotion when configurable thresholds are not met.

    Usage:
        comparator = MetricComparator(
            primary_metric="f1",
            thresholds={"f1": 0.85, "auc_roc": 0.90, "precision": 0.80, "recall": 0.75}
        )
        result = comparator.compare_runs(runs, production_metrics)
    """

    DEFAULT_THRESHOLDS = {
        "f1": 0.85,
        "auc_roc": 0.90,
        "precision": 0.80,
        "recall": 0.75,
    }

    def __init__(
        self,
        primary_metric: str = "f1",
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize the MetricComparator.

        Args:
            primary_metric: Metric name used for ranking (descending). Default "f1".
            thresholds: Minimum metric thresholds for promotion eligibility.
                       Uses defaults if None.
        """
        self._primary_metric = primary_metric
        self._thresholds = thresholds if thresholds is not None else dict(self.DEFAULT_THRESHOLDS)

    @property
    def primary_metric(self) -> str:
        """Return the configured primary ranking metric."""
        return self._primary_metric

    @property
    def thresholds(self) -> dict[str, float]:
        """Return the configured promotion thresholds."""
        return self._thresholds

    def compare_runs(
        self,
        runs: list[dict[str, Any]],
        production_metrics: dict[str, float] | None = None,
    ) -> ComparisonResult:
        """Rank runs by primary metric and compute deltas vs production.

        Args:
            runs: List of run dicts, each containing at minimum:
                  - "run_id": str
                  - "metrics": dict[str, float] with f1, auc_roc, precision, recall
            production_metrics: Metrics of the current production model.
                              If None, deltas are computed as the metric values themselves.

        Returns:
            ComparisonResult with rankings, deltas, and promotion eligibility.

        Raises:
            MLflowError: If runs list is empty or primary metric is missing.
        """
        if not runs:
            raise MLflowError(
                "Cannot compare empty list of runs",
                details={"primary_metric": self._primary_metric},
            )

        # Validate that primary metric exists in at least one run
        valid_runs = []
        for run in runs:
            metrics = run.get("metrics", {})
            if self._primary_metric in metrics:
                valid_runs.append(run)

        if not valid_runs:
            raise MLflowError(
                f"No runs contain the primary metric '{self._primary_metric}'",
                details={"run_count": len(runs)},
            )

        # Rank runs by primary metric descending
        rankings = self._rank_runs(valid_runs)

        # Compute deltas vs production for the top-ranked run
        top_run_metrics = rankings[0]["metrics"]
        deltas = self._compute_deltas(top_run_metrics, production_metrics)

        # Check promotion eligibility
        failed_thresholds = self._check_thresholds(top_run_metrics)
        promotion_eligible = len(failed_thresholds) == 0

        result = ComparisonResult(
            rankings=rankings,
            deltas=deltas,
            promotion_eligible=promotion_eligible,
            failed_thresholds=failed_thresholds,
        )

        logger.info(
            f"Compared {len(valid_runs)} runs. "
            f"Top run: {rankings[0]['run_id']} "
            f"({self._primary_metric}={rankings[0]['metrics'].get(self._primary_metric, 'N/A')}). "
            f"Promotion eligible: {promotion_eligible}"
        )

        return result

    def _rank_runs(self, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank runs by primary metric in descending order.

        Args:
            runs: List of run dicts with 'run_id' and 'metrics'.

        Returns:
            Sorted list of run dicts with added 'rank' field.
        """
        sorted_runs = sorted(
            runs,
            key=lambda r: r.get("metrics", {}).get(self._primary_metric, 0.0),
            reverse=True,
        )

        rankings = []
        for i, run in enumerate(sorted_runs, start=1):
            rankings.append({
                "rank": i,
                "run_id": run.get("run_id", "unknown"),
                "metrics": run.get("metrics", {}),
            })

        return rankings

    def _compute_deltas(
        self,
        candidate_metrics: dict[str, float],
        production_metrics: dict[str, float] | None,
    ) -> dict[str, float]:
        """Compute signed deltas (candidate - production) for key metrics.

        Args:
            candidate_metrics: Metrics of the candidate (top-ranked) run.
            production_metrics: Metrics of the current production model.

        Returns:
            Dict of metric_name -> signed delta.
        """
        metric_keys = ["f1", "auc_roc", "precision", "recall"]
        deltas: dict[str, float] = {}

        for key in metric_keys:
            candidate_val = candidate_metrics.get(key, 0.0)
            if production_metrics is not None:
                production_val = production_metrics.get(key, 0.0)
                deltas[key] = candidate_val - production_val
            else:
                # No production model; delta is the value itself
                deltas[key] = candidate_val

        return deltas

    def _check_thresholds(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Check if metrics meet all promotion thresholds.

        Args:
            metrics: Candidate model metrics.

        Returns:
            List of failed threshold dicts. Empty if all pass.
        """
        failed = []

        for metric_name, min_value in self._thresholds.items():
            actual = metrics.get(metric_name, 0.0)
            if actual < min_value:
                failed.append({
                    "metric": metric_name,
                    "actual": actual,
                    "required": min_value,
                })

        return failed
