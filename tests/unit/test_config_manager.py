"""Unit tests for ConfigManager."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config.config_manager import (
    ConfigFileError,
    ConfigManager,
    ConfigValidationError,
)


def _make_valid_config() -> dict:
    """Return a minimal valid configuration dictionary."""
    return {
        "ingestion": {
            "kafka": {
                "bootstrap_servers": "localhost:9092",
                "topic": "transactions",
                "consumer_group": "fraud-detection-group",
                "reconnect_timeout_seconds": 5,
            },
            "batch": {"default_batch_size": 10000},
            "dead_letter_queue": {"enabled": True},
        },
        "preprocessing": {
            "smote": {"target_ratio": 1.0, "k_neighbors": 5},
            "scaler": {"type": "StandardScaler"},
            "feature_selection": {"correlation_threshold": 0.95},
            "splitting": {"test_size": 0.2, "k_folds": 5, "random_seed": 42},
        },
        "ensemble": {
            "xgboost": {
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 300,
                "scale_pos_weight": 1.0,
            },
            "lightgbm": {
                "learning_rate": 0.1,
                "num_leaves": 31,
                "n_estimators": 300,
                "is_unbalance": True,
            },
            "catboost": {
                "learning_rate": 0.1,
                "depth": 6,
                "iterations": 300,
                "auto_class_weights": "Balanced",
            },
            "voting": {
                "weights": {"xgboost": 0.333, "lightgbm": 0.333, "catboost": 0.334},
                "decision_threshold": 0.5,
            },
            "logging_interval": 10,
        },
        "explainability": {
            "shap": {"top_n_features": 20, "timeout_seconds": 300},
            "lime": {
                "num_samples": 5000,
                "top_n_features": 10,
                "low_fidelity_threshold": 0.6,
            },
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "fraud-detection",
            "promotion_thresholds": {
                "f1_min": 0.85,
                "auc_roc_min": 0.90,
                "precision_min": 0.80,
                "recall_min": 0.75,
            },
            "opentelemetry": {
                "collector_endpoint": "http://localhost:4317",
                "buffer_max_spans": 1000,
                "retry_interval_seconds": 30,
            },
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "auth": {"enabled": True, "max_concurrent_keys": 50},
            "rate_limit": {"requests_per_window": 100, "window_seconds": 60},
        },
        "monitoring": {
            "drift": {
                "psi_threshold": 0.2,
                "ks_p_value_threshold": 0.05,
                "psi_bins": 10,
                "schedule_seconds": 3600,
                "window_size": 10000,
                "min_samples": 100,
            },
            "alerting": {
                "slack_webhook_url": "",
                "pagerduty_routing_key": "",
                "retry_attempts": 3,
                "retry_base_interval_seconds": 2,
                "suppression_window_minutes": 15,
                "f1_alert_threshold": 0.90,
            },
            "retraining": {
                "drift_feature_threshold": 3,
                "data_window_days": 7,
                "min_records": 1000,
            },
            "prometheus": {"metrics_port": 9090, "scrape_interval_seconds": 15},
            "grafana": {"dashboard_output_path": "./dashboards/"},
        },
        "pipeline": {"timeout_hours": 6, "dry_run": False},
    }


def _write_config(config: dict, path: Path) -> None:
    """Write a config dict to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class TestConfigManagerLoadYAML:
    """Tests for YAML loading behavior."""

    def test_loads_valid_config(self, tmp_path):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        cm = ConfigManager(config_file)
        assert cm.get("ingestion", "kafka", "topic") == "transactions"
        assert cm.get("preprocessing", "scaler", "type") == "StandardScaler"
        assert cm.get("pipeline", "dry_run") is False

    def test_missing_file_raises_config_file_error(self, tmp_path):
        missing_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigFileError) as exc_info:
            ConfigManager(missing_file)
        assert "does not exist" in str(exc_info.value)
        assert str(missing_file) in exc_info.value.file_path

    def test_invalid_yaml_raises_config_file_error(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(":\n  bad: [unclosed")
        with pytest.raises(ConfigFileError) as exc_info:
            ConfigManager(config_file)
        assert "YAML parsing error" in str(exc_info.value)

    def test_empty_file_raises_config_file_error(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises(ConfigFileError) as exc_info:
            ConfigManager(config_file)
        assert "empty" in str(exc_info.value).lower()

    def test_non_dict_yaml_raises_config_file_error(self, tmp_path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigFileError) as exc_info:
            ConfigManager(config_file)
        assert "mapping" in str(exc_info.value).lower()


class TestConfigManagerEnvOverrides:
    """Tests for environment variable override behavior."""

    def test_env_var_overrides_yaml_value(self, tmp_path, monkeypatch):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        monkeypatch.setenv("AIOPS_PREPROCESSING__SCALER__TYPE", "MinMaxScaler")
        cm = ConfigManager(config_file)
        assert cm.get("preprocessing", "scaler", "type") == "MinMaxScaler"

    def test_env_var_overrides_numeric_value(self, tmp_path, monkeypatch):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        monkeypatch.setenv("AIOPS_API__PORT", "9000")
        cm = ConfigManager(config_file)
        assert cm.get("api", "port") == 9000

    def test_env_var_overrides_boolean_value(self, tmp_path, monkeypatch):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        monkeypatch.setenv("AIOPS_PIPELINE__DRY_RUN", "true")
        cm = ConfigManager(config_file)
        assert cm.get("pipeline", "dry_run") is True

    def test_env_var_creates_missing_intermediate_keys(self, tmp_path, monkeypatch):
        config_data = _make_valid_config()
        # Remove a nested key
        del config_data["preprocessing"]["scaler"]
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        monkeypatch.setenv("AIOPS_PREPROCESSING__SCALER__TYPE", "StandardScaler")
        cm = ConfigManager(config_file)
        assert cm.get("preprocessing", "scaler", "type") == "StandardScaler"

    def test_env_var_case_insensitive_path(self, tmp_path, monkeypatch):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        monkeypatch.setenv("AIOPS_ENSEMBLE__LOGGING_INTERVAL", "20")
        cm = ConfigManager(config_file)
        assert cm.get("ensemble", "logging_interval") == 20


class TestConfigManagerValidation:
    """Tests for schema validation behavior."""

    def test_missing_required_field_raises_validation_error(self, tmp_path):
        config_data = _make_valid_config()
        del config_data["preprocessing"]["scaler"]["type"]
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert len(violations) >= 1
        assert any(v["key"] == "type" for v in violations)

    def test_wrong_type_raises_validation_error(self, tmp_path):
        config_data = _make_valid_config()
        config_data["api"]["port"] = "not_a_number"
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert any(v["key"] == "port" for v in violations)

    def test_value_below_min_raises_validation_error(self, tmp_path):
        config_data = _make_valid_config()
        config_data["preprocessing"]["splitting"]["k_folds"] = 0
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert any(v["key"] == "k_folds" for v in violations)

    def test_value_above_max_raises_validation_error(self, tmp_path):
        config_data = _make_valid_config()
        config_data["preprocessing"]["splitting"]["k_folds"] = 25
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert any(v["key"] == "k_folds" for v in violations)

    def test_disallowed_value_raises_validation_error(self, tmp_path):
        config_data = _make_valid_config()
        config_data["preprocessing"]["scaler"]["type"] = "RobustScaler"
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert any(v["key"] == "type" and "allowed" in v["violation"] for v in violations)

    def test_multiple_violations_reported_at_once(self, tmp_path):
        config_data = _make_valid_config()
        config_data["api"]["port"] = "bad"
        config_data["preprocessing"]["splitting"]["k_folds"] = 0
        config_data["pipeline"]["timeout_hours"] = 100
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert len(violations) >= 3

    def test_missing_entire_section_reports_all_fields(self, tmp_path):
        config_data = _make_valid_config()
        del config_data["pipeline"]
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        # pipeline section has timeout_hours and dry_run
        assert len(violations) >= 2

    def test_exclusive_min_boundary(self, tmp_path):
        config_data = _make_valid_config()
        # decision_threshold has min_exclusive=0.0
        config_data["ensemble"]["voting"]["decision_threshold"] = 0.0
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigManager(config_file)
        violations = exc_info.value.violations
        assert any(v["key"] == "decision_threshold" for v in violations)


class TestConfigManagerGetMethod:
    """Tests for the get() accessor method."""

    def test_get_nested_value(self, tmp_path):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        cm = ConfigManager(config_file)
        assert cm.get("ensemble", "xgboost", "learning_rate") == 0.1

    def test_get_nonexistent_key_raises_key_error(self, tmp_path):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        cm = ConfigManager(config_file)
        with pytest.raises(KeyError):
            cm.get("nonexistent", "path")

    def test_config_property_returns_full_dict(self, tmp_path):
        config_data = _make_valid_config()
        config_file = tmp_path / "config.yaml"
        _write_config(config_data, config_file)

        cm = ConfigManager(config_file)
        assert isinstance(cm.config, dict)
        assert "ingestion" in cm.config
        assert "pipeline" in cm.config
