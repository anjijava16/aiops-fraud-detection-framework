"""Configuration manager with YAML loading, environment variable overrides, and schema validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Environment variable prefix for overrides
ENV_PREFIX = "AIOPS_"
# Delimiter used in env vars to separate nested sections (e.g., AIOPS_PREPROCESSING__SCALER__TYPE)
ENV_SECTION_DELIMITER = "__"

# Required top-level sections
REQUIRED_SECTIONS = [
    "ingestion",
    "preprocessing",
    "ensemble",
    "explainability",
    "mlflow",
    "api",
    "monitoring",
    "pipeline",
]


class ConfigValidationError(Exception):
    """Raised when configuration validation fails at startup.

    Attributes:
        violations: List of dicts with keys 'section', 'key', 'violation'.
    """

    def __init__(self, violations: list[dict[str, str]]) -> None:
        self.violations = violations
        details = "; ".join(
            f"[{v['section']}] {v['key']}: {v['violation']}" for v in violations
        )
        super().__init__(f"Configuration validation failed: {details}")


class ConfigFileError(Exception):
    """Raised when the YAML configuration file is missing or unparseable."""

    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Configuration file error at '{file_path}': {reason}"
        )


# Schema definition for validation.
# Each leaf entry: {"type": <python_type or tuple>, "required": bool, ...optional range/allowed}
# Supported constraints: "min", "max", "allowed", "min_exclusive", "max_exclusive"
CONFIG_SCHEMA: dict[str, Any] = {
    "ingestion": {
        "kafka": {
            "bootstrap_servers": {"type": str, "required": True},
            "topic": {"type": str, "required": True},
            "consumer_group": {"type": str, "required": True},
            "reconnect_timeout_seconds": {"type": (int, float), "required": True, "min": 1},
        },
        "batch": {
            "default_batch_size": {"type": int, "required": True, "min": 1},
        },
        "dead_letter_queue": {
            "enabled": {"type": bool, "required": True},
        },
    },
    "preprocessing": {
        "smote": {
            "target_ratio": {"type": (int, float), "required": True, "min_exclusive": 0.0},
            "k_neighbors": {"type": int, "required": True, "min": 1},
        },
        "scaler": {
            "type": {"type": str, "required": True, "allowed": ["StandardScaler", "MinMaxScaler"]},
        },
        "feature_selection": {
            "correlation_threshold": {"type": (int, float), "required": True, "min": 0.5, "max": 1.0},
        },
        "splitting": {
            "test_size": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max_exclusive": 1.0},
            "k_folds": {"type": int, "required": True, "min": 2, "max": 20},
            "random_seed": {"type": int, "required": True},
        },
    },
    "ensemble": {
        "xgboost": {
            "learning_rate": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max": 1.0},
            "max_depth": {"type": int, "required": True, "min": 1, "max": 50},
            "n_estimators": {"type": int, "required": True, "min": 1, "max": 10000},
            "scale_pos_weight": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max": 10000.0},
        },
        "lightgbm": {
            "learning_rate": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max": 1.0},
            "num_leaves": {"type": int, "required": True, "min": 2, "max": 131072},
            "n_estimators": {"type": int, "required": True, "min": 1, "max": 10000},
            "is_unbalance": {"type": bool, "required": True},
        },
        "catboost": {
            "learning_rate": {"type": (int, float), "required": True, "min": 0.001, "max": 1.0},
            "depth": {"type": int, "required": True, "min": 1, "max": 16},
            "iterations": {"type": int, "required": True, "min": 1, "max": 10000},
            "auto_class_weights": {"type": str, "required": True},
        },
        "voting": {
            "weights": {"type": dict, "required": True},
            "decision_threshold": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max_exclusive": 1.0},
        },
        "logging_interval": {"type": int, "required": True, "min": 1},
    },
    "explainability": {
        "shap": {
            "top_n_features": {"type": int, "required": True, "min": 1, "max": 30},
            "timeout_seconds": {"type": (int, float), "required": True, "min": 1},
        },
        "lime": {
            "num_samples": {"type": int, "required": True, "min": 1000, "max": 50000},
            "top_n_features": {"type": int, "required": True, "min": 1, "max": 30},
            "low_fidelity_threshold": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
        },
    },
    "mlflow": {
        "tracking_uri": {"type": str, "required": True},
        "experiment_name": {"type": str, "required": True},
        "promotion_thresholds": {
            "f1_min": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
            "auc_roc_min": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
            "precision_min": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
            "recall_min": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
        },
        "opentelemetry": {
            "collector_endpoint": {"type": str, "required": True},
            "buffer_max_spans": {"type": int, "required": True, "min": 1},
            "retry_interval_seconds": {"type": (int, float), "required": True, "min": 1},
        },
    },
    "api": {
        "host": {"type": str, "required": True},
        "port": {"type": int, "required": True, "min": 1, "max": 65535},
        "auth": {
            "enabled": {"type": bool, "required": True},
            "max_concurrent_keys": {"type": int, "required": True, "min": 1},
        },
        "rate_limit": {
            "requests_per_window": {"type": int, "required": True, "min": 1},
            "window_seconds": {"type": (int, float), "required": True, "min": 1},
        },
    },
    "monitoring": {
        "drift": {
            "psi_threshold": {"type": (int, float), "required": True, "min_exclusive": 0.0},
            "ks_p_value_threshold": {"type": (int, float), "required": True, "min_exclusive": 0.0, "max_exclusive": 1.0},
            "psi_bins": {"type": int, "required": True, "min": 2},
            "schedule_seconds": {"type": (int, float), "required": True, "min": 1},
            "window_size": {"type": int, "required": True, "min": 1},
            "min_samples": {"type": int, "required": True, "min": 1},
        },
        "alerting": {
            "slack_webhook_url": {"type": str, "required": True},
            "pagerduty_routing_key": {"type": str, "required": True},
            "retry_attempts": {"type": int, "required": True, "min": 1},
            "retry_base_interval_seconds": {"type": (int, float), "required": True, "min": 1},
            "suppression_window_minutes": {"type": (int, float), "required": True, "min": 1},
            "f1_alert_threshold": {"type": (int, float), "required": True, "min": 0.0, "max": 1.0},
        },
        "retraining": {
            "drift_feature_threshold": {"type": int, "required": True, "min": 1},
            "data_window_days": {"type": int, "required": True, "min": 1, "max": 90},
            "min_records": {"type": int, "required": True, "min": 1},
        },
        "prometheus": {
            "metrics_port": {"type": int, "required": True, "min": 1, "max": 65535},
            "scrape_interval_seconds": {"type": (int, float), "required": True, "min": 1},
        },
        "grafana": {
            "dashboard_output_path": {"type": str, "required": True},
        },
    },
    "pipeline": {
        "timeout_hours": {"type": (int, float), "required": True, "min": 1, "max": 48},
        "dry_run": {"type": bool, "required": True},
    },
}


def _is_leaf_schema(schema_node: dict[str, Any]) -> bool:
    """Check if a schema node is a leaf validation rule (has 'type' key)."""
    return "type" in schema_node and "required" in schema_node


def _coerce_env_value(value_str: str, expected_type: Any) -> Any:
    """Attempt to coerce a string environment variable value to the expected type."""
    if expected_type is bool or (isinstance(expected_type, tuple) and bool in expected_type):
        lower = value_str.lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
        # If bool isn't the only option, try other types
        if expected_type is bool:
            raise ValueError(f"Cannot convert '{value_str}' to bool")

    if expected_type is int or (isinstance(expected_type, tuple) and int in expected_type):
        try:
            return int(value_str)
        except ValueError:
            pass

    if expected_type is float or (isinstance(expected_type, tuple) and float in expected_type):
        try:
            return float(value_str)
        except ValueError:
            pass

    if isinstance(expected_type, tuple):
        # For (int, float) tuples, try float as fallback
        if int in expected_type and float in expected_type:
            try:
                return float(value_str)
            except ValueError:
                pass

    if expected_type is str or (isinstance(expected_type, tuple) and str in expected_type):
        return value_str

    # Default: return as string
    return value_str


class ConfigManager:
    """Manages application configuration from YAML files with environment variable overrides.

    Loads configuration from a YAML file, applies environment variable overrides
    (using AIOPS_ prefix and double-underscore section delimiters), and validates
    all values against a type/range schema at startup.

    Usage:
        config = ConfigManager("config.yaml")
        kafka_servers = config.get("ingestion", "kafka", "bootstrap_servers")
        # Or access the full config dict:
        full_config = config.config
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize the ConfigManager.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            ConfigFileError: If the file does not exist or cannot be parsed.
            ConfigValidationError: If validation fails (missing/invalid fields).
        """
        self._config_path = Path(config_path)
        self._config: dict[str, Any] = {}

        self._load_yaml()
        self._apply_env_overrides()
        self._validate()

    @property
    def config(self) -> dict[str, Any]:
        """Return the full validated configuration dictionary."""
        return self._config

    def get(self, *keys: str) -> Any:
        """Get a configuration value by traversing nested keys.

        Args:
            *keys: Sequence of keys to traverse (e.g., "ingestion", "kafka", "topic").

        Returns:
            The configuration value at the specified path.

        Raises:
            KeyError: If the key path does not exist.
        """
        current = self._config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"Configuration key not found: {'.'.join(keys)}")
            current = current[key]
        return current

    def _load_yaml(self) -> None:
        """Load configuration from YAML file.

        Raises:
            ConfigFileError: If the file does not exist or is not valid YAML.
        """
        if not self._config_path.exists():
            raise ConfigFileError(
                str(self._config_path),
                "File does not exist",
            )

        try:
            with open(self._config_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigFileError(
                str(self._config_path),
                f"YAML parsing error: {e}",
            )

        if data is None:
            raise ConfigFileError(
                str(self._config_path),
                "File is empty or contains no valid YAML content",
            )

        if not isinstance(data, dict):
            raise ConfigFileError(
                str(self._config_path),
                "YAML content must be a mapping (dictionary) at the top level",
            )

        self._config = data

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides using AIOPS_ prefix.

        Environment variables with the AIOPS_ prefix override YAML values.
        Double-underscore (__) is used as a section delimiter.

        Example:
            AIOPS_PREPROCESSING__SCALER__TYPE=MinMaxScaler
            -> overrides config["preprocessing"]["scaler"]["type"]
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(ENV_PREFIX):
                continue

            # Strip prefix and split by delimiter
            config_path = env_key[len(ENV_PREFIX):].lower()
            parts = config_path.split(ENV_SECTION_DELIMITER.lower())

            if not parts or parts == [""]:
                continue

            # Look up the schema for this path to determine the expected type
            schema_node = CONFIG_SCHEMA
            for part in parts:
                if isinstance(schema_node, dict) and part in schema_node:
                    schema_node = schema_node[part]
                else:
                    schema_node = None
                    break

            # Coerce value to expected type if we found a schema
            if schema_node is not None and _is_leaf_schema(schema_node):
                try:
                    coerced_value = _coerce_env_value(env_value, schema_node["type"])
                except (ValueError, TypeError):
                    coerced_value = env_value
            else:
                # No schema found — try numeric coercion, fall back to string
                try:
                    coerced_value = int(env_value)
                except ValueError:
                    try:
                        coerced_value = float(env_value)
                    except ValueError:
                        lower = env_value.lower()
                        if lower in ("true", "1", "yes"):
                            coerced_value = True
                        elif lower in ("false", "0", "no"):
                            coerced_value = False
                        else:
                            coerced_value = env_value

            # Set the value in the config dict, creating intermediate dicts as needed
            current = self._config
            for i, part in enumerate(parts[:-1]):
                if part not in current or not isinstance(current.get(part), dict):
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = coerced_value

    def _validate(self) -> None:
        """Validate the configuration against the schema.

        Raises:
            ConfigValidationError: If any fields are missing or invalid.
        """
        violations: list[dict[str, str]] = []
        self._validate_node(self._config, CONFIG_SCHEMA, "", violations)

        if violations:
            raise ConfigValidationError(violations)

    def _validate_node(
        self,
        config_node: Any,
        schema_node: dict[str, Any],
        path: str,
        violations: list[dict[str, str]],
    ) -> None:
        """Recursively validate a config node against its schema node.

        Args:
            config_node: The current config value to validate.
            schema_node: The schema definition for this node.
            path: Dot-separated path for error reporting.
            violations: Accumulator for validation errors.
        """
        if _is_leaf_schema(schema_node):
            self._validate_leaf(config_node, schema_node, path, violations)
            return

        # schema_node is a nested dict of sub-schemas
        for key, sub_schema in schema_node.items():
            sub_path = f"{path}.{key}" if path else key

            if not isinstance(config_node, dict) or key not in config_node:
                if _is_leaf_schema(sub_schema) and sub_schema.get("required", False):
                    section = sub_path.rsplit(".", 1)[0] if "." in sub_path else sub_path
                    violations.append({
                        "section": section,
                        "key": key,
                        "violation": "required field is missing",
                    })
                elif not _is_leaf_schema(sub_schema):
                    # Recurse into sub-schema to find all missing required fields
                    self._validate_node({}, sub_schema, sub_path, violations)
                continue

            sub_value = config_node[key]

            if _is_leaf_schema(sub_schema):
                self._validate_leaf(sub_value, sub_schema, sub_path, violations)
            else:
                # Recurse into nested sections
                self._validate_node(sub_value, sub_schema, sub_path, violations)

    def _validate_leaf(
        self,
        value: Any,
        schema: dict[str, Any],
        path: str,
        violations: list[dict[str, str]],
    ) -> None:
        """Validate a single leaf value against its schema constraints.

        Args:
            value: The configuration value.
            schema: The schema definition with type and constraint info.
            path: Dot-separated path for error reporting.
            violations: Accumulator for validation errors.
        """
        section = path.rsplit(".", 1)[0] if "." in path else path
        key = path.rsplit(".", 1)[-1] if "." in path else path

        expected_type = schema["type"]

        # Type check
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"expected type {' or '.join(t.__name__ for t in expected_type)}, got {type(value).__name__}",
                })
                return
        else:
            # Special handling: int values are acceptable for float fields
            if expected_type is float and isinstance(value, int):
                pass  # int is acceptable where float is expected
            elif expected_type is dict and isinstance(value, dict):
                pass  # dict matches
            elif not isinstance(value, expected_type):
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"expected type {expected_type.__name__}, got {type(value).__name__}",
                })
                return

        # Range checks (inclusive bounds)
        if "min" in schema and isinstance(value, (int, float)):
            if value < schema["min"]:
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"value {value} is below minimum {schema['min']}",
                })
                return

        if "max" in schema and isinstance(value, (int, float)):
            if value > schema["max"]:
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"value {value} is above maximum {schema['max']}",
                })
                return

        # Range checks (exclusive bounds)
        if "min_exclusive" in schema and isinstance(value, (int, float)):
            if value <= schema["min_exclusive"]:
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"value {value} must be greater than {schema['min_exclusive']}",
                })
                return

        if "max_exclusive" in schema and isinstance(value, (int, float)):
            if value >= schema["max_exclusive"]:
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"value {value} must be less than {schema['max_exclusive']}",
                })
                return

        # Allowed values check
        if "allowed" in schema:
            if value not in schema["allowed"]:
                violations.append({
                    "section": section,
                    "key": key,
                    "violation": f"value '{value}' not in allowed values: {schema['allowed']}",
                })
                return
