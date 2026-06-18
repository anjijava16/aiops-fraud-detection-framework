"""Unit tests for the CLI entry point (main.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.main import main, parse_args


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])
        assert args.config == "config.yaml"
        assert args.dry_run is False
        assert args.mode == "train"
        assert args.log_level == "INFO"

    def test_config_arg(self):
        """Test --config argument."""
        args = parse_args(["--config", "custom.yaml"])
        assert args.config == "custom.yaml"

    def test_dry_run_flag(self):
        """Test --dry-run flag."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_mode_train(self):
        """Test --mode train."""
        args = parse_args(["--mode", "train"])
        assert args.mode == "train"

    def test_mode_serve(self):
        """Test --mode serve."""
        args = parse_args(["--mode", "serve"])
        assert args.mode == "serve"

    def test_mode_full(self):
        """Test --mode full."""
        args = parse_args(["--mode", "full"])
        assert args.mode == "full"

    def test_invalid_mode_rejected(self):
        """Test that invalid mode is rejected."""
        with pytest.raises(SystemExit):
            parse_args(["--mode", "invalid"])

    def test_log_level_arg(self):
        """Test --log-level argument."""
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


class TestMain:
    """Tests for main() function."""

    def test_missing_config_returns_1(self):
        """Test that missing config file returns exit code 1."""
        exit_code = main(["--config", "/nonexistent/config.yaml"])
        assert exit_code == 1

    @patch("src.config.config_manager.ConfigManager.__init__", return_value=None)
    def test_train_mode_dry_run(self, mock_init):
        """Test train mode with dry-run completes successfully."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.config.config_manager.ConfigManager.config", new_callable=lambda: property(lambda self: {"pipeline": {"dry_run": True}})):
                exit_code = main(["--config", "config.yaml", "--mode", "train", "--dry-run"])

        assert exit_code == 0

    @patch("src.config.config_manager.ConfigManager.__init__", return_value=None)
    def test_serve_mode_dry_run(self, mock_init):
        """Test serve mode with dry-run validates and returns 0."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.config.config_manager.ConfigManager.config", new_callable=lambda: property(lambda self: {"api": {"host": "0.0.0.0", "port": 8000}})):
                exit_code = main(["--config", "config.yaml", "--mode", "serve", "--dry-run"])

        assert exit_code == 0

    def test_config_load_failure_returns_1(self):
        """Test that config load failure returns exit code 1."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.config.config_manager.ConfigManager.__init__", side_effect=Exception("Parse error")):
                exit_code = main(["--config", "config.yaml"])

        assert exit_code == 1
