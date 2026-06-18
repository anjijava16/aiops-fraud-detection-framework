"""CLI entry point for the AIOps Fraud Detection Framework.

Provides command-line interface with --config, --dry-run, and --mode arguments
for pipeline execution and API serving.

Usage:
    python -m src.main --config config.yaml --mode train
    python -m src.main --config config.yaml --mode serve
    python -m src.main --config config.yaml --mode train --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with config, dry_run, mode, and log_level.
    """
    parser = argparse.ArgumentParser(
        prog="aiops-fraud-detection",
        description="AIOps Fraud Detection Framework - End-to-end ML pipeline",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate pipeline configuration without executing layers",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "serve", "full"],
        default="train",
        help="Execution mode: 'train' (pipeline), 'serve' (API), 'full' (train + serve)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args(argv)


def run_training_pipeline(config: dict[str, Any], dry_run: bool = False) -> int:
    """Execute the training pipeline.

    Args:
        config: Loaded configuration dictionary.
        dry_run: If True, validate without executing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from src.pipeline.orchestrator import PipelineOrchestrator, PipelineStatus

    orchestrator = PipelineOrchestrator(config=config, dry_run=dry_run)

    # Register layer handlers (these would be replaced with actual implementations)
    # For now, register placeholder handlers that log execution
    for layer in PipelineOrchestrator.LAYER_ORDER:
        orchestrator.register_layer(
            layer.value,
            lambda cfg, name=layer.value: logger.info(f"Executing layer: {name}"),
        )

    result = orchestrator.execute()

    if result.status in (PipelineStatus.COMPLETED, PipelineStatus.DRY_RUN_COMPLETED):
        logger.info(
            f"Pipeline finished successfully in {result.total_duration_seconds:.2f}s"
        )
        return 0
    else:
        logger.error(
            f"Pipeline failed at layer '{result.failed_layer}': {result.error}"
        )
        return 1


def run_api_server(config: dict[str, Any]) -> int:
    """Start the FastAPI serving layer.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        import uvicorn

        from src.api.app import create_app

        api_config = config.get("api", {})
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)

        rate_limit_config = api_config.get("rate_limit", {})
        auth_config = api_config.get("auth", {})

        app = create_app(
            model=None,  # Model would be loaded from registry
            model_version="0.0.0",
            api_keys=set(),
            requests_per_window=rate_limit_config.get("requests_per_window", 100),
            window_seconds=rate_limit_config.get("window_seconds", 60),
            auth_enabled=auth_config.get("enabled", True),
        )

        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        return 0

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info(f"AIOps Fraud Detection Framework starting (mode={args.mode})")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    try:
        from src.config.config_manager import ConfigManager

        config_manager = ConfigManager(str(config_path))
        config = config_manager.config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Execute based on mode
    if args.mode == "train":
        return run_training_pipeline(config, dry_run=args.dry_run)
    elif args.mode == "serve":
        if args.dry_run:
            logger.info("[DRY-RUN] API server configuration validated")
            return 0
        return run_api_server(config)
    elif args.mode == "full":
        # Run training first, then serve
        exit_code = run_training_pipeline(config, dry_run=args.dry_run)
        if exit_code != 0:
            return exit_code
        if args.dry_run:
            logger.info("[DRY-RUN] Full pipeline configuration validated")
            return 0
        return run_api_server(config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
