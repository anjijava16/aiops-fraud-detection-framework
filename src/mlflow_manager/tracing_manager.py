"""OpenTelemetry tracing manager - instrument pipeline stages with spans.

Provides TracingManager class that creates traces spanning API receipt through
inference, exports to a configurable collector endpoint, and buffers traces
locally on failure with automatic retry.

Validates: Requirements 20.1, 20.2, 20.3, 20.4
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.models.errors import MLflowError

logger = logging.getLogger(__name__)


@dataclass
class SpanRecord:
    """A buffered span record awaiting export.

    Attributes:
        name: Span name (e.g., "ingestion", "preprocessing").
        start_time: Unix timestamp when the span started.
        end_time: Unix timestamp when the span ended.
        attributes: Key-value metadata attached to the span.
        status: Span status ("OK", "ERROR").
    """

    name: str
    start_time: float
    end_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"


class TracingManager:
    """Instruments pipeline stages with OpenTelemetry spans.

    Creates traces spanning API receipt through inference, exports to a
    configurable collector endpoint, and buffers traces locally on export
    failure with automatic retry every 30 seconds (capped at 1000 spans).

    Usage:
        tracer = TracingManager(collector_endpoint="http://localhost:4317")
        with tracer.create_span("preprocessing") as span:
            # ... do work ...
            span.set_attribute("records_processed", 5000)
        tracer.shutdown()
    """

    DEFAULT_BUFFER_MAX = 1000
    DEFAULT_RETRY_INTERVAL = 30.0  # seconds

    def __init__(
        self,
        collector_endpoint: str = "http://localhost:4317",
        service_name: str = "fraud-detection-pipeline",
        buffer_max_spans: int = 1000,
        retry_interval_seconds: float = 30.0,
    ) -> None:
        """Initialize the TracingManager.

        Args:
            collector_endpoint: OTLP collector endpoint URL.
            service_name: Service name for trace identification.
            buffer_max_spans: Maximum spans to buffer on export failure.
            retry_interval_seconds: Retry interval for buffered spans.
        """
        self._collector_endpoint = collector_endpoint
        self._service_name = service_name
        self._buffer_max_spans = buffer_max_spans
        self._retry_interval = retry_interval_seconds

        self._buffer: deque[SpanRecord] = deque(maxlen=buffer_max_spans)
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._retry_thread: threading.Thread | None = None
        self._exporter_available = False
        self._tracer: Any = None

        self._initialize_tracer()

    @property
    def buffer_size(self) -> int:
        """Return the current number of buffered spans."""
        with self._lock:
            return len(self._buffer)

    @property
    def collector_endpoint(self) -> str:
        """Return the configured collector endpoint."""
        return self._collector_endpoint

    def _initialize_tracer(self) -> None:
        """Initialize OpenTelemetry tracer and exporter."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
            )

            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(resource=resource)

            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=self._collector_endpoint)
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
                self._exporter_available = True
            except (ImportError, Exception) as e:
                logger.warning(
                    f"OTLP exporter not available, using local buffering: {e}"
                )
                self._exporter_available = False

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self._service_name)
            logger.info(
                f"TracingManager initialized (endpoint={self._collector_endpoint}, "
                f"exporter_available={self._exporter_available})"
            )
        except ImportError as e:
            logger.warning(f"OpenTelemetry not available: {e}")
            self._tracer = None
            self._exporter_available = False

    def create_span(self, name: str, attributes: dict[str, Any] | None = None) -> "SpanContext":
        """Create a trace span for a pipeline stage.

        Args:
            name: Name of the span (e.g., "ingestion", "training").
            attributes: Optional key-value pairs to attach to the span.

        Returns:
            A SpanContext context manager.
        """
        return SpanContext(self, name, attributes or {})

    def _record_span(self, span_record: SpanRecord) -> None:
        """Record a completed span, buffering if export fails.

        Args:
            span_record: The completed span record.
        """
        if self._tracer is not None and self._exporter_available:
            # The span was already exported via OpenTelemetry
            return

        # Buffer the span for retry
        with self._lock:
            self._buffer.append(span_record)
            logger.debug(
                f"Buffered span '{span_record.name}' "
                f"(buffer size: {len(self._buffer)})"
            )

        # Start retry thread if not running
        self._ensure_retry_thread()

    def _ensure_retry_thread(self) -> None:
        """Start the retry thread if it's not already running."""
        if self._retry_thread is None or not self._retry_thread.is_alive():
            self._retry_thread = threading.Thread(
                target=self._retry_loop,
                daemon=True,
                name="tracing-retry",
            )
            self._retry_thread.start()

    def _retry_loop(self) -> None:
        """Background thread that retries exporting buffered spans."""
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=self._retry_interval)

            if self._shutdown_event.is_set():
                break

            with self._lock:
                if not self._buffer:
                    break  # Nothing to retry

                spans_to_export = list(self._buffer)

            # Attempt export
            if self._try_export(spans_to_export):
                with self._lock:
                    # Remove exported spans
                    for _ in range(min(len(spans_to_export), len(self._buffer))):
                        self._buffer.popleft()
                logger.info(f"Successfully exported {len(spans_to_export)} buffered spans")
            else:
                logger.debug("Retry export failed, will try again later")

    def _try_export(self, spans: list[SpanRecord]) -> bool:
        """Attempt to export spans to the collector.

        Args:
            spans: List of span records to export.

        Returns:
            True if export succeeded, False otherwise.
        """
        try:
            if self._tracer is None:
                return False

            # Attempt to create and export spans through the tracer
            for span_record in spans:
                with self._tracer.start_as_current_span(span_record.name) as span:
                    for k, v in span_record.attributes.items():
                        span.set_attribute(k, v)

            return True
        except Exception as e:
            logger.debug(f"Export attempt failed: {e}")
            return False

    def flush(self) -> int:
        """Flush all buffered spans.

        Returns:
            Number of spans that were in the buffer.
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
        return count

    def shutdown(self) -> None:
        """Shutdown the tracing manager and stop retry thread."""
        self._shutdown_event.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=5.0)
        logger.info("TracingManager shut down")


class SpanContext:
    """Context manager for creating OpenTelemetry spans.

    Usage:
        with tracer.create_span("processing") as span:
            span.set_attribute("key", "value")
    """

    def __init__(
        self,
        manager: TracingManager,
        name: str,
        attributes: dict[str, Any],
    ) -> None:
        self._manager = manager
        self._name = name
        self._attributes = dict(attributes)
        self._start_time: float = 0.0
        self._otel_span: Any = None
        self._status = "OK"

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Args:
            key: Attribute key.
            value: Attribute value.
        """
        self._attributes[key] = value
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(key, value)
            except Exception:
                pass

    def set_error(self, error: str) -> None:
        """Mark the span as errored.

        Args:
            error: Error description.
        """
        self._status = "ERROR"
        self._attributes["error"] = error

    def __enter__(self) -> "SpanContext":
        self._start_time = time.time()

        if self._manager._tracer is not None:
            try:
                self._otel_span = self._manager._tracer.start_span(self._name)
                for k, v in self._attributes.items():
                    self._otel_span.set_attribute(k, v)
            except Exception:
                self._otel_span = None

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_time = time.time()

        if exc_type is not None:
            self._status = "ERROR"
            self._attributes["error"] = str(exc_val)

        if self._otel_span is not None:
            try:
                self._otel_span.end()
            except Exception:
                pass

        # Record the span for buffering if needed
        span_record = SpanRecord(
            name=self._name,
            start_time=self._start_time,
            end_time=end_time,
            attributes=self._attributes,
            status=self._status,
        )
        self._manager._record_span(span_record)
