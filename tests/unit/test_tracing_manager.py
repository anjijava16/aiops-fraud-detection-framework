"""Unit tests for TracingManager."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.mlflow_manager.tracing_manager import SpanContext, SpanRecord, TracingManager


class TestSpanRecord:
    """Tests for SpanRecord dataclass."""

    def test_default_values(self):
        """Test SpanRecord default attribute values."""
        record = SpanRecord(name="test", start_time=1.0, end_time=2.0)
        assert record.name == "test"
        assert record.start_time == 1.0
        assert record.end_time == 2.0
        assert record.attributes == {}
        assert record.status == "OK"

    def test_custom_attributes(self):
        """Test SpanRecord with custom attributes."""
        record = SpanRecord(
            name="preprocessing",
            start_time=1.0,
            end_time=5.0,
            attributes={"records": 1000},
            status="ERROR",
        )
        assert record.attributes["records"] == 1000
        assert record.status == "ERROR"


class TestTracingManager:
    """Tests for TracingManager class."""

    def _create_manager(self):
        """Create a TracingManager with mocked OpenTelemetry."""
        with patch("src.mlflow_manager.tracing_manager.TracingManager._initialize_tracer"):
            manager = TracingManager(
                collector_endpoint="http://localhost:4317",
                buffer_max_spans=1000,
                retry_interval_seconds=30.0,
            )
            manager._tracer = None
            manager._exporter_available = False
            return manager

    def test_initialization(self):
        """Test TracingManager initializes with correct defaults."""
        manager = self._create_manager()
        assert manager.collector_endpoint == "http://localhost:4317"
        assert manager.buffer_size == 0
        assert manager._buffer_max_spans == 1000
        assert manager._retry_interval == 30.0

    def test_create_span_returns_context(self):
        """Test that create_span returns a SpanContext."""
        manager = self._create_manager()
        span_ctx = manager.create_span("test-span")
        assert isinstance(span_ctx, SpanContext)

    def test_span_context_records_duration(self):
        """Test that span context records start and end times."""
        manager = self._create_manager()

        with manager.create_span("test-span") as span:
            time.sleep(0.01)

        # Span should be buffered since no exporter is available
        assert manager.buffer_size == 1

    def test_span_context_set_attribute(self):
        """Test setting attributes on a span."""
        manager = self._create_manager()

        with manager.create_span("test-span") as span:
            span.set_attribute("records_processed", 5000)
            span.set_attribute("layer", "preprocessing")

        # Check the buffered span has attributes
        buffered = manager._buffer[0]
        assert buffered.attributes["records_processed"] == 5000
        assert buffered.attributes["layer"] == "preprocessing"

    def test_span_context_set_error(self):
        """Test marking a span as errored."""
        manager = self._create_manager()

        with manager.create_span("test-span") as span:
            span.set_error("Something went wrong")

        buffered = manager._buffer[0]
        assert buffered.status == "ERROR"
        assert buffered.attributes["error"] == "Something went wrong"

    def test_span_context_captures_exception(self):
        """Test that exceptions in span context are captured."""
        manager = self._create_manager()

        with pytest.raises(ValueError):
            with manager.create_span("test-span") as span:
                raise ValueError("test error")

        buffered = manager._buffer[0]
        assert buffered.status == "ERROR"
        assert "test error" in buffered.attributes["error"]

    def test_buffer_cap(self):
        """Test that buffer respects max size."""
        with patch("src.mlflow_manager.tracing_manager.TracingManager._initialize_tracer"):
            manager = TracingManager(
                collector_endpoint="http://localhost:4317",
                buffer_max_spans=5,
                retry_interval_seconds=30.0,
            )
            manager._tracer = None
            manager._exporter_available = False

        # Add more than the max
        for i in range(10):
            with manager.create_span(f"span-{i}"):
                pass

        assert manager.buffer_size == 5

    def test_flush_clears_buffer(self):
        """Test that flush clears all buffered spans."""
        manager = self._create_manager()

        for i in range(3):
            with manager.create_span(f"span-{i}"):
                pass

        assert manager.buffer_size == 3
        count = manager.flush()
        assert count == 3
        assert manager.buffer_size == 0

    def test_shutdown(self):
        """Test that shutdown sets the event and cleans up."""
        manager = self._create_manager()
        manager.shutdown()
        assert manager._shutdown_event.is_set()

    def test_spans_not_buffered_when_exporter_available(self):
        """Test that spans are not buffered when exporter is available."""
        manager = self._create_manager()
        manager._exporter_available = True
        manager._tracer = MagicMock()

        with manager.create_span("test-span"):
            pass

        # Should not be buffered when exporter is available
        assert manager.buffer_size == 0
