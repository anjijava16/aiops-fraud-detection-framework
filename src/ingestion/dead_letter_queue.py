"""Dead-letter queue for routing malformed messages that fail ingestion.

Messages that cannot be parsed or validated are stored here for later
analysis and reprocessing, along with the error reason and original offset.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterMessage:
    """A message that failed ingestion processing.

    Attributes:
        raw_message: The original raw message bytes or string.
        error: Description of why the message failed.
        offset: The Kafka message offset (for traceability).
        topic: The source Kafka topic.
        partition: The source Kafka partition.
        timestamp: Unix timestamp when the message was routed to DLQ.
    """

    raw_message: bytes | str | None
    error: str
    offset: int
    topic: str
    partition: int = 0
    timestamp: float = field(default_factory=time.time)


class DeadLetterQueue:
    """Stores failed messages for later analysis and reprocessing.

    Thread-safe in-memory dead-letter queue. In production, this would be
    backed by a persistent store (e.g., a dedicated Kafka topic or database).

    Usage:
        dlq = DeadLetterQueue()
        dlq.enqueue(raw_message=b'bad data', error="Invalid JSON", offset=42, topic="transactions")
        failed = dlq.get_messages()
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the DeadLetterQueue.

        Args:
            max_size: Maximum number of messages to retain. Oldest messages
                      are discarded when the limit is reached.
        """
        self._messages: list[DeadLetterMessage] = []
        self._max_size = max_size
        self._lock = Lock()

    def enqueue(
        self,
        raw_message: bytes | str | None,
        error: str,
        offset: int,
        topic: str,
        partition: int = 0,
    ) -> None:
        """Route a failed message to the dead-letter queue.

        Logs the error with the message offset for traceability.

        Args:
            raw_message: The original raw message content.
            error: Description of why the message failed processing.
            offset: The Kafka message offset.
            topic: The source Kafka topic.
            partition: The source Kafka partition.
        """
        entry = DeadLetterMessage(
            raw_message=raw_message,
            error=error,
            offset=offset,
            topic=topic,
            partition=partition,
        )

        logger.warning(
            "Message routed to dead-letter queue: topic=%s, partition=%d, offset=%d, error=%s",
            topic,
            partition,
            offset,
            error,
        )

        with self._lock:
            self._messages.append(entry)
            # Evict oldest messages if over capacity
            if len(self._messages) > self._max_size:
                overflow = len(self._messages) - self._max_size
                self._messages = self._messages[overflow:]

    def get_messages(self, limit: int | None = None) -> list[DeadLetterMessage]:
        """Retrieve messages from the dead-letter queue.

        Args:
            limit: Maximum number of messages to return. None returns all.

        Returns:
            List of DeadLetterMessage entries, oldest first.
        """
        with self._lock:
            if limit is None:
                return list(self._messages)
            return list(self._messages[:limit])

    def size(self) -> int:
        """Return the current number of messages in the queue."""
        with self._lock:
            return len(self._messages)

    def clear(self) -> None:
        """Remove all messages from the queue."""
        with self._lock:
            self._messages.clear()
