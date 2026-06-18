"""Kafka consumer for real-time transaction ingestion with dead-letter queue routing.

Consumes transaction messages from a configured Kafka topic, validates them
against the schema, and yields valid TransactionRecords. Malformed messages
are routed to a dead-letter queue with offset logging.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Generator

from confluent_kafka import Consumer, KafkaError, KafkaException, Message

from src.ingestion.dead_letter_queue import DeadLetterQueue
from src.ingestion.schema_validator import SchemaValidator
from src.models.errors import IngestionError
from src.models.transaction import TransactionRecord

logger = logging.getLogger(__name__)


class TransactionKafkaConsumer:
    """Consumes transaction messages from Kafka and validates them.

    Connects to configured Kafka bootstrap servers, subscribes to the
    configured topic, and consumes messages. Valid messages are parsed and
    validated using SchemaValidator, yielding TransactionRecord instances.
    Malformed messages (invalid JSON or schema failures) are routed to the
    dead-letter queue with offset logging.

    Supports:
    - Configurable consumer group IDs for horizontal scaling
    - Auto-reconnect within 5 seconds on disconnection
    - Dead-letter queue routing for malformed messages

    Usage:
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="fraud-detection-group",
        )
        for record in consumer.consume():
            process(record)
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        consumer_group: str,
        dead_letter_queue: DeadLetterQueue | None = None,
        schema_validator: SchemaValidator | None = None,
        reconnect_timeout_seconds: float = 5.0,
        poll_timeout_seconds: float = 1.0,
        auto_offset_reset: str = "earliest",
    ) -> None:
        """Initialize the TransactionKafkaConsumer.

        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses.
            topic: Kafka topic to subscribe to.
            consumer_group: Consumer group ID for coordinated consumption.
            dead_letter_queue: Queue for malformed messages. Created if None.
            schema_validator: Validator for message schema. Created if None.
            reconnect_timeout_seconds: Max seconds to wait for reconnection.
            poll_timeout_seconds: Timeout for each poll call.
            auto_offset_reset: Where to start consuming if no offset exists.
        """
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._consumer_group = consumer_group
        self._reconnect_timeout_seconds = reconnect_timeout_seconds
        self._poll_timeout_seconds = poll_timeout_seconds
        self._auto_offset_reset = auto_offset_reset

        self._dlq = dead_letter_queue or DeadLetterQueue()
        self._validator = schema_validator or SchemaValidator()
        self._consumer: Consumer | None = None
        self._running = False

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """Return the dead-letter queue instance."""
        return self._dlq

    @property
    def is_running(self) -> bool:
        """Return whether the consumer is actively consuming."""
        return self._running

    def _create_consumer_config(self) -> dict[str, Any]:
        """Build the confluent-kafka consumer configuration dictionary."""
        return {
            "bootstrap.servers": self._bootstrap_servers,
            "group.id": self._consumer_group,
            "auto.offset.reset": self._auto_offset_reset,
            "enable.auto.commit": True,
            "session.timeout.ms": 30000,
            "heartbeat.interval.ms": 10000,
            "reconnect.backoff.ms": 100,
            "reconnect.backoff.max.ms": int(self._reconnect_timeout_seconds * 1000),
        }

    def _connect(self) -> None:
        """Create and configure the Kafka consumer, subscribing to the topic.

        Raises:
            IngestionError: If connection cannot be established.
        """
        try:
            config = self._create_consumer_config()
            self._consumer = Consumer(config)
            self._consumer.subscribe([self._topic])
            logger.info(
                "Kafka consumer connected: servers=%s, topic=%s, group=%s",
                self._bootstrap_servers,
                self._topic,
                self._consumer_group,
            )
        except KafkaException as e:
            raise IngestionError(
                f"Failed to connect to Kafka: {e}",
                details={
                    "bootstrap_servers": self._bootstrap_servers,
                    "topic": self._topic,
                    "consumer_group": self._consumer_group,
                },
            )

    def _reconnect(self) -> bool:
        """Attempt to reconnect to Kafka within the configured timeout.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < self._reconnect_timeout_seconds:
            attempt += 1
            try:
                logger.info(
                    "Reconnection attempt %d to Kafka (elapsed: %.1fs)",
                    attempt,
                    time.time() - start_time,
                )
                self._close_consumer()
                self._connect()
                logger.info("Reconnected to Kafka after %d attempt(s)", attempt)
                return True
            except IngestionError:
                backoff = min(0.5 * attempt, self._reconnect_timeout_seconds / 2)
                remaining = self._reconnect_timeout_seconds - (time.time() - start_time)
                if remaining > 0:
                    time.sleep(min(backoff, remaining))

        logger.error(
            "Failed to reconnect to Kafka within %.1f seconds",
            self._reconnect_timeout_seconds,
        )
        return False

    def _close_consumer(self) -> None:
        """Safely close the current consumer instance."""
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception:
                pass
            self._consumer = None

    def _parse_message(self, msg: Message) -> TransactionRecord | None:
        """Parse and validate a Kafka message, routing failures to DLQ.

        Args:
            msg: The raw Kafka message.

        Returns:
            A valid TransactionRecord, or None if the message was malformed.
        """
        raw_value = msg.value()
        offset = msg.offset()
        topic = msg.topic() or self._topic
        partition = msg.partition() or 0

        # Attempt JSON parsing
        try:
            if isinstance(raw_value, bytes):
                payload = json.loads(raw_value.decode("utf-8"))
            elif isinstance(raw_value, str):
                payload = json.loads(raw_value)
            else:
                self._dlq.enqueue(
                    raw_message=raw_value,
                    error="Message value is None or unsupported type",
                    offset=offset,
                    topic=topic,
                    partition=partition,
                )
                logger.error(
                    "Malformed message at offset %d: value is None or unsupported type",
                    offset,
                )
                return None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._dlq.enqueue(
                raw_message=raw_value,
                error=f"JSON parse error: {e}",
                offset=offset,
                topic=topic,
                partition=partition,
            )
            logger.error(
                "Malformed message at offset %d: JSON parse error: %s",
                offset,
                e,
            )
            return None

        # Validate schema
        if not isinstance(payload, dict):
            self._dlq.enqueue(
                raw_message=raw_value,
                error="Message payload is not a JSON object",
                offset=offset,
                topic=topic,
                partition=partition,
            )
            logger.error(
                "Malformed message at offset %d: payload is not a JSON object",
                offset,
            )
            return None

        validation_result = self._validator.validate(payload)

        if not validation_result.is_valid:
            error_details = "; ".join(
                f"{err.field_name}: {err.message}" for err in validation_result.errors
            )
            self._dlq.enqueue(
                raw_message=raw_value,
                error=f"Schema validation failed: {error_details}",
                offset=offset,
                topic=topic,
                partition=partition,
            )
            logger.error(
                "Schema validation failed at offset %d: %s",
                offset,
                error_details,
            )
            return None

        # Build TransactionRecord from validated payload
        transaction_id = payload.get("transaction_id", str(uuid.uuid4()))
        features = {f"V{i}": float(payload[f"V{i}"]) for i in range(1, 29)}

        record = TransactionRecord(
            transaction_id=str(transaction_id),
            time=float(payload["Time"]),
            features=features,
            amount=float(payload["Amount"]),
            class_label=int(payload["Class"]) if "Class" in payload else None,
        )

        return record

    def consume(self) -> Generator[TransactionRecord, None, None]:
        """Consume messages from Kafka, yielding valid TransactionRecords.

        Connects to Kafka if not already connected. On disconnection,
        attempts auto-reconnect within the configured timeout. Malformed
        messages are routed to the dead-letter queue.

        Yields:
            TransactionRecord instances for each valid message.

        Raises:
            IngestionError: If initial connection fails or reconnection
                           is exhausted.
        """
        if self._consumer is None:
            self._connect()

        self._running = True
        try:
            while self._running:
                try:
                    msg = self._consumer.poll(timeout=self._poll_timeout_seconds)
                except KafkaException as e:
                    logger.warning("Kafka poll error, attempting reconnect: %s", e)
                    if not self._reconnect():
                        raise IngestionError(
                            "Kafka reconnection failed",
                            details={"error": str(e)},
                        )
                    continue

                if msg is None:
                    # No message within poll timeout
                    continue

                error = msg.error()
                if error is not None:
                    if error.code() == KafkaError._PARTITION_EOF:
                        # End of partition, not an error
                        logger.debug(
                            "Reached end of partition %s[%d] at offset %d",
                            msg.topic(),
                            msg.partition(),
                            msg.offset(),
                        )
                        continue
                    elif error.code() == KafkaError._ALL_BROKERS_DOWN:
                        logger.warning("All brokers down, attempting reconnect")
                        if not self._reconnect():
                            raise IngestionError(
                                "All Kafka brokers are down and reconnection failed",
                                details={"error_code": str(error.code())},
                            )
                        continue
                    else:
                        logger.error("Kafka error: %s", error)
                        # Route message to DLQ for unrecoverable message errors
                        if error.code() in (
                            KafkaError._KEY_DESERIALIZATION,
                            KafkaError._VALUE_DESERIALIZATION,
                        ):
                            self._dlq.enqueue(
                                raw_message=msg.value(),
                                error=f"Kafka deserialization error: {error}",
                                offset=msg.offset(),
                                topic=msg.topic() or self._topic,
                                partition=msg.partition() or 0,
                            )
                            continue
                        # For other errors, attempt reconnect
                        if not self._reconnect():
                            raise IngestionError(
                                f"Kafka error: {error}",
                                details={"error_code": str(error.code())},
                            )
                        continue

                # Parse and validate the message
                record = self._parse_message(msg)
                if record is not None:
                    yield record

        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the consumer loop gracefully."""
        self._running = False

    def close(self) -> None:
        """Stop consuming and close the Kafka connection."""
        self.stop()
        self._close_consumer()
        logger.info("Kafka consumer closed")
