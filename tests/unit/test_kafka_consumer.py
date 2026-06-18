"""Unit tests for TransactionKafkaConsumer and DeadLetterQueue.

Tests use mocked Kafka client to verify:
- Message consumption and TransactionRecord parsing
- Malformed JSON routing to dead-letter queue
- Schema validation failure routing to DLQ
- Configurable consumer group IDs
- Auto-reconnect behavior on disconnection
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from confluent_kafka import KafkaError, KafkaException

from src.ingestion.dead_letter_queue import DeadLetterQueue, DeadLetterMessage
from src.ingestion.kafka_consumer import TransactionKafkaConsumer
from src.ingestion.schema_validator import SchemaValidator
from src.models.transaction import TransactionRecord


# --- DeadLetterQueue Tests ---


class TestDeadLetterQueue:
    """Tests for the DeadLetterQueue class."""

    def test_enqueue_stores_message(self):
        """Enqueuing a message stores it with correct attributes."""
        dlq = DeadLetterQueue()
        dlq.enqueue(
            raw_message=b"bad data",
            error="Invalid JSON",
            offset=42,
            topic="transactions",
            partition=1,
        )

        assert dlq.size() == 1
        messages = dlq.get_messages()
        assert len(messages) == 1
        assert messages[0].raw_message == b"bad data"
        assert messages[0].error == "Invalid JSON"
        assert messages[0].offset == 42
        assert messages[0].topic == "transactions"
        assert messages[0].partition == 1

    def test_enqueue_multiple_messages(self):
        """Multiple messages are stored in order."""
        dlq = DeadLetterQueue()
        dlq.enqueue(raw_message=b"msg1", error="err1", offset=1, topic="t")
        dlq.enqueue(raw_message=b"msg2", error="err2", offset=2, topic="t")
        dlq.enqueue(raw_message=b"msg3", error="err3", offset=3, topic="t")

        assert dlq.size() == 3
        messages = dlq.get_messages()
        assert [m.offset for m in messages] == [1, 2, 3]

    def test_max_size_eviction(self):
        """Oldest messages are evicted when max_size is exceeded."""
        dlq = DeadLetterQueue(max_size=3)
        for i in range(5):
            dlq.enqueue(raw_message=f"msg{i}".encode(), error="err", offset=i, topic="t")

        assert dlq.size() == 3
        messages = dlq.get_messages()
        # Should retain offsets 2, 3, 4 (oldest evicted)
        assert [m.offset for m in messages] == [2, 3, 4]

    def test_get_messages_with_limit(self):
        """get_messages respects the limit parameter."""
        dlq = DeadLetterQueue()
        for i in range(10):
            dlq.enqueue(raw_message=f"msg{i}".encode(), error="err", offset=i, topic="t")

        messages = dlq.get_messages(limit=3)
        assert len(messages) == 3
        assert [m.offset for m in messages] == [0, 1, 2]

    def test_clear_removes_all_messages(self):
        """clear() empties the queue."""
        dlq = DeadLetterQueue()
        dlq.enqueue(raw_message=b"msg", error="err", offset=0, topic="t")
        dlq.clear()

        assert dlq.size() == 0
        assert dlq.get_messages() == []

    def test_enqueue_with_none_message(self):
        """Enqueuing with None raw_message is allowed."""
        dlq = DeadLetterQueue()
        dlq.enqueue(raw_message=None, error="Empty message", offset=5, topic="t")

        assert dlq.size() == 1
        assert dlq.get_messages()[0].raw_message is None


# --- TransactionKafkaConsumer Tests ---


def _make_valid_transaction() -> dict:
    """Create a valid transaction payload."""
    record = {"Time": 0.0, "Amount": 149.62}
    for i in range(1, 29):
        record[f"V{i}"] = float(i) * -0.1
    return record


def _make_kafka_message(
    value: bytes | None = None,
    offset: int = 0,
    topic: str = "transactions",
    partition: int = 0,
    error: KafkaError | None = None,
) -> MagicMock:
    """Create a mock Kafka message."""
    msg = MagicMock()
    msg.value.return_value = value
    msg.offset.return_value = offset
    msg.topic.return_value = topic
    msg.partition.return_value = partition
    msg.error.return_value = error
    return msg


class TestTransactionKafkaConsumer:
    """Tests for TransactionKafkaConsumer."""

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_consumer_connects_with_correct_config(self, mock_consumer_class):
        """Consumer passes correct config to confluent-kafka."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="broker1:9092,broker2:9092",
            topic="test-topic",
            consumer_group="test-group",
            reconnect_timeout_seconds=5.0,
        )
        consumer._connect()

        call_args = mock_consumer_class.call_args[0][0]
        assert call_args["bootstrap.servers"] == "broker1:9092,broker2:9092"
        assert call_args["group.id"] == "test-group"
        mock_consumer.subscribe.assert_called_once_with(["test-topic"])

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_configurable_consumer_group_id(self, mock_consumer_class):
        """Consumer group ID is configurable for horizontal scaling."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="my-custom-group-42",
        )
        consumer._connect()

        call_args = mock_consumer_class.call_args[0][0]
        assert call_args["group.id"] == "my-custom-group-42"

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_consume_valid_message_yields_transaction_record(self, mock_consumer_class):
        """Valid JSON messages are parsed into TransactionRecord."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        payload = _make_valid_transaction()
        payload["transaction_id"] = "txn-123"
        msg = _make_kafka_message(value=json.dumps(payload).encode("utf-8"), offset=10)

        # Consumer returns one message then None to break loop
        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                return msg
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()  # Stop after first record

        assert len(records) == 1
        assert isinstance(records[0], TransactionRecord)
        assert records[0].transaction_id == "txn-123"
        assert records[0].time == 0.0
        assert records[0].amount == 149.62
        assert records[0].class_label is None

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_consume_valid_message_with_class_label(self, mock_consumer_class):
        """Valid messages with Class field include class_label."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        payload = _make_valid_transaction()
        payload["Class"] = 1
        msg = _make_kafka_message(value=json.dumps(payload).encode("utf-8"))

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                return msg
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        assert records[0].class_label == 1

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_invalid_json_routes_to_dlq(self, mock_consumer_class):
        """Malformed JSON messages are routed to dead-letter queue."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        msg = _make_kafka_message(
            value=b"this is not json {{{",
            offset=55,
            topic="transactions",
            partition=2,
        )

        dlq = DeadLetterQueue()
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            dead_letter_queue=dlq,
        )

        # Directly test _parse_message which doesn't need the consume loop
        consumer._connect()
        result = consumer._parse_message(msg)

        assert result is None
        assert dlq.size() == 1
        dlq_msg = dlq.get_messages()[0]
        assert dlq_msg.offset == 55
        assert dlq_msg.topic == "transactions"
        assert dlq_msg.partition == 2
        assert "JSON parse error" in dlq_msg.error

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_schema_validation_failure_routes_to_dlq(self, mock_consumer_class):
        """Messages failing schema validation are routed to DLQ."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        # Missing V1 field and negative Amount
        payload = {"Time": 0.0, "Amount": -5.0}
        for i in range(2, 29):
            payload[f"V{i}"] = 0.0

        msg = _make_kafka_message(
            value=json.dumps(payload).encode("utf-8"),
            offset=100,
        )

        dlq = DeadLetterQueue()
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            dead_letter_queue=dlq,
        )

        consumer._connect()
        result = consumer._parse_message(msg)

        assert result is None
        assert dlq.size() == 1
        dlq_msg = dlq.get_messages()[0]
        assert dlq_msg.offset == 100
        assert "Schema validation failed" in dlq_msg.error

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_non_dict_json_routes_to_dlq(self, mock_consumer_class):
        """JSON that parses to a non-object (e.g., list) is routed to DLQ."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        msg = _make_kafka_message(
            value=json.dumps([1, 2, 3]).encode("utf-8"),
            offset=77,
        )

        dlq = DeadLetterQueue()
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            dead_letter_queue=dlq,
        )

        consumer._connect()
        result = consumer._parse_message(msg)

        assert result is None
        assert dlq.size() == 1
        assert "not a JSON object" in dlq.get_messages()[0].error

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_none_poll_result_is_skipped(self, mock_consumer_class):
        """None poll result (timeout) is skipped without error."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] <= 3:
                return None
            # After 3 None polls, return a valid message to exit
            payload = _make_valid_transaction()
            return _make_kafka_message(
                value=json.dumps(payload).encode("utf-8"), offset=1
            )

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            poll_timeout_seconds=0.01,
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        # First 3 polls returned None (skipped), 4th returned valid message
        assert len(records) == 1
        assert call_count[0] == 4

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_partition_eof_is_handled_gracefully(self, mock_consumer_class):
        """Partition EOF messages are handled without error."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        eof_error = MagicMock()
        eof_error.code.return_value = KafkaError._PARTITION_EOF

        eof_msg = _make_kafka_message(offset=999)
        eof_msg.error.return_value = eof_error

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                return eof_msg
            # After EOF, send a valid message so we can exit the loop
            payload = _make_valid_transaction()
            return _make_kafka_message(
                value=json.dumps(payload).encode("utf-8"), offset=1000
            )

        mock_consumer.poll.side_effect = poll_side_effect

        dlq = DeadLetterQueue()
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            dead_letter_queue=dlq,
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        # EOF was handled gracefully, no DLQ entries
        assert dlq.size() == 0
        # The second poll produced a valid record
        assert len(records) == 1

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_auto_reconnect_on_poll_error(self, mock_consumer_class):
        """Consumer attempts reconnection when poll raises KafkaException."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KafkaException(KafkaError(KafkaError._TRANSPORT))
            # After reconnect, return a valid message
            if call_count[0] == 2:
                payload = _make_valid_transaction()
                return _make_kafka_message(
                    value=json.dumps(payload).encode("utf-8"),
                    offset=1,
                )
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            reconnect_timeout_seconds=5.0,
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        # Reconnection should have succeeded and produced a record
        assert len(records) == 1

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_consumer_close_cleans_up(self, mock_consumer_class):
        """close() stops the consumer and cleans up resources."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
        )
        consumer._connect()
        consumer.close()

        mock_consumer.close.assert_called_once()
        assert consumer._consumer is None
        assert not consumer.is_running

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_transaction_id_generated_if_not_in_payload(self, mock_consumer_class):
        """A UUID transaction_id is generated if not present in the message."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        payload = _make_valid_transaction()
        # No transaction_id field
        assert "transaction_id" not in payload
        msg = _make_kafka_message(value=json.dumps(payload).encode("utf-8"))

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                return msg
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        assert len(records) == 1
        # Should have a UUID-format transaction_id
        assert len(records[0].transaction_id) > 0

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_features_extracted_correctly(self, mock_consumer_class):
        """V1-V28 features are correctly extracted into features dict."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        payload = _make_valid_transaction()
        payload["V1"] = -1.35
        payload["V14"] = 7.77
        msg = _make_kafka_message(value=json.dumps(payload).encode("utf-8"))

        call_count = [0]

        def poll_side_effect(timeout=1.0):
            call_count[0] += 1
            if call_count[0] == 1:
                return msg
            return None

        mock_consumer.poll.side_effect = poll_side_effect

        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
        )

        records = []
        for record in consumer.consume():
            records.append(record)
            consumer.stop()

        assert records[0].features["V1"] == -1.35
        assert records[0].features["V14"] == 7.77
        assert len(records[0].features) == 28

    @patch("src.ingestion.kafka_consumer.Consumer")
    def test_offset_logged_for_malformed_messages(self, mock_consumer_class):
        """DLQ entries include correct offset for traceability."""
        mock_consumer = MagicMock()
        mock_consumer_class.return_value = mock_consumer

        # Test _parse_message directly for multiple malformed messages
        malformed_messages = [
            _make_kafka_message(value=b"not json", offset=10),
            _make_kafka_message(value=b"also bad", offset=20),
        ]

        dlq = DeadLetterQueue()
        consumer = TransactionKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topic="transactions",
            consumer_group="test-group",
            dead_letter_queue=dlq,
        )
        consumer._connect()

        for msg in malformed_messages:
            result = consumer._parse_message(msg)
            assert result is None

        assert dlq.size() == 2
        messages_in_dlq = dlq.get_messages()
        assert messages_in_dlq[0].offset == 10
        assert messages_in_dlq[1].offset == 20
