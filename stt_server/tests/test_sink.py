"""Unit tests for SinkStage."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from stt_server.data_types import TranscriptionSegment
from stt_server.stages.sink import SinkStage


class TestSinkStage(unittest.IsolatedAsyncioTestCase):
    """Tests for SinkStage."""

    async def test_sends_segment_via_websocket(self):
        """SinkStage should send segment as JSON via WebSocket."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        sink = SinkStage(mock_websocket)

        segment = TranscriptionSegment(
            text="Hello world",
            is_final=False,
            is_end_of_turn=False,
        )
        await sink.process(segment)
        await sink.process(None)  # Sentinel to stop
        await sink.join()

        mock_websocket.send_json.assert_called_once_with({
            "text": "Hello world",
            "isFinal": False,
            "isEndOfTurn": False,
        })

    async def test_sends_multiple_segments(self):
        """SinkStage should send multiple segments in order."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        sink = SinkStage(mock_websocket)

        segments = [
            TranscriptionSegment(text="First", is_final=False),
            TranscriptionSegment(text="Second", is_final=False),
            TranscriptionSegment(text="Final", is_final=True, is_end_of_turn=True),
        ]
        for segment in segments:
            await sink.process(segment)
        await sink.process(None)
        await sink.join()

        self.assertEqual(mock_websocket.send_json.call_count, 3)

        calls = mock_websocket.send_json.call_args_list
        self.assertEqual(calls[0][0][0]["text"], "First")
        self.assertEqual(calls[1][0][0]["text"], "Second")
        self.assertEqual(calls[2][0][0]["text"], "Final")
        self.assertTrue(calls[2][0][0]["isFinal"])
        self.assertTrue(calls[2][0][0]["isEndOfTurn"])

    async def test_uses_camel_case_serialization(self):
        """SinkStage should serialize with camelCase keys."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        sink = SinkStage(mock_websocket)

        segment = TranscriptionSegment(
            text="Test",
            is_final=True,
            is_end_of_turn=True,
        )
        await sink.process(segment)
        await sink.process(None)
        await sink.join()

        call_args = mock_websocket.send_json.call_args[0][0]
        self.assertIn("isFinal", call_args)
        self.assertIn("isEndOfTurn", call_args)
        self.assertNotIn("is_final", call_args)
        self.assertNotIn("is_end_of_turn", call_args)


if __name__ == "__main__":
    unittest.main()
