"""Unit tests for PipelineSession."""

import base64
import unittest
from unittest.mock import AsyncMock, MagicMock

from stt_server.data_types import AudioFrame
from stt_server.session import PipelineSession


def make_audio_frame(duration_ms: int = 32, sample_rate: int = 16000) -> AudioFrame:
    """Create a test audio frame (512 samples = 32ms at 16kHz)."""
    num_samples = int(duration_ms * sample_rate / 1000)
    raw_samples = bytes(num_samples * 2)  # 16-bit PCM silence
    samples = base64.b64encode(raw_samples)
    return AudioFrame(samples=samples, sample_rate=sample_rate, channels=1)


class TestPipelineSession(unittest.IsolatedAsyncioTestCase):
    """Tests for PipelineSession."""

    async def test_sends_segments_to_websocket(self):
        """PipelineSession should send segments to WebSocket when VAD detects speech."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Speech then silence (large gap triggers end of turn)
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.1, 0.1]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = "hello world"

        session = PipelineSession(
            websocket=mock_websocket,
            vad=mock_vad,
            model=mock_model,
            large_gap_seconds=0.1,  # 1600 samples = ~3 chunks of silence
            min_speech_seconds=0.01,
        )

        # Push 5 audio frames to trigger VAD end-of-turn
        for _ in range(5):
            await session.push_audio(make_audio_frame())

        await session.close()

        # WebSocket should have received segments
        self.assertTrue(mock_websocket.send_json.called)

        # Check that at least one segment contains "hello"
        calls = mock_websocket.send_json.call_args_list
        texts = [call[0][0].get("text", "") for call in calls]
        self.assertTrue(any("hello" in text.lower() for text in texts))

    async def test_close_shuts_down_pipeline(self):
        """close() should shut down the pipeline gracefully."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        mock_vad.return_value = 0.1  # Silence

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ""

        session = PipelineSession(
            websocket=mock_websocket,
            vad=mock_vad,
            model=mock_model,
        )

        # Push some audio
        await session.push_audio(make_audio_frame())

        # Close should complete without error
        await session.close()

        # Double close should be safe
        await session.close()

    async def test_push_after_close_is_ignored(self):
        """push_audio() after close() should be ignored."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        mock_vad.return_value = 0.1

        mock_model = MagicMock()

        session = PipelineSession(
            websocket=mock_websocket,
            vad=mock_vad,
            model=mock_model,
        )
        await session.close()

        # Should not raise
        await session.push_audio(make_audio_frame())

    async def test_multiple_turns(self):
        """Session should handle multiple speech turns."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Two turns: speech, then 3 silence frames for large gap, repeat
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = ["first turn", "second turn"]

        session = PipelineSession(
            websocket=mock_websocket,
            vad=mock_vad,
            model=mock_model,
            large_gap_seconds=0.05,  # ~800 samples = ~2 chunks
            min_speech_seconds=0.01,
        )

        # Push frames for two turns
        for _ in range(8):
            await session.push_audio(make_audio_frame())

        await session.close()

        # WebSocket should have received segments from both turns
        calls = mock_websocket.send_json.call_args_list
        texts = [call[0][0].get("text", "") for call in calls]

        # Should have segments containing "first" and "second"
        self.assertTrue(any("first" in text.lower() for text in texts))
        self.assertTrue(any("second" in text.lower() for text in texts))

        # Should have end-of-turn segments
        end_of_turns = [call[0][0].get("isEndOfTurn", False) for call in calls]
        self.assertGreaterEqual(sum(end_of_turns), 2)


if __name__ == "__main__":
    unittest.main()
