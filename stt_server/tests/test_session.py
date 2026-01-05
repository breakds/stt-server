"""Unit tests for PipelineSession."""

import asyncio
import unittest
from unittest.mock import MagicMock

from stt_server.data_types import AudioFrame, TranscriptionSegment
from stt_server.session import PipelineSession


def make_audio_frame(duration_ms: int = 32, sample_rate: int = 16000) -> AudioFrame:
    """Create a test audio frame (512 samples = 32ms at 16kHz)."""
    num_samples = int(duration_ms * sample_rate / 1000)
    samples = bytes(num_samples * 2)  # 16-bit PCM silence
    return AudioFrame(samples=samples, sample_rate=sample_rate, channels=1)


class TestPipelineSession(unittest.IsolatedAsyncioTestCase):
    """Tests for PipelineSession."""

    async def test_push_audio_and_get_segment(self):
        """PipelineSession should emit segments when VAD detects speech."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Speech then silence (large gap triggers end of turn)
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.1, 0.1]

        mock_model = MagicMock()
        mock_model.transcribe.return_value = "hello world"

        session = PipelineSession(
            vad=mock_vad,
            model=mock_model,
            large_gap_seconds=0.1,  # 1600 samples = ~3 chunks of silence
            min_speech_seconds=0.01,
        )

        # Push 5 audio frames to trigger VAD end-of-turn
        for _ in range(5):
            await session.push_audio(make_audio_frame())

        # Should get transcription segments (tentative then final)
        segment = await asyncio.wait_for(session.get_segment(), timeout=1.0)
        self.assertIsInstance(segment, TranscriptionSegment)
        self.assertIn("hello", segment.text.lower())

        await session.close()

    async def test_close_shuts_down_pipeline(self):
        """close() should shut down the pipeline gracefully."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        mock_vad.return_value = 0.1  # Silence

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ""

        session = PipelineSession(vad=mock_vad, model=mock_model)

        # Push some audio
        await session.push_audio(make_audio_frame())

        # Close should complete without error
        await session.close()

        # Double close should be safe
        await session.close()

    async def test_push_after_close_is_ignored(self):
        """push_audio() after close() should be ignored."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        mock_vad.return_value = 0.1

        mock_model = MagicMock()

        session = PipelineSession(vad=mock_vad, model=mock_model)
        await session.close()

        # Should not raise
        await session.push_audio(make_audio_frame())

    async def test_get_segment_raises_on_shutdown(self):
        """get_segment() should raise CancelledError when pipeline shuts down."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        mock_vad.return_value = 0.1  # No speech

        mock_model = MagicMock()

        session = PipelineSession(vad=mock_vad, model=mock_model)

        # Start waiting for segment in background
        get_task = asyncio.create_task(session.get_segment())

        # Give the task a chance to start waiting
        await asyncio.sleep(0.01)

        # Close the session
        await session.close()

        # get_segment should raise CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await get_task

    async def test_multiple_turns(self):
        """Session should handle multiple speech turns."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Two turns: speech, then 3 silence frames for large gap, repeat
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = ["first turn", "second turn"]

        session = PipelineSession(
            vad=mock_vad,
            model=mock_model,
            large_gap_seconds=0.05,  # ~800 samples = ~2 chunks
            min_speech_seconds=0.01,
        )

        # Push frames for two turns
        for _ in range(8):
            await session.push_audio(make_audio_frame())

        # First turn: tentative segment then final
        segment1 = await asyncio.wait_for(session.get_segment(), timeout=1.0)
        self.assertIn("first", segment1.text.lower())

        final1 = await asyncio.wait_for(session.get_segment(), timeout=1.0)
        self.assertTrue(final1.is_end_of_turn)

        # Second turn: tentative then final
        segment2 = await asyncio.wait_for(session.get_segment(), timeout=1.0)
        self.assertIn("second", segment2.text.lower())

        final2 = await asyncio.wait_for(session.get_segment(), timeout=1.0)
        self.assertTrue(final2.is_end_of_turn)

        await session.close()


if __name__ == "__main__":
    unittest.main()
