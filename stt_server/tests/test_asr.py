"""Unit tests for ASRStage."""

import asyncio
import unittest
from unittest.mock import MagicMock

import numpy as np

from stt_server.data_types import AudioChunk, EndOfTurnSignal, TranscriptionSegment
from stt_server.stages.asr import ASRStage
from stt_server.pipeline import SingleStage


class CollectSink(SingleStage[TranscriptionSegment, None]):
    """Test sink that collects segments into a list."""

    def __init__(self, result: list[TranscriptionSegment]):
        super().__init__()
        self._result = result

    async def _process_item(self, item: TranscriptionSegment) -> None:
        self._result.append(item)


def make_audio_chunk(duration_ms: int = 100, sample_rate: int = 16000) -> AudioChunk:
    """Create a test audio chunk with silence."""
    num_samples = int(duration_ms * sample_rate / 1000)
    # Create silent audio as 16-bit signed PCM
    samples = np.zeros(num_samples, dtype=np.int16).tobytes()
    return AudioChunk(samples=samples, sample_rate=sample_rate)


class TestASRStage(unittest.IsolatedAsyncioTestCase):
    """Tests for ASRStage."""

    async def test_transcribes_audio_chunk(self):
        """ASRStage should transcribe audio chunk and emit tentative segment."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "hello world"

        results: list[TranscriptionSegment] = []
        asr = ASRStage(mock_model, overlap_duration=5.0)
        pipeline = asr + CollectSink(results)

        await pipeline.process(make_audio_chunk())
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "hello world")
        self.assertFalse(results[0].is_final)
        self.assertFalse(results[0].is_end_of_turn)

    async def test_emits_final_on_end_of_turn(self):
        """ASRStage should emit final segment on EndOfTurnSignal."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = "test transcript"

        results: list[TranscriptionSegment] = []
        asr = ASRStage(mock_model)
        pipeline = asr + CollectSink(results)

        await pipeline.process(make_audio_chunk())
        await pipeline.process(EndOfTurnSignal())
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(len(results), 2)
        # First: tentative
        self.assertFalse(results[0].is_final)
        # Second: final
        self.assertTrue(results[1].is_final)
        self.assertTrue(results[1].is_end_of_turn)
        self.assertEqual(results[1].text, "test transcript")

    async def test_merges_transcripts_with_overlap(self):
        """ASRStage should merge transcripts using strops.merge_by_overlap."""
        mock_model = MagicMock()
        # First chunk transcribes to "the quick brown"
        # Second chunk (with overlap) transcribes to "brown fox jumps"
        mock_model.transcribe.side_effect = [
            "the quick brown",
            "brown fox jumps",
        ]

        results: list[TranscriptionSegment] = []
        asr = ASRStage(mock_model, overlap_duration=0.1)
        pipeline = asr + CollectSink(results)

        await pipeline.process(make_audio_chunk(duration_ms=200))
        await pipeline.process(make_audio_chunk(duration_ms=200))
        await pipeline.process(None)
        await pipeline.join()

        self.assertEqual(len(results), 2)
        # Second segment should be merged
        self.assertEqual(results[1].text, "the quick brown fox jumps")

    async def test_clears_state_after_end_of_turn(self):
        """ASRStage should clear state after EndOfTurnSignal."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = [
            "first turn",
            "second turn",
        ]

        results: list[TranscriptionSegment] = []
        asr = ASRStage(mock_model)
        pipeline = asr + CollectSink(results)

        # First turn
        await pipeline.process(make_audio_chunk())
        await pipeline.process(EndOfTurnSignal())
        # Second turn
        await pipeline.process(make_audio_chunk())
        await pipeline.process(EndOfTurnSignal())
        await pipeline.process(None)
        await pipeline.join()

        # Should have 4 segments: tentative, final, tentative, final
        self.assertEqual(len(results), 4)
        self.assertEqual(results[1].text, "first turn")
        self.assertEqual(results[3].text, "second turn")
        # Transcripts should NOT merge across turns
        self.assertNotIn("first", results[3].text)

    async def test_handles_empty_transcription(self):
        """ASRStage should handle empty transcription gracefully."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ""

        results: list[TranscriptionSegment] = []
        asr = ASRStage(mock_model)
        pipeline = asr + CollectSink(results)

        await pipeline.process(make_audio_chunk())
        await pipeline.process(EndOfTurnSignal())
        await pipeline.process(None)
        await pipeline.join()

        # Should still emit segments (just with empty text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "")
        self.assertEqual(results[1].text, "")


if __name__ == "__main__":
    unittest.main()
