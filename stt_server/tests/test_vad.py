"""Unit tests for VADStage."""

import asyncio
import base64
import unittest
from unittest.mock import MagicMock

from stt_server.data_types import AudioChunk, AudioFrame, EndOfTurnSignal
from stt_server.pipeline import SingleStage
from stt_server.stages.vad import SpeechState, VADStage


class CollectSink(SingleStage[AudioChunk | EndOfTurnSignal, None]):
    """Test sink that collects output items into a list."""

    def __init__(self, result: list[AudioChunk | EndOfTurnSignal]):
        super().__init__()
        self._result = result

    async def _process_item(self, item: AudioChunk | EndOfTurnSignal) -> None:
        self._result.append(item)


def make_audio_frame(duration_ms: int = 32, sample_rate: int = 16000) -> AudioFrame:
    """Create a test audio frame with silence (512 samples = 32ms at 16kHz)."""
    num_samples = int(duration_ms * sample_rate / 1000)
    # Silent audio as 16-bit signed PCM (all zeros), base64-encoded
    raw_samples = bytes(num_samples * 2)
    samples = base64.b64encode(raw_samples)
    return AudioFrame(samples=samples, sample_rate=sample_rate, channels=1)


def make_vad_chunk_frame() -> AudioFrame:
    """Create an audio frame that matches VAD chunk size (512 samples)."""
    return make_audio_frame(duration_ms=32)  # 512 samples at 16kHz


class TestVADStageHysteresis(unittest.IsolatedAsyncioTestCase):
    """Tests for VAD hysteresis state machine."""

    async def test_inactive_to_speech_transition(self):
        """State should transition from inactive to speech at high threshold."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512

        vad_stage = VADStage(
            mock_vad,
            silence_to_speech_threshold=0.5,
            speech_to_silence_threshold=0.35,
        )

        # Below threshold - should stay inactive
        vad_stage._update_state(0.4)
        self.assertEqual(vad_stage._state, SpeechState.INACTIVE)

        # At threshold - should transition to speaking
        vad_stage._update_state(0.5)
        self.assertEqual(vad_stage._state, SpeechState.SPEAKING)

    async def test_speech_to_silence_transition(self):
        """State should transition from speech to silence at low threshold."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512

        vad_stage = VADStage(
            mock_vad,
            silence_to_speech_threshold=0.5,
            speech_to_silence_threshold=0.35,
        )

        # Start in speaking state
        vad_stage._state = SpeechState.SPEAKING

        # Above low threshold - should stay in speaking
        vad_stage._update_state(0.4)
        self.assertEqual(vad_stage._state, SpeechState.SPEAKING)

        # Below low threshold - should transition to silence
        vad_stage._update_state(0.3)
        self.assertEqual(vad_stage._state, SpeechState.SILENCE)

    async def test_hysteresis_prevents_rapid_toggling(self):
        """Hysteresis should prevent toggling between 0.35 and 0.5."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512

        vad_stage = VADStage(
            mock_vad,
            silence_to_speech_threshold=0.5,
            speech_to_silence_threshold=0.35,
        )

        # Start inactive, probability at 0.4 (between thresholds)
        vad_stage._update_state(0.4)
        self.assertEqual(vad_stage._state, SpeechState.INACTIVE)

        # Still at 0.4 - should stay inactive
        vad_stage._update_state(0.4)
        self.assertEqual(vad_stage._state, SpeechState.INACTIVE)

        # Jump to 0.6 - should become speaking
        vad_stage._update_state(0.6)
        self.assertEqual(vad_stage._state, SpeechState.SPEAKING)

        # Back to 0.4 - should stay speaking (above 0.35)
        vad_stage._update_state(0.4)
        self.assertEqual(vad_stage._state, SpeechState.SPEAKING)


class TestVADStageGapDetection(unittest.IsolatedAsyncioTestCase):
    """Tests for VAD gap detection and emission."""

    async def test_emits_chunk_on_large_gap(self):
        """VADStage should emit AudioChunk + EndOfTurnSignal on large gap."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Use side_effect to return different values for sequential calls
        # 1 speech frame, then 4 silence frames
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.1, 0.1]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=0.1,  # 1600 samples = 100ms
            small_gap_seconds=0.5,  # Higher than large gap, so small gap won't trigger
            min_speech_seconds=0.01,
        )
        pipeline = vad_stage + CollectSink(results)

        # First: speech detected, then silence
        # 100ms = 1600 samples, each frame is 512 samples, need ~4 frames
        for _ in range(5):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should have AudioChunk followed by EndOfTurnSignal
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], AudioChunk)
        self.assertIsInstance(results[1], EndOfTurnSignal)

    async def test_emits_chunk_on_small_gap_with_enough_speech(self):
        """VADStage should emit AudioChunk on small gap if min speech met."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # 2 speech frames, then 2 silence frames
        mock_vad.side_effect = [0.9, 0.9, 0.1, 0.1]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=0.5,
            small_gap_seconds=0.05,  # 800 samples = 50ms
            min_speech_seconds=0.05,  # 800 samples = 50ms
        )
        pipeline = vad_stage + CollectSink(results)

        # 4 frames total
        for _ in range(4):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should have emitted AudioChunk (no EndOfTurnSignal for small gap)
        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(results[0], AudioChunk)

    async def test_no_emit_on_small_gap_without_enough_speech(self):
        """VADStage should NOT emit on small gap if min speech not met."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # 1 speech frame, then 2 silence frames
        mock_vad.side_effect = [0.9, 0.1, 0.1]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=1.0,  # Very long - won't trigger
            small_gap_seconds=0.05,
            min_speech_seconds=1.0,  # Very long - won't be met
        )
        pipeline = vad_stage + CollectSink(results)

        # 3 frames total
        for _ in range(3):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should NOT have emitted anything (min speech not met)
        self.assertEqual(len(results), 0)


class TestVADStageBufferManagement(unittest.IsolatedAsyncioTestCase):
    """Tests for VAD buffer management."""

    async def test_trims_leading_silence(self):
        """VADStage should trim excess leading silence from emitted chunks."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # 3 silence, 1 speech, 2 silence
        mock_vad.side_effect = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=0.05,  # Short for testing
            max_leading_silence_seconds=0.032,  # 512 samples = 1 chunk
        )
        pipeline = vad_stage + CollectSink(results)

        # 6 frames total
        for _ in range(6):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should have emitted with trimmed leading silence
        self.assertGreaterEqual(len(results), 1)
        if isinstance(results[0], AudioChunk):
            # With max_leading_silence of 512 samples and 3 chunks of silence before speech,
            # we should have trimmed 2 chunks worth of silence
            # Original: 3 silence + 1 speech + 2 silence = 6 chunks = 6144 bytes
            # After trim: 1 silence + 1 speech + 2 silence = 4 chunks = 4096 bytes
            self.assertLess(len(results[0].samples), 6 * 1024)

    async def test_emits_on_max_buffer_size(self):
        """VADStage should force emit when buffer reaches max size."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Continuous speech
        mock_vad.side_effect = [0.9, 0.9, 0.9, 0.9, 0.9]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=100.0,  # Very long - won't trigger normally
            small_gap_seconds=100.0,
            max_buffer_seconds=0.1,  # 1600 samples = ~3 chunks
        )
        pipeline = vad_stage + CollectSink(results)

        # 5 speech frames
        for _ in range(5):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should have force-emitted due to max buffer size
        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(results[0], AudioChunk)


class TestVADStageStateReset(unittest.IsolatedAsyncioTestCase):
    """Tests for VAD state management across turns."""

    async def test_resets_buffer_after_end_of_turn(self):
        """VADStage should reset buffer state after emitting end-of-turn."""
        mock_vad = MagicMock()
        mock_vad.chunk_bytes.return_value = 1024
        mock_vad.chunk_samples.return_value = 512
        # Two turns: speech-silence-silence, then speech-silence-silence
        mock_vad.side_effect = [0.9, 0.1, 0.1, 0.9, 0.1, 0.1]

        results: list[AudioChunk | EndOfTurnSignal] = []
        vad_stage = VADStage(
            mock_vad,
            large_gap_seconds=0.05,  # 800 samples, need 2 silence frames
            min_speech_seconds=0.01,
        )
        pipeline = vad_stage + CollectSink(results)

        # Process all 6 frames
        for _ in range(6):
            await pipeline.process(make_vad_chunk_frame())

        await pipeline.process(None)
        await pipeline.join()

        # Should have 4 items: AudioChunk + EndOfTurnSignal for each turn
        self.assertEqual(len(results), 4)
        self.assertIsInstance(results[0], AudioChunk)
        self.assertIsInstance(results[1], EndOfTurnSignal)
        self.assertIsInstance(results[2], AudioChunk)
        self.assertIsInstance(results[3], EndOfTurnSignal)


if __name__ == "__main__":
    unittest.main()
