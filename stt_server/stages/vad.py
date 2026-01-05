"""VAD stage that segments audio using Silero VAD with hysteresis."""

from enum import Enum
from typing import override

from pysilero_vad import SileroVoiceActivityDetector

from stt_server.data_types import AudioChunk, AudioFrame, EndOfTurnSignal
from stt_server.pipeline import SingleStage


class SpeechState(Enum):
    """Binary state for hysteresis-based VAD."""

    SILENCE = "silence"
    SPEAKING = "speaking"

# TODOs
# 1. We assume 16kHz, need to check that the input audio frame matches it. If not match, do not process and log error.
# 2. _trim_leading_silence is an unnecessary wrapper
# 3. 
class VADStage(SingleStage[AudioFrame, AudioChunk | EndOfTurnSignal]):
    """VAD stage that segments audio based on speech/silence detection.

    Uses Silero VAD with hysteresis to classify audio frames and emits
    AudioChunk segments based on gap detection.

    Args:
        vad: The SileroVoiceActivityDetector instance.
        silence_to_speech_threshold: VAD probability to transition silence→speaking.
        speech_to_silence_threshold: VAD probability to transition speaking→silence.
        small_gap_seconds: Silence duration to trigger chunk emission.
        large_gap_seconds: Silence duration to trigger end-of-turn.
        min_speech_seconds: Minimum speech before small gap triggers emission.
        max_buffer_seconds: Maximum buffer size before forced emission.
        max_leading_silence_seconds: Maximum silence kept at buffer start.
    """

    # VAD and parameters
    _vad: SileroVoiceActivityDetector
    _silence_to_speech_threshold: float
    _speech_to_silence_threshold: float
    _small_gap_samples: int
    _large_gap_samples: int
    _min_speech_samples: int
    _max_buffer_samples: int
    _max_leading_silence_samples: int

    # State tracking
    _state: SpeechState
    _sample_rate: int

    # Audio buffer (accumulated samples as bytes)
    _audio_buffer: bytearray

    # VAD processing buffer (for 512-sample chunks)
    _vad_buffer: bytearray

    # Per-sample state tracking for gap detection
    # Each entry is True if that sample was in SPEAKING state
    _sample_states: list[bool]

    # Silence tracking
    _current_silence_samples: int
    _total_speech_samples: int

    def __init__(
        self,
        vad: SileroVoiceActivityDetector,
        *,
        silence_to_speech_threshold: float = 0.5,
        speech_to_silence_threshold: float = 0.35,
        small_gap_seconds: float = 0.8,
        large_gap_seconds: float = 1.5,
        min_speech_seconds: float = 3.0,
        max_buffer_seconds: float = 25.0,
        max_leading_silence_seconds: float = 3.0,
    ):
        super().__init__()
        self._vad = vad
        self._silence_to_speech_threshold = silence_to_speech_threshold
        self._speech_to_silence_threshold = speech_to_silence_threshold

        # Convert seconds to samples (assuming 16kHz)
        self._sample_rate = 16000
        self._small_gap_samples = int(small_gap_seconds * self._sample_rate)
        self._large_gap_samples = int(large_gap_seconds * self._sample_rate)
        self._min_speech_samples = int(min_speech_seconds * self._sample_rate)
        self._max_buffer_samples = int(max_buffer_seconds * self._sample_rate)
        self._max_leading_silence_samples = int(
            max_leading_silence_seconds * self._sample_rate
        )

        # Initialize state
        self._state = SpeechState.SILENCE
        self._audio_buffer = bytearray()
        self._vad_buffer = bytearray()
        self._sample_states = []
        self._current_silence_samples = 0
        self._total_speech_samples = 0

    @override
    async def _process_item(self, item: AudioFrame) -> None:
        """Process an audio frame through VAD and emit chunks as needed."""
        # Add frame to VAD buffer
        self._vad_buffer.extend(item.samples)

        # Process complete 512-sample chunks through VAD
        vad_chunk_bytes = self._vad.chunk_bytes()  # 1024 bytes (512 samples * 2)
        vad_chunk_samples = self._vad.chunk_samples()  # 512 samples

        while len(self._vad_buffer) >= vad_chunk_bytes:
            # Extract chunk for VAD processing
            chunk = bytes(self._vad_buffer[:vad_chunk_bytes])
            del self._vad_buffer[:vad_chunk_bytes]

            # Get speech probability
            prob = self._vad(chunk)

            # Update state with hysteresis
            new_state = self._update_state(prob)

            # Add chunk to audio buffer
            self._audio_buffer.extend(chunk)

            # Track per-sample states for this chunk
            is_speaking = new_state == SpeechState.SPEAKING
            self._sample_states.extend([is_speaking] * vad_chunk_samples)

            # Update counters
            if is_speaking:
                self._total_speech_samples += vad_chunk_samples
                self._current_silence_samples = 0
            else:
                self._current_silence_samples += vad_chunk_samples

            # Check for gap-based emission
            await self._check_emit_conditions()

    def _update_state(self, prob: float) -> SpeechState:
        """Update speech state using hysteresis thresholds."""
        if self._state == SpeechState.SILENCE:
            if prob >= self._silence_to_speech_threshold:
                self._state = SpeechState.SPEAKING
        else:  # SPEAKING
            if prob < self._speech_to_silence_threshold:
                self._state = SpeechState.SILENCE
        return self._state

    async def _check_emit_conditions(self) -> None:
        """Check if we should emit a chunk based on gap or buffer size."""
        buffer_samples = len(self._audio_buffer) // 2  # 16-bit = 2 bytes per sample

        # Check for large gap (end of turn)
        if self._current_silence_samples >= self._large_gap_samples:
            if self._total_speech_samples > 0:
                await self._emit_buffer(is_end_of_turn=True)
            else:
                # Silence-only buffer, just emit end-of-turn signal
                await self._emit_end_of_turn_only()
            return

        # Check for small gap (emit chunk if enough speech accumulated)
        if (
            self._current_silence_samples >= self._small_gap_samples
            and self._total_speech_samples >= self._min_speech_samples
        ):
            await self._emit_buffer(is_end_of_turn=False)
            return

        # Check for max buffer size
        if buffer_samples >= self._max_buffer_samples:
            await self._emit_at_largest_gap()

    async def _emit_buffer(self, *, is_end_of_turn: bool) -> None:
        """Emit the current buffer as an AudioChunk."""
        if not self._audio_buffer:
            if is_end_of_turn:
                await self._emit_end_of_turn_only()
            return

        # Trim leading silence
        trimmed_audio = self._trim_leading_silence()

        if self._output_queue is not None:
            if trimmed_audio:
                chunk = AudioChunk(
                    samples=bytes(trimmed_audio), sample_rate=self._sample_rate
                )
                await self._output_queue.put(chunk)

            if is_end_of_turn:
                await self._output_queue.put(EndOfTurnSignal())

        # Reset state for next segment
        self._reset_buffer_state()

    async def _emit_end_of_turn_only(self) -> None:
        """Emit only an EndOfTurnSignal without an AudioChunk."""
        if self._output_queue is not None:
            await self._output_queue.put(EndOfTurnSignal())
        self._reset_buffer_state()

    async def _emit_at_largest_gap(self) -> None:
        """Find largest silence gap in buffer and emit up to that point."""
        if not self._sample_states:
            return

        # Find the largest continuous silence gap
        best_gap_start = -1
        best_gap_length = 0
        current_gap_start = -1
        current_gap_length = 0

        for i, is_speaking in enumerate(self._sample_states):
            if not is_speaking:
                if current_gap_start == -1:
                    current_gap_start = i
                current_gap_length += 1
            else:
                if current_gap_length > best_gap_length:
                    best_gap_start = current_gap_start
                    best_gap_length = current_gap_length
                current_gap_start = -1
                current_gap_length = 0

        # Check trailing gap
        if current_gap_length > best_gap_length:
            best_gap_start = current_gap_start
            best_gap_length = current_gap_length

        # If we found a gap, split there; otherwise emit everything
        if best_gap_start > 0 and best_gap_length > 0:
            # Split at the middle of the gap
            split_point = best_gap_start + best_gap_length // 2
            split_byte = split_point * 2  # 16-bit = 2 bytes per sample

            # Emit first part
            first_part = self._audio_buffer[:split_byte]
            first_states = self._sample_states[:split_point]

            # Trim leading silence from first part
            trimmed = self._trim_leading_silence_from(first_part, first_states)

            if self._output_queue is not None and trimmed:
                chunk = AudioChunk(samples=bytes(trimmed), sample_rate=self._sample_rate)
                await self._output_queue.put(chunk)

            # Keep remainder in buffer
            self._audio_buffer = self._audio_buffer[split_byte:]
            self._sample_states = self._sample_states[split_point:]

            # Recalculate speech samples
            self._total_speech_samples = sum(1 for s in self._sample_states if s)
        else:
            # No good gap found, emit everything
            await self._emit_buffer(is_end_of_turn=False)

    def _trim_leading_silence(self) -> bytearray:
        """Trim leading silence from buffer, keeping at most max_leading_silence."""
        return self._trim_leading_silence_from(self._audio_buffer, self._sample_states)

    def _trim_leading_silence_from(
        self, audio: bytearray, states: list[bool]
    ) -> bytearray:
        """Trim leading silence from given audio/states pair."""
        if not states:
            return bytearray()

        # Find first speech sample
        first_speech = -1
        for i, is_speaking in enumerate(states):
            if is_speaking:
                first_speech = i
                break

        if first_speech == -1:
            # All silence - keep max_leading_silence worth
            keep_samples = min(len(states), self._max_leading_silence_samples)
            return audio[: keep_samples * 2]

        # Calculate how much leading silence to keep
        leading_silence = first_speech
        keep_silence = min(leading_silence, self._max_leading_silence_samples)
        trim_samples = leading_silence - keep_silence

        return audio[trim_samples * 2 :]

    def _reset_buffer_state(self) -> None:
        """Reset buffer state for the next segment."""
        self._audio_buffer = bytearray()
        self._sample_states = []
        self._current_silence_samples = 0
        self._total_speech_samples = 0
        # Note: We keep _state and _vad_buffer intact for continuity
