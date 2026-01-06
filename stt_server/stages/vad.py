"""VAD stage that segments audio using Silero VAD with hysteresis."""

from enum import Enum
from typing import NamedTuple, override
from collections import deque

from loguru import logger
from pysilero_vad import SileroVoiceActivityDetector

from stt_server.data_types import AudioChunk, AudioFrame, EndOfTurnSignal
from stt_server.pipeline import SingleStage


class SpeechState(Enum):
    """Binary state for hysteresis-based VAD."""

    INACTIVE = "inactive"    # Fresh or has sent END OF TURN, waiting for next utterance
    SILENCE = "silence"      # Silence during utterance
    SPEAKING = "speaking"    # Speaking during utterance


class PartitionRecord(NamedTuple):
    is_speaking: bool
    num_samples: int

    def adjust(self, num_samples: int) -> "PartitionRecord":
        return self._replace(num_samples=self.num_samples + num_samples)


class AudioBuffer:
    _bytes: bytearray
    _partitions: deque[PartitionRecord]
    _speaking_samples: int

    def __init__(self):
        self.reset()

    def __len__(self) -> int:
        return len(self._bytes) // 2

    @property
    def speaking_samples(self) -> int:
        return self._speaking_samples

    def push(self, chunk: bytes, state: SpeechState) -> None:
        self._bytes.extend(chunk)
        is_speaking = state == SpeechState.SPEAKING
        if is_speaking:
            self._speaking_samples += len(chunk) // 2
        if len(self._partitions) == 0 or self._partitions[-1].is_speaking != is_speaking:
            self._partitions.append(PartitionRecord(is_speaking, len(chunk) // 2))
        else:
            self._partitions[-1] = self._partitions[-1].adjust(len(chunk) // 2)

    def trim_leading(self, num_samples: int) -> bytes:
        num_bytes = min(num_samples * 2, len(self._bytes))
        result = bytes(self._bytes[:num_bytes])
        self._bytes = self._bytes[num_bytes:]

        samples_to_trim = num_bytes // 2
        while samples_to_trim > 0:
            remaining = samples_to_trim - self._partitions[0].num_samples
            if remaining < 0:
                if self._partitions[0].is_speaking:
                    self._speaking_samples -= samples_to_trim
                self._partitions[0] = self._partitions[0].adjust(-samples_to_trim)
                break
            else:
                if self._partitions[0].is_speaking:
                    self._speaking_samples -= self._partitions[0].num_samples
                samples_to_trim = remaining
                _ = self._partitions.popleft()
        return result

    def trim_leading_silence_until(self, num_samples: int):
        if len(self._partitions) > 0 and not self._partitions[0].is_speaking:
            exceeding = self._partitions[0].num_samples - num_samples
            if exceeding > 0:
                self._partitions[0] = self._partitions[0].adjust(-exceeding)
                self._bytes = self._bytes[exceeding * 2:]

    def reset(self) -> None:
        self._bytes = bytearray()
        self._partitions = deque()
        self._speaking_samples = 0

    def pop(self) -> bytes:
        result = bytes(self._bytes)
        self.reset()
        return result

    def pop_until_largest_gap(self) -> bytes:
        """Find the largest silence gap, split at its middle, return the first part."""
        if len(self._partitions) == 0:
            return self.pop()

        # Find the largest silence gap (skip index 0 - leading gap shouldn't be a split point)
        best_gap_index = -1
        best_gap_size = 0
        for i, partition in enumerate(self._partitions):
            if i == 0:
                continue
            if not partition.is_speaking and partition.num_samples > best_gap_size:
                best_gap_index = i
                best_gap_size = partition.num_samples

        # If no silence gap found, return everything
        if best_gap_index == -1:
            return self.pop()

        # Calculate sample offset to the middle of the largest gap
        split_offset = 0
        for i, partition in enumerate(self._partitions):
            if i < best_gap_index:
                split_offset += partition.num_samples
            elif i == best_gap_index:
                # Split at the middle of the gap
                split_offset += partition.num_samples // 2
                break

        # If split point is at the very beginning or end, just pop everything
        if split_offset == 0 or split_offset >= len(self):
            return self.pop()

        return self.trim_leading(split_offset)


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
    _audio_buffer: AudioBuffer

    # VAD processing buffer (for 512-sample chunks)
    _vad_buffer: bytearray

    # Silence tracking
    _current_silence_samples: int

    def __init__(
        self,
        vad: SileroVoiceActivityDetector,
        *,
        silence_to_speech_threshold: float = 0.5,
        speech_to_silence_threshold: float = 0.35,
        small_gap_seconds: float = 0.3,
        large_gap_seconds: float = 1.5,
        min_speech_seconds: float = 1.0,
        max_buffer_seconds: float = 20.0,
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
        self._state = SpeechState.INACTIVE
        self._audio_buffer = AudioBuffer()
        self._vad_buffer = bytearray()
        self._current_silence_samples = 0

    @override
    async def _process_item(self, item: AudioFrame) -> None:
        """Process an audio frame through VAD and emit chunks as needed."""
        if item.sample_rate != self._sample_rate:
            logger.error(f"Require 16kHz audio, but got {item.sample_rate} Hz")
            return

        self._vad_buffer.extend(item.samples)

        # VAD requires fixed 512-sample chunks
        vad_chunk_bytes = self._vad.chunk_bytes()
        vad_chunk_samples = self._vad.chunk_samples()

        while len(self._vad_buffer) >= vad_chunk_bytes:
            chunk = bytes(self._vad_buffer[:vad_chunk_bytes])
            del self._vad_buffer[:vad_chunk_bytes]

            prob = self._vad(chunk)
            new_state = self._update_state(prob)
            self._audio_buffer.push(chunk, new_state)

            # Update silence counter and trim leading silence when inactive
            match new_state:
                case SpeechState.INACTIVE:
                    self._audio_buffer.trim_leading_silence_until(self._max_leading_silence_samples)
                    self._current_silence_samples += vad_chunk_samples
                case SpeechState.SILENCE:
                    self._current_silence_samples += vad_chunk_samples
                case SpeechState.SPEAKING:
                    self._current_silence_samples = 0

            # Force emit if buffer exceeds max size
            if len(self._audio_buffer) > self._max_buffer_samples:
                await self._emit_at_largest_gap()
                continue

            # Emission only happens during SILENCE state (gap detection)
            if new_state != SpeechState.SILENCE:
                continue

            # Large gap → end of turn; small gap → intermediate chunk
            if self._current_silence_samples >= self._large_gap_samples:
                await self._emit_with_end_of_turn()
            elif (self._current_silence_samples >= self._small_gap_samples and
                  self._audio_buffer.speaking_samples >= self._min_speech_samples):
                await self._emit_normally()

    def _update_state(self, prob: float) -> SpeechState:
        """Update speech state using hysteresis thresholds."""
        match self._state:
            case SpeechState.INACTIVE:
                if prob >= self._silence_to_speech_threshold:
                    self._state = SpeechState.SPEAKING
            case SpeechState.SILENCE:
                if prob >= self._silence_to_speech_threshold:
                    self._state = SpeechState.SPEAKING
            case SpeechState.SPEAKING:
                if prob < self._speech_to_silence_threshold:
                    self._state = SpeechState.SILENCE
        return self._state

    async def _emit_at_largest_gap(self):
        chunk = AudioChunk(
            samples=self._audio_buffer.pop_until_largest_gap(), sample_rate=self._sample_rate
        )
        if self._output_queue is None:
            return
        await self._output_queue.put(chunk)

    async def _emit_with_end_of_turn(self):
        speaking_samples = self._audio_buffer.speaking_samples
        chunk = AudioChunk(
            samples=self._audio_buffer.pop(), sample_rate=self._sample_rate
        )
        self._current_silence_samples = 0
        self._state = SpeechState.INACTIVE
        if self._output_queue is None:
            return
        if speaking_samples > 0:
            await self._output_queue.put(chunk)
        await self._output_queue.put(EndOfTurnSignal())

    async def _emit_normally(self):
        chunk = AudioChunk(
            samples=self._audio_buffer.pop(), sample_rate=self._sample_rate
        )
        if self._output_queue is None:
            return
        await self._output_queue.put(chunk)
