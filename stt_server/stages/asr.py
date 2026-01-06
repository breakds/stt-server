"""ASR stage that transcribes audio chunks using Canary-Qwen."""

import re
from typing import override

import numpy as np
from strops import merge_by_overlap

from nemo_lite import CanaryQwen
from stt_server.data_types import AudioChunk, EndOfTurnSignal, TranscriptionSegment
from stt_server.pipeline import SingleStage

# Pattern for tokenizing ASR output: words (with contractions) or punctuation
_TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for ASR overlap detection.

    Splits text into lowercase words and punctuation tokens. This improves
    overlap detection by ensuring "Hello," and "hello" can match.

    Contractions are kept together (e.g., "don't" stays as one token).

    Examples:
        "Hello, world!" → ["hello", ",", "world", "!"]
        "What's up?" → ["what's", "up", "?"]
    """
    return _TOKEN_PATTERN.findall(text.lower())


class ASRStage(SingleStage[AudioChunk | EndOfTurnSignal, TranscriptionSegment]):
    """ASR stage that transcribes audio using Canary-Qwen-2.5B.

    Processes AudioChunk items by transcribing them and emitting
    TranscriptionSegment items. Handles overlap by maintaining an
    audio buffer and using semi-global alignment for transcript merging.

    Args:
        model: The CanaryQwen model instance for transcription.
        overlap_duration: Duration of audio overlap in seconds. Default: 5.0.
    """

    _model: CanaryQwen
    _overlap_duration: float
    _sample_rate: int
    _transcript_words: list[str]
    _overlap_buffer: np.ndarray | None

    def __init__(self, model: CanaryQwen, overlap_duration: float = 5.0):
        super().__init__()
        self._model = model
        self._overlap_duration = overlap_duration
        self._sample_rate = 16000  # Canary-Qwen expects 16kHz
        self._transcript_words = []
        self._overlap_buffer = None

    @override
    async def _process_item(self, item: AudioChunk | EndOfTurnSignal) -> None:
        """Process an audio chunk or end-of-turn signal."""
        if isinstance(item, EndOfTurnSignal):
            await self._handle_end_of_turn()
        else:
            await self._handle_audio_chunk(item)

    async def _handle_audio_chunk(self, chunk: AudioChunk) -> None:
        """Transcribe an audio chunk and emit a tentative segment."""
        # Convert bytes to numpy array (16-bit signed PCM)
        audio = np.frombuffer(chunk.samples, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0  # Normalize to [-1, 1]

        # Prepend overlap from previous chunk if available
        if self._overlap_buffer is not None:
            combined_audio = np.concatenate([self._overlap_buffer, audio])
        else:
            combined_audio = audio

        # Transcribe the combined audio
        text = self._model.transcribe(combined_audio, sample_rate=chunk.sample_rate)
        new_words = tokenize(text)

        # Merge with existing transcript
        if self._transcript_words and new_words:
            self._transcript_words = merge_by_overlap(self._transcript_words, new_words)
        elif new_words:
            self._transcript_words = new_words

        # Update overlap buffer (keep last N seconds of this chunk's audio)
        overlap_samples = int(self._overlap_duration * chunk.sample_rate)
        if len(audio) > overlap_samples:
            self._overlap_buffer = audio[-overlap_samples:]
        else:
            # If chunk is shorter than overlap, keep it all
            if self._overlap_buffer is not None:
                # Combine with existing overlap, keeping only last overlap_samples
                combined = np.concatenate([self._overlap_buffer, audio])
                self._overlap_buffer = combined[-overlap_samples:]
            else:
                self._overlap_buffer = audio

        # Emit tentative segment
        if self._output_queue is not None:
            segment = TranscriptionSegment(
                text=" ".join(self._transcript_words),
                is_final=False,
                is_end_of_turn=False,
            )
            await self._output_queue.put(segment)

    async def _handle_end_of_turn(self) -> None:
        """Handle end-of-turn signal by emitting final segment and clearing state."""
        if self._output_queue is not None:
            segment = TranscriptionSegment(
                text=" ".join(self._transcript_words),
                is_final=True,
                is_end_of_turn=True,
            )
            await self._output_queue.put(segment)

        # Clear state for next turn
        self._transcript_words = []
        self._overlap_buffer = None
