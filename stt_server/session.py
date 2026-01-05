"""Transcription session management for the STT server.

A session manages the streaming transcription for a single WebSocket connection.
It receives audio frames and produces transcription segments.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import override

from nemo_lite import CanaryQwen
from pysilero_vad import SileroVoiceActivityDetector

from stt_server.data_types import AudioFrame, TranscriptionSegment
from stt_server.pipeline import ChainedStage
from stt_server.stages import ASRStage, VADStage


class TranscriptionSession(ABC):
    """Abstract base class for transcription sessions.

    A session manages the streaming transcription for a single WebSocket
    connection. It receives audio frames and produces transcription segments.
    """

    @abstractmethod
    async def push_audio(self, frame: AudioFrame) -> None:
        """Push an audio frame into the session.

        Args:
            frame: Audio frame from the client.
        """
        pass

    @abstractmethod
    async def get_segment(self) -> TranscriptionSegment:
        """Wait for and return the next transcription segment.

        This method blocks until a segment is available.

        Returns:
            The next transcription segment.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up session resources.

        This should unblock any pending get_segment() calls.
        """
        pass


class MockTranscriptionSession(TranscriptionSession):
    """Mock session for testing the WebSocket skeleton.

    Simulates transcription by echoing back information about received audio.
    Produces a tentative segment every N frames and a final segment periodically.
    """

    def __init__(self):
        self._frame_count = 0
        self._total_samples = 0
        self._sample_rate = 16000
        self._segment_queue: asyncio.Queue[TranscriptionSegment] = asyncio.Queue()
        self._closed = False

    async def push_audio(self, frame: AudioFrame) -> None:
        self._frame_count += 1
        self._sample_rate = frame.sample_rate
        # Each sample is 2 bytes (16-bit)
        num_samples = len(frame.samples) // 2
        self._total_samples += num_samples

        # Produce tentative segment every 10 frames
        if self._frame_count % 10 == 0:
            duration = self._total_samples / self._sample_rate
            await self._segment_queue.put(
                TranscriptionSegment(
                    text=f"[Receiving audio... {duration:.1f}s]",
                    is_final=False,
                    is_end_of_turn=False,
                )
            )

        # Produce final segment every 50 frames (simulating end of utterance)
        if self._frame_count % 50 == 0:
            duration = self._total_samples / self._sample_rate
            await self._segment_queue.put(
                TranscriptionSegment(
                    text=f"[Mock transcription: {duration:.1f}s of audio received]",
                    is_final=True,
                    is_end_of_turn=True,
                )
            )
            # Reset counters for next turn
            self._total_samples = 0

    async def get_segment(self) -> TranscriptionSegment:
        return await self._segment_queue.get()

    async def close(self) -> None:
        self._closed = True
        # The caller should handle CancelledError from task cancellation
        pass


class PipelineSession(TranscriptionSession):
    """Real transcription session using the VAD + ASR pipeline.

    Creates and manages a pipeline of:
        VADStage → ASRStage → segment_queue

    Audio frames are pushed to VAD, which emits AudioChunks to ASR.
    ASR transcribes and emits TranscriptionSegments to the output queue.
    """

    _pipeline: ChainedStage[AudioFrame, TranscriptionSegment]
    _segment_queue: asyncio.Queue[TranscriptionSegment | None]
    _closed: bool

    def __init__(
        self,
        vad: SileroVoiceActivityDetector,
        model: CanaryQwen,
        *,
        small_gap_seconds: float = 0.8,
        large_gap_seconds: float = 1.5,
        min_speech_seconds: float = 3.0,
    ):
        self._segment_queue = asyncio.Queue()
        self._closed = False

        # Build pipeline: VAD → ASR
        vad_stage = VADStage(
            vad,
            small_gap_seconds=small_gap_seconds,
            large_gap_seconds=large_gap_seconds,
            min_speech_seconds=min_speech_seconds,
        )
        asr_stage = ASRStage(model)
        self._pipeline = vad_stage + asr_stage

        # Wire ASR output to the segment queue
        asr_stage._output_queue = self._segment_queue

    @override
    async def push_audio(self, frame: AudioFrame) -> None:
        if not self._closed:
            await self._pipeline.process(frame)

    @override
    async def get_segment(self) -> TranscriptionSegment:
        segment = await self._segment_queue.get()
        if segment is None:
            # Pipeline has shut down; raise to signal no more segments
            raise asyncio.CancelledError()
        return segment

    @override
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Send sentinel to shut down the pipeline
        await self._pipeline.process(None)
        await self._pipeline.join()


# Global shared resources (initialized lazily)
_shared_vad: SileroVoiceActivityDetector | None = None
_shared_model: CanaryQwen | None = None


def _get_shared_vad() -> SileroVoiceActivityDetector:
    global _shared_vad
    if _shared_vad is None:
        _shared_vad = SileroVoiceActivityDetector()
    return _shared_vad


def _get_shared_model() -> CanaryQwen:
    global _shared_model
    if _shared_model is None:
        _shared_model = CanaryQwen()
    return _shared_model


def create_session() -> TranscriptionSession:
    """Factory function to create a transcription session.

    Returns:
        A new TranscriptionSession instance.
    """
    return PipelineSession(
        vad=_get_shared_vad(),
        model=_get_shared_model(),
    )
