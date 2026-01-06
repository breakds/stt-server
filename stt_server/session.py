"""Transcription session management for the STT server.

A session manages the streaming transcription for a single WebSocket connection.
It receives audio frames and the pipeline sends transcription segments directly.
"""

from abc import ABC, abstractmethod
from typing import override

from fastapi import WebSocket
from nemo_lite import CanaryQwen
from pysilero_vad import SileroVoiceActivityDetector

from stt_server.data_types import AudioFrame
from stt_server.pipeline import ChainedStage
from stt_server.stages import ASRStage, SinkStage, VADStage


class TranscriptionSession(ABC):
    """Abstract base class for transcription sessions.

    A session manages the streaming transcription for a single WebSocket
    connection. It receives audio frames and the pipeline sends segments
    directly to the WebSocket.
    """

    @abstractmethod
    async def push_audio(self, frame: AudioFrame) -> None:
        """Push an audio frame into the session.

        Args:
            frame: Audio frame from the client.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up session resources."""
        pass


class PipelineSession(TranscriptionSession):
    """Real transcription session using the VAD + ASR + Sink pipeline.

    Creates and manages a pipeline of:
        VADStage → ASRStage → SinkStage → WebSocket

    Audio frames are pushed to VAD, which emits AudioChunks to ASR.
    ASR transcribes and emits TranscriptionSegments to Sink.
    Sink sends segments directly to the WebSocket.
    """

    _pipeline: ChainedStage[AudioFrame, None]
    _closed: bool

    def __init__(
        self,
        websocket: WebSocket,
        vad: SileroVoiceActivityDetector,
        model: CanaryQwen,
        *,
        small_gap_seconds: float = 0.3,
        large_gap_seconds: float = 1.5,
        min_speech_seconds: float = 1.0,
    ):
        self._closed = False

        # Build pipeline: VAD → ASR → Sink
        vad_stage = VADStage(
            vad,
            small_gap_seconds=small_gap_seconds,
            large_gap_seconds=large_gap_seconds,
            min_speech_seconds=min_speech_seconds,
        )
        asr_stage = ASRStage(model)
        sink_stage = SinkStage(websocket)
        self._pipeline = vad_stage + asr_stage + sink_stage

    @override
    async def push_audio(self, frame: AudioFrame) -> None:
        if not self._closed:
            await self._pipeline.process(frame)

    @override
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Send sentinel to shut down the pipeline
        await self._pipeline.process(None)
        await self._pipeline.join()


# Global shared resources (initialized at startup via init_shared_resources)
_shared_vad: SileroVoiceActivityDetector | None = None
_shared_model: CanaryQwen | None = None


def init_shared_resources(device: str = "cuda") -> None:
    """Initialize shared VAD and ASR model at startup.

    Call this before accepting connections to avoid timeout during
    model loading on the first request.

    Args:
        device: Device for ASR model ('cuda' or 'cpu'). Default: 'cuda'.
    """
    global _shared_vad, _shared_model
    if _shared_vad is None:
        _shared_vad = SileroVoiceActivityDetector()
    if _shared_model is None:
        _shared_model = CanaryQwen(device=device)


def _get_shared_vad() -> SileroVoiceActivityDetector:
    if _shared_vad is None:
        raise RuntimeError("Shared resources not initialized. Call init_shared_resources() first.")
    return _shared_vad


def _get_shared_model() -> CanaryQwen:
    if _shared_model is None:
        raise RuntimeError("Shared resources not initialized. Call init_shared_resources() first.")
    return _shared_model


def create_session(websocket: WebSocket) -> TranscriptionSession:
    """Factory function to create a transcription session.

    Args:
        websocket: The WebSocket connection for sending segments.

    Returns:
        A new TranscriptionSession instance.
    """
    return PipelineSession(
        websocket=websocket,
        vad=_get_shared_vad(),
        model=_get_shared_model(),
    )
