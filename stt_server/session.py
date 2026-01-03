"""Transcription session management for the STT server.

A session manages the streaming transcription for a single WebSocket connection.
It receives audio frames and produces transcription segments.
"""

import asyncio
from abc import ABC, abstractmethod

from stt_server.data_types import AudioFrame, TranscriptionSegment


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


def create_session() -> TranscriptionSession:
    """Factory function to create a transcription session.

    Returns:
        A new TranscriptionSession instance.
    """
    # TODO: Replace with real implementation using CanaryQwen
    return MockTranscriptionSession()
