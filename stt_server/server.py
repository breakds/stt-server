"""WebSocket-based Speech-to-Text server using FastAPI.

Usage:
    uvicorn stt_server.server:app --host 0.0.0.0 --port 8000

WebSocket Protocol:
    1. Client connects to /ws/transcribe
    2. Client sends AudioFrame JSON messages (continuous streaming)
    3. Server streams TranscriptionSegment JSON messages back
    4. Connection stays open for multiple turns
    5. Client disconnects when done
"""

import asyncio
from abc import ABC, abstractmethod

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from stt_server.data_types import AudioFrame, TranscriptionSegment, ErrorResponse

app = FastAPI(
    title="STT Server",
    description="Speech-to-Text WebSocket service for conversational agents",
    version="0.1.0",
)


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
        # Put a sentinel to unblock any waiting get_segment() call
        # The caller should handle CancelledError from task cancellation
        pass


def create_session() -> TranscriptionSession:
    """Factory function to create a transcription session.

    Returns:
        A new TranscriptionSession instance.
    """
    # TODO: Replace with real implementation using CanaryQwen
    return MockTranscriptionSession()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for streaming audio transcription.

    Protocol:
        - Client sends: AudioFrame JSON messages
        - Server sends: TranscriptionSegment JSON messages

    The connection stays open for multiple speaking turns. The client
    disconnects when they no longer need the service.
    """
    await websocket.accept()
    session = create_session()

    # Task to wait for and send transcription segments
    async def segment_sender():
        try:
            while True:
                segment = await session.get_segment()
                await websocket.send_json(segment.model_dump(by_alias=True))
        except asyncio.CancelledError:
            pass

    sender_task = asyncio.create_task(segment_sender())

    try:
        while True:
            data = await websocket.receive_json()

            try:
                frame = AudioFrame.model_validate(data)
                await session.push_audio(frame)
            except ValidationError as e:
                logger.warning(f"Invalid AudioFrame: {e}")
                response = ErrorResponse(
                    error=f"Invalid AudioFrame: {e}",
                    code="INVALID_MESSAGE",
                )
                await websocket.send_json(response.model_dump(by_alias=True))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            response = ErrorResponse(error=str(e), code="INTERNAL_ERROR")
            await websocket.send_json(response.model_dump(by_alias=True))
        except Exception:
            pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        await session.close()
