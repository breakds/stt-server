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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from stt_server.data_types import AudioFrame, ErrorResponse
from stt_server.session import create_session

app = FastAPI(
    title="STT Server",
    description="Speech-to-Text WebSocket service for conversational agents",
    version="0.1.0",
)


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
        _ = sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        await session.close()
