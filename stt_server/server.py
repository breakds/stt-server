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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import click
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from stt_server.data_types import AudioFrame, ErrorResponse
from stt_server.session import create_session, init_shared_resources


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load models at startup to avoid delays on first connection."""
    logger.info("Loading VAD and ASR models...")
    init_shared_resources()
    logger.info("Models loaded successfully")
    yield


app = FastAPI(
    title="STT Server",
    description="Speech-to-Text WebSocket service for conversational agents",
    version="0.1.0",
    lifespan=lifespan,
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
    session = create_session(websocket)

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
        await session.close()


@click.command()
@click.option("--port", default=15751, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
def main(port: int, host: str):
    """Run the STT WebSocket server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
