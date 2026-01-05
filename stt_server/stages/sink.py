"""Sink stage that delivers transcription segments via WebSocket."""

from typing import override

from fastapi import WebSocket

from stt_server.data_types import TranscriptionSegment
from stt_server.pipeline import SingleStage


class SinkStage(SingleStage[TranscriptionSegment, None]):
    """Final pipeline stage that sends segments directly via WebSocket.

    This stage takes TranscriptionSegment items and sends them as JSON
    to the connected WebSocket client.
    """

    _websocket: WebSocket

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket

    @override
    async def _process_item(self, item: TranscriptionSegment) -> None:
        """Send the segment to the WebSocket client."""
        await self._websocket.send_json(item.model_dump(by_alias=True))
