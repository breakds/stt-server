#!/usr/bin/env python3
"""WebSocket client for the STT server.

Usage:
    # Stream from microphone (interactive device selection)
    python -m stt_server.scripts.stt_client

    # Stream from audio file
    python -m stt_server.scripts.stt_client audio.mp3

Supports WAV, FLAC, MP3, OGG, and other formats via librosa.
Audio is automatically resampled to 16kHz mono.
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text
from websockets.asyncio.client import ClientConnection

TARGET_SAMPLE_RATE = 16000


def list_input_devices() -> list[tuple[int, str, int]]:
    """List available audio input devices.

    Returns:
        List of (device_index, device_name, num_channels) tuples.
    """
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices: list[tuple[int, str, int]] = []
    for i, dev in enumerate(devices):
        dev_dict: dict[str, str | int] = dev  # type: ignore[assignment]
        max_channels = int(dev_dict["max_input_channels"])
        if max_channels > 0:
            input_devices.append((i, str(dev_dict["name"]), max_channels))
    return input_devices


def select_input_device() -> int:
    """Prompt user to select an input device.

    Returns:
        Selected device index.
    """
    console = Console()
    input_devices = list_input_devices()

    if not input_devices:
        raise click.ClickException("No audio input devices found")

    if len(input_devices) == 1:
        idx, name, _ = input_devices[0]
        console.print(f"Using input device: [bold]{name}[/bold]")
        return idx

    # Multiple devices - show table and prompt
    table = Table(title="Available Input Devices")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Device Name", style="green")
    table.add_column("Channels", justify="right")

    for idx, name, channels in input_devices:
        table.add_row(str(idx), name, str(channels))

    console.print(table)
    console.print()

    valid_ids = [d[0] for d in input_devices]
    default_id = input_devices[0][0]

    while True:
        choice = IntPrompt.ask(
            "Select input device",
            default=default_id,
            console=console,
        )
        if choice in valid_ids:
            return choice
        console.print(f"[red]Invalid choice: {choice}[/red]")


class TranscriptDisplay:
    """Manages transcript display with rich formatting."""

    def __init__(self) -> None:
        self._final_segments: list[str] = []
        self._tentative_text: str = ""
        self._turn_count: int = 0

    def add_segment(self, text: str, is_final: bool, is_end_of_turn: bool) -> None:
        """Update the transcript with a new segment."""
        if is_end_of_turn:
            if text:
                self._final_segments.append(text)
            self._tentative_text = ""
            self._turn_count += 1
        elif is_final:
            self._final_segments.append(text)
            self._tentative_text = ""
        else:
            self._tentative_text = text

    def render(self) -> Panel:
        """Render the current transcript state."""
        parts: list[Text] = []

        # Render final segments in green/bold
        if self._final_segments:
            final_text = Text()
            for i, segment in enumerate(self._final_segments):
                if i > 0:
                    final_text.append(" ")
                final_text.append(segment, style="green")
            parts.append(final_text)

        # Render tentative text in dim/italic
        if self._tentative_text:
            tentative = Text()
            if parts:
                tentative.append(" ")
            tentative.append(self._tentative_text, style="dim italic yellow")
            parts.append(tentative)

        # Combine all parts
        if parts:
            combined = Text()
            for part in parts:
                combined.append_text(part)
            content = combined
        else:
            content = Text("Waiting for speech...", style="dim")

        title = "Transcript"
        if self._turn_count > 0:
            title += f" (Turn {self._turn_count})"

        return Panel(content, title=title, border_style="blue")


async def receive_transcriptions(
    ws: ClientConnection, display: TranscriptDisplay, live: Live
) -> None:
    """Receive and print transcription segments from the server."""
    try:
        async for message in ws:
            segment: dict[str, str | bool] = json.loads(message)
            text = str(segment.get("text", ""))
            is_final = bool(segment.get("isFinal", False))
            is_end_of_turn = bool(segment.get("isEndOfTurn", False))

            display.add_segment(text, is_final, is_end_of_turn)
            live.update(display.render())
    except websockets.ConnectionClosed:
        pass


async def stream_from_microphone(uri: str, device_id: int, chunk_ms: int = 32):
    """Stream audio from microphone to the STT server.

    Args:
        uri: WebSocket URI
        device_id: Audio input device index
        chunk_ms: Chunk size in milliseconds
    """
    import sounddevice as sd

    console = Console()
    samples_per_chunk = int(TARGET_SAMPLE_RATE * chunk_ms / 1000)

    # Queue for passing audio from callback to async sender
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def audio_callback(
        indata: npt.NDArray[np.float32], frames: int, time: Any, status: Any
    ) -> None:
        _ = frames, time  # unused
        if status:
            console.print(f"[yellow]Audio status: {status}[/yellow]")
        # Convert float32 [-1, 1] to int16 bytes
        audio_int16: npt.NDArray[np.int16] = (indata[:, 0] * 32767).astype(np.int16)
        # Put bytes in queue (non-blocking)
        try:
            audio_queue.put_nowait(audio_int16.tobytes())
        except asyncio.QueueFull:
            pass  # Drop frames if queue is full

    async with websockets.connect(uri) as ws:
        console.print(f"Connected to [cyan]{uri}[/cyan]")
        console.print("[dim]Recording from microphone. Press Ctrl+C to stop.[/dim]\n")

        display = TranscriptDisplay()

        with Live(display.render(), console=console, refresh_per_second=10) as live:
            receiver_task = asyncio.create_task(
                receive_transcriptions(ws, display, live)
            )

            # Start audio stream
            stream = sd.InputStream(
                device=device_id,
                samplerate=TARGET_SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=samples_per_chunk,
                callback=audio_callback,
            )

            try:
                with stream:
                    while True:
                        # Get audio chunk from queue
                        chunk = await audio_queue.get()

                        # Send to server
                        frame = {
                            "samples": base64.b64encode(chunk).decode("ascii"),
                            "sampleRate": TARGET_SAMPLE_RATE,
                            "channels": 1,
                        }
                        await ws.send(json.dumps(frame))
            except asyncio.CancelledError:
                pass
            finally:
                _ = receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass

    console.print("\n[green]Done.[/green]")


async def stream_from_file(uri: str, audio_path: Path, chunk_ms: int = 32):
    """Send an audio file to the STT server and print transcriptions.

    Args:
        uri: WebSocket URI (e.g., ws://localhost:8000/ws/transcribe)
        audio_path: Path to the audio file (WAV, FLAC, MP3, etc.)
        chunk_ms: Chunk size in milliseconds (default: 32ms)
    """
    import librosa

    console = Console()

    # Load audio file with librosa (resamples to target rate, converts to mono)
    audio, original_sr = librosa.load(
        str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True
    )

    console.print(f"Audio file: [cyan]{audio_path}[/cyan]")
    console.print(f"  Sample rate: {original_sr} Hz → {TARGET_SAMPLE_RATE} Hz")
    console.print(f"  Duration: {len(audio) / TARGET_SAMPLE_RATE:.2f}s\n")

    # Convert float32 [-1, 1] to int16 bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_data = audio_int16.tobytes()

    # Calculate chunk size in bytes (16-bit mono = 2 bytes per sample)
    samples_per_chunk = int(TARGET_SAMPLE_RATE * chunk_ms / 1000)
    chunk_bytes = samples_per_chunk * 2

    async with websockets.connect(uri) as ws:
        console.print(f"Connected to [cyan]{uri}[/cyan]\n")

        display = TranscriptDisplay()

        with Live(display.render(), console=console, refresh_per_second=10) as live:
            receiver_task = asyncio.create_task(
                receive_transcriptions(ws, display, live)
            )

            # Send audio in chunks
            offset = 0
            chunks_sent = 0
            while offset < len(audio_data):
                chunk = audio_data[offset : offset + chunk_bytes]
                offset += chunk_bytes

                # Create AudioFrame message
                frame = {
                    "samples": base64.b64encode(chunk).decode("ascii"),
                    "sampleRate": TARGET_SAMPLE_RATE,
                    "channels": 1,
                }
                await ws.send(json.dumps(frame))
                chunks_sent += 1

                # Simulate real-time streaming
                await asyncio.sleep(chunk_ms / 1000)

            # Wait a bit for final transcriptions
            await asyncio.sleep(2.0)

            _ = receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

    console.print(f"\n[dim]Sent {chunks_sent} chunks ({len(audio_data)} bytes)[/dim]")
    console.print("[green]Done.[/green]")


@click.command()
@click.argument(
    "audio_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "--uri",
    default="ws://localhost:15751/ws/transcribe",
    help="WebSocket URI",
)
@click.option(
    "--chunk-ms",
    default=32,
    help="Chunk size in milliseconds",
)
def main(audio_file: Path | None, uri: str, chunk_ms: int):
    """Stream audio to the STT server for transcription.

    If AUDIO_FILE is provided, streams from the file.
    Otherwise, streams from the microphone.
    """
    if audio_file is not None:
        asyncio.run(stream_from_file(uri, audio_file, chunk_ms))
    else:
        device_id = select_input_device()
        try:
            asyncio.run(stream_from_microphone(uri, device_id, chunk_ms))
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
