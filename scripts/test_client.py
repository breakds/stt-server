#!/usr/bin/env python3
"""Sample WebSocket client for testing the STT server.

Usage:
    python scripts/test_client.py <audio_file.wav>

The audio file should be:
    - WAV format
    - 16kHz sample rate (will resample if different)
    - Mono channel (will convert if stereo)
"""

import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets


async def send_audio_file(uri: str, audio_path: Path, chunk_ms: int = 32):
    """Send an audio file to the STT server and print transcriptions.

    Args:
        uri: WebSocket URI (e.g., ws://localhost:8000/ws/transcribe)
        audio_path: Path to the WAV audio file
        chunk_ms: Chunk size in milliseconds (default: 32ms)
    """
    # Read the WAV file
    with wave.open(str(audio_path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        audio_data = wav.readframes(n_frames)

    print(f"Audio file: {audio_path}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {channels}")
    print(f"  Sample width: {sample_width} bytes")
    print(f"  Duration: {n_frames / sample_rate:.2f}s")
    print()

    if sample_width != 2:
        print(f"Warning: Expected 16-bit audio, got {sample_width * 8}-bit")

    # Calculate chunk size in bytes
    bytes_per_sample = sample_width * channels
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = samples_per_chunk * bytes_per_sample

    async with websockets.connect(uri) as ws:
        print(f"Connected to {uri}")
        print("-" * 50)

        # Task to receive and print transcription segments
        async def receiver():
            try:
                async for message in ws:
                    segment = json.loads(message)
                    text = segment.get("text", "")
                    is_final = segment.get("isFinal", False)
                    is_end_of_turn = segment.get("isEndOfTurn", False)

                    if is_end_of_turn:
                        print(f"\n[END OF TURN] {text}")
                        print("-" * 50)
                    elif is_final:
                        print(f"\n[FINAL] {text}")
                    else:
                        # Tentative - overwrite line
                        print(f"\r[...] {text[:80]:<80}", end="", flush=True)
            except websockets.ConnectionClosed:
                pass

        receiver_task = asyncio.create_task(receiver())

        # Send audio in chunks
        offset = 0
        chunks_sent = 0
        while offset < len(audio_data):
            chunk = audio_data[offset : offset + chunk_bytes]
            offset += chunk_bytes

            # Create AudioFrame message
            frame = {
                "samples": base64.b64encode(chunk).decode("ascii"),
                "sampleRate": sample_rate,
                "channels": channels,
            }
            await ws.send(json.dumps(frame))
            chunks_sent += 1

            # Simulate real-time streaming
            await asyncio.sleep(chunk_ms / 1000)

        print(f"\nSent {chunks_sent} chunks ({len(audio_data)} bytes)")

        # Wait a bit for final transcriptions
        await asyncio.sleep(2.0)

        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Test client for STT WebSocket server"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to WAV audio file",
    )
    parser.add_argument(
        "--uri",
        default="ws://localhost:8000/ws/transcribe",
        help="WebSocket URI (default: ws://localhost:8000/ws/transcribe)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=32,
        help="Chunk size in milliseconds (default: 32)",
    )
    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}")
        return 1

    asyncio.run(send_audio_file(args.uri, args.audio_file, args.chunk_ms))
    return 0


if __name__ == "__main__":
    exit(main())
