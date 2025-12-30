#!/usr/bin/env python3
"""Example script to transcribe audio using Canary-Qwen-2.5B.

Usage:
    python examples/transcribe.py <audio_file>
    python examples/transcribe.py audio.wav
    python examples/transcribe.py audio.mp3 --device cuda
    python examples/transcribe.py long_audio.mp3 --chunk-duration 30

Supported formats: wav, mp3, flac, ogg (anything librosa can read)
"""

import time
from pathlib import Path

import click
import librosa
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Maximum audio duration (seconds) before chunking is required
# ~30s of audio produces ~375 embeddings after 8x subsampling, which is safe
MAX_SINGLE_CHUNK_DURATION = 30.0


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """Load audio file and resample to target sample rate."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def chunk_audio(audio: np.ndarray, sr: int, chunk_duration: float) -> list[np.ndarray]:
    """Split audio into chunks of specified duration.

    Args:
        audio: Audio array.
        sr: Sample rate.
        chunk_duration: Duration of each chunk in seconds.

    Returns:
        List of audio chunks.
    """
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) > sr * 0.5:  # Skip chunks shorter than 0.5s
            chunks.append(chunk)
    return chunks


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on (default: cuda if available, else cpu)",
)
@click.option(
    "--max-tokens",
    default=448,
    type=int,
    help="Maximum tokens to generate per chunk (default: 448)",
)
@click.option(
    "--chunk-duration",
    default=30.0,
    type=float,
    help="Duration of each audio chunk in seconds (default: 30)",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory to cache downloaded models (default: HuggingFace default)",
)
def main(audio_file: Path, device: str, max_tokens: int, chunk_duration: float, cache_dir: Path | None):
    """Transcribe audio using Canary-Qwen-2.5B."""

    console.print(f"\n[bold blue]Canary-Qwen-2.5B Transcription[/bold blue]\n")

    # Load audio
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Loading audio: {audio_file.name}", total=None)
        audio, sr = load_audio(str(audio_file))

    duration = len(audio) / sr
    console.print(f"[dim]Audio:[/dim] {audio_file.name}")
    console.print(f"[dim]Duration:[/dim] {duration:.2f}s")
    console.print(f"[dim]Sample rate:[/dim] {sr}Hz")

    # Check if chunking is needed
    needs_chunking = duration > MAX_SINGLE_CHUNK_DURATION
    if needs_chunking:
        chunks = chunk_audio(audio, sr, chunk_duration)
        console.print(f"[dim]Chunks:[/dim] {len(chunks)} x {chunk_duration}s")
    console.print()

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Loading model on {device}...", total=None)
        t0 = time.time()

        from nemo_lite import CanaryQwen

        model = CanaryQwen(device=device, cache_dir=str(cache_dir) if cache_dir else None)
        load_time = time.time() - t0

    console.print(f"[dim]Model loaded in[/dim] {load_time:.1f}s\n")

    # Transcribe
    t0 = time.time()

    if needs_chunking:
        # Transcribe each chunk with progress bar
        transcripts = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing...", total=len(chunks))
            for i, chunk in enumerate(chunks):
                chunk_transcript = model.transcribe(
                    chunk,
                    sample_rate=sr,
                    max_new_tokens=max_tokens,
                )
                transcripts.append(chunk_transcript.strip())
                progress.update(task, advance=1)

        transcript = " ".join(transcripts)
    else:
        # Single transcription
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Transcribing...", total=None)
            transcript = model.transcribe(
                audio,
                sample_rate=sr,
                max_new_tokens=max_tokens,
            )

    transcribe_time = time.time() - t0

    # Output
    console.print(
        Panel(
            transcript.strip(),
            title="[bold green]Transcript[/bold green]",
            border_style="green",
        )
    )

    rtf = transcribe_time / duration
    console.print(f"\n[dim]Transcription time:[/dim] {transcribe_time:.2f}s")
    console.print(f"[dim]Real-time factor:[/dim] {rtf:.2f}x")


if __name__ == "__main__":
    main()
