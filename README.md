# STT Server

Real-time Speech-to-Text WebSocket server for conversational agents.

## Overview

STT Server provides low-latency streaming transcription with turn-taking awareness. It processes continuous audio streams and returns transcription segments as they become available, distinguishing between intermediate (tentative) and final results.

The server solves the problem of integrating speech recognition into conversational AI systems where:
- Low latency is critical for natural conversation flow
- Turn boundaries need to be detected automatically
- Transcription should be progressive (showing partial results)

### Architecture

The system uses a multi-stage async pipeline:

```
Audio → [VAD] → [ASR] → [Sink] → WebSocket
         │        │        │
         │        │        └─ Serializes to JSON and sends to client
         │        └─ Transcribes audio using Canary-Qwen-2.5B
         └─ Detects speech/silence, segments by turn boundaries
```

**VAD Stage** - Uses Silero VAD to classify audio frames as speech or silence. Implements hysteresis-based state machine to prevent chatter at silence boundaries. Emits audio chunks on small gaps (0.3s) for continuous transcription, and end-of-turn signals on large gaps (1.5s).

**ASR Stage** - Transcribes audio chunks using Canary-Qwen-2.5B (2.5B parameter model). Maintains audio overlap for context continuity and uses semi-global alignment to merge overlapping transcriptions.

**Sink Stage** - Terminal stage that serializes transcription segments to JSON and sends them directly to the WebSocket client.

## Sub-Projects

### nemo_lite

Lightweight inference-only wrapper for Canary-Qwen-2.5B. Located in `nemo_lite/`.

The official NVIDIA NeMo toolkit has heavy dependencies that are problematic for deployment:
- `lhotse`, `nv-one-logger-*`, `fiddle`, `lightning`, `hydra` - none needed for inference

nemo_lite provides the same transcription capability with minimal dependencies:
- `torch`, `torchaudio` - Core tensor operations
- `transformers`, `peft`, `safetensors` - Model loading
- `librosa` - Mel filterbank (matches NeMo exactly)

Key components:
- `AudioPreprocessor` - Converts PCM to mel spectrogram (128 features, 16kHz)
- `FastConformer` - 32-layer encoder (1024 dim, 8x temporal downsampling)
- `Qwen3-1.7B + LoRA` - Text generation via HuggingFace transformers

### strops

Rust/Python library for word sequence merging. Located in `strops-rs/`.

Provides `merge_by_overlap(prev, new)` function that uses semi-global alignment to find where the suffix of previous transcription overlaps with the prefix of new transcription. This maintains context continuity when processing audio in overlapping chunks.

```python
>>> from strops import merge_by_overlap
>>> merge_by_overlap(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
["The", "quick", "brown", "fox", "jumps"]
```

Built with maturin and pyo3 for Python bindings.

## Usage

### Prerequisites

- **NVIDIA GPU with CUDA** - Required for real-time performance (RTX 2070 or better recommended)
- **CPU mode** - Works but significantly slower, not suitable for real-time use
- **Nix** - For dependency management (or manually install Python dependencies)
- **~5GB disk space** - For model weights (downloaded on first run)

### Running the Server

**Development mode:**
```bash
# Enter the Nix development shell
nix develop

# Run the server
python stt_server/server.py --port 15751 --host 0.0.0.0

# Or with CPU mode
STT_DEVICE=cpu python stt_server/server.py
```

**Using the built package:**
```bash
nix build .#stt-server
./result/bin/stt-server --port 15751
```

The server exposes:
- `GET /health` - Health check endpoint
- `WebSocket /ws/transcribe` - Streaming transcription

### WebSocket Protocol

**Client sends** `AudioFrame` messages:
```json
{
  "samples": "<base64-encoded 16-bit PCM>",
  "sampleRate": 16000,
  "channels": 1
}
```

**Server sends** `TranscriptionSegment` messages:
```json
{
  "text": "transcribed text here",
  "isFinal": false,
  "isEndOfTurn": false
}
```

### Test Client

**Stream from microphone:**
```bash
python -m stt_server.scripts.stt_client
```
Interactive device selection if multiple microphones are available.

**Stream from audio file:**
```bash
python -m stt_server.scripts.stt_client path/to/audio.mp3
```
Supports WAV, FLAC, MP3, OGG with automatic resampling to 16kHz mono.

### NixOS Service

Add to your NixOS configuration:

```nix
{
  imports = [ stt-server.nixosModules.default ];

  nixpkgs.overlays = [ stt-server.overlays.default ];

  services.stt-server = {
    enable = true;
    port = 15751;
    host = "0.0.0.0";
    device = "cuda";  # or "cpu"
    openFirewall = true;  # for internal network access
  };
}
```

Configuration options:
- `port` - Server port (default: 15751)
- `host` - Bind address (default: "0.0.0.0")
- `device` - "cuda" or "cpu" (default: "cuda")
- `package` - The stt-server package to use
- `openFirewall` - Open TCP port in firewall (default: false)

Model weights are cached in `/var/cache/stt-server` (managed by systemd).

## For Developers

### Development Shells

**Main development:**
```bash
nix develop
```
Python environment with all ML dependencies (torch, transformers, etc.) and dev tools (basedpyright, ruff).

**strops development:**
```bash
nix develop .#strops
```
Rust toolchain for developing the strops library.

### Testing

```bash
python -m unittest discover -s stt_server/tests
```

### Project Structure

```
stt-server/
├── stt_server/           # Main Python package
│   ├── server.py         # FastAPI WebSocket server
│   ├── session.py        # Transcription session management
│   ├── pipeline.py       # Async pipeline infrastructure
│   ├── data_types.py     # Pydantic models for protocol
│   ├── stages/           # Pipeline stages (VAD, ASR, Sink)
│   ├── scripts/          # CLI tools (stt_client)
│   └── tests/            # Unit tests
├── nemo_lite/            # Lightweight Canary-Qwen wrapper
│   ├── model.py          # Main CanaryQwen class
│   ├── preprocessing.py  # Mel spectrogram extraction
│   ├── conformer_lite/   # FastConformer encoder
│   ├── qwen/             # Qwen3 LLM wrapper
│   └── weights.py        # Weight loading utilities
├── strops-rs/            # Rust/Python sequence alignment
│   ├── src/              # Rust source
│   └── nix/              # Nix packaging
└── nix/                  # Nix configuration
    ├── development.nix   # Dev shell
    ├── release.nix       # Package/module exports
    ├── packages/         # Nix packages
    └── modules/          # NixOS modules
```

## License

MIT
