# STT Server Design Document

## Goal

Build a lightweight, inference-only implementation for running NVIDIA's Canary-Qwen-2.5B speech-to-text model without depending on the full NeMo framework.

### Motivation

The official NeMo toolkit has heavy dependencies that are problematic for deployment:
- `lhotse` - Audio data loading library (used even for simple inference)
- `nv-one-logger-*` - NVIDIA internal logging packages
- `fiddle` - Configuration library
- `lightning` - Training framework (not needed for inference)
- `hydra` / `omegaconf` - Configuration management

These dependencies are either hard to package, require internal NVIDIA infrastructure, or are simply unnecessary for inference workloads.

### Use Case

The primary use case is transcribing audio from PCM data:
- Input: 16-bit PCM audio as uint16 array, mono channel, known sample rate
- Output: Transcribed text

No file I/O for audio is required.

## Architecture

The Canary-Qwen-2.5B model consists of:

```
PCM Audio (16-bit, mono)
    │
    ▼
┌─────────────────────────┐
│   Mel Spectrogram       │  ← librosa + torch.stft
│   (128 features)        │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   FastConformer Encoder │  ← Custom implementation
│   (32 layers, 1024 dim) │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Linear Projection     │  ← nn.Linear(1024, 2048)
│   (audio → LLM space)   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Qwen3-1.7B + LoRA     │  ← transformers + peft
│   (text generation)     │
└─────────────────────────┘
    │
    ▼
Transcribed Text
```

## Modules

### 1. Audio Preprocessing (`preprocessing.py`) ✅

Converts raw PCM audio to mel spectrogram features.

- Input: float32 tensor of audio samples `(B, T)`, normalized to [-1, 1]
- Output: tuple of `(mel, mel_lengths)` where mel is `(B, 128, T')` and mel_lengths is `(B,)`

Pipeline:
1. Pre-emphasis filter: `y[n] = x[n] - 0.97 * x[n-1]`
2. STFT with Hann window
3. Power spectrum (magnitude squared)
4. Mel filterbank projection (librosa, slaney-normalized)
5. Log scaling: `log(mel + 2^-24)`
6. Per-feature normalization (z-score per mel bin)

Parameters (matching NeMo's Canary config):
- Sample rate: 16000 Hz
- Mel features: 128
- FFT size: 512
- Window: 25ms / 400 samples (Hann)
- Stride: 10ms / 160 samples
- Pre-emphasis: 0.97

Implementation: librosa for mel filterbank (matches NeMo exactly), PyTorch `torch.stft` for STFT.

### 2. Conformer Encoder (`conformer_lite/`)

The FastConformer speech encoder that processes mel spectrograms.

- Input: mel spectrogram `(B, 128, T)` transposed to `(B, T, 128)`
- Output: encoded audio features `(B, T', 1024)` where `T' = T / 8`

#### Configuration (from `nvidia/canary-qwen-2.5b`)

| Parameter | Value |
|-----------|-------|
| n_layers | 32 |
| d_model | 1024 |
| n_heads | 8 |
| head_dim | 128 |
| ff_expansion_factor | 4 |
| ff_hidden | 4096 |
| conv_kernel_size | 9 |
| conv_norm_type | batch_norm |
| subsampling | dw_striding |
| subsampling_factor | 8 |
| subsampling_conv_channels | 256 |
| self_attention_model | rel_pos |
| untie_biases | true |
| pos_emb_max_len | 5000 |
| dropout | 0.1 |
| dropout_att | 0.1 |
| dropout_pre_encoder | 0.1 |

#### Subsampling Module (dw_striding, 8x)

Reduces time dimension by 8x using depthwise-separable convolutions:

```
Input: (B, T, 128)
    │
    ├─ Conv2D(1 → 256, kernel=3, stride=2) + ReLU
    │  Output: (B, 256, T/2, freq/2)
    │
    ├─ DepthwiseConv2D(256, kernel=3, stride=2, groups=256)
    ├─ PointwiseConv2D(256 → 256, kernel=1) + ReLU
    │  Output: (B, 256, T/4, freq/4)
    │
    ├─ DepthwiseConv2D(256, kernel=3, stride=2, groups=256)
    ├─ PointwiseConv2D(256 → 256, kernel=1) + ReLU
    │  Output: (B, 256, T/8, freq/8)
    │
    └─ Flatten + Linear(256 * freq/8 → 1024)
       Output: (B, T/8, 1024)
```

#### Relative Positional Encoding (Transformer-XL style)

- Sinusoidal positional embeddings for relative positions
- Linear projection: `Linear(1024, 1024, bias=False)`
- Learnable biases per head: `pos_bias_u`, `pos_bias_v` (untied, shape `[n_heads, head_dim]`)
- No xscaling (unlike some variants)

#### Conformer Block Structure (Pre-Norm)

Each of the 32 blocks follows this exact order:

```
Input: x (B, T, 1024)
    │
    ├─ FFN1 (half-step):
    │    out = LayerNorm(x)
    │    out = Linear(1024 → 4096) → Swish → Dropout → Linear(4096 → 1024)
    │    x = x + 0.5 * Dropout(out)
    │
    ├─ Self-Attention:
    │    out = LayerNorm(x)
    │    out = RelPosMultiHeadAttention(out, pos_emb)
    │    x = x + Dropout(out)
    │
    ├─ Convolution:
    │    out = LayerNorm(x)
    │    out = PointwiseConv(1024 → 2048) → GLU → (1024)
    │    out = DepthwiseConv(1024, kernel=9, groups=1024)
    │    out = BatchNorm → Swish
    │    out = PointwiseConv(1024 → 1024)
    │    x = x + Dropout(out)
    │
    ├─ FFN2 (half-step):
    │    out = LayerNorm(x)
    │    out = Linear(1024 → 4096) → Swish → Dropout → Linear(4096 → 1024)
    │    x = x + 0.5 * Dropout(out)
    │
    └─ Final LayerNorm:
       output = LayerNorm(x)
```

#### Key Details for Weight Compatibility

1. **Normalization**: Pre-Norm (LayerNorm before each sub-module)
2. **Conv norm**: BatchNorm (not LayerNorm) inside convolution module
3. **FFN scale**: 0.5 for both feed-forward residuals (Macaron-style)
4. **Attention biases**: Untied per head (`pos_bias_u`, `pos_bias_v`)
5. **GLU activation**: Splits channels in half, applies sigmoid gate

This is the main implementation effort (~800-1000 lines).

### 3. Projection Layer (`projection.py`)

Simple linear projection from audio encoder space to LLM embedding space.

- Input: (B, T, 1024)
- Output: (B, T, 2048)

Implementation: Single `nn.Linear(1024, 2048)`.

### 4. Language Model (`llm.py`)

Qwen3-1.7B with LoRA adapters for text generation.

- Base model: `Qwen/Qwen3-1.7B` via HuggingFace transformers
- LoRA config: rank=128, alpha=256, target_modules=[q_proj, v_proj]

The model receives concatenated embeddings:
1. Text prompt tokens (embedded via LLM's embedding layer)
2. Audio features (from projection layer, inserted at `<|audioplaceholder|>` position)

### 5. Model Wrapper (`model.py`)

Top-level class that orchestrates inference.

Responsibilities:
- Load weights from safetensors
- Coordinate preprocessing → encoder → projection → LLM
- Provide simple `transcribe(audio, sample_rate)` API
- Handle prompt formatting

### 6. Weight Loading (`weights.py`)

Utility to load and map weights from the official checkpoint.

- Source: `nvidia/canary-qwen-2.5b` on HuggingFace Hub
- Format: safetensors (5.12 GB)
- Task: Map weight names from NeMo format to our module names

## Dependencies

### Required
- `torch` - Core tensor operations, STFT
- `librosa` - Mel filterbank (matches NeMo exactly)
- `torchaudio` - Audio resampling (if needed)
- `transformers` - Qwen3 LLM loading
- `peft` - LoRA adapter support
- `safetensors` - Weight loading
- `numpy` - PCM array handling

### Not Required (vs NeMo)
- ~~lhotse~~ - We handle audio as tensors directly
- ~~nv-one-logger-*~~ - No telemetry needed
- ~~fiddle~~ - No dynamic configuration
- ~~lightning~~ - No training framework
- ~~hydra/omegaconf~~ - Hardcoded inference config
- ~~webdataset~~ - No dataset loading
- ~~many others...~~

## File Structure

```
stt_server/
├── __init__.py
├── model.py           # Main CanaryQwen class
├── preprocessing.py   # Mel spectrogram
├── conformer.py       # FastConformer encoder
├── projection.py      # Linear projection
├── llm.py            # Qwen3 + LoRA wrapper
├── weights.py        # Weight loading utilities
└── tokenizer.py      # Tokenizer wrapper (if needed)
```

## Open Questions

1. **Conformer implementation**: Should we extract and simplify NeMo's implementation, or reimplement from scratch based on the paper?

2. **Streaming support**: The original model supports streaming ASR. Do we need this for the initial version?

3. **Batch inference**: Should we support batched transcription from the start?

4. **Model precision**: The original uses bfloat16. Should we support float16/float32 fallbacks?
