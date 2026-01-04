# STT Server Design

## Overview

The STT server provides a WebSocket-based speech-to-text service for conversational agents. Once a client establishes a WebSocket connection, it streams audio frames (typically 10ms PCM chunks) to the server. The server processes the audio through a multi-stage pipeline and streams back transcription segments.

The pipeline is built on top of `nemo_lite` (a lightweight Canary-Qwen-2.5B implementation) and adds:
- Voice Activity Detection (VAD) for speech/silence classification
- Gap detection for turn segmentation
- Context-aware transcription with overlap handling

## Pipeline Architecture

The processing pipeline is implemented using the `Stage` abstraction from `stt_server/pipeline.py`. Each stage runs its own async task, communicates via queues, and can be composed with the `+` operator:

```
AudioFrame → [VAD] → AudioChunk | EndOfTurnSignal → [ASR] → TranscriptionSegment → [Sink]
```

The pipeline is instantiated per WebSocket session and wired to the session's output queue.

## Data Types

### Input

**AudioFrame** (from client)
- `samples: bytes` - Raw PCM audio (16-bit signed, mono)
- `sample_rate: int` - Sample rate in Hz (typically 16000)
- `channels: int` - Number of channels (typically 1)

### Inter-Stage Types

**AudioChunk** (VAD → ASR)
- `samples: bytes` - Accumulated audio buffer
- `sample_rate: int` - Sample rate

**EndOfTurnSignal** (VAD → ASR)
- Signals that a long silence was detected, indicating end of speaking turn

### Output

**TranscriptionSegment** (to client)
- `text: str` - Transcribed text
- `is_final: bool` - Whether this segment is final (won't change) or tentative
- `is_end_of_turn: bool` - Whether this marks the end of a speaking turn

## Stage Details

### Stage 1: VAD

Uses Silero VAD to classify audio frames, detect speech/silence states with hysteresis, and segment audio based on gaps.

**Input:** `AudioFrame`
**Output:** `AudioChunk | EndOfTurnSignal`

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `silence_to_speech_threshold` | 0.5 | VAD probability to transition silence→speaking |
| `speech_to_silence_threshold` | 0.35 | VAD probability to transition speaking→silence |
| `small_gap_threshold` | TBD | Silence duration to trigger chunk emission |
| `large_gap_threshold` | TBD | Silence duration to trigger end-of-turn |
| `min_speech_duration` | 3s | Minimum speech before small gap triggers emission |
| `max_buffer_duration` | 25s | Maximum buffer size before forced emission |
| `max_leading_silence` | 3s | Maximum silence kept at buffer start |

**State Machine:**

The stage maintains a binary state (`speaking` or `silence`) using hysteresis to prevent rapid toggling:

```
                    vad_prob >= silence_to_speech_threshold (0.5)
           ┌─────────────────────────────────────────────────────┐
           │                                                     ▼
       ┌───────┐                                           ┌──────────┐
       │silence│                                           │ speaking │
       └───────┘                                           └──────────┘
           ▲                                                     │
           └─────────────────────────────────────────────────────┘
                    vad_prob < speech_to_silence_threshold (0.35)
```

**Behavior:**
- Runs Silero VAD on each frame to get speech probability
- Updates state using hysteresis thresholds
- Accumulates frames in a buffer, tracking silence duration based on state
- On small gap (if speech > min_speech_duration): emit buffer, continue accumulating
- On large gap:
  - If buffer has speech: emit buffer + EndOfTurnSignal
  - If buffer is silence-only: emit EndOfTurnSignal only
- On max buffer reached: find largest gap, split there, emit first part
- Trims leading silence to max_leading_silence

### Stage 2: ASR

Runs Canary-Qwen-2.5B transcription with context overlap.

**Input:** `AudioChunk | EndOfTurnSignal`
**Output:** `TranscriptionSegment`

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_duration` | 5s | Audio overlap for context |

**Behavior:**
- Maintains transcript buffer and last N seconds of audio
- On AudioChunk:
  1. Prepend overlap from previous audio (if available)
  2. Run ASR on combined audio
  3. Merge new transcript with existing (see Transcript Merge below)
  4. Emit tentative TranscriptionSegment
- On EndOfTurnSignal:
  1. Emit final TranscriptionSegment with `is_end_of_turn=True`
  2. Clear transcript buffer

### Stage 3: Sink

Delivers transcription segments to the WebSocket.

**Input:** `TranscriptionSegment`
**Output:** (to WebSocket client)

**Behavior:**
- Serializes segment to JSON
- Sends via WebSocket

## Transcript Merge Algorithm

When ASR processes overlapped audio, we need to merge the new transcript with the existing one. Since Canary-Qwen-2.5B doesn't provide word timestamps, we use a word-level semi-global alignment algorithm.

### Problem Statement

Given:
- `prev`: Previous transcript as a list of words
- `new`: New transcript (from overlapped audio) as a list of words

Find the optimal alignment between a **suffix** of `prev` and a **prefix** of `new`, then merge by keeping `prev`'s non-overlapping prefix and all of `new`.

**Example:**
```
prev: ["The", "quick", "brown", "fox", "jumps"]
new:  ["brown", "fox", "jumps", "over", "the", "lazy", "dog"]

Alignment finds: prev[-3:] ≈ new[:3]  ("brown fox jumps")
Result: ["The", "quick"] + ["brown", "fox", "jumps", "over", "the", "lazy", "dog"]
      = "The quick brown fox jumps over the lazy dog"
```

### Semi-Global Alignment Algorithm

We use a DP-based semi-global alignment that:
- Allows free gaps at the **start** of `prev` (we only care about matching a suffix)
- Penalizes gaps elsewhere (insertions/deletions in the overlap region)
- Finds the optimal prefix of `new` that aligns with a suffix of `prev`

**DP Formulation:**

```
Let m = len(prev), n = len(new)
dp[i][j] = best alignment score for prev[:i] vs new[:j]

Initialize:
  dp[i][0] = 0          for all i (free to skip any prefix of prev)
  dp[0][j] = j × gap    for j > 0 (gaps at start of new are penalized)

Recurrence:
  dp[i][j] = max(
      dp[i-1][j-1] + score(prev[i-1], new[j-1]),  # align words
      dp[i-1][j] + gap,                            # skip word in prev
      dp[i][j-1] + gap                             # skip word in new
  )

Answer:
  Find j* = argmax(dp[m][j]) for j in 0..n
  Traceback from dp[m][j*] to find the alignment boundary
```

**Merge Strategy:**

After finding the alignment:
1. Identify where the overlap starts in `prev` (call it index `k`)
2. Result = `prev[:k] + new` (keep new's version for the overlapping region)

### Scoring Function

The `score(word_a, word_b)` function and `gap` penalty are configurable parameters. We will experiment with different schemes:

- **Match/mismatch scoring:** Exact match bonus, mismatch penalty, or continuous similarity based on edit distance
- **Gap penalty:** Constant or affine (open + extend)
- **Word normalization:** Lowercase, strip punctuation for comparison

The optimal scoring scheme will be determined empirically.

## Session Lifecycle

1. Client connects to `/ws/transcribe`
2. Server creates a new pipeline: `VAD + ASR + Sink`
3. Server wires sink output to WebSocket
4. Client streams `AudioFrame` messages
5. Server streams `TranscriptionSegment` messages
6. Client disconnects when done
7. Server sends sentinel through pipeline and awaits `join()`
