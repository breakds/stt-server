# STT Server Implementation Log

## Overview

This document tracks the implementation progress of the STT server pipeline.

## Current State

**Implemented:**
- `pipeline.py` - Stage, SingleStage, ChainedStage infrastructure
- `data_types.py` - AudioFrame, TranscriptionSegment, ErrorResponse, AudioChunk, EndOfTurnSignal
- `server.py` - WebSocket skeleton with FastAPI
- `session.py` - TranscriptionSession ABC and MockTranscriptionSession
- `stages/sink.py` - SinkStage (sends segments via WebSocket)
- `stages/asr.py` - ASRStage (Canary-Qwen transcription with overlap handling)

**To Implement:**
- VAD Stage (Silero VAD + hysteresis + gap detection)
- PipelineSession (real TranscriptionSession using the pipeline)

---

## Implementation Plan

### Phase 1: Data Types and Sink Stage [DONE]

**Goal:** Complete the data type definitions and implement the simplest stage first.

- [x] **1.1** Add inter-stage types to `data_types.py`
  - `AudioChunk`: accumulated audio buffer with samples and sample_rate
  - `EndOfTurnSignal`: marker class for end-of-turn

- [x] **1.2** Implement `SinkStage` in `stages/sink.py`
  - Takes `TranscriptionSegment`, sends directly via WebSocket
  - Constructed with WebSocket handle

- [x] **1.3** Write unit tests for SinkStage

---

### Phase 2: ASR Stage [DONE]

**Goal:** Implement ASR with transcript merging and overlap handling.

- [x] **2.1** Create `stages/asr.py` with `ASRStage`
  - Input: `AudioChunk | EndOfTurnSignal`
  - Output: `TranscriptionSegment`
  - Uses `nemo_lite.CanaryQwen` for transcription

- [x] **2.2** Implement basic transcription
  - On `AudioChunk`: transcribe, emit tentative segment
  - On `EndOfTurnSignal`: emit final segment, clear state

- [x] **2.3** Add overlap handling
  - Maintains audio buffer for overlap (default 5s)
  - Prepends overlap to new chunks
  - Uses `strops.merge_by_overlap` to merge transcripts

- [x] **2.4** Write unit tests for ASRStage
  - 5 tests with mocked model
  - Tests transcription, merging, state clearing, empty handling

---

### Phase 3: VAD Stage [DONE]

**Goal:** Implement VAD with hysteresis and gap detection.

- [x] **3.1** Create `stages/vad.py` with `VADStage`
  - Input: `AudioFrame`
  - Output: `AudioChunk | EndOfTurnSignal`

- [x] **3.2** Integrate Silero VAD (pysilero-vad)
  - Run VAD on each frame
  - Get speech probability

- [x] **3.3** Implement hysteresis state machine
  - Track `speaking` / `silence` state
  - Use dual thresholds (0.5 for silence→speaking, 0.35 for speaking→silence)

- [x] **3.4** Implement gap detection and buffering
  - Accumulate frames in buffer
  - Track silence duration
  - Emit on small gap (if enough speech)
  - Emit + EndOfTurnSignal on large gap
  - Force emit on max buffer

- [x] **3.5** Implement buffer management
  - Trim leading silence to max_leading_silence
  - Find largest gap for forced splits

- [x] **3.6** Write unit tests for VADStage
  - 9 tests covering hysteresis, gap detection, buffer management, state reset
  - All tests passing

---

### Phase 4: Pipeline Integration [DONE]

**Goal:** Wire everything together and replace the mock session.

- [x] **4.1** Create `PipelineSession` in `session.py`
  - Implements `TranscriptionSession`
  - Creates and manages `VAD + ASR + Sink` pipeline
  - Pipeline sends segments directly to WebSocket (no intermediate queue)

- [x] **4.2** Simplify `TranscriptionSession` interface
  - Removed `get_segment()` - pipeline handles WebSocket sending
  - Only `push_audio()` and `close()` remain

- [x] **4.3** Update `server.py`
  - Removed `segment_sender` task
  - Server just pushes audio; pipeline handles output

- [x] **4.4** Update `create_session()` to accept WebSocket
  - Uses shared VAD and model instances (lazy initialization)
  - Configurable VAD parameters (small_gap, large_gap, min_speech)

- [x] **4.5** Write unit tests for PipelineSession
  - 4 tests covering WebSocket output, close, multiple turns

---

### Phase 5: Tuning and Optimization

**Goal:** Tune parameters and optimize performance.

- [ ] **5.1** Determine optimal gap thresholds
  - `small_gap_threshold`: TBD
  - `large_gap_threshold`: TBD

- [ ] **5.2** Performance profiling
  - Measure latency through pipeline
  - Identify bottlenecks

- [ ] **5.3** Add configuration support
  - Make parameters configurable
  - Environment variables or config file

---

## Progress Log

### [Date TBD] - Project Setup
- Created initial implementation plan
- Reviewed existing code structure

---

## Notes

- All stages run in separate async tasks via the pipeline infrastructure
- Sentinel (None) propagates through pipeline for clean shutdown
- Use `join()` to wait for pipeline completion
