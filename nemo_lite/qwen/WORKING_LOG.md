# Qwen LLM Integration Working Log

This document tracks the implementation progress for integrating Qwen3-1.7B LLM with LoRA adapters for the Canary-Qwen-2.5B ASR model.

## Reference Code

NeMo source code: `~/projects/other/NeMo`

Key files:
- `nemo/collections/speechlm2/models/salm.py` - Main Speech-Audio LLM model
- `nemo/collections/speechlm2/modules/perception.py` - Audio perception module
- `nemo/collections/speechlm2/parts/lora.py` - LoRA installation utilities
- `nemo/collections/common/prompts/qwen.py` - Qwen prompt formatter

## Model Configuration (from `nvidia/canary-qwen-2.5b`)

### Base LLM
- Model: `Qwen/Qwen3-1.7B`
- Architecture: 28 transformer layers
- Hidden dimension: 2048
- Grouped Query Attention (GQA):
  - Q heads: 2048 / head_dim
  - KV heads: 1024 / head_dim (fewer heads for keys/values)

### LoRA Configuration
| Parameter | Value |
|-----------|-------|
| rank (r) | 128 |
| alpha | 256 |
| dropout | 0.01 |
| target_modules | q_proj, v_proj |
| task_type | CAUSAL_LM |

### Audio Integration
- Audio placeholder token: `<|audioplaceholder|>`
- Prompt format: "qwen"
- Perception output dim: 2048 (matches LLM hidden dim)

## Checkpoint Structure

The checkpoint contains LLM weights with the following structure:

```
llm.base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight  # (2048, 2048)
llm.base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight  # (128, 2048)
llm.base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.default.weight  # (2048, 128)
llm.base_model.model.model.layers.{i}.self_attn.k_proj.weight  # (1024, 2048) - no LoRA
llm.base_model.model.model.layers.{i}.self_attn.v_proj.base_layer.weight  # (1024, 2048)
llm.base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.default.weight  # (128, 2048)
llm.base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default.weight  # (1024, 128)
llm.base_model.model.model.layers.{i}.self_attn.o_proj.weight  # (2048, 2048)
llm.base_model.model.model.layers.{i}.self_attn.q_norm.weight  # RMSNorm
llm.base_model.model.model.layers.{i}.self_attn.k_norm.weight  # RMSNorm
llm.base_model.model.model.layers.{i}.mlp.gate_proj.weight  # (6144, 2048)
llm.base_model.model.model.layers.{i}.mlp.up_proj.weight    # (6144, 2048)
llm.base_model.model.model.layers.{i}.mlp.down_proj.weight  # (2048, 6144)
llm.base_model.model.model.layers.{i}.input_layernorm.weight
llm.base_model.model.model.layers.{i}.post_attention_layernorm.weight
llm.base_model.model.model.norm.weight  # Final RMSNorm
```

**Total LLM keys**: 421
- 112 LoRA weights (4 per layer × 28 layers)
- 309 base model weights

**Not in checkpoint (loaded from base model)**:
- `embed_tokens.weight` - Token embeddings (vocab_size, 2048)
- `lm_head.weight` - Output projection (may be tied with embed_tokens)

## Implementation Plan

### Phase 1: Basic LLM Wrapper

**Goal**: Load Qwen3-1.7B with LoRA adapters from checkpoint.

**Components**:
1. `QwenWrapper` class that wraps HuggingFace Qwen model
2. Weight loading from checkpoint (merge LoRA into base weights or keep separate)
3. Forward pass with `inputs_embeds` instead of `input_ids`

**Key decision**: LoRA weight handling
- Option A: Use `peft` library to install LoRA adapters
- Option B: Manually merge LoRA weights into base weights at load time

**Decision**: Option A - Use `peft` library.
- Already in the dev shell, no additional dependency burden
- Cleaner separation of base and LoRA weights
- Can leverage `peft` utilities for weight loading

**Files to create**: `qwen_wrapper.py`

---

### Phase 2: Weight Loading

**Goal**: Load and map checkpoint weights to Qwen model with LoRA.

**Strategy**: Use `peft` library to install LoRA adapters, then load weights.

**Weight key mapping**:
```python
# Base model weights (checkpoint -> HuggingFace)
"llm.base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight"
    -> "base_model.model.layers.{i}.self_attn.q_proj.base_layer.weight"

# LoRA weights (checkpoint -> peft model)
"llm.base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight"
    -> "base_model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight"

# Non-LoRA layers (no base_layer wrapper)
"llm.base_model.model.model.layers.{i}.self_attn.k_proj.weight"
    -> "base_model.model.layers.{i}.self_attn.k_proj.weight"
```

**Loading approach**:
1. Load base Qwen model from HuggingFace
2. Install LoRA adapters using `peft.get_peft_model()`
3. Load checkpoint weights with key mapping
4. Weights match peft's internal structure

**Files to update**: `nemo_lite/weights.py`

---

### Phase 3: Audio Embedding Injection

**Goal**: Replace `<|audioplaceholder|>` tokens with audio embeddings.

**Pipeline**:
1. Tokenize input prompt (contains `<|audioplaceholder|>` token)
2. Get text embeddings from `embed_tokens` layer
3. Find position(s) of placeholder token
4. Replace placeholder embedding with audio embeddings from projection layer
5. Adjust sequence length (audio embeddings may be longer than single token)

**Implementation**:
```python
def inject_audio_embeddings(
    input_ids: Tensor,        # (B, T_text)
    audio_embeds: Tensor,     # (B, T_audio, 2048)
    embed_tokens: nn.Embedding,
    placeholder_id: int,
) -> tuple[Tensor, Tensor]:
    """Returns (inputs_embeds, attention_mask)."""
    # 1. Get text embeddings
    text_embeds = embed_tokens(input_ids)  # (B, T_text, 2048)

    # 2. Find placeholder positions
    placeholder_mask = (input_ids == placeholder_id)

    # 3. Replace placeholder with audio embeddings
    # (Implementation handles variable-length replacement)

    return combined_embeds, attention_mask
```

**Files to create**: `embedding_injection.py` or include in `qwen_wrapper.py`

---

### Phase 4: Prompt Formatting

**Goal**: Format prompts for Qwen3 with proper special tokens.

**Qwen prompt format** (from NeMo):
```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

**Canary-specific prompt template**:
```
<|im_start|>user
Transcribe the following audio:<|audioplaceholder|><|im_end|>
<|im_start|>assistant
```

**Implementation**:
```python
def format_transcription_prompt(
    source_lang: str = "en",
    task: str = "transcribe",
    pnc: bool = True,
) -> str:
    """Create prompt for transcription task."""
    return f"""<|im_start|>user
Transcribe the following audio in {source_lang}:<|audioplaceholder|><|im_end|>
<|im_start|>assistant
"""
```

**Files to create**: `prompts.py`

---

### Phase 5: Text Generation

**Goal**: Generate transcription text from combined embeddings.

**Implementation**:
- Use HuggingFace `model.generate()` with `inputs_embeds`
- Configure generation parameters (max_length, temperature, etc.)

**Decoding**:
- Use Qwen tokenizer to decode output token IDs
- Strip special tokens and prompt prefix

**Files to update**: `qwen_wrapper.py`

---

### Phase 6: Full Model Integration

**Goal**: Create top-level `CanaryQwen` class that orchestrates everything.

**API**:
```python
class CanaryQwen:
    def __init__(self, device: str = "cuda"):
        self.preprocessor = AudioPreprocessor()
        self.encoder = FastConformerEncoder(**get_encoder_config())
        self.projection = AudioProjection()
        self.llm = QwenWrapper()

        # Load weights
        load_encoder_weights(self.encoder, "nvidia/canary-qwen-2.5b")
        load_projection_weights(self.projection, "nvidia/canary-qwen-2.5b")
        load_llm_weights(self.llm, "nvidia/canary-qwen-2.5b")

    def transcribe(
        self,
        audio: np.ndarray,  # (T,) float32 normalized to [-1, 1]
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio to text."""
        # 1. Preprocess
        mel, mel_lengths = self.preprocessor(audio)

        # 2. Encode
        audio_features, _ = self.encoder(mel, mel_lengths)

        # 3. Project
        audio_embeds = self.projection(audio_features)

        # 4. Generate text
        text = self.llm.generate(audio_embeds)

        return text
```

**Files to create**: `nemo_lite/model.py`

---

## Verification Strategy

1. **Weight loading test**: Verify all 421 LLM weights load correctly with peft
2. **Embedding injection test**: Verify placeholder replacement works
3. **Generation test**: Verify text generation produces sensible output
4. **End-to-end test**: Compare transcription with NeMo reference

---

## Open Questions

1. **Qwen version**: The config says `Qwen3-1.7B` but the architecture looks more like Qwen2.
   Need to verify which base model to use.

2. **Special tokens**: Need to verify the exact special token IDs for Qwen.
   - `<|im_start|>`, `<|im_end|>`, `<|audioplaceholder|>`

3. **Generation config**: What sampling parameters does NeMo use?
   - Temperature, top_p, max_length, etc.

4. **Prompt template**: Need to verify exact prompt format from NeMo training.

---

## Progress Log

### 2024-12-28: Initial Research Complete
- [x] Explored NeMo codebase for LLM integration patterns
- [x] Analyzed checkpoint structure (421 LLM keys, 112 LoRA weights)
- [x] Identified model architecture:
  - Base: Qwen/Qwen3-1.7B (28 layers, 2048 hidden dim)
  - LoRA: rank=128, alpha=256, targets q_proj and v_proj
- [x] Documented weight key mapping strategy
- [x] Created implementation plan

### 2024-12-28: Exploration Scripts Complete
- [x] Created exploration scripts in `exploration/canary_qwen/`
- [x] Step 1: Qwen model structure (28 layers, 2048 hidden, GQA 16/8 heads)
- [x] Step 2: LoRA mechanics (peft wraps layers, scaling=2.0)
- [x] Step 3: Weight loading (strip `llm.` prefix, 421 weights load correctly)
- [x] Step 4: Embedding injection (replace placeholder with N audio embeds)
- [x] Step 5: Text generation (use `inputs_embeds`, output is new tokens only)
- [x] Documented key learnings in `exploration/canary_qwen/README.md`

**Key discoveries:**
1. Weight loading is simple: `ckpt_key[4:]` to strip "llm." prefix
2. embed_tokens and lm_head come from base Qwen, not checkpoint
3. `model.generate(inputs_embeds=...)` returns only NEW tokens
4. Placeholder token ID: 151669 (after adding to tokenizer)

### 2024-12-28: QwenWrapper Implementation Complete
- [x] Created `QwenWrapper` class in `wrapper.py`
  - Loads Qwen3-1.7B with LoRA adapters
  - Adds `<|audioplaceholder|>` token
  - Provides `inject_audio_embeddings()` method
  - Provides `generate()` method for transcription
- [x] Added LLM weight loading to `weights.py`
  - `map_llm_weight_key()` - strips "llm." prefix
  - `load_llm_weights()` - loads 421 weights from checkpoint
- [x] Created tests in `tests/test_qwen.py` (7 tests)
- [x] Integration test passes:
  - Weight loading: 421 loaded, 2 missing (embed_tokens, lm_head as expected)
  - Embedding injection: correct sequence length
  - Generation: works (gibberish with fake audio)
- [x] All 122 tests passing

### Next: Full Model Integration
- [ ] Create `CanaryQwen` class in `model.py`
- [ ] Wire up: preprocessor → encoder → projection → LLM
- [ ] Test end-to-end with real audio

