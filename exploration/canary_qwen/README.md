# Canary-Qwen LLM Integration Exploration

This directory contains exploration scripts for understanding how to integrate Qwen3-1.7B with audio embeddings for the Canary-Qwen-2.5B ASR model.

## Scripts

| Script | Purpose |
|--------|---------|
| `explore_qwen.py` | Understand Qwen model structure |
| `explore_lora.py` | Understand LoRA adapter mechanics |
| `explore_weight_loading.py` | Load Canary checkpoint weights |
| `explore_embedding_injection.py` | Replace placeholder with audio |
| `explore_generation.py` | Generate text from embeddings |

Run any script with:
```bash
python -m exploration.canary_qwen.explore_qwen
```

---

## Key Learnings

### 1. Qwen Model Structure

```
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens: Embedding(151936, 2048)
│   ├── layers: 28x Qwen3DecoderLayer
│   │   ├── self_attn: Qwen3Attention (GQA: 16 Q heads, 8 KV heads)
│   │   ├── mlp: Qwen3MLP (gate_proj, up_proj, down_proj)
│   │   ├── input_layernorm: RMSNorm
│   │   └── post_attention_layernorm: RMSNorm
│   ├── norm: RMSNorm
│   └── rotary_emb: Qwen3RotaryEmbedding
└── lm_head: Linear(2048, 151936)
```

**Key dimensions:**
- Hidden: 2048
- Layers: 28
- Attention heads: 16 (Q), 8 (KV) - Grouped Query Attention
- FFN intermediate: 6144
- Vocab: 151,936

### 2. LoRA Mechanics

**Configuration (matching Canary-Qwen):**
```python
LoraConfig(
    r=128,                          # Rank
    lora_alpha=256,                 # Scaling = alpha/r = 2.0
    lora_dropout=0.01,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

**How LoRA wraps layers:**
```
Before: Linear(2048, 2048)
After:  lora.Linear
        ├── base_layer: Linear(2048, 2048)   # Frozen
        ├── lora_A: Linear(2048, 128)        # Trainable
        └── lora_B: Linear(128, 2048)        # Trainable

Forward: output = base(x) + 2.0 * lora_B(lora_A(x))
```

**Parameter counts:**
- Total: 1.75B
- Trainable (LoRA): 25.7M (1.47%)
- Frozen: 1.72B (98.53%)

### 3. Weight Loading

**Key mapping:** Just strip `llm.` prefix
```python
def map_llm_key(ckpt_key: str) -> str | None:
    if not ckpt_key.startswith("llm."):
        return None
    return ckpt_key[4:]  # "llm.base_model..." -> "base_model..."
```

**Checkpoint structure:**
- 421 LLM weights total
- 112 LoRA weights (4 per layer × 28 layers)
- 309 base weights

**Missing keys (expected):**
- `embed_tokens.weight` - From base Qwen model
- `lm_head.weight` - From base Qwen model

### 4. Audio Embedding Injection

**Placeholder token:** `<|audioplaceholder|>` (ID: 151669 after adding)

**Injection process:**
```
Before: [9 text embeds] [1 placeholder] [5 text embeds]  = 15 tokens
After:  [9 text embeds] [N audio embeds] [5 text embeds] = 14 + N embeddings
```

**Function signature:**
```python
def inject_audio_embeddings(
    input_ids: Tensor,      # (B, T_text)
    text_embeds: Tensor,    # (B, T_text, 2048)
    audio_embeds: Tensor,   # (B, T_audio, 2048)
    placeholder_id: int,
) -> tuple[Tensor, Tensor]:  # (combined_embeds, attention_mask)
```

### 5. Text Generation

**Two ways to generate:**
```python
# With token IDs (normal)
output = model.generate(input_ids=input_ids)
# Returns: input tokens + new tokens

# With embeddings (our approach)
output = model.generate(inputs_embeds=combined_embeds)
# Returns: only new tokens (convenient!)
```

**Generation config for ASR:**
```python
model.generate(
    inputs_embeds=combined_embeds,
    attention_mask=attention_mask,
    max_new_tokens=448,
    do_sample=False,              # Greedy decoding
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

---

## Complete Pipeline

```
1. Audio → Preprocessor → Mel spectrogram (B, 128, T)
2. Mel → Encoder → Audio features (B, T/8, 1024)
3. Features → Projection → Audio embeddings (B, T/8, 2048)
4. Prompt → Tokenizer → Token IDs
5. Token IDs → embed_tokens → Text embeddings
6. Find <|audioplaceholder|> position
7. Replace placeholder with audio embeddings
8. Combined embeddings → model.generate() → Token IDs
9. Token IDs → Tokenizer.decode() → Transcription text
```

---

## Prompt Format

```
<|im_start|>user
Transcribe the following audio:<|audioplaceholder|><|im_end|>
<|im_start|>assistant
[model generates transcription here]
```
