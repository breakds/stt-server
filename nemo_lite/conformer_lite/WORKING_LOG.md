# FastConformer Implementation Working Log

This document tracks the implementation progress and details for the FastConformer encoder.

## Reference Code

NeMo source code: `~/projects/other/NeMo`

Key files:
- `nemo/collections/asr/modules/conformer_encoder.py` - Main encoder
- `nemo/collections/asr/parts/submodules/subsampling.py` - Subsampling modules
- `nemo/collections/asr/parts/submodules/conformer_modules.py` - Conformer blocks
- `nemo/collections/asr/parts/submodules/multi_head_attention.py` - Attention with rel pos

## Implementation Plan

### Phase 1: Subsampling Module

**Goal**: Implement `dw_striding` subsampling that reduces time by 8x.

**Architecture**:
```python
# Layer 1: Regular Conv2D
Conv2d(1, 256, kernel_size=3, stride=2)  # (B, 1, T, 128) -> (B, 256, T/2, 64)
ReLU()

# Layer 2: Depthwise-separable
Conv2d(256, 256, kernel_size=3, stride=2, groups=256)  # Depthwise
Conv2d(256, 256, kernel_size=1, stride=1)  # Pointwise
ReLU()

# Layer 3: Depthwise-separable
Conv2d(256, 256, kernel_size=3, stride=2, groups=256)  # Depthwise
Conv2d(256, 256, kernel_size=1, stride=1)  # Pointwise
ReLU()

# Output projection
# After 3 layers: (B, 256, T/8, 128/8) = (B, 256, T/8, 16)
# Flatten: (B, T/8, 256*16) = (B, T/8, 4096)
Linear(4096, 1024)
```

**Padding**: For kernel=3, stride=2:
- left_pad = (kernel - 1) // 2 = 1
- right_pad = (kernel - 1) // 2 = 1

**Verification**:
- Check output shape matches NeMo for same input
- Compare intermediate activations if needed

**Files to create**: `subsampling.py`

---

### Phase 2: Relative Positional Encoding

**Goal**: Implement Transformer-XL style relative positional encoding.

**Components**:
1. `RelPositionalEncoding` - Generates sinusoidal position embeddings
2. Position biases `pos_bias_u`, `pos_bias_v` - Learnable, per-head

**Key formulas**:
```python
# Sinusoidal encoding
pe[pos, 2i] = sin(pos / 10000^(2i/d_model))
pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

# Position embedding range: [-(L-1), ..., 0, ..., (L-1)]
# Total positions: 2*L - 1 for sequence length L
```

**Verification**:
- Compare position embeddings with NeMo output
- Check that biases have correct shape [n_heads, head_dim]

**Files to create**: `pos_encoding.py`

---

### Phase 3: Multi-Head Attention with Relative Position

**Goal**: Implement `RelPositionMultiHeadAttention`.

**Architecture**:
```python
# Projections
linear_q = Linear(1024, 1024)  # Query
linear_k = Linear(1024, 1024)  # Key
linear_v = Linear(1024, 1024)  # Value
linear_out = Linear(1024, 1024)  # Output
linear_pos = Linear(1024, 1024, bias=False)  # Position (no bias!)

# Attention computation
q = linear_q(x).view(B, T, n_heads, head_dim).transpose(1, 2)
k = linear_k(x).view(B, T, n_heads, head_dim).transpose(1, 2)
v = linear_v(x).view(B, T, n_heads, head_dim).transpose(1, 2)
p = linear_pos(pos_emb).view(B, T_pos, n_heads, head_dim).transpose(1, 2)

# Relative attention (Transformer-XL style)
q_with_bias_u = q + pos_bias_u  # Content bias
q_with_bias_v = q + pos_bias_v  # Position bias

# Content attention: (B, heads, T, T)
matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

# Position attention: (B, heads, T, T_pos) -> (B, heads, T, T)
matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
matrix_bd = rel_shift(matrix_bd)  # Skew trick for relative positions

scores = (matrix_ac + matrix_bd) / sqrt(head_dim)
attn = softmax(scores)
attn = dropout(attn)
out = torch.matmul(attn, v)
```

**rel_shift operation**: Implements the skewing trick from Transformer-XL paper.

**Verification**:
- Test with known input, compare attention scores with NeMo
- Check that linear_pos has no bias

**Files to create**: `attention.py`

---

### Phase 4: Convolution Module

**Goal**: Implement Conformer convolution block.

**Architecture**:
```python
# Pointwise expansion with GLU
pointwise_conv1 = Conv1d(1024, 2048, kernel_size=1)
# GLU splits to 1024, applies sigmoid gate

# Depthwise conv
depthwise_conv = Conv1d(1024, 1024, kernel_size=9, groups=1024, padding=4)

# BatchNorm + Swish
batch_norm = BatchNorm1d(1024)
# Swish activation

# Pointwise projection
pointwise_conv2 = Conv1d(1024, 1024, kernel_size=1)
```

**GLU (Gated Linear Unit)**:
```python
def glu(x):
    a, b = x.chunk(2, dim=-1)
    return a * torch.sigmoid(b)
```

**Verification**:
- Check BatchNorm (not LayerNorm) is used
- Verify padding gives same output length

**Files to create**: `convolution.py`

---

### Phase 5: Feed-Forward Module

**Goal**: Implement Conformer feed-forward with 0.5 residual scaling.

**Architecture**:
```python
# Expansion
linear1 = Linear(1024, 4096)
# Swish activation
dropout1 = Dropout(0.1)

# Projection
linear2 = Linear(4096, 1024)
dropout2 = Dropout(0.1)
```

**Residual**: `x = x + 0.5 * ffn(layer_norm(x))`

**Files to create**: `feed_forward.py`

---

### Phase 6: Conformer Block

**Goal**: Combine all modules into a single Conformer layer.

**Order of operations**:
1. FFN1 with 0.5 residual
2. Self-attention with full residual
3. Convolution with full residual
4. FFN2 with 0.5 residual
5. Final LayerNorm

**LayerNorm placement**: Pre-norm (before each sub-module)

**Files to create**: `conformer_block.py`

---

### Phase 7: Full Encoder

**Goal**: Stack 32 Conformer blocks with subsampling and positional encoding.

**Architecture**:
```python
class FastConformerEncoder(nn.Module):
    def __init__(self):
        self.subsampling = ConvSubsampling(...)
        self.pos_encoding = RelPositionalEncoding(...)
        self.dropout_pre = Dropout(0.1)
        self.layers = nn.ModuleList([ConformerBlock(...) for _ in range(32)])

    def forward(self, x, lengths):
        # x: (B, 128, T) mel spectrogram
        x = x.transpose(1, 2)  # (B, T, 128)
        x, lengths = self.subsampling(x, lengths)  # (B, T/8, 1024)
        x, pos_emb = self.pos_encoding(x)
        x = self.dropout_pre(x)

        for layer in self.layers:
            x = layer(x, pos_emb)

        return x, lengths
```

**Files to create**: `encoder.py`, `__init__.py`

---

## Weight Mapping

NeMo uses `nn.Sequential` for conv layers, so indices skip activation layers (which have no parameters).

### Subsampling Module (`pre_encode`)

NeMo's structure:
```python
self.pre_encode = ConvSubsampling(...)
# Inside ConvSubsampling:
self.conv = MaskedConvSequential(
    Conv2d(...),      # index 0
    ReLU(),           # index 1 (no params)
    DepthwiseConv2d,  # index 2
    PointwiseConv2d,  # index 3
    ReLU(),           # index 4 (no params)
    DepthwiseConv2d,  # index 5
    PointwiseConv2d,  # index 6
    ReLU(),           # index 7 (no params)
)
self.out = nn.Linear(...)
```

| NeMo weight name | Our weight name |
|-----------------|-----------------|
| `encoder.pre_encode.conv.0.weight` | `subsampling.conv1.weight` |
| `encoder.pre_encode.conv.0.bias` | `subsampling.conv1.bias` |
| `encoder.pre_encode.conv.2.weight` | `subsampling.dwconv2.weight` |
| `encoder.pre_encode.conv.2.bias` | `subsampling.dwconv2.bias` |
| `encoder.pre_encode.conv.3.weight` | `subsampling.pwconv2.weight` |
| `encoder.pre_encode.conv.3.bias` | `subsampling.pwconv2.bias` |
| `encoder.pre_encode.conv.5.weight` | `subsampling.dwconv3.weight` |
| `encoder.pre_encode.conv.5.bias` | `subsampling.dwconv3.bias` |
| `encoder.pre_encode.conv.6.weight` | `subsampling.pwconv3.weight` |
| `encoder.pre_encode.conv.6.bias` | `subsampling.pwconv3.bias` |
| `encoder.pre_encode.out.weight` | `subsampling.out.weight` |
| `encoder.pre_encode.out.bias` | `subsampling.out.bias` |

### Self-Attention Module (`self_attn`)

Each Conformer layer has a self-attention module with the following weights:

| NeMo weight name | Our weight name |
|-----------------|-----------------|
| `encoder.layers.{i}.self_attn.linear_q.weight` | `layers.{i}.self_attn.linear_q.weight` |
| `encoder.layers.{i}.self_attn.linear_q.bias` | `layers.{i}.self_attn.linear_q.bias` |
| `encoder.layers.{i}.self_attn.linear_k.weight` | `layers.{i}.self_attn.linear_k.weight` |
| `encoder.layers.{i}.self_attn.linear_k.bias` | `layers.{i}.self_attn.linear_k.bias` |
| `encoder.layers.{i}.self_attn.linear_v.weight` | `layers.{i}.self_attn.linear_v.weight` |
| `encoder.layers.{i}.self_attn.linear_v.bias` | `layers.{i}.self_attn.linear_v.bias` |
| `encoder.layers.{i}.self_attn.linear_out.weight` | `layers.{i}.self_attn.linear_out.weight` |
| `encoder.layers.{i}.self_attn.linear_out.bias` | `layers.{i}.self_attn.linear_out.bias` |
| `encoder.layers.{i}.self_attn.linear_pos.weight` | `layers.{i}.self_attn.linear_pos.weight` |
| `encoder.layers.{i}.self_attn.pos_bias_u` | `layers.{i}.self_attn.pos_bias_u` |
| `encoder.layers.{i}.self_attn.pos_bias_v` | `layers.{i}.self_attn.pos_bias_v` |

**Note**: `linear_pos` has NO bias (critical for weight loading).

### Convolution Module (`conv_module`)

Each Conformer layer has a convolution module with the following weights:

| NeMo weight name | Our weight name |
|-----------------|-----------------|
| `encoder.layers.{i}.conv_module.pointwise_conv1.weight` | `layers.{i}.conv_module.pointwise_conv1.weight` |
| `encoder.layers.{i}.conv_module.pointwise_conv1.bias` | `layers.{i}.conv_module.pointwise_conv1.bias` |
| `encoder.layers.{i}.conv_module.depthwise_conv.weight` | `layers.{i}.conv_module.depthwise_conv.weight` |
| `encoder.layers.{i}.conv_module.depthwise_conv.bias` | `layers.{i}.conv_module.depthwise_conv.bias` |
| `encoder.layers.{i}.conv_module.batch_norm.weight` | `layers.{i}.conv_module.batch_norm.weight` |
| `encoder.layers.{i}.conv_module.batch_norm.bias` | `layers.{i}.conv_module.batch_norm.bias` |
| `encoder.layers.{i}.conv_module.batch_norm.running_mean` | `layers.{i}.conv_module.batch_norm.running_mean` |
| `encoder.layers.{i}.conv_module.batch_norm.running_var` | `layers.{i}.conv_module.batch_norm.running_var` |
| `encoder.layers.{i}.conv_module.pointwise_conv2.weight` | `layers.{i}.conv_module.pointwise_conv2.weight` |
| `encoder.layers.{i}.conv_module.pointwise_conv2.bias` | `layers.{i}.conv_module.pointwise_conv2.bias` |

**Note**: BatchNorm has running statistics (running_mean, running_var) that are buffers, not parameters.

### Conformer Layers (to be filled)

```
encoder.layers.0.norm_feed_forward1.weight → layers.0.norm_ffn1.weight
encoder.layers.0.feed_forward1.linear1.weight → layers.0.ffn1.linear1.weight
...
```

---

## Verification Strategy

1. **Unit tests per module**: Each module tested against NeMo output
2. **Shape tests**: Verify all intermediate shapes match expected
3. **Numerical tests**: Compare outputs within tolerance (1e-4 for float32)
4. **Weight loading test**: Load NeMo weights, verify forward pass matches

---

## Progress Log

### 2024-12-28: Subsampling Module Complete
- [x] Created module structure (`conformer_lite/`)
- [x] Implemented `subsampling.py` with `ConvSubsampling` class
- [x] Created unit tests in `tests/test_subsampling.py` (8 tests passing)
- [x] Documented weight mapping for subsampling module
- [ ] TODO: Implement weight loading utility in `nemo_lite/weights.py`

### 2024-12-28: Positional Encoding Complete
- [x] Implemented `pos_encoding.py` with `RelPositionalEncoding` class
- [x] Created unit tests in `tests/test_pos_encoding.py` (11 tests passing)
- [x] No learnable parameters - uses sinusoidal embeddings as buffer
- [x] Verified: position ordering, sinusoidal values, div_term formula

### 2024-12-28: Multi-Head Attention Complete
- [x] Implemented `attention.py` with `RelPositionMultiHeadAttention` class
- [x] Created unit tests in `tests/test_attention.py` (17 tests passing)
- [x] Key implementation details:
  - `linear_pos` has NO bias (critical for weight compatibility)
  - `pos_bias_u`, `pos_bias_v`: shape (n_heads, d_k), initialized to zeros
  - `rel_shift`: Transformer-XL skewing trick for relative positions
  - Uses SDPA (scaled_dot_product_attention) for efficiency
  - Pre-scales matrix_bd for SDPA's additive attention mask
- [x] Documented weight mapping in WORKING_LOG.md

### 2024-12-28: Convolution Module Complete
- [x] Implemented `convolution.py` with `ConvolutionModule` class
- [x] Created unit tests in `tests/test_convolution.py` (17 tests passing)
- [x] Key implementation details:
  - GLU on dim=1 (channel dimension) after pointwise expansion
  - Depthwise Conv1d with symmetric padding for non-streaming
  - BatchNorm1d (NOT LayerNorm!) - critical for correctness
  - Swish/SiLU activation
- [x] Documented weight mapping in WORKING_LOG.md

### 2024-12-28: Feed-Forward Module Complete
- [x] Implemented `feed_forward.py` with `FeedForwardModule` class
- [x] Created unit tests in `tests/test_feed_forward.py` (10 tests passing)
- [x] Simple structure: Linear → Swish → Dropout → Linear

### 2024-12-28: Conformer Block Complete
- [x] Implemented `conformer_block.py` with `ConformerBlock` class
- [x] Created unit tests in `tests/test_conformer_block.py` (11 tests passing)
- [x] Key implementation details:
  - Macaron-style: FFN1 → Attention → Conv → FFN2 → LayerNorm
  - FFN residual scaling factor = 0.5
  - Pre-norm LayerNorm before each sub-module
  - Separate dropout wrapping each sub-module output

### 2024-12-28: FastConformer Encoder Complete
- [x] Implemented `encoder.py` with `FastConformerEncoder` class
- [x] Created unit tests in `tests/test_encoder.py` (12 tests passing)
- [x] Complete pipeline:
  - ConvSubsampling (8x time reduction)
  - RelPositionalEncoding (sinusoidal, no learnable params)
  - 32× ConformerBlock (configurable)
  - Padding mask creation for variable-length inputs
- [x] Total: 86 tests passing

## Implementation Complete!

All core components of the FastConformer encoder are implemented:
- ConvSubsampling (dw_striding, 8x reduction)
- RelPositionalEncoding (Transformer-XL style)
- RelPositionMultiHeadAttention (with SDPA)
- ConvolutionModule (BatchNorm, not LayerNorm)
- FeedForwardModule (Swish activation)
- ConformerBlock (Macaron-style)
- FastConformerEncoder (complete pipeline)

Next steps:
- [ ] Implement weight loading utility
- [ ] Verify numerical equivalence with NeMo
- [ ] Integrate with Qwen decoder
