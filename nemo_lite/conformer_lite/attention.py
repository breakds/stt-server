"""Multi-head attention with relative positional encoding for FastConformer.

Implements Transformer-XL style relative position attention used in FastConformer's
self-attention layers. Uses PyTorch's scaled_dot_product_attention (SDPA) for
efficient computation.

Note: Due to the relative position bias (matrix_bd), SDPA will use the
EFFICIENT_ATTENTION backend (not FLASH_ATTENTION, which doesn't support
custom attention masks).

Reference: NeMo's RelPositionMultiHeadAttention in
    nemo/collections/asr/parts/submodules/multi_head_attention.py

See: Section 3.3 in https://arxiv.org/abs/1901.02860 (Transformer-XL)

Weight Loading:
    NeMo weight names use the following structure:
        encoder.layers.{i}.self_attn.linear_q.weight
        encoder.layers.{i}.self_attn.linear_q.bias
        encoder.layers.{i}.self_attn.linear_k.weight
        encoder.layers.{i}.self_attn.linear_k.bias
        encoder.layers.{i}.self_attn.linear_v.weight
        encoder.layers.{i}.self_attn.linear_v.bias
        encoder.layers.{i}.self_attn.linear_out.weight
        encoder.layers.{i}.self_attn.linear_out.bias
        encoder.layers.{i}.self_attn.linear_pos.weight  (no bias!)
        encoder.layers.{i}.self_attn.pos_bias_u
        encoder.layers.{i}.self_attn.pos_bias_v
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Value used to mask attention scores (before softmax)
_NEG_INF = -10000.0


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding (Transformer-XL style).

    Uses PyTorch's scaled_dot_product_attention (SDPA) for efficient computation.
    The relative position bias is passed as attn_mask to SDPA, which computes:
        attn = softmax(Q @ K^T / sqrt(d_k) + attn_mask) @ V

    where attn_mask contains the pre-scaled relative position scores.

    Input/Output:
        Input:
            - query: (B, T, d_model)
            - key: (B, T, d_model)
            - value: (B, T, d_model)
            - mask: (B, T, T) or None - True indicates positions to mask
            - pos_emb: (1, 2T-1, d_model) from RelPositionalEncoding
        Output: (B, T, d_model)

    Args:
        n_heads: Number of attention heads. Default: 8.
        d_model: Model dimension. Default: 1024.
        dropout_rate: Dropout rate for attention weights. Default: 0.1.
    """

    def __init__(
        self,
        n_heads: int = 8,
        d_model: int = 1024,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)  # Pre-compute for SDPA
        self.dropout_rate = dropout_rate

        # Q, K, V projections (with bias, matching NeMo default)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Output projection
        self.linear_out = nn.Linear(d_model, d_model)

        # Position projection (NO BIAS - critical for weight compatibility)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        # Learnable position biases (Transformer-XL style)
        # Shape: (n_heads, d_k)
        # Note: With untie_biases=True (Canary default), each layer has its own biases.
        # Initialized to zeros (NeMo's current initialization).
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shifting (skewing trick from Transformer-XL).

        Converts the position attention matrix from absolute position indexing
        to relative position indexing.

        Args:
            x: Position attention matrix of shape (B, n_heads, T, 2T-1).

        Returns:
            Shifted matrix of shape (B, n_heads, T, 2T-1).

        Example:
            For T=3, the input has positions [2, 1, 0, -1, -2] (5 positions).
            After rel_shift, for query at position i, we get attention to
            relative positions [0, -1, -2, ...] aligned with keys at [0, 1, 2].
        """
        b, h, qlen, pos_len = x.size()

        # Pad left side with zeros: (B, h, T, 2T-1) -> (B, h, T, 2T)
        x = F.pad(x, pad=(1, 0))

        # Reshape to transpose last two dims: (B, h, T, 2T) -> (B, h, 2T, T)
        x = x.view(b, h, -1, qlen)

        # Drop first row and reshape back: (B, h, 2T-1, T) -> (B, h, T, 2T-1)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention with relative positional encoding using SDPA.

        Args:
            query: Query tensor of shape (B, T, d_model).
            key: Key tensor of shape (B, T, d_model).
            value: Value tensor of shape (B, T, d_model).
            pos_emb: Positional embeddings of shape (1, 2T-1, d_model).
            mask: Attention mask of shape (B, T, T). True = masked position.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Project Q, K, V: (B, T, d_model) -> (B, n_heads, T, d_k)
        q = self.linear_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.linear_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.linear_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose to (B, n_heads, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Transform positional embeddings: (1, 2T-1, d_model) -> (1, n_heads, 2T-1, d_k)
        pos_len = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(1, pos_len, self.n_heads, self.d_k)
        p = p.transpose(1, 2)  # (1, n_heads, 2T-1, d_k)

        # Add position biases to queries
        # NeMo does this in a specific order: transpose q first, add bias, transpose back
        # q: (B, n_heads, T, d_k) -> transpose to (B, T, n_heads, d_k)
        # add pos_bias: (n_heads, d_k) broadcasts to (B, T, n_heads, d_k)
        # transpose back to (B, n_heads, T, d_k)
        q_transposed = q.transpose(1, 2)  # (B, T, n_heads, d_k)
        q_with_bias_u = (q_transposed + self.pos_bias_u).transpose(1, 2)  # (B, n_heads, T, d_k)
        q_with_bias_v = (q_transposed + self.pos_bias_v).transpose(1, 2)  # (B, n_heads, T, d_k)

        # Position attention: Q_v @ P^T -> (B, n_heads, T, 2T-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        # Apply relative shift to align positions
        matrix_bd = self._rel_shift(matrix_bd)

        # Trim to key length and pre-scale for SDPA
        # SDPA computes: softmax(Q @ K^T / sqrt(d_k) + attn_mask) @ V
        # We want: softmax((Q @ K^T + matrix_bd) / sqrt(d_k)) @ V
        # So we pre-scale matrix_bd by 1/sqrt(d_k) to get equivalent result
        matrix_bd = matrix_bd[:, :, :, :seq_len] * self.scale

        # Apply mask to matrix_bd (SDPA uses additive mask)
        if mask is not None:
            # Expand mask for heads: (B, T, T) -> (B, 1, T, T)
            mask = mask.unsqueeze(1)
            matrix_bd = matrix_bd.masked_fill(mask, _NEG_INF)

        # Use SDPA for efficient attention computation
        # Note: dropout is only applied during training
        dropout_p = self.dropout_rate if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_with_bias_u, k, v,
            attn_mask=matrix_bd,
            dropout_p=dropout_p,
        )

        # Handle fully masked rows (set output to zero)
        # This matches NeMo's behavior and avoids NaN from softmax of all -inf
        if mask is not None:
            # Check if entire rows are masked: (B, 1, T, T) -> all True in last dim
            all_masked_rows = torch.all(mask, dim=-1)  # (B, 1, T)
            all_masked_rows = all_masked_rows.unsqueeze(-1)  # (B, 1, T, 1)
            out = out.masked_fill(all_masked_rows, 0.0)

        # Reshape back: (B, n_heads, T, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        out = self.linear_out(out)

        return out
