"""Positional encoding for FastConformer encoder.

Implements relative positional encoding (Transformer-XL style) used in
FastConformer's self-attention layers.

Reference: NeMo's RelPositionalEncoding in
    nemo/collections/asr/parts/submodules/multi_head_attention.py

See: Appendix B in https://arxiv.org/abs/1901.02860 (Transformer-XL)

Weight Loading:
    This module has no learnable parameters. The positional embeddings are
    computed from sinusoidal functions and stored as a buffer.
"""

import math

import torch
import torch.nn as nn


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding for Transformer-XL style attention.

    Creates sinusoidal positional embeddings for relative positions. For a
    sequence of length L, positions range from (L-1) to -(L-1), giving 2L-1
    total positions. Positive positions represent "left" (past) tokens,
    negative positions represent "right" (future) tokens.

    Input/Output:
        Input: x of shape (B, T, d_model)
        Output: (x', pos_emb) where:
            - x' is dropout(xscale * x) of shape (B, T, d_model)
            - pos_emb is (1, 2T-1, d_model) positional embeddings

    Args:
        d_model: Model dimension. Default: 1024.
        dropout_rate: Dropout rate applied to x. Default: 0.1.
        max_len: Maximum sequence length for pre-computed embeddings. Default: 5000.
        xscale: If not None, scale input by this value (typically sqrt(d_model)).
            Default: None (no scaling, matching Canary config).
        dropout_rate_emb: Dropout rate for positional embeddings. Default: 0.0.
    """

    def __init__(
        self,
        d_model: int = 1024,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
        xscale: float | None = None,
        dropout_rate_emb: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.max_len = max_len
        # Note: Dropout is a no-op during inference (model.eval()).
        # Kept for NeMo compatibility during development/verification.
        self.dropout = nn.Dropout(p=dropout_rate)

        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(p=dropout_rate_emb)
        else:
            self.dropout_emb = None

        # Pre-compute positional encodings for max_len
        # Will be extended if needed
        self._extend_pe(max_len, torch.device("cpu"), torch.float32)

    def _create_pe(self, positions: torch.Tensor, dtype: torch.dtype) -> None:
        """Create sinusoidal positional encodings.

        Args:
            positions: Position indices of shape (num_positions, 1).
            dtype: Data type for the encodings.
        """
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)

        # div_term = exp(arange(0, d_model, 2) * -(log(10000) / d_model))
        # This is equivalent to: 1 / (10000 ^ (2i / d_model))
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )

        # pe[:, 0::2] = sin(pos / 10000^(2i/d_model))
        # pe[:, 1::2] = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Add batch dimension: (1, num_positions, d_model)
        pe = pe.unsqueeze(0).to(dtype)

        # Register as buffer (not a parameter, but moves with .to())
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def _extend_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        """Extend positional encodings if needed.

        For relative positions, we need 2*length - 1 positions:
        positions from (length-1) down to -(length-1).

        Args:
            length: Required sequence length.
            device: Device to create tensors on.
            dtype: Data type for the encodings.
        """
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            return

        # Positions: [length-1, length-2, ..., 1, 0, -1, ..., -(length-1)]
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)

        self._create_pe(positions, dtype)

    def forward(
        self,
        x: torch.Tensor,
        cache_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.

        Args:
            x: Input tensor of shape (B, T, d_model).
            cache_len: Length of cached context (for streaming). Default: 0.

        Returns:
            Tuple of:
                - x after optional scaling and dropout, shape (B, T, d_model).
                - pos_emb: Positional embeddings, shape (1, 2*T-1, d_model).
        """
        # Extend PE buffer if needed
        input_len = x.size(1) + cache_len
        self._extend_pe(input_len, x.device, x.dtype)

        # Optional scaling (disabled for Canary: xscale=None)
        if self.xscale is not None:
            x = x * self.xscale

        # Extract position embeddings for current input length
        # PE buffer has 2*max_len - 1 positions centered at max_len - 1
        # For input_len, we need positions from (input_len-1) to -(input_len-1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        # Optional dropout on positional embeddings
        if self.dropout_emb is not None:
            pos_emb = self.dropout_emb(pos_emb)

        # Dropout on input
        x = self.dropout(x)

        return x, pos_emb
