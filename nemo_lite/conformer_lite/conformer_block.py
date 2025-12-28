"""Conformer block for FastConformer encoder.

Implements a single Conformer layer that combines:
- Feed-Forward Module (×2, with 0.5 residual scaling)
- Multi-Head Self-Attention with Relative Position Encoding
- Convolution Module
- Pre-norm LayerNorm before each sub-module
- Final LayerNorm

Reference: NeMo's ConformerLayer in
    nemo/collections/asr/parts/submodules/conformer_modules.py

Architecture (Macaron-style):
    x = x + 0.5 * dropout(FFN1(norm1(x)))
    x = x + dropout(Attention(norm2(x), pos_emb))
    x = x + dropout(Conv(norm3(x)))
    x = x + 0.5 * dropout(FFN2(norm4(x)))
    x = norm_out(x)

Weight Loading:
    NeMo weight names for layer i:
        encoder.layers.{i}.norm_feed_forward1.weight/bias
        encoder.layers.{i}.feed_forward1.linear1.weight/bias
        encoder.layers.{i}.feed_forward1.linear2.weight/bias
        encoder.layers.{i}.norm_self_att.weight/bias
        encoder.layers.{i}.self_attn.* (see attention.py)
        encoder.layers.{i}.norm_conv.weight/bias
        encoder.layers.{i}.conv_module.* (see convolution.py)
        encoder.layers.{i}.norm_feed_forward2.weight/bias
        encoder.layers.{i}.feed_forward2.linear1.weight/bias
        encoder.layers.{i}.feed_forward2.linear2.weight/bias
        encoder.layers.{i}.norm_out.weight/bias
"""

import torch
import torch.nn as nn

from nemo_lite.conformer_lite.attention import RelPositionMultiHeadAttention
from nemo_lite.conformer_lite.convolution import ConvolutionModule
from nemo_lite.conformer_lite.feed_forward import FeedForwardModule


class ConformerBlock(nn.Module):
    """A single Conformer encoder block.

    Combines FFN, self-attention, and convolution with pre-norm and residual
    connections. Uses Macaron-style architecture with two FFN modules.

    Input/Output:
        Input:
            - x: (B, T, d_model)
            - pos_emb: (1, 2T-1, d_model) from RelPositionalEncoding
            - att_mask: Optional (B, T, T), True = masked
            - pad_mask: Optional (B, T), True = padded
        Output: (B, T, d_model)

    Args:
        d_model: Model dimension. Default: 1024.
        d_ff: Feed-forward dimension. Default: 4096.
        n_heads: Number of attention heads. Default: 8.
        conv_kernel_size: Kernel size for convolution module. Default: 9.
        dropout_rate: Dropout rate. Default: 0.1.
        dropout_att: Dropout rate for attention. Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_ff: int = 4096,
        n_heads: int = 8,
        conv_kernel_size: int = 9,
        dropout_rate: float = 0.1,
        dropout_att: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.fc_factor = 0.5  # Residual scaling for FFN modules

        # FFN1: First feed-forward module
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
        )

        # Self-attention
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model,
            dropout_rate=dropout_att,
        )

        # Convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv_module = ConvolutionModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
        )

        # FFN2: Second feed-forward module
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForwardModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
        )

        # Final layer norm
        self.norm_out = nn.LayerNorm(d_model)

        # Dropout applied to each sub-module output before residual addition
        # Note: This is separate from dropout inside each sub-module
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        att_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).
            pos_emb: Positional embeddings of shape (1, 2T-1, d_model).
            att_mask: Attention mask of shape (B, T, T). True = masked.
            pad_mask: Padding mask of shape (B, T). True = padded.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        # FFN1 with 0.5 residual
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # Self-attention with full residual
        x = self.norm_self_att(residual)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            pos_emb=pos_emb,
            mask=att_mask,
        )
        residual = residual + self.dropout(x)

        # Convolution with full residual
        x = self.norm_conv(residual)
        x = self.conv_module(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)

        # FFN2 with 0.5 residual
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        # Final layer norm
        x = self.norm_out(residual)

        return x
