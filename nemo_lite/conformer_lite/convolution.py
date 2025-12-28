"""Convolution module for FastConformer encoder.

Implements the convolution block used in each Conformer layer. This module
captures local context through depthwise separable convolutions.

Reference: NeMo's ConformerConvolution in
    nemo/collections/asr/parts/submodules/conformer_modules.py

Architecture:
    1. Pointwise Conv1d (expansion with GLU)
    2. Depthwise Conv1d (local context)
    3. BatchNorm1d (NOT LayerNorm!)
    4. Swish activation
    5. Pointwise Conv1d (projection)

Weight Loading:
    NeMo weight names:
        encoder.layers.{i}.conv_module.pointwise_conv1.weight
        encoder.layers.{i}.conv_module.pointwise_conv1.bias
        encoder.layers.{i}.conv_module.depthwise_conv.weight
        encoder.layers.{i}.conv_module.depthwise_conv.bias
        encoder.layers.{i}.conv_module.batch_norm.weight
        encoder.layers.{i}.conv_module.batch_norm.bias
        encoder.layers.{i}.conv_module.batch_norm.running_mean
        encoder.layers.{i}.conv_module.batch_norm.running_var
        encoder.layers.{i}.conv_module.batch_norm.num_batches_tracked
        encoder.layers.{i}.conv_module.pointwise_conv2.weight
        encoder.layers.{i}.conv_module.pointwise_conv2.bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Applies pointwise-depthwise-pointwise convolution pattern with GLU gating
    and BatchNorm normalization.

    Input/Output:
        Input: (B, T, d_model)
        Output: (B, T, d_model)

    Args:
        d_model: Model dimension. Default: 1024.
        kernel_size: Kernel size for depthwise convolution. Default: 9.
    """

    def __init__(
        self,
        d_model: int = 1024,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Pointwise expansion: d_model -> d_model * 2 (for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Depthwise convolution: each channel processed independently
        # Symmetric padding for non-streaming (full context) mode
        # For kernel_size=9: padding = (9-1)//2 = 4
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,  # Depthwise: each channel has its own filter
            bias=True,
        )

        # BatchNorm (NOT LayerNorm!) - critical for correctness
        # Operates on channel dimension, expects (B, C, T) input
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Swish activation (also known as SiLU)
        self.activation = nn.SiLU()

        # Pointwise projection: d_model -> d_model
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).
            pad_mask: Optional padding mask of shape (B, T). True = padded position.

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        # Transpose for Conv1d: (B, T, d_model) -> (B, d_model, T)
        x = x.transpose(1, 2)

        # Pointwise expansion with GLU
        x = self.pointwise_conv1(x)  # (B, d_model*2, T)
        x = F.glu(x, dim=1)  # GLU on channel dim -> (B, d_model, T)

        # Apply padding mask before depthwise conv (if provided)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        # Depthwise convolution
        x = self.depthwise_conv(x)  # (B, d_model, T)

        # BatchNorm + Swish
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)  # (B, d_model, T)

        # Transpose back: (B, d_model, T) -> (B, T, d_model)
        x = x.transpose(1, 2)

        return x
