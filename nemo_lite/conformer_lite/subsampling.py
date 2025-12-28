"""Convolutional subsampling for FastConformer encoder.

Implements dw_striding (depthwise-separable striding) subsampling that reduces
the time dimension by 8x while projecting features to the model dimension.

Reference: NeMo's ConvSubsampling in
    nemo/collections/asr/parts/submodules/subsampling.py

Weight Loading:
    NeMo uses nn.Sequential indices for conv layers. The mapping is:
        encoder.pre_encode.conv.0  -> conv1
        encoder.pre_encode.conv.2  -> dwconv2
        encoder.pre_encode.conv.3  -> pwconv2
        encoder.pre_encode.conv.5  -> dwconv3
        encoder.pre_encode.conv.6  -> pwconv3
        encoder.pre_encode.out     -> out
    See WORKING_LOG.md for the full mapping table.

    TODO: Implement weight loading utility in nemo_lite/weights.py
"""

import torch
import torch.nn as nn


class ConvSubsampling(nn.Module):
    """Depthwise-separable striding subsampling (dw_striding).

    Reduces time dimension by 8x using three stride-2 convolution stages:
    1. Regular Conv2d: 1 channel → conv_channels
    2. Depthwise-separable Conv2d (×2): maintains conv_channels

    After convolutions, flattens the channel and frequency dimensions and
    projects to the model dimension.

    Input/Output:
        Input: (B, T, feat_in) where feat_in=128 (mel features)
        Output: (B, T//8, d_model) where d_model=1024

    Args:
        feat_in: Input feature dimension (mel bins). Default: 128.
        d_model: Output model dimension. Default: 1024.
        conv_channels: Number of channels in conv layers. Default: 256.
    """

    def __init__(
        self,
        feat_in: int = 128,
        d_model: int = 1024,
        conv_channels: int = 256,
    ):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model
        self.conv_channels = conv_channels
        self.subsampling_factor = 8

        # Convolution parameters
        kernel_size = 3
        stride = 2
        padding = (kernel_size - 1) // 2  # = 1, for "same" padding with stride

        # Layer 1: Regular Conv2d
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Layer 2: Depthwise-separable Conv2d
        #
        #  Depthwise Conv2d (groups=out_channels=in_channels):
        # 
        #  Each output channel sees only ONE input channel
        #  Input channels:   [0]  [1]  [2]  ... [255]
        #                     │    │    │         │
        #                     ▼    ▼    ▼         ▼
        #  Output channels:  [0]  [1]  [2]  ... [255]
        self.dwconv2 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=conv_channels,  # depthwise
        )
        self.pwconv2 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Layer 3: Depthwise-separable Conv2d
        self.dwconv3 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=conv_channels,  # depthwise
        )
        self.pwconv3 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.activation = nn.ReLU()

        # Calculate output frequency dimension after 3x stride-2
        # feat_in=128 → 64 → 32 → 16
        freq_out = feat_in
        for _ in range(3):  # 3 stride-2 operations
            freq_out = (freq_out + 2 * padding - kernel_size) // stride + 1

        # Final linear projection
        self.out = nn.Linear(conv_channels * freq_out, d_model)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, feat_in).
            lengths: Valid lengths for each sample in the batch, shape (B,).

        Returns:
            Tuple of:
                - Output tensor of shape (B, T//8, d_model).
                - Output lengths of shape (B,).
        """
        # Add channel dimension: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)

        # Layer 1: Conv + ReLU
        x = self.activation(self.conv1(x))

        # Layer 2: Depthwise + Pointwise + ReLU
        x = self.dwconv2(x)
        x = self.activation(self.pwconv2(x))

        # Layer 3: Depthwise + Pointwise + ReLU
        x = self.dwconv3(x)
        x = self.activation(self.pwconv3(x))

        # x is now (B, C, T', F') where T' = T//8, F' = F//8

        # Transpose and flatten: (B, C, T', F') -> (B, T', C*F')
        b, c, t, f = x.shape
        x = x.transpose(1, 2)  # (B, T', C, F')
        x = x.reshape(b, t, c * f)  # (B, T', C*F')

        # Project to model dimension
        x = self.out(x)  # (B, T', d_model)

        # Calculate output lengths
        # Each stride-2 with padding=1, kernel=3: out = floor((in + 2*1 - 3) / 2) + 1 = floor((in - 1) / 2) + 1
        out_lengths = lengths
        for _ in range(3):
            out_lengths = (out_lengths - 1) // 2 + 1

        return x, out_lengths
