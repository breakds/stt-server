"""FastConformer encoder for ASR.

Implements the complete FastConformer encoder that combines subsampling,
positional encoding, and stacked Conformer blocks.

Reference: NeMo's ConformerEncoder in
    nemo/collections/asr/modules/conformer_encoder.py

Architecture:
    1. ConvSubsampling: (B, T, 128) -> (B, T/8, 1024)
    2. RelPositionalEncoding: adds position embeddings
    3. 32× ConformerBlock: self-attention + convolution + feed-forward
    4. Output: (B, T/8, 1024)

Weight Loading:
    NeMo weight names:
        encoder.pre_encode.* (subsampling)
        encoder.pos_enc.* (positional encoding - no learnable params)
        encoder.layers.{i}.* (32 conformer blocks)
"""

import torch
import torch.nn as nn

from nemo_lite.conformer_lite.conformer_block import ConformerBlock
from nemo_lite.conformer_lite.pos_encoding import RelPositionalEncoding
from nemo_lite.conformer_lite.subsampling import ConvSubsampling


class FastConformerEncoder(nn.Module):
    """FastConformer encoder for ASR (inference only).

    Combines subsampling, positional encoding, and Conformer blocks to encode
    audio features into high-level representations.

    Input/Output:
        Input:
            - audio_signal: (B, feat_in, T) mel spectrogram features
            - length: (B,) valid lengths for each sample
        Output:
            - encoded: (B, T/8, d_model) encoded features
            - length: (B,) output lengths after subsampling

    Args:
        feat_in: Input feature dimension (mel bins). Default: 128.
        n_layers: Number of Conformer blocks. Default: 32.
        d_model: Model dimension. Default: 1024.
        d_ff: Feed-forward dimension. Default: 4096.
        n_heads: Number of attention heads. Default: 8.
        conv_kernel_size: Kernel size for convolution module. Default: 9.
        subsampling_factor: Time reduction factor. Default: 8.
        subsampling_conv_channels: Channels in subsampling convs. Default: 256.
        dropout_rate: Dropout rate. Default: 0.1.
        dropout_att: Dropout rate for attention. Default: 0.1.
    """

    def __init__(
        self,
        feat_in: int = 128,
        n_layers: int = 32,
        d_model: int = 1024,
        d_ff: int = 4096,
        n_heads: int = 8,
        conv_kernel_size: int = 9,
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        dropout_rate: float = 0.1,
        dropout_att: float = 0.1,
    ):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model
        self.n_layers = n_layers

        # Subsampling (pre_encode in NeMo)
        # Reduces time by subsampling_factor (8x for dw_striding)
        self.pre_encode = ConvSubsampling(
            feat_in=feat_in,
            d_model=d_model,
            conv_channels=subsampling_conv_channels,
        )

        # Positional encoding
        # xscale=None for Canary (no input scaling)
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_rate,
            xscale=None,
        )

        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate,
                dropout_att=dropout_att,
            )
            for _ in range(n_layers)
        ])

    def _create_pad_mask(
        self,
        length: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Create padding mask from lengths.

        Args:
            length: Valid lengths of shape (B,).
            max_len: Maximum sequence length.

        Returns:
            Padding mask of shape (B, max_len). True = padded position.
        """
        # Create position indices: (1, max_len)
        positions = torch.arange(max_len, device=length.device).unsqueeze(0)
        # Compare with lengths: (B, 1)
        lengths = length.unsqueeze(1)
        # True where position >= length (i.e., padded)
        pad_mask = positions >= lengths
        return pad_mask

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            audio_signal: Input mel spectrogram of shape (B, feat_in, T).
            length: Valid lengths for each sample of shape (B,).

        Returns:
            Tuple of:
                - Encoded features of shape (B, T/8, d_model).
                - Output lengths of shape (B,).
        """
        # Transpose: (B, feat_in, T) -> (B, T, feat_in)
        audio_signal = audio_signal.transpose(1, 2)

        # Subsampling: (B, T, feat_in) -> (B, T/8, d_model)
        audio_signal, length = self.pre_encode(audio_signal, length)
        length = length.to(torch.int64)

        # Positional encoding
        audio_signal, pos_emb = self.pos_enc(audio_signal)

        # Create padding mask
        max_len = audio_signal.size(1)
        pad_mask = self._create_pad_mask(length, max_len)

        # For full attention (non-streaming), we still need to mask padded positions.
        # NeMo builds att_mask from pad_mask even for unlimited context (att_context_size=[-1,-1]).
        # The attention mask prevents queries from attending to padded key/value positions.
        # Shape: (B, T, T) where True = masked (should not attend)
        # pad_mask[:, None, :] broadcasts query dim, pad_mask[:, :, None] broadcasts key dim
        # We mask position (i,j) if position j is padded (regardless of i)
        att_mask = pad_mask.unsqueeze(1).expand(-1, max_len, -1)

        # Apply Conformer blocks
        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                pos_emb=pos_emb,
                att_mask=att_mask,
                pad_mask=pad_mask,
            )

        return audio_signal, length
