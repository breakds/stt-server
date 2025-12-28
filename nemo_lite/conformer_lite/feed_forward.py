"""Feed-forward module for FastConformer encoder.

Implements the feed-forward block used in each Conformer layer.
Two FFN modules are used per block with 0.5 residual scaling.

Reference: NeMo's ConformerFeedForward in
    nemo/collections/asr/parts/submodules/conformer_modules.py

Architecture:
    Linear(d_model, d_ff) → Swish → Dropout → Linear(d_ff, d_model)

Weight Loading:
    NeMo weight names (for FFN1 and FFN2 in each layer):
        encoder.layers.{i}.feed_forward1.linear1.weight
        encoder.layers.{i}.feed_forward1.linear1.bias
        encoder.layers.{i}.feed_forward1.linear2.weight
        encoder.layers.{i}.feed_forward1.linear2.bias
        encoder.layers.{i}.feed_forward2.linear1.weight
        encoder.layers.{i}.feed_forward2.linear1.bias
        encoder.layers.{i}.feed_forward2.linear2.weight
        encoder.layers.{i}.feed_forward2.linear2.bias
"""

import torch.nn as nn


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module.

    Applies expansion → activation → dropout → projection.

    Input/Output:
        Input: (B, T, d_model)
        Output: (B, T, d_model)

    Args:
        d_model: Model dimension. Default: 1024.
        d_ff: Feed-forward dimension. Default: 4096 (4x expansion).
        dropout_rate: Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_ff: int = 4096,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Expansion: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Swish activation (SiLU)
        self.activation = nn.SiLU()

        # Dropout (no-op during inference)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Projection: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
