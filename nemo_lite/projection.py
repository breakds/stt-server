"""Projection layer for Canary-Qwen model.

Projects FastConformer encoder output to Qwen LLM embedding space.

The projection is a simple linear layer:
    Input: (B, T, 1024) - encoder output
    Output: (B, T, 2048) - LLM embedding dimension

Weight Loading:
    Checkpoint key: perception.proj.weight, perception.proj.bias
    Our key: proj.weight, proj.bias
"""

import torch
import torch.nn as nn


class AudioProjection(nn.Module):
    """Projects audio encoder output to LLM embedding space.

    Args:
        encoder_dim: Encoder output dimension. Default: 1024.
        llm_dim: LLM embedding dimension. Default: 2048.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        llm_dim: int = 2048,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.proj = nn.Linear(encoder_dim, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output to LLM space.

        Args:
            x: Encoder output of shape (B, T, encoder_dim).

        Returns:
            Projected features of shape (B, T, llm_dim).
        """
        return self.proj(x)
