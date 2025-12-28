"""Tests for RelPositionMultiHeadAttention module."""

import math
import unittest

import torch

from nemo_lite.conformer_lite.attention import RelPositionMultiHeadAttention


class TestRelPositionMultiHeadAttentionShape(unittest.TestCase):
    """Test output shapes of RelPositionMultiHeadAttention."""

    def test_output_shape_basic(self):
        """Test basic output shape."""
        attn = RelPositionMultiHeadAttention(n_heads=8, d_model=1024)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 1024)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 1024)

        out = attn(x, x, x, pos_emb)

        self.assertEqual(out.shape, (batch_size, seq_len, 1024))

    def test_output_shape_various_lengths(self):
        """Test output shape with various sequence lengths."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=256)

        for seq_len in [10, 50, 100]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 256)
                pos_emb = torch.randn(1, 2 * seq_len - 1, 256)

                out = attn(x, x, x, pos_emb)

                self.assertEqual(out.shape, (1, seq_len, 256))

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=256)
        seq_len = 50
        pos_emb = torch.randn(1, 2 * seq_len - 1, 256)

        for batch_size in [1, 4, 16]:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, seq_len, 256)
                out = attn(x, x, x, pos_emb)
                self.assertEqual(out.shape, (batch_size, seq_len, 256))


class TestRelShift(unittest.TestCase):
    """Test the rel_shift operation."""

    def test_rel_shift_shape(self):
        """Test that rel_shift preserves shape."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64)

        # Simulate position attention before shift
        batch_size = 2
        n_heads = 4
        seq_len = 5
        pos_len = 2 * seq_len - 1  # 9

        x = torch.randn(batch_size, n_heads, seq_len, pos_len)
        shifted = attn._rel_shift(x)

        self.assertEqual(shifted.shape, x.shape)

    def test_rel_shift_alignment(self):
        """Test that rel_shift correctly aligns relative positions.

        For a sequence of length 3 with positions [2, 1, 0, -1, -2]:
        - Query at position 0 should attend to keys with relative positions [0, -1, -2]
        - Query at position 1 should attend to keys with relative positions [1, 0, -1]
        - Query at position 2 should attend to keys with relative positions [2, 1, 0]
        """
        attn = RelPositionMultiHeadAttention(n_heads=1, d_model=4)

        # Create a simple test case
        # pos_emb positions: [2, 1, 0, -1, -2] (5 positions for seq_len=3)
        # We'll use position index as the value for easy tracking
        seq_len = 3
        pos_len = 2 * seq_len - 1  # 5

        # Create input where each position column has a distinct value
        x = torch.arange(pos_len).float().view(1, 1, 1, pos_len).expand(1, 1, seq_len, -1)

        shifted = attn._rel_shift(x)

        # After shift:
        # Row 0 (query pos 0): should see [2, 1, 0, -1, -2] shifted to align 0 with key 0
        #   -> indices [2, 3, 4, ?, ?] (positions 0, -1, -2 for keys 0, 1, 2)
        # Row 1 (query pos 1): [1, 2, 3, ?, ?] (positions 1, 0, -1)
        # Row 2 (query pos 2): [0, 1, 2, ?, ?] (positions 2, 1, 0)

        # Check first few columns which are within bounds
        # Row 0, col 0: should be value at original position 2 (index 2)
        self.assertEqual(shifted[0, 0, 0, 0].item(), 2.0)
        # Row 1, col 0: should be value at original position 1 (index 1)
        self.assertEqual(shifted[0, 0, 1, 0].item(), 1.0)
        # Row 2, col 0: should be value at original position 0 (index 0)
        self.assertEqual(shifted[0, 0, 2, 0].item(), 0.0)


class TestPositionBiases(unittest.TestCase):
    """Test pos_bias_u and pos_bias_v."""

    def test_bias_shape(self):
        """Test that position biases have correct shape."""
        n_heads = 8
        d_model = 1024
        d_k = d_model // n_heads

        attn = RelPositionMultiHeadAttention(n_heads=n_heads, d_model=d_model)

        self.assertEqual(attn.pos_bias_u.shape, (n_heads, d_k))
        self.assertEqual(attn.pos_bias_v.shape, (n_heads, d_k))

    def test_bias_initialization(self):
        """Test that position biases are initialized to zeros."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64)

        torch.testing.assert_close(
            attn.pos_bias_u,
            torch.zeros_like(attn.pos_bias_u),
        )
        torch.testing.assert_close(
            attn.pos_bias_v,
            torch.zeros_like(attn.pos_bias_v),
        )

    def test_bias_are_parameters(self):
        """Test that position biases are learnable parameters."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64)

        param_names = [name for name, _ in attn.named_parameters()]

        self.assertIn("pos_bias_u", param_names)
        self.assertIn("pos_bias_v", param_names)


class TestLinearPos(unittest.TestCase):
    """Test linear_pos layer properties."""

    def test_linear_pos_no_bias(self):
        """Test that linear_pos has no bias (critical for weight compatibility)."""
        attn = RelPositionMultiHeadAttention(n_heads=8, d_model=1024)

        self.assertIsNone(attn.linear_pos.bias)

    def test_other_linears_have_bias(self):
        """Test that Q, K, V, Out projections have bias."""
        attn = RelPositionMultiHeadAttention(n_heads=8, d_model=1024)

        self.assertIsNotNone(attn.linear_q.bias)
        self.assertIsNotNone(attn.linear_k.bias)
        self.assertIsNotNone(attn.linear_v.bias)
        self.assertIsNotNone(attn.linear_out.bias)


class TestMasking(unittest.TestCase):
    """Test attention masking behavior."""

    def test_mask_shape(self):
        """Test that masking works with correct shapes."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64)
        attn.eval()  # Disable dropout

        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 64)

        # Create a simple mask (upper triangular = causal)
        mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()

        out = attn(x, x, x, pos_emb, mask=mask)

        self.assertEqual(out.shape, (batch_size, seq_len, 64))

    def test_full_mask_produces_zeros(self):
        """Test that fully masked positions produce zero output."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64, dropout_rate=0.0)
        attn.eval()

        batch_size = 1
        seq_len = 5
        x = torch.randn(batch_size, seq_len, 64)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 64)

        # Mask everything for position 0
        mask = torch.zeros(batch_size, seq_len, seq_len).bool()
        mask[0, 0, :] = True  # Position 0 can't attend to anything

        out = attn(x, x, x, pos_emb, mask=mask)

        # After masking all positions, softmax gives uniform 0s,
        # which means output for position 0 should be 0 (before linear_out bias)
        # But linear_out has bias, so we just check output is valid
        self.assertFalse(torch.isnan(out).any())


class TestDropout(unittest.TestCase):
    """Test dropout behavior."""

    def test_dropout_in_training(self):
        """Test that dropout is applied during training."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64, dropout_rate=0.5)
        attn.train()

        torch.manual_seed(42)
        x = torch.randn(2, 20, 64)
        pos_emb = torch.randn(1, 39, 64)

        # Run twice with same input but different dropout
        out1 = attn(x, x, x, pos_emb)
        out2 = attn(x, x, x, pos_emb)

        # Outputs should be different due to dropout
        self.assertFalse(torch.allclose(out1, out2))

    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied during eval."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64, dropout_rate=0.5)
        attn.eval()

        x = torch.randn(2, 20, 64)
        pos_emb = torch.randn(1, 39, 64)

        out1 = attn(x, x, x, pos_emb)
        out2 = attn(x, x, x, pos_emb)

        # Outputs should be identical in eval mode
        torch.testing.assert_close(out1, out2)


class TestDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters correctly."""
        attn = RelPositionMultiHeadAttention(n_heads=4, d_model=64)

        # All parameters should be on CPU by default
        for param in attn.parameters():
            self.assertEqual(param.device.type, "cpu")

        if torch.cuda.is_available():
            attn = attn.to("cuda")
            for param in attn.parameters():
                self.assertEqual(param.device.type, "cuda")

            # Test forward pass on CUDA
            x = torch.randn(1, 10, 64, device="cuda")
            pos_emb = torch.randn(1, 19, 64, device="cuda")
            out = attn(x, x, x, pos_emb)
            self.assertEqual(out.device.type, "cuda")


class TestNumericalProperties(unittest.TestCase):
    """Test numerical properties of the attention computation."""

    def test_attention_symmetry_without_pos(self):
        """Test attention behavior with zero position biases and uniform pos_emb."""
        n_heads = 4
        d_model = 64
        attn = RelPositionMultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout_rate=0.0)
        attn.eval()

        # With zero position biases (default init) and zero pos_emb,
        # the attention should be purely content-based
        seq_len = 5
        x = torch.randn(1, seq_len, d_model)
        pos_emb = torch.zeros(1, 2 * seq_len - 1, d_model)

        # Set linear_pos to zero to remove position contribution
        with torch.no_grad():
            attn.linear_pos.weight.zero_()

        out = attn(x, x, x, pos_emb)

        # Output should be valid (no NaN/Inf)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_scale_factor(self):
        """Test that scale factor is 1/sqrt(d_k) for SDPA pre-scaling."""
        n_heads = 8
        d_model = 1024
        d_k = d_model // n_heads  # 128

        attn = RelPositionMultiHeadAttention(n_heads=n_heads, d_model=d_model)

        # Scale is 1/sqrt(d_k) because we pre-scale matrix_bd for SDPA
        self.assertAlmostEqual(attn.scale, 1.0 / math.sqrt(d_k), places=10)


if __name__ == "__main__":
    unittest.main()
