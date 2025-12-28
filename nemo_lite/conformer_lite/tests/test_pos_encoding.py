"""Tests for RelPositionalEncoding module."""

import math
import unittest

import torch

from nemo_lite.conformer_lite.pos_encoding import RelPositionalEncoding


class TestRelPositionalEncodingShape(unittest.TestCase):
    """Test output shapes of RelPositionalEncoding."""

    def test_output_shape_basic(self):
        """Test basic output shape."""
        pos_enc = RelPositionalEncoding(d_model=1024)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 1024)

        x_out, pos_emb = pos_enc(x)

        # x shape should be unchanged
        self.assertEqual(x_out.shape, (batch_size, seq_len, 1024))

        # pos_emb should be (1, 2*seq_len - 1, d_model)
        expected_pos_len = 2 * seq_len - 1
        self.assertEqual(pos_emb.shape, (1, expected_pos_len, 1024))

    def test_output_shape_various_lengths(self):
        """Test output shape with various sequence lengths."""
        pos_enc = RelPositionalEncoding(d_model=512)

        for seq_len in [10, 50, 100, 500]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 512)
                x_out, pos_emb = pos_enc(x)

                expected_pos_len = 2 * seq_len - 1
                self.assertEqual(pos_emb.shape, (1, expected_pos_len, 512))


class TestRelPositionalEncodingValues(unittest.TestCase):
    """Test correctness of positional encoding values."""

    def test_sinusoidal_pattern(self):
        """Test that encodings follow sinusoidal pattern."""
        d_model = 64
        pos_enc = RelPositionalEncoding(d_model=d_model, dropout_rate=0.0)

        x = torch.randn(1, 10, d_model)
        _, pos_emb = pos_enc(x)

        # pos_emb has 19 positions for seq_len=10
        # Center position (index 9) corresponds to position 0
        center_idx = 9

        # At position 0, sin(0) = 0 for all even indices
        pos_zero = pos_emb[0, center_idx, :]
        for i in range(0, d_model, 2):
            self.assertAlmostEqual(
                pos_zero[i].item(), 0.0, places=5,
                msg=f"sin(0) should be 0 at index {i}"
            )

        # At position 0, cos(0) = 1 for all odd indices
        for i in range(1, d_model, 2):
            self.assertAlmostEqual(
                pos_zero[i].item(), 1.0, places=5,
                msg=f"cos(0) should be 1 at index {i}"
            )

    def test_position_ordering(self):
        """Test that positions are ordered correctly (positive to negative)."""
        pos_enc = RelPositionalEncoding(d_model=64, dropout_rate=0.0)

        x = torch.randn(1, 5, 64)
        _, pos_emb = pos_enc(x)

        # For seq_len=5, positions should be [4, 3, 2, 1, 0, -1, -2, -3, -4]
        # pos_emb shape: (1, 9, 64)

        # Check that position 4 (index 0) has sin(4 * div_term) pattern
        # and position -4 (index 8) has sin(-4 * div_term) pattern
        # These should be negatives of each other for sin (even indices)
        pos_4 = pos_emb[0, 0, 0::2]  # sin components at position 4
        pos_neg4 = pos_emb[0, 8, 0::2]  # sin components at position -4

        # sin(-x) = -sin(x)
        torch.testing.assert_close(pos_4, -pos_neg4, rtol=1e-5, atol=1e-5)

    def test_div_term_formula(self):
        """Test the div_term calculation matches expected formula."""
        d_model = 8
        pos_enc = RelPositionalEncoding(d_model=d_model, dropout_rate=0.0)

        x = torch.randn(1, 3, d_model)
        _, pos_emb = pos_enc(x)

        # For position 1 (index 1 in pos_emb, since positions are [2, 1, 0, -1, -2])
        pos_1 = pos_emb[0, 1, :]

        # Manually compute expected values
        # div_term[i] = exp(-i * log(10000) / d_model) for i in [0, 2, 4, 6]
        for i in range(0, d_model, 2):
            div_term = math.exp(-i * math.log(10000.0) / d_model)
            expected_sin = math.sin(1.0 * div_term)
            expected_cos = math.cos(1.0 * div_term)

            self.assertAlmostEqual(
                pos_1[i].item(), expected_sin, places=5,
                msg=f"sin mismatch at index {i}"
            )
            self.assertAlmostEqual(
                pos_1[i + 1].item(), expected_cos, places=5,
                msg=f"cos mismatch at index {i + 1}"
            )


class TestRelPositionalEncodingXScale(unittest.TestCase):
    """Test xscale functionality."""

    def test_no_xscale(self):
        """Test that xscale=None doesn't modify input magnitude."""
        pos_enc = RelPositionalEncoding(d_model=64, xscale=None, dropout_rate=0.0)

        x = torch.ones(1, 10, 64)
        x_out, _ = pos_enc(x)

        # Without scaling, output should equal input
        torch.testing.assert_close(x_out, x)

    def test_with_xscale(self):
        """Test that xscale multiplies input."""
        xscale = math.sqrt(64)
        pos_enc = RelPositionalEncoding(d_model=64, xscale=xscale, dropout_rate=0.0)

        x = torch.ones(1, 10, 64)
        x_out, _ = pos_enc(x)

        expected = x * xscale
        torch.testing.assert_close(x_out, expected)


class TestRelPositionalEncodingDropout(unittest.TestCase):
    """Test dropout behavior."""

    def test_dropout_in_training(self):
        """Test that dropout is applied during training."""
        pos_enc = RelPositionalEncoding(d_model=64, dropout_rate=0.5)
        pos_enc.train()

        torch.manual_seed(42)
        x = torch.ones(1, 100, 64)
        x_out, _ = pos_enc(x)

        # With 50% dropout, some values should be zeroed
        zero_count = (x_out == 0).sum().item()
        self.assertGreater(zero_count, 0, "Dropout should zero some values")

    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied during eval."""
        pos_enc = RelPositionalEncoding(d_model=64, dropout_rate=0.5)
        pos_enc.eval()

        x = torch.ones(1, 100, 64)
        x_out, _ = pos_enc(x)

        # In eval mode, no values should be zeroed
        zero_count = (x_out == 0).sum().item()
        self.assertEqual(zero_count, 0, "No dropout in eval mode")


class TestRelPositionalEncodingDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves buffers correctly."""
        pos_enc = RelPositionalEncoding(d_model=64)

        # Check PE buffer is on CPU by default
        self.assertEqual(pos_enc.pe.device.type, "cpu")

        # Skip CUDA test if not available
        if torch.cuda.is_available():
            pos_enc = pos_enc.to("cuda")
            self.assertEqual(pos_enc.pe.device.type, "cuda")

            # Test forward pass on CUDA
            x = torch.randn(1, 10, 64, device="cuda")
            x_out, pos_emb = pos_enc(x)
            self.assertEqual(x_out.device.type, "cuda")
            self.assertEqual(pos_emb.device.type, "cuda")


class TestRelPositionalEncodingExtension(unittest.TestCase):
    """Test PE buffer extension for long sequences."""

    def test_extend_for_long_sequence(self):
        """Test that PE is extended for sequences longer than initial max_len."""
        pos_enc = RelPositionalEncoding(d_model=64, max_len=100)

        initial_pe_size = pos_enc.pe.size(1)
        self.assertEqual(initial_pe_size, 2 * 100 - 1)  # 199

        # Process a longer sequence
        x = torch.randn(1, 200, 64)
        x_out, pos_emb = pos_enc(x)

        # PE should have been extended
        new_pe_size = pos_enc.pe.size(1)
        self.assertGreaterEqual(new_pe_size, 2 * 200 - 1)  # At least 399


if __name__ == "__main__":
    unittest.main()
