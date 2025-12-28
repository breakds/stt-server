"""Tests for ConformerBlock."""

import unittest

import torch

from nemo_lite.conformer_lite.conformer_block import ConformerBlock


class TestConformerBlockShape(unittest.TestCase):
    """Test output shapes of ConformerBlock."""

    def test_output_shape_basic(self):
        """Test basic output shape."""
        block = ConformerBlock(d_model=1024, d_ff=4096, n_heads=8)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 1024)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 1024)

        out = block(x, pos_emb)

        self.assertEqual(out.shape, (batch_size, seq_len, 1024))

    def test_output_shape_various_lengths(self):
        """Test output shape with various sequence lengths."""
        block = ConformerBlock(d_model=256, d_ff=1024, n_heads=4)

        for seq_len in [10, 50, 100]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 256)
                pos_emb = torch.randn(1, 2 * seq_len - 1, 256)
                out = block(x, pos_emb)
                self.assertEqual(out.shape, (1, seq_len, 256))


class TestConformerBlockStructure(unittest.TestCase):
    """Test structure of ConformerBlock."""

    def test_has_two_ffn_modules(self):
        """Test that block has two FFN modules."""
        block = ConformerBlock(d_model=256, d_ff=1024, n_heads=4)

        self.assertTrue(hasattr(block, "feed_forward1"))
        self.assertTrue(hasattr(block, "feed_forward2"))

    def test_has_layer_norms(self):
        """Test that block has all required LayerNorms."""
        block = ConformerBlock(d_model=256, d_ff=1024, n_heads=4)

        self.assertIsInstance(block.norm_feed_forward1, torch.nn.LayerNorm)
        self.assertIsInstance(block.norm_self_att, torch.nn.LayerNorm)
        self.assertIsInstance(block.norm_conv, torch.nn.LayerNorm)
        self.assertIsInstance(block.norm_feed_forward2, torch.nn.LayerNorm)
        self.assertIsInstance(block.norm_out, torch.nn.LayerNorm)

    def test_fc_factor(self):
        """Test that FFN residual factor is 0.5."""
        block = ConformerBlock(d_model=256, d_ff=1024, n_heads=4)

        self.assertEqual(block.fc_factor, 0.5)


class TestConformerBlockMasking(unittest.TestCase):
    """Test masking behavior."""

    def test_with_attention_mask(self):
        """Test that attention mask works."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4)
        block.eval()

        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 64)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 64)

        # Causal mask
        att_mask = torch.triu(
            torch.ones(batch_size, seq_len, seq_len), diagonal=1
        ).bool()

        out = block(x, pos_emb, att_mask=att_mask)

        self.assertEqual(out.shape, (batch_size, seq_len, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_with_pad_mask(self):
        """Test that padding mask works."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4)
        block.eval()

        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 64)
        pos_emb = torch.randn(1, 2 * seq_len - 1, 64)

        # Last 5 positions are padded
        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        pad_mask[:, -5:] = True

        out = block(x, pos_emb, pad_mask=pad_mask)

        self.assertEqual(out.shape, (batch_size, seq_len, 64))


class TestConformerBlockDropout(unittest.TestCase):
    """Test dropout behavior."""

    def test_different_outputs_in_training(self):
        """Test that outputs differ in training mode due to dropout."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4, dropout_rate=0.5)
        block.train()

        torch.manual_seed(42)
        x = torch.randn(2, 20, 64)
        pos_emb = torch.randn(1, 39, 64)

        out1 = block(x, pos_emb)
        out2 = block(x, pos_emb)

        self.assertFalse(torch.allclose(out1, out2))

    def test_deterministic_in_eval(self):
        """Test that output is deterministic in eval mode."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4, dropout_rate=0.5)
        block.eval()

        x = torch.randn(2, 20, 64)
        pos_emb = torch.randn(1, 39, 64)

        out1 = block(x, pos_emb)
        out2 = block(x, pos_emb)

        torch.testing.assert_close(out1, out2)


class TestConformerBlockDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters correctly."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4)

        for param in block.parameters():
            self.assertEqual(param.device.type, "cpu")

        if torch.cuda.is_available():
            block = block.to("cuda")
            for param in block.parameters():
                self.assertEqual(param.device.type, "cuda")

            x = torch.randn(1, 20, 64, device="cuda")
            pos_emb = torch.randn(1, 39, 64, device="cuda")
            out = block(x, pos_emb)
            self.assertEqual(out.device.type, "cuda")


class TestConformerBlockGradient(unittest.TestCase):
    """Test gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through all sub-modules."""
        block = ConformerBlock(d_model=64, d_ff=256, n_heads=4)
        block.train()

        x = torch.randn(2, 20, 64, requires_grad=True)
        pos_emb = torch.randn(1, 39, 64)

        out = block(x, pos_emb)
        loss = out.sum()
        loss.backward()

        # Check key parameters have gradients
        key_params = [
            "norm_feed_forward1.weight",
            "feed_forward1.linear1.weight",
            "norm_self_att.weight",
            "self_attn.linear_q.weight",
            "norm_conv.weight",
            "conv_module.pointwise_conv1.weight",
            "norm_feed_forward2.weight",
            "feed_forward2.linear1.weight",
            "norm_out.weight",
        ]

        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")


if __name__ == "__main__":
    unittest.main()
