"""Tests for FeedForwardModule."""

import unittest

import torch

from nemo_lite.conformer_lite.feed_forward import FeedForwardModule


class TestFeedForwardModuleShape(unittest.TestCase):
    """Test output shapes of FeedForwardModule."""

    def test_output_shape_basic(self):
        """Test basic output shape."""
        ffn = FeedForwardModule(d_model=1024, d_ff=4096)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 1024)

        out = ffn(x)

        self.assertEqual(out.shape, (batch_size, seq_len, 1024))

    def test_output_shape_various_lengths(self):
        """Test output shape with various sequence lengths."""
        ffn = FeedForwardModule(d_model=256, d_ff=1024)

        for seq_len in [10, 50, 100, 500]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 256)
                out = ffn(x)
                self.assertEqual(out.shape, (1, seq_len, 256))


class TestFeedForwardModuleLayers(unittest.TestCase):
    """Test layer structure of FeedForwardModule."""

    def test_linear1_expansion(self):
        """Test that linear1 expands to d_ff."""
        d_model = 256
        d_ff = 1024
        ffn = FeedForwardModule(d_model=d_model, d_ff=d_ff)

        self.assertEqual(ffn.linear1.in_features, d_model)
        self.assertEqual(ffn.linear1.out_features, d_ff)

    def test_linear2_projection(self):
        """Test that linear2 projects back to d_model."""
        d_model = 256
        d_ff = 1024
        ffn = FeedForwardModule(d_model=d_model, d_ff=d_ff)

        self.assertEqual(ffn.linear2.in_features, d_ff)
        self.assertEqual(ffn.linear2.out_features, d_model)

    def test_activation_is_silu(self):
        """Test that SiLU (Swish) activation is used."""
        ffn = FeedForwardModule(d_model=256, d_ff=1024)

        self.assertIsInstance(ffn.activation, torch.nn.SiLU)

    def test_all_layers_have_bias(self):
        """Test that all linear layers have bias."""
        ffn = FeedForwardModule(d_model=256, d_ff=1024)

        self.assertIsNotNone(ffn.linear1.bias)
        self.assertIsNotNone(ffn.linear2.bias)


class TestFeedForwardModuleDropout(unittest.TestCase):
    """Test dropout behavior."""

    def test_dropout_in_training(self):
        """Test that dropout is applied during training."""
        ffn = FeedForwardModule(d_model=64, d_ff=256, dropout_rate=0.5)
        ffn.train()

        torch.manual_seed(42)
        x = torch.randn(2, 20, 64)

        out1 = ffn(x)
        out2 = ffn(x)

        # Outputs should differ due to dropout
        self.assertFalse(torch.allclose(out1, out2))

    def test_no_dropout_in_eval(self):
        """Test that dropout is not applied during eval."""
        ffn = FeedForwardModule(d_model=64, d_ff=256, dropout_rate=0.5)
        ffn.eval()

        x = torch.randn(2, 20, 64)

        out1 = ffn(x)
        out2 = ffn(x)

        torch.testing.assert_close(out1, out2)


class TestFeedForwardModuleDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters correctly."""
        ffn = FeedForwardModule(d_model=64, d_ff=256)

        for param in ffn.parameters():
            self.assertEqual(param.device.type, "cpu")

        if torch.cuda.is_available():
            ffn = ffn.to("cuda")
            for param in ffn.parameters():
                self.assertEqual(param.device.type, "cuda")

            x = torch.randn(1, 20, 64, device="cuda")
            out = ffn(x)
            self.assertEqual(out.device.type, "cuda")


class TestFeedForwardModuleGradient(unittest.TestCase):
    """Test gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through all layers."""
        ffn = FeedForwardModule(d_model=64, d_ff=256)
        ffn.train()

        x = torch.randn(2, 20, 64, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()

        for name, param in ffn.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")


if __name__ == "__main__":
    unittest.main()
