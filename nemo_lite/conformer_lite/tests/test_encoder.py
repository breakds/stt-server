"""Tests for FastConformerEncoder."""

import unittest

import torch

from nemo_lite.conformer_lite.encoder import FastConformerEncoder


class TestFastConformerEncoderShape(unittest.TestCase):
    """Test output shapes of FastConformerEncoder."""

    def test_output_shape_basic(self):
        """Test basic output shape with small model."""
        encoder = FastConformerEncoder(
            feat_in=128,
            n_layers=2,  # Small for testing
            d_model=256,
            d_ff=1024,
            n_heads=4,
        )

        batch_size = 2
        time_steps = 800  # 800 frames -> 100 after 8x subsampling
        audio = torch.randn(batch_size, 128, time_steps)
        lengths = torch.tensor([800, 600])

        encoded, out_lengths = encoder(audio, lengths)

        # After 8x subsampling: 800 -> 100, 600 -> 75
        self.assertEqual(encoded.shape[0], batch_size)
        self.assertEqual(encoded.shape[2], 256)  # d_model
        self.assertEqual(encoded.shape[1], 100)  # 800 / 8

    def test_output_lengths(self):
        """Test that output lengths are correctly computed."""
        encoder = FastConformerEncoder(
            feat_in=128,
            n_layers=2,
            d_model=256,
            d_ff=1024,
            n_heads=4,
        )

        audio = torch.randn(2, 128, 800)
        lengths = torch.tensor([800, 400])

        _, out_lengths = encoder(audio, lengths)

        # Subsampling formula: ((x - 1) // 2 + 1) applied 3 times
        # 800 -> 400 -> 200 -> 100
        # 400 -> 200 -> 100 -> 50
        self.assertEqual(out_lengths[0].item(), 100)
        self.assertEqual(out_lengths[1].item(), 50)

    def test_various_input_lengths(self):
        """Test with various input lengths."""
        encoder = FastConformerEncoder(
            feat_in=128,
            n_layers=2,
            d_model=128,
            d_ff=512,
            n_heads=4,
        )

        for time_steps in [160, 320, 800]:
            with self.subTest(time_steps=time_steps):
                audio = torch.randn(1, 128, time_steps)
                lengths = torch.tensor([time_steps])

                encoded, out_lengths = encoder(audio, lengths)

                self.assertEqual(encoded.shape[0], 1)
                self.assertEqual(encoded.shape[2], 128)


class TestFastConformerEncoderStructure(unittest.TestCase):
    """Test structure of FastConformerEncoder."""

    def test_has_subsampling(self):
        """Test that encoder has subsampling module."""
        encoder = FastConformerEncoder(n_layers=2, d_model=256)

        self.assertTrue(hasattr(encoder, "pre_encode"))

    def test_has_pos_encoding(self):
        """Test that encoder has positional encoding."""
        encoder = FastConformerEncoder(n_layers=2, d_model=256)

        self.assertTrue(hasattr(encoder, "pos_enc"))

    def test_has_correct_num_layers(self):
        """Test that encoder has correct number of layers."""
        for n_layers in [2, 8, 16]:
            with self.subTest(n_layers=n_layers):
                encoder = FastConformerEncoder(n_layers=n_layers, d_model=256)
                self.assertEqual(len(encoder.layers), n_layers)

    def test_canary_config(self):
        """Test with Canary-like configuration."""
        encoder = FastConformerEncoder(
            feat_in=128,
            n_layers=32,
            d_model=1024,
            d_ff=4096,
            n_heads=8,
            conv_kernel_size=9,
        )

        self.assertEqual(encoder.n_layers, 32)
        self.assertEqual(encoder.d_model, 1024)
        self.assertEqual(len(encoder.layers), 32)


class TestFastConformerEncoderDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters correctly."""
        encoder = FastConformerEncoder(n_layers=2, d_model=128)

        for param in encoder.parameters():
            self.assertEqual(param.device.type, "cpu")

        if torch.cuda.is_available():
            encoder = encoder.to("cuda")
            for param in encoder.parameters():
                self.assertEqual(param.device.type, "cuda")

            audio = torch.randn(1, 128, 160, device="cuda")
            lengths = torch.tensor([160], device="cuda")

            encoded, out_lengths = encoder(audio, lengths)

            self.assertEqual(encoded.device.type, "cuda")


class TestFastConformerEncoderTrainEval(unittest.TestCase):
    """Test train/eval mode behavior."""

    def test_deterministic_in_eval(self):
        """Test that output is deterministic in eval mode."""
        encoder = FastConformerEncoder(n_layers=2, d_model=128, dropout_rate=0.5)
        encoder.eval()

        audio = torch.randn(1, 128, 160)
        lengths = torch.tensor([160])

        out1, _ = encoder(audio, lengths)
        out2, _ = encoder(audio, lengths)

        torch.testing.assert_close(out1, out2)

    def test_different_outputs_in_training(self):
        """Test that outputs differ in training due to dropout."""
        encoder = FastConformerEncoder(n_layers=2, d_model=128, dropout_rate=0.5)
        encoder.train()

        torch.manual_seed(42)
        audio = torch.randn(1, 128, 160)
        lengths = torch.tensor([160])

        out1, _ = encoder(audio, lengths)
        out2, _ = encoder(audio, lengths)

        self.assertFalse(torch.allclose(out1, out2))


class TestFastConformerEncoderGradient(unittest.TestCase):
    """Test gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through entire encoder."""
        encoder = FastConformerEncoder(n_layers=2, d_model=128)
        encoder.train()

        audio = torch.randn(1, 128, 160, requires_grad=True)
        lengths = torch.tensor([160])

        encoded, _ = encoder(audio, lengths)
        loss = encoded.sum()
        loss.backward()

        # Check that input has gradient
        self.assertIsNotNone(audio.grad)

        # Check key parameters have gradients
        self.assertIsNotNone(encoder.pre_encode.conv1.weight.grad)
        self.assertIsNotNone(encoder.layers[0].feed_forward1.linear1.weight.grad)
        self.assertIsNotNone(encoder.layers[-1].norm_out.weight.grad)


class TestFastConformerEncoderPadMask(unittest.TestCase):
    """Test padding mask creation."""

    def test_pad_mask_creation(self):
        """Test that padding mask is created correctly."""
        encoder = FastConformerEncoder(n_layers=2, d_model=128)

        lengths = torch.tensor([10, 5, 8])
        max_len = 10

        pad_mask = encoder._create_pad_mask(lengths, max_len)

        # Check shape
        self.assertEqual(pad_mask.shape, (3, 10))

        # Check values
        # Sample 0: length 10, no padding
        self.assertFalse(pad_mask[0].any())

        # Sample 1: length 5, positions 5-9 are padded
        self.assertFalse(pad_mask[1, :5].any())
        self.assertTrue(pad_mask[1, 5:].all())

        # Sample 2: length 8, positions 8-9 are padded
        self.assertFalse(pad_mask[2, :8].any())
        self.assertTrue(pad_mask[2, 8:].all())


if __name__ == "__main__":
    unittest.main()
