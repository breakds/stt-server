"""Tests for ConvSubsampling module."""

import unittest

import torch

from nemo_lite.conformer_lite.subsampling import ConvSubsampling


class TestConvSubsamplingShape(unittest.TestCase):
    """Test output shapes of ConvSubsampling."""

    def test_output_shape_basic(self):
        """Test basic output shape with default parameters."""
        subsampling = ConvSubsampling()

        batch_size = 2
        time_steps = 800  # 8 seconds at 100 frames/sec
        feat_in = 128

        x = torch.randn(batch_size, time_steps, feat_in)
        lengths = torch.tensor([800, 600])

        out, out_lengths = subsampling(x, lengths)

        # Time should be reduced by 8x
        expected_time = 100  # 800 / 8
        self.assertEqual(out.shape, (batch_size, expected_time, 1024))

        # Check output lengths
        # For length 800: (800-1)//2 + 1 = 400 -> (400-1)//2 + 1 = 200 -> (200-1)//2 + 1 = 100
        # For length 600: (600-1)//2 + 1 = 300 -> (300-1)//2 + 1 = 150 -> (150-1)//2 + 1 = 75
        self.assertEqual(out_lengths[0].item(), 100)
        self.assertEqual(out_lengths[1].item(), 75)

    def test_output_shape_various_lengths(self):
        """Test output shape with various input lengths."""
        subsampling = ConvSubsampling()

        test_cases = [
            (100, 13),   # 100 -> 50 -> 25 -> 13
            (160, 20),   # 160 -> 80 -> 40 -> 20
            (200, 25),   # 200 -> 100 -> 50 -> 25
            (1000, 125), # 1000 -> 500 -> 250 -> 125
        ]

        for input_len, expected_out_len in test_cases:
            with self.subTest(input_len=input_len):
                x = torch.randn(1, input_len, 128)
                lengths = torch.tensor([input_len])

                out, out_lengths = subsampling(x, lengths)

                self.assertEqual(out.shape[1], expected_out_len)
                self.assertEqual(out_lengths[0].item(), expected_out_len)

    def test_frequency_dimension_calculation(self):
        """Test that frequency dimension is correctly reduced."""
        subsampling = ConvSubsampling(feat_in=128, d_model=1024, conv_channels=256)

        # After 3x stride-2 on freq=128: 128 -> 64 -> 32 -> 16
        # Linear input size should be 256 * 16 = 4096
        self.assertEqual(subsampling.out.in_features, 4096)
        self.assertEqual(subsampling.out.out_features, 1024)


class TestConvSubsamplingGradient(unittest.TestCase):
    """Test gradient flow through ConvSubsampling."""

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        subsampling = ConvSubsampling()

        x = torch.randn(2, 100, 128, requires_grad=True)
        lengths = torch.tensor([100, 80])

        out, _ = subsampling(x, lengths)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestConvSubsamplingDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters."""
        subsampling = ConvSubsampling()

        # Check all parameters are on CPU by default
        for param in subsampling.parameters():
            self.assertEqual(param.device.type, "cpu")

        # Skip CUDA test if not available
        if torch.cuda.is_available():
            subsampling = subsampling.to("cuda")
            for param in subsampling.parameters():
                self.assertEqual(param.device.type, "cuda")


class TestConvSubsamplingParameters(unittest.TestCase):
    """Test parameter counts and layer structure."""

    def test_layer_structure(self):
        """Test that all expected layers exist."""
        subsampling = ConvSubsampling()

        # Check layer existence
        self.assertIsInstance(subsampling.conv1, torch.nn.Conv2d)
        self.assertIsInstance(subsampling.dwconv2, torch.nn.Conv2d)
        self.assertIsInstance(subsampling.pwconv2, torch.nn.Conv2d)
        self.assertIsInstance(subsampling.dwconv3, torch.nn.Conv2d)
        self.assertIsInstance(subsampling.pwconv3, torch.nn.Conv2d)
        self.assertIsInstance(subsampling.out, torch.nn.Linear)

    def test_depthwise_conv_groups(self):
        """Test that depthwise convs have correct groups setting."""
        subsampling = ConvSubsampling(conv_channels=256)

        self.assertEqual(subsampling.dwconv2.groups, 256)
        self.assertEqual(subsampling.dwconv3.groups, 256)

    def test_conv1_channels(self):
        """Test first conv layer has correct channel configuration."""
        subsampling = ConvSubsampling(conv_channels=256)

        self.assertEqual(subsampling.conv1.in_channels, 1)
        self.assertEqual(subsampling.conv1.out_channels, 256)


if __name__ == "__main__":
    unittest.main()
