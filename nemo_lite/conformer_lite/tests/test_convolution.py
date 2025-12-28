"""Tests for ConvolutionModule."""

import unittest

import torch

from nemo_lite.conformer_lite.convolution import ConvolutionModule


class TestConvolutionModuleShape(unittest.TestCase):
    """Test output shapes of ConvolutionModule."""

    def test_output_shape_basic(self):
        """Test basic output shape."""
        conv = ConvolutionModule(d_model=1024, kernel_size=9)

        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 1024)

        out = conv(x)

        self.assertEqual(out.shape, (batch_size, seq_len, 1024))

    def test_output_shape_various_lengths(self):
        """Test output shape with various sequence lengths."""
        conv = ConvolutionModule(d_model=256, kernel_size=9)

        for seq_len in [10, 50, 100, 500]:
            with self.subTest(seq_len=seq_len):
                x = torch.randn(1, seq_len, 256)
                out = conv(x)
                self.assertEqual(out.shape, (1, seq_len, 256))

    def test_different_kernel_sizes(self):
        """Test with various kernel sizes."""
        for kernel_size in [3, 9, 15, 31]:
            with self.subTest(kernel_size=kernel_size):
                conv = ConvolutionModule(d_model=64, kernel_size=kernel_size)
                x = torch.randn(2, 50, 64)
                out = conv(x)
                # Output should have same shape as input
                self.assertEqual(out.shape, x.shape)


class TestConvolutionModuleLayers(unittest.TestCase):
    """Test layer structure of ConvolutionModule."""

    def test_pointwise_conv1_expansion(self):
        """Test that pointwise_conv1 expands to 2x channels for GLU."""
        d_model = 256
        conv = ConvolutionModule(d_model=d_model, kernel_size=9)

        self.assertEqual(conv.pointwise_conv1.in_channels, d_model)
        self.assertEqual(conv.pointwise_conv1.out_channels, d_model * 2)
        self.assertEqual(conv.pointwise_conv1.kernel_size, (1,))

    def test_depthwise_conv_groups(self):
        """Test that depthwise conv uses groups=d_model."""
        d_model = 256
        conv = ConvolutionModule(d_model=d_model, kernel_size=9)

        self.assertEqual(conv.depthwise_conv.in_channels, d_model)
        self.assertEqual(conv.depthwise_conv.out_channels, d_model)
        self.assertEqual(conv.depthwise_conv.groups, d_model)

    def test_depthwise_conv_padding(self):
        """Test that depthwise conv has correct symmetric padding."""
        kernel_size = 9
        expected_padding = (kernel_size - 1) // 2  # 4

        conv = ConvolutionModule(d_model=256, kernel_size=kernel_size)

        self.assertEqual(conv.depthwise_conv.padding, (expected_padding,))

    def test_pointwise_conv2_projection(self):
        """Test that pointwise_conv2 maintains d_model dimension."""
        d_model = 256
        conv = ConvolutionModule(d_model=d_model, kernel_size=9)

        self.assertEqual(conv.pointwise_conv2.in_channels, d_model)
        self.assertEqual(conv.pointwise_conv2.out_channels, d_model)
        self.assertEqual(conv.pointwise_conv2.kernel_size, (1,))

    def test_batch_norm_not_layer_norm(self):
        """Test that BatchNorm1d is used (not LayerNorm)."""
        conv = ConvolutionModule(d_model=256, kernel_size=9)

        self.assertIsInstance(conv.batch_norm, torch.nn.BatchNorm1d)
        self.assertEqual(conv.batch_norm.num_features, 256)

    def test_activation_is_silu(self):
        """Test that SiLU (Swish) activation is used."""
        conv = ConvolutionModule(d_model=256, kernel_size=9)

        self.assertIsInstance(conv.activation, torch.nn.SiLU)

    def test_all_layers_have_bias(self):
        """Test that all conv layers have bias (matching NeMo default)."""
        conv = ConvolutionModule(d_model=256, kernel_size=9)

        self.assertIsNotNone(conv.pointwise_conv1.bias)
        self.assertIsNotNone(conv.depthwise_conv.bias)
        self.assertIsNotNone(conv.pointwise_conv2.bias)


class TestConvolutionModuleGLU(unittest.TestCase):
    """Test GLU gating behavior."""

    def test_glu_halves_channels(self):
        """Test that GLU reduces channels by half."""
        d_model = 64
        conv = ConvolutionModule(d_model=d_model, kernel_size=9)

        x = torch.randn(1, 10, d_model)
        x_transposed = x.transpose(1, 2)  # (1, d_model, 10)

        # After pointwise_conv1, we have d_model*2 channels
        after_pw1 = conv.pointwise_conv1(x_transposed)
        self.assertEqual(after_pw1.shape[1], d_model * 2)

        # After GLU, back to d_model channels
        after_glu = torch.nn.functional.glu(after_pw1, dim=1)
        self.assertEqual(after_glu.shape[1], d_model)


class TestConvolutionModuleMasking(unittest.TestCase):
    """Test padding mask behavior."""

    def test_mask_shape(self):
        """Test that masking works with correct shapes."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)
        conv.eval()

        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 64)

        # Create padding mask (last 5 positions are padded)
        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        pad_mask[:, -5:] = True

        out = conv(x, pad_mask=pad_mask)

        self.assertEqual(out.shape, (batch_size, seq_len, 64))


class TestConvolutionModuleTrainEval(unittest.TestCase):
    """Test train/eval mode behavior (especially BatchNorm)."""

    def test_batchnorm_train_mode(self):
        """Test that BatchNorm updates stats in train mode."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)
        conv.train()

        # Initial running mean should be zeros
        initial_mean = conv.batch_norm.running_mean.clone()

        x = torch.randn(4, 20, 64) + 5.0  # Shifted input
        _ = conv(x)

        # Running mean should have changed
        self.assertFalse(torch.allclose(conv.batch_norm.running_mean, initial_mean))

    def test_batchnorm_eval_mode(self):
        """Test that BatchNorm uses fixed stats in eval mode."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)
        conv.eval()

        initial_mean = conv.batch_norm.running_mean.clone()

        x = torch.randn(4, 20, 64) + 5.0
        _ = conv(x)

        # Running mean should NOT change in eval mode
        torch.testing.assert_close(conv.batch_norm.running_mean, initial_mean)

    def test_deterministic_in_eval(self):
        """Test that output is deterministic in eval mode."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)
        conv.eval()

        x = torch.randn(2, 20, 64)

        out1 = conv(x)
        out2 = conv(x)

        torch.testing.assert_close(out1, out2)


class TestConvolutionModuleDevice(unittest.TestCase):
    """Test device handling."""

    def test_to_device(self):
        """Test that .to() moves all parameters correctly."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)

        for param in conv.parameters():
            self.assertEqual(param.device.type, "cpu")

        if torch.cuda.is_available():
            conv = conv.to("cuda")
            for param in conv.parameters():
                self.assertEqual(param.device.type, "cuda")

            # BatchNorm buffers should also move
            self.assertEqual(conv.batch_norm.running_mean.device.type, "cuda")
            self.assertEqual(conv.batch_norm.running_var.device.type, "cuda")

            # Test forward pass on CUDA
            x = torch.randn(1, 20, 64, device="cuda")
            out = conv(x)
            self.assertEqual(out.device.type, "cuda")


class TestConvolutionModuleGradient(unittest.TestCase):
    """Test gradient flow."""

    def test_gradient_flow(self):
        """Test that gradients flow through all layers."""
        conv = ConvolutionModule(d_model=64, kernel_size=9)
        conv.train()

        x = torch.randn(2, 20, 64, requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in conv.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(
                torch.all(param.grad == 0), f"Zero gradient for {name}"
            )


if __name__ == "__main__":
    unittest.main()
