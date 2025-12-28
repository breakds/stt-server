"""Tests for AudioProjection layer."""

import unittest

import torch

from nemo_lite.projection import AudioProjection
from nemo_lite.weights import map_projection_weight_key


class TestAudioProjectionShape(unittest.TestCase):
    """Test output shapes of AudioProjection."""

    def test_output_shape_default(self):
        """Test output shape with default dimensions."""
        projection = AudioProjection()

        x = torch.randn(2, 100, 1024)
        out = projection(x)

        self.assertEqual(out.shape, (2, 100, 2048))

    def test_output_shape_custom(self):
        """Test output shape with custom dimensions."""
        projection = AudioProjection(encoder_dim=512, llm_dim=1024)

        x = torch.randn(1, 50, 512)
        out = projection(x)

        self.assertEqual(out.shape, (1, 50, 1024))


class TestAudioProjectionStructure(unittest.TestCase):
    """Test structure of AudioProjection."""

    def test_has_proj_layer(self):
        """Test that projection has linear layer."""
        projection = AudioProjection()

        self.assertTrue(hasattr(projection, "proj"))
        self.assertIsInstance(projection.proj, torch.nn.Linear)

    def test_dimensions_stored(self):
        """Test that dimensions are stored correctly."""
        projection = AudioProjection(encoder_dim=512, llm_dim=1024)

        self.assertEqual(projection.encoder_dim, 512)
        self.assertEqual(projection.llm_dim, 1024)


class TestProjectionWeightMapping(unittest.TestCase):
    """Test weight key mapping for projection layer."""

    def test_weight_mapping(self):
        """Test mapping of weight key."""
        ckpt_key = "perception.proj.weight"
        expected = "proj.weight"
        self.assertEqual(map_projection_weight_key(ckpt_key), expected)

    def test_bias_mapping(self):
        """Test mapping of bias key."""
        ckpt_key = "perception.proj.bias"
        expected = "proj.bias"
        self.assertEqual(map_projection_weight_key(ckpt_key), expected)

    def test_non_projection_keys_return_none(self):
        """Test that non-projection keys return None."""
        self.assertIsNone(map_projection_weight_key("perception.encoder.layers.0.weight"))
        self.assertIsNone(map_projection_weight_key("llm.model.weight"))
        self.assertIsNone(map_projection_weight_key("random.key"))


if __name__ == "__main__":
    unittest.main()
