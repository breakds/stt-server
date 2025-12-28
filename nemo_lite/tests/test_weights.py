"""Tests for weight loading utilities."""

import unittest

from nemo_lite.weights import map_weight_key


class TestWeightKeyMapping(unittest.TestCase):
    """Test weight key mapping from checkpoint format to our format."""

    def test_non_encoder_keys_return_none(self):
        """Test that non-encoder keys return None."""
        # LLM weights
        self.assertIsNone(map_weight_key("llm.base_model.model.layers.0.self_attn.q_proj.weight"))
        # Projection weights (not encoder)
        self.assertIsNone(map_weight_key("perception.proj.weight"))
        # Preprocessor weights
        self.assertIsNone(map_weight_key("perception.preprocessor.featurizer.fb"))
        # Random keys
        self.assertIsNone(map_weight_key("some.random.key"))

    def test_subsampling_conv1_mapping(self):
        """Test mapping for first conv layer (conv.0 -> conv1)."""
        ckpt_key = "perception.encoder.pre_encode.conv.0.weight"
        expected = "pre_encode.conv1.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

        ckpt_key = "perception.encoder.pre_encode.conv.0.bias"
        expected = "pre_encode.conv1.bias"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_subsampling_dwconv2_mapping(self):
        """Test mapping for depthwise conv layer 2 (conv.2 -> dwconv2)."""
        ckpt_key = "perception.encoder.pre_encode.conv.2.weight"
        expected = "pre_encode.dwconv2.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

        ckpt_key = "perception.encoder.pre_encode.conv.2.bias"
        expected = "pre_encode.dwconv2.bias"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_subsampling_pwconv2_mapping(self):
        """Test mapping for pointwise conv layer 2 (conv.3 -> pwconv2)."""
        ckpt_key = "perception.encoder.pre_encode.conv.3.weight"
        expected = "pre_encode.pwconv2.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_subsampling_dwconv3_mapping(self):
        """Test mapping for depthwise conv layer 3 (conv.5 -> dwconv3)."""
        ckpt_key = "perception.encoder.pre_encode.conv.5.weight"
        expected = "pre_encode.dwconv3.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_subsampling_pwconv3_mapping(self):
        """Test mapping for pointwise conv layer 3 (conv.6 -> pwconv3)."""
        ckpt_key = "perception.encoder.pre_encode.conv.6.weight"
        expected = "pre_encode.pwconv3.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_subsampling_out_mapping(self):
        """Test mapping for output linear layer (out stays the same)."""
        ckpt_key = "perception.encoder.pre_encode.out.weight"
        expected = "pre_encode.out.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

        ckpt_key = "perception.encoder.pre_encode.out.bias"
        expected = "pre_encode.out.bias"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_attention_linear_q_mapping(self):
        """Test mapping for attention query projection."""
        ckpt_key = "perception.encoder.layers.0.self_attn.linear_q.weight"
        expected = "layers.0.self_attn.linear_q.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

        ckpt_key = "perception.encoder.layers.0.self_attn.linear_q.bias"
        expected = "layers.0.self_attn.linear_q.bias"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_attention_linear_pos_mapping(self):
        """Test mapping for position projection (no bias)."""
        ckpt_key = "perception.encoder.layers.0.self_attn.linear_pos.weight"
        expected = "layers.0.self_attn.linear_pos.weight"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_attention_pos_bias_mapping(self):
        """Test mapping for position biases."""
        ckpt_key = "perception.encoder.layers.0.self_attn.pos_bias_u"
        expected = "layers.0.self_attn.pos_bias_u"
        self.assertEqual(map_weight_key(ckpt_key), expected)

        ckpt_key = "perception.encoder.layers.0.self_attn.pos_bias_v"
        expected = "layers.0.self_attn.pos_bias_v"
        self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_conv_module_mapping(self):
        """Test mapping for convolution module weights (conv -> conv_module)."""
        test_cases = [
            ("perception.encoder.layers.0.conv.pointwise_conv1.weight",
             "layers.0.conv_module.pointwise_conv1.weight"),
            ("perception.encoder.layers.0.conv.pointwise_conv1.bias",
             "layers.0.conv_module.pointwise_conv1.bias"),
            ("perception.encoder.layers.0.conv.depthwise_conv.weight",
             "layers.0.conv_module.depthwise_conv.weight"),
            ("perception.encoder.layers.0.conv.depthwise_conv.bias",
             "layers.0.conv_module.depthwise_conv.bias"),
            ("perception.encoder.layers.0.conv.pointwise_conv2.weight",
             "layers.0.conv_module.pointwise_conv2.weight"),
            ("perception.encoder.layers.0.conv.pointwise_conv2.bias",
             "layers.0.conv_module.pointwise_conv2.bias"),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_batch_norm_mapping(self):
        """Test mapping for BatchNorm weights and buffers."""
        test_cases = [
            ("perception.encoder.layers.0.conv.batch_norm.weight",
             "layers.0.conv_module.batch_norm.weight"),
            ("perception.encoder.layers.0.conv.batch_norm.bias",
             "layers.0.conv_module.batch_norm.bias"),
            ("perception.encoder.layers.0.conv.batch_norm.running_mean",
             "layers.0.conv_module.batch_norm.running_mean"),
            ("perception.encoder.layers.0.conv.batch_norm.running_var",
             "layers.0.conv_module.batch_norm.running_var"),
            ("perception.encoder.layers.0.conv.batch_norm.num_batches_tracked",
             "layers.0.conv_module.batch_norm.num_batches_tracked"),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_feed_forward_mapping(self):
        """Test mapping for feed-forward module weights."""
        test_cases = [
            ("perception.encoder.layers.0.feed_forward1.linear1.weight",
             "layers.0.feed_forward1.linear1.weight"),
            ("perception.encoder.layers.0.feed_forward1.linear1.bias",
             "layers.0.feed_forward1.linear1.bias"),
            ("perception.encoder.layers.0.feed_forward1.linear2.weight",
             "layers.0.feed_forward1.linear2.weight"),
            ("perception.encoder.layers.0.feed_forward1.linear2.bias",
             "layers.0.feed_forward1.linear2.bias"),
            ("perception.encoder.layers.0.feed_forward2.linear1.weight",
             "layers.0.feed_forward2.linear1.weight"),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_layer_norm_mapping(self):
        """Test mapping for LayerNorm weights."""
        test_cases = [
            ("perception.encoder.layers.0.norm_feed_forward1.weight",
             "layers.0.norm_feed_forward1.weight"),
            ("perception.encoder.layers.0.norm_feed_forward1.bias",
             "layers.0.norm_feed_forward1.bias"),
            ("perception.encoder.layers.0.norm_self_att.weight",
             "layers.0.norm_self_att.weight"),
            ("perception.encoder.layers.0.norm_conv.weight",
             "layers.0.norm_conv.weight"),
            ("perception.encoder.layers.0.norm_feed_forward2.weight",
             "layers.0.norm_feed_forward2.weight"),
            ("perception.encoder.layers.0.norm_out.weight",
             "layers.0.norm_out.weight"),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_weight_key(ckpt_key), expected)

    def test_various_layer_indices(self):
        """Test mapping for various layer indices."""
        for i in [0, 15, 31]:
            with self.subTest(layer_index=i):
                ckpt_key = f"perception.encoder.layers.{i}.self_attn.linear_q.weight"
                expected = f"layers.{i}.self_attn.linear_q.weight"
                self.assertEqual(map_weight_key(ckpt_key), expected)

                # Also test conv module mapping with different indices
                ckpt_key = f"perception.encoder.layers.{i}.conv.batch_norm.weight"
                expected = f"layers.{i}.conv_module.batch_norm.weight"
                self.assertEqual(map_weight_key(ckpt_key), expected)


if __name__ == "__main__":
    unittest.main()
