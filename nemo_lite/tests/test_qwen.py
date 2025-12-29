"""Tests for Qwen LLM wrapper and weight loading."""

import unittest

from nemo_lite.weights import map_llm_weight_key


class TestLLMWeightKeyMapping(unittest.TestCase):
    """Test LLM weight key mapping."""

    def test_non_llm_keys_return_none(self):
        """Test that non-LLM keys return None."""
        self.assertIsNone(map_llm_weight_key("perception.encoder.layers.0.weight"))
        self.assertIsNone(map_llm_weight_key("perception.proj.weight"))
        self.assertIsNone(map_llm_weight_key("random.key"))

    def test_strips_llm_prefix(self):
        """Test that llm. prefix is stripped."""
        ckpt_key = "llm.base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
        expected = "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
        self.assertEqual(map_llm_weight_key(ckpt_key), expected)

    def test_lora_a_mapping(self):
        """Test mapping for LoRA A weights."""
        ckpt_key = "llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        expected = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        self.assertEqual(map_llm_weight_key(ckpt_key), expected)

    def test_lora_b_mapping(self):
        """Test mapping for LoRA B weights."""
        ckpt_key = "llm.base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight"
        expected = "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight"
        self.assertEqual(map_llm_weight_key(ckpt_key), expected)

    def test_non_lora_layer_mapping(self):
        """Test mapping for non-LoRA layers (k_proj, o_proj, mlp)."""
        test_cases = [
            (
                "llm.base_model.model.model.layers.0.self_attn.k_proj.weight",
                "base_model.model.model.layers.0.self_attn.k_proj.weight",
            ),
            (
                "llm.base_model.model.model.layers.0.self_attn.o_proj.weight",
                "base_model.model.model.layers.0.self_attn.o_proj.weight",
            ),
            (
                "llm.base_model.model.model.layers.0.mlp.gate_proj.weight",
                "base_model.model.model.layers.0.mlp.gate_proj.weight",
            ),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_llm_weight_key(ckpt_key), expected)

    def test_norm_mapping(self):
        """Test mapping for normalization layers."""
        test_cases = [
            (
                "llm.base_model.model.model.layers.0.input_layernorm.weight",
                "base_model.model.model.layers.0.input_layernorm.weight",
            ),
            (
                "llm.base_model.model.model.norm.weight",
                "base_model.model.model.norm.weight",
            ),
        ]
        for ckpt_key, expected in test_cases:
            with self.subTest(ckpt_key=ckpt_key):
                self.assertEqual(map_llm_weight_key(ckpt_key), expected)

    def test_various_layer_indices(self):
        """Test mapping for various layer indices."""
        for i in [0, 14, 27]:
            with self.subTest(layer_index=i):
                ckpt_key = f"llm.base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight"
                expected = f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer.weight"
                self.assertEqual(map_llm_weight_key(ckpt_key), expected)


if __name__ == "__main__":
    unittest.main()
