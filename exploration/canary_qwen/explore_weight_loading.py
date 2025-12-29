"""Exploration script to load Canary-Qwen weights into peft model.

Run this to see how weight loading works:
    python -m nemo_lite.qwen.explore_weight_loading
"""

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import torch


def main():
    print("=" * 60)
    print("Step 1: Load base Qwen model with LoRA")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.01,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    print("Peft model created.")

    print("\n" + "=" * 60)
    print("Step 2: Download and inspect checkpoint")
    print("=" * 60)

    checkpoint_path = hf_hub_download(
        "nvidia/canary-qwen-2.5b",
        "model.safetensors",
    )
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint keys
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        ckpt_keys = list(f.keys())
        llm_keys = [k for k in ckpt_keys if k.startswith("llm.")]

    print(f"\nTotal checkpoint keys: {len(ckpt_keys)}")
    print(f"LLM keys: {len(llm_keys)}")

    print("\n" + "=" * 60)
    print("Step 3: Map checkpoint keys to peft model keys")
    print("=" * 60)

    def map_llm_key(ckpt_key: str) -> str | None:
        """Map checkpoint key to peft model key."""
        if not ckpt_key.startswith("llm."):
            return None
        # llm.base_model.model.model... -> base_model.model.model...
        return ckpt_key[4:]  # Remove "llm." prefix

    # Get peft model keys
    peft_state_dict = peft_model.state_dict()
    peft_keys = set(peft_state_dict.keys())

    # Map and verify
    mapped_keys = {}
    missing_in_ckpt = []
    unexpected_in_ckpt = []

    for ckpt_key in llm_keys:
        peft_key = map_llm_key(ckpt_key)
        if peft_key in peft_keys:
            mapped_keys[ckpt_key] = peft_key
        else:
            unexpected_in_ckpt.append(ckpt_key)

    for peft_key in peft_keys:
        ckpt_key = "llm." + peft_key
        if ckpt_key not in llm_keys:
            missing_in_ckpt.append(peft_key)

    print(f"\nSuccessfully mapped: {len(mapped_keys)}")
    print(f"Missing in checkpoint: {len(missing_in_ckpt)}")
    print(f"Unexpected in checkpoint: {len(unexpected_in_ckpt)}")

    if missing_in_ckpt:
        print("\nKeys in peft model but NOT in checkpoint:")
        for k in missing_in_ckpt[:10]:
            print(f"  {k}")
        if len(missing_in_ckpt) > 10:
            print(f"  ... ({len(missing_in_ckpt)} total)")

    if unexpected_in_ckpt:
        print("\nKeys in checkpoint but NOT in peft model:")
        for k in unexpected_in_ckpt[:10]:
            print(f"  {k}")
        if len(unexpected_in_ckpt) > 10:
            print(f"  ... ({len(unexpected_in_ckpt)} total)")

    print("\n" + "=" * 60)
    print("Step 4: Load weights")
    print("=" * 60)

    # Load weights from checkpoint
    loaded_state_dict = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for ckpt_key, peft_key in mapped_keys.items():
            tensor = f.get_tensor(ckpt_key)
            loaded_state_dict[peft_key] = tensor

    print(f"Loaded {len(loaded_state_dict)} tensors from checkpoint")

    # Verify shapes match
    shape_mismatches = []
    for key, tensor in loaded_state_dict.items():
        expected_shape = peft_state_dict[key].shape
        if tensor.shape != expected_shape:
            shape_mismatches.append((key, tensor.shape, expected_shape))

    if shape_mismatches:
        print("\nShape mismatches:")
        for key, got, expected in shape_mismatches:
            print(f"  {key}: got {got}, expected {expected}")
    else:
        print("All shapes match!")

    # Load into model
    result = peft_model.load_state_dict(loaded_state_dict, strict=False)
    print(f"\nMissing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    print("\n" + "=" * 60)
    print("Step 5: Verify loaded weights")
    print("=" * 60)

    # Compare a few weights
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        # Check layer 0 q_proj lora_A
        ckpt_key = "llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        ckpt_tensor = f.get_tensor(ckpt_key)
        model_tensor = peft_model.state_dict()["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]

        print(f"\nComparing {ckpt_key}:")
        print(f"  Checkpoint: {ckpt_tensor.shape}, dtype={ckpt_tensor.dtype}")
        print(f"  Model: {model_tensor.shape}, dtype={model_tensor.dtype}")
        print(f"  Values match: {torch.allclose(ckpt_tensor.float(), model_tensor.float())}")
        print(f"  Max diff: {(ckpt_tensor.float() - model_tensor.float()).abs().max().item():.6e}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"""
Weight loading successful!

- Loaded {len(loaded_state_dict)} weights from checkpoint
- Missing {len(result.missing_keys)} keys (expected: lm_head, embed_tokens)
- These come from the base Qwen model, not the checkpoint

The missing keys are:
""")
    for k in result.missing_keys[:5]:
        print(f"  {k}")
    if len(result.missing_keys) > 5:
        print(f"  ... ({len(result.missing_keys)} total)")


if __name__ == "__main__":
    main()
