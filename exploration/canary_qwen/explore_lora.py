"""Exploration script to understand LoRA with peft.

Run this to see how LoRA adapters work:
    python -m nemo_lite.qwen.explore_lora
"""

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch


def main():
    print("=" * 60)
    print("Loading Qwen/Qwen3-1.7B...")
    print("=" * 60)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("\n" + "=" * 60)
    print("Before LoRA: q_proj structure")
    print("=" * 60)

    # Show q_proj before LoRA
    q_proj = model.model.layers[0].self_attn.q_proj
    print(f"\nq_proj type: {type(q_proj).__name__}")
    print(f"q_proj: {q_proj}")

    print("\n" + "=" * 60)
    print("Adding LoRA adapters (matching Canary-Qwen config)")
    print("=" * 60)

    # LoRA config matching the Canary-Qwen checkpoint
    lora_config = LoraConfig(
        r=128,                          # Rank
        lora_alpha=256,                 # Scaling factor
        lora_dropout=0.01,              # Dropout
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        task_type="CAUSAL_LM",
    )
    print(f"\nLoRA config:")
    print(f"  rank (r): {lora_config.r}")
    print(f"  alpha: {lora_config.lora_alpha}")
    print(f"  scaling: alpha/r = {lora_config.lora_alpha / lora_config.r}")
    print(f"  target_modules: {lora_config.target_modules}")

    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)

    print("\n" + "=" * 60)
    print("After LoRA: q_proj structure")
    print("=" * 60)

    # Show q_proj after LoRA
    q_proj_lora = peft_model.model.model.layers[0].self_attn.q_proj
    print(f"\nq_proj type: {type(q_proj_lora).__name__}")
    print(f"q_proj: {q_proj_lora}")

    print("\n" + "=" * 60)
    print("LoRA layer internals")
    print("=" * 60)

    # Show the internal structure
    print(f"\nbase_layer: {q_proj_lora.base_layer}")
    print(f"lora_A shape: {q_proj_lora.lora_A['default'].weight.shape}")
    print(f"lora_B shape: {q_proj_lora.lora_B['default'].weight.shape}")
    print(f"scaling: {q_proj_lora.scaling}")

    print("\n" + "=" * 60)
    print("How LoRA forward pass works")
    print("=" * 60)

    print("""
    For input x of shape (batch, seq, 2048):

    1. base_output = base_layer(x)           # (batch, seq, 2048)

    2. lora_down = lora_A(x)                 # (batch, seq, 128)  <- compress
       lora_up = lora_B(lora_down)           # (batch, seq, 2048) <- expand
       lora_output = lora_up * scaling       # scale by alpha/r = 2.0

    3. output = base_output + lora_output    # combine

    The LoRA matrices are initialized so lora_output starts at 0,
    then learns the "delta" needed for the task.
    """)

    print("\n" + "=" * 60)
    print("Parameter counts")
    print("=" * 60)

    # Count parameters
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Frozen (base): {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    # Show which parameters are trainable
    print("\nTrainable parameter names (first 10):")
    trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    for name in trainable_names[:10]:
        print(f"  {name}")
    print(f"  ... ({len(trainable_names)} total)")

    print("\n" + "=" * 60)
    print("State dict structure (for weight loading)")
    print("=" * 60)

    # Show state dict keys for layer 0
    state_dict = peft_model.state_dict()
    layer0_keys = [k for k in state_dict.keys() if "layers.0.self_attn" in k]
    print("\nLayer 0 self_attn keys:")
    for k in sorted(layer0_keys):
        shape = state_dict[k].shape
        print(f"  {k}: {shape}")


if __name__ == "__main__":
    main()
