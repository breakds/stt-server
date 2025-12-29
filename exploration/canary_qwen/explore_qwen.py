"""Exploration script to understand Qwen model structure.

Run this to see how the Qwen model is organized:
    python -m nemo_lite.qwen.explore_qwen
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    print("=" * 60)
    print("Loading Qwen/Qwen3-1.7B from HuggingFace...")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    # Load model (use float16 to save memory, don't load to GPU yet)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu",  # Keep on CPU for exploration
    )

    print("\n" + "=" * 60)
    print("Model Architecture")
    print("=" * 60)

    # Print model config
    config = model.config
    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")

    print("\n" + "=" * 60)
    print("Model Structure (top-level modules)")
    print("=" * 60)

    # Print top-level structure
    for name, module in model.named_children():
        print(f"\n{name}: {type(module).__name__}")
        for subname, submodule in module.named_children():
            print(f"  {subname}: {type(submodule).__name__}")
            # Only show first layer details for layers
            if subname == "layers":
                first_layer = list(submodule.children())[0]
                print(f"    [0]: {type(first_layer).__name__}")
                for layername, layermodule in first_layer.named_children():
                    print(f"      {layername}: {type(layermodule).__name__}")

    print("\n" + "=" * 60)
    print("Key Components for Audio Integration")
    print("=" * 60)

    # Show embedding layer
    embed = model.model.embed_tokens
    print(f"\nEmbed tokens: {embed}")
    print(f"  Shape: ({embed.num_embeddings}, {embed.embedding_dim})")

    # Show how to get embeddings
    print("\n" + "=" * 60)
    print("Example: Text to Embeddings")
    print("=" * 60)

    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"\nInput text: '{text}'")
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    # Get embeddings
    with torch.no_grad():
        embeddings = embed(input_ids)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  (batch=1, seq_len={embeddings.shape[1]}, hidden={embeddings.shape[2]})")

    print("\n" + "=" * 60)
    print("Forward pass with inputs_embeds")
    print("=" * 60)

    # Show that we can pass embeddings directly instead of token IDs
    print("\nThe model can accept `inputs_embeds` instead of `input_ids`:")
    print("  output = model(inputs_embeds=embeddings)")
    print("\nThis is how we'll inject audio embeddings!")


if __name__ == "__main__":
    main()
