"""Exploration script to understand audio embedding injection.

Run this to see how placeholder replacement works:
    python -m nemo_lite.qwen.explore_embedding_injection
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    print("=" * 60)
    print("Step 1: Load tokenizer and add placeholder token")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # The audio placeholder token used by Canary
    AUDIO_PLACEHOLDER = "<|audioplaceholder|>"

    # Add it as a special token
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [AUDIO_PLACEHOLDER]
    })
    print(f"Added {num_added} new token(s)")

    placeholder_id = tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)
    print(f"Placeholder token: '{AUDIO_PLACEHOLDER}'")
    print(f"Placeholder ID: {placeholder_id}")

    print("\n" + "=" * 60)
    print("Step 2: Tokenize a prompt with placeholder")
    print("=" * 60)

    # Example prompt for transcription
    prompt = f"""<|im_start|>user
Transcribe the following audio:{AUDIO_PLACEHOLDER}<|im_end|>
<|im_start|>assistant
"""

    print(f"Prompt:\n{prompt}")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    print(f"\nToken IDs shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids.tolist()[0]}")

    # Show each token
    print("\nTokens breakdown:")
    for i, tid in enumerate(input_ids[0]):
        token = tokenizer.decode([tid])
        marker = " ← PLACEHOLDER" if tid == placeholder_id else ""
        print(f"  [{i}] {tid:6d}: {repr(token)}{marker}")

    # Find placeholder position
    placeholder_positions = (input_ids == placeholder_id).nonzero(as_tuple=True)
    print(f"\nPlaceholder at position: {placeholder_positions[1].tolist()}")

    print("\n" + "=" * 60)
    print("Step 3: Get text embeddings")
    print("=" * 60)

    # Load model (just need embed_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Resize embeddings to include new token
    model.resize_token_embeddings(len(tokenizer))

    embed_tokens = model.model.embed_tokens
    print(f"Embedding layer: {embed_tokens}")

    with torch.no_grad():
        text_embeds = embed_tokens(input_ids)
    print(f"Text embeddings shape: {text_embeds.shape}")

    print("\n" + "=" * 60)
    print("Step 4: Simulate audio embeddings")
    print("=" * 60)

    # Simulate audio embeddings from our projection layer
    # In reality: audio -> preprocessor -> encoder -> projection -> audio_embeds
    batch_size = 1
    audio_seq_len = 50  # e.g., 50 time steps from encoder
    hidden_dim = 2048

    audio_embeds = torch.randn(batch_size, audio_seq_len, hidden_dim, dtype=torch.float16)
    print(f"Audio embeddings shape: {audio_embeds.shape}")
    print(f"  (batch={batch_size}, audio_seq_len={audio_seq_len}, hidden={hidden_dim})")

    print("\n" + "=" * 60)
    print("Step 5: Replace placeholder with audio embeddings")
    print("=" * 60)

    def inject_audio_embeddings(
        input_ids: torch.Tensor,      # (B, T_text)
        text_embeds: torch.Tensor,    # (B, T_text, hidden)
        audio_embeds: torch.Tensor,   # (B, T_audio, hidden)
        placeholder_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace placeholder token embedding with audio embeddings.

        Returns:
            combined_embeds: (B, T_text - 1 + T_audio, hidden)
            attention_mask: (B, T_text - 1 + T_audio)
        """
        batch_size = input_ids.shape[0]
        text_len = input_ids.shape[1]
        audio_len = audio_embeds.shape[1]
        hidden_dim = text_embeds.shape[2]

        # Find placeholder position (assuming one per sequence)
        placeholder_mask = (input_ids == placeholder_id)
        placeholder_pos = placeholder_mask.nonzero(as_tuple=True)[1]  # (B,)

        # For simplicity, assume batch_size=1 and single placeholder
        pos = placeholder_pos[0].item()

        # Split text embeddings at placeholder position
        before = text_embeds[:, :pos, :]      # Everything before placeholder
        after = text_embeds[:, pos + 1:, :]   # Everything after placeholder

        # Concatenate: [before] + [audio] + [after]
        combined = torch.cat([before, audio_embeds, after], dim=1)

        # Create attention mask (all 1s for now)
        new_len = combined.shape[1]
        attention_mask = torch.ones(batch_size, new_len, dtype=torch.long)

        return combined, attention_mask

    combined_embeds, attention_mask = inject_audio_embeddings(
        input_ids, text_embeds, audio_embeds, placeholder_id
    )

    print(f"Original text length: {text_embeds.shape[1]}")
    print(f"Audio length: {audio_embeds.shape[1]}")
    print(f"Combined length: {combined_embeds.shape[1]}")
    print(f"  = {text_embeds.shape[1]} - 1 + {audio_embeds.shape[1]}")

    print(f"\nCombined embeddings shape: {combined_embeds.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    print("\n" + "=" * 60)
    print("Step 6: Visualize the sequence")
    print("=" * 60)

    pos = (input_ids == placeholder_id).nonzero(as_tuple=True)[1][0].item()

    print("\nBefore injection:")
    print("  [text tokens] [PLACEHOLDER] [text tokens]")
    print(f"  [{pos} tokens]  [1 token]    [{text_embeds.shape[1] - pos - 1} tokens]")

    print("\nAfter injection:")
    print("  [text embeds] [AUDIO EMBEDS] [text embeds]")
    print(f"  [{pos} embeds]  [{audio_embeds.shape[1]} embeds]   [{text_embeds.shape[1] - pos - 1} embeds]")

    print(f"\nTotal sequence: {combined_embeds.shape[1]} embeddings")

    print("\n" + "=" * 60)
    print("Key insight")
    print("=" * 60)

    print("""
The placeholder token gets REPLACED by multiple audio embeddings:

    "Transcribe:" [PLACEHOLDER] "\\n"
         ↓            ↓          ↓
    [embed]     [50 audio]   [embed]
    [embed]      [embeds]    [embed]
      ...                      ...

This is why the sequence grows: 1 placeholder token → N audio embeddings.

For Canary-Qwen:
- Audio: ~10 seconds → ~1000 mel frames → 125 encoder frames (8x reduction)
- Each encoder frame becomes one embedding in the LLM sequence
""")


if __name__ == "__main__":
    main()
