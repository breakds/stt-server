"""Exploration script to understand text generation with inputs_embeds.

Run this to see how generation works:
    python -m nemo_lite.qwen.explore_generation
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    print("=" * 60)
    print("Step 1: Load model and tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()

    print(f"Model loaded: {type(model).__name__}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    print("\n" + "=" * 60)
    print("Step 2: Normal generation with input_ids")
    print("=" * 60)

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    print(f"Prompt: '{prompt}'")
    print(f"Input IDs: {input_ids.tolist()[0]}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")

    print("\n" + "=" * 60)
    print("Step 3: Generation with inputs_embeds")
    print("=" * 60)

    # Get embeddings for the same prompt
    embed_tokens = model.model.embed_tokens
    with torch.no_grad():
        inputs_embeds = embed_tokens(input_ids)

    print(f"Inputs embeds shape: {inputs_embeds.shape}")

    # Generate using embeddings instead of token IDs
    with torch.no_grad():
        output_ids_from_embeds = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_from_embeds = tokenizer.decode(output_ids_from_embeds[0], skip_special_tokens=True)
    print(f"Generated from embeds: '{generated_from_embeds}'")

    # Verify outputs match
    print(f"\nOutputs match: {generated_text == generated_from_embeds}")

    print("\n" + "=" * 60)
    print("Step 4: Generation with modified embeddings")
    print("=" * 60)

    # Simulate what happens with audio injection:
    # Replace some embeddings with "audio" (random noise here)
    modified_embeds = inputs_embeds.clone()

    # Insert some "audio embeddings" (random) in the middle
    # In reality, these would be from the audio encoder + projection
    audio_embeds = torch.randn(1, 10, 2048, dtype=torch.float16)  # 10 "audio" frames

    # Concatenate: [first 2 tokens] + [audio] + [rest of tokens]
    combined_embeds = torch.cat([
        inputs_embeds[:, :2, :],   # "The capital"
        audio_embeds,              # [fake audio]
        inputs_embeds[:, 2:, :],   # "of France is"
    ], dim=1)

    print(f"Original embeds: {inputs_embeds.shape}")
    print(f"Combined embeds: {combined_embeds.shape}")

    with torch.no_grad():
        output_modified = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"Generated with fake audio: '{tokenizer.decode(output_modified[0], skip_special_tokens=True)}'")
    print("(gibberish expected - the 'audio' is random noise)")

    print("\n" + "=" * 60)
    print("Step 5: Generation parameters for transcription")
    print("=" * 60)

    print("""
For ASR transcription, typical generation settings:

    model.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=448,      # Max transcription length
        do_sample=False,         # Greedy (deterministic)
        # OR for sampling:
        # do_sample=True,
        # temperature=0.7,
        # top_p=0.9,
        num_beams=1,             # No beam search (faster)
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

The model will generate until:
- max_new_tokens is reached, OR
- eos_token (<|im_end|> for Qwen) is generated
""")

    print("\n" + "=" * 60)
    print("Step 6: Decoding the output")
    print("=" * 60)

    # The generate() output includes ALL token IDs (not just new ones)
    # But with inputs_embeds, we don't have input token IDs
    # So output_ids only contains the NEW generated tokens

    print("When using inputs_embeds:")
    print(f"  - Output shape: {output_ids_from_embeds.shape}")
    print(f"  - Contains only NEW tokens (not input)")

    print("\nWhen using input_ids:")
    print(f"  - Output shape: {output_ids.shape}")
    print(f"  - Contains input + new tokens")

    # For transcription, we just decode everything
    print("\nDecoding strategy:")
    print("  text = tokenizer.decode(output_ids[0], skip_special_tokens=True)")

    print("\n" + "=" * 60)
    print("Key takeaways")
    print("=" * 60)

    print("""
1. model.generate() works with EITHER input_ids OR inputs_embeds
   - inputs_embeds lets us inject audio embeddings

2. With inputs_embeds, output contains ONLY new tokens
   - No need to strip the prompt

3. Generation is autoregressive:
   - Model generates one token at a time
   - Each new token's embedding is computed and fed back
   - Continues until max_new_tokens or eos_token

4. The model's forward() expects:
   - inputs_embeds: (batch, seq_len, hidden_dim)
   - attention_mask: (batch, seq_len) - optional but recommended
""")


if __name__ == "__main__":
    main()
