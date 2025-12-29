"""Integration test for QwenWrapper with weight loading.

Run this to verify the full pipeline works:
    python -m nemo_lite.qwen.test_integration
"""

import torch
from nemo_lite.qwen import QwenWrapper
from nemo_lite.weights import load_llm_weights


def main():
    print("=" * 60)
    print("Step 1: Create QwenWrapper")
    print("=" * 60)

    wrapper = QwenWrapper(device="cpu", dtype=torch.float16)
    print(f"Wrapper created successfully")
    print(f"  Placeholder token: '{wrapper.tokenizer.decode([wrapper.placeholder_id])}'")
    print(f"  Placeholder ID: {wrapper.placeholder_id}")
    print(f"  Hidden size: {wrapper.hidden_size}")

    print("\n" + "=" * 60)
    print("Step 2: Load weights from checkpoint")
    print("=" * 60)

    missing, unexpected = load_llm_weights(
        wrapper,
        "nvidia/canary-qwen-2.5b",
        device="cpu",
    )

    print(f"Missing keys: {len(missing)}")
    for k in missing:
        print(f"  {k}")

    print(f"\nUnexpected keys: {len(unexpected)}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"  {k}")

    print("\n" + "=" * 60)
    print("Step 3: Test embedding injection")
    print("=" * 60)

    # Create fake audio embeddings
    audio_embeds = torch.randn(1, 50, 2048, dtype=torch.float16)
    print(f"Fake audio embeddings: {audio_embeds.shape}")

    # Get default prompt
    prompt = wrapper.default_prompt()
    print(f"Prompt:\n{prompt}")

    # Tokenize
    input_ids = wrapper.tokenizer(prompt, return_tensors="pt")["input_ids"]
    print(f"Token IDs shape: {input_ids.shape}")

    # Get text embeddings
    with torch.no_grad():
        text_embeds = wrapper.get_text_embeddings(input_ids)
    print(f"Text embeddings: {text_embeds.shape}")

    # Inject audio
    combined, mask = wrapper.inject_audio_embeddings(input_ids, text_embeds, audio_embeds)
    print(f"Combined embeddings: {combined.shape}")
    print(f"Attention mask: {mask.shape}")

    expected_len = text_embeds.shape[1] - 1 + audio_embeds.shape[1]
    print(f"Expected length: {text_embeds.shape[1]} - 1 + {audio_embeds.shape[1]} = {expected_len}")
    assert combined.shape[1] == expected_len, "Length mismatch!"
    print("Length check: PASSED")

    print("\n" + "=" * 60)
    print("Step 4: Test generation (short)")
    print("=" * 60)

    print("Generating with fake audio (expect gibberish)...")
    with torch.no_grad():
        # Generate just a few tokens for testing
        text = wrapper.generate(audio_embeds, max_new_tokens=10)

    print(f"Generated: '{text}'")
    print("(Gibberish expected - using random embeddings)")

    print("\n" + "=" * 60)
    print("Integration test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
