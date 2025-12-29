"""Integration test for the full CanaryQwen model.

Run this to verify the complete pipeline works:
    python -m exploration.canary_qwen.test_full_model
"""

import torch
import numpy as np


def main():
    print("=" * 60)
    print("Testing full CanaryQwen model")
    print("=" * 60)

    print("\nStep 1: Import and create model...")
    from nemo_lite import CanaryQwen

    # Use CPU for testing (GPU would be faster)
    model = CanaryQwen(device="cpu", dtype=torch.float16)
    print("Model created successfully!")

    print("\n" + "=" * 60)
    print("Step 2: Check component shapes")
    print("=" * 60)

    # Create fake audio (1 second at 16kHz)
    sample_rate = 16000
    duration = 1.0  # seconds
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    audio = audio / np.abs(audio).max()  # Normalize to [-1, 1]

    print(f"Audio shape: {audio.shape}")
    print(f"Audio duration: {duration}s at {sample_rate}Hz")

    # Test preprocessing
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(model.device)
    mel, mel_lengths = model.preprocessor(audio_tensor)
    print(f"\nMel spectrogram: {mel.shape}")
    print(f"  Expected: (1, 128, ~{int(sample_rate * duration / 160)}) frames")

    # Test encoder
    mel = mel.to(model.dtype)
    features, feature_lengths = model.encoder(mel, mel_lengths)
    print(f"\nEncoder output: {features.shape}")
    print(f"  Expected: (1, ~{mel.shape[2] // 8}, 1024)")

    # Test projection
    projected = model.projection(features)
    print(f"\nProjection output: {projected.shape}")
    print(f"  Expected: (1, {features.shape[1]}, 2048)")

    print("\n" + "=" * 60)
    print("Step 3: Test transcription (with random audio)")
    print("=" * 60)

    print("\nTranscribing random noise (expect gibberish)...")
    text = model.transcribe(audio, max_new_tokens=20)
    print(f"Transcription: '{text}'")
    print("(Random audio produces random text - this is expected)")

    print("\n" + "=" * 60)
    print("Step 4: Test with sine wave (still no real speech)")
    print("=" * 60)

    # Create a simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    sine_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz

    print("Transcribing 440Hz sine wave...")
    text = model.transcribe(sine_audio, max_new_tokens=20)
    print(f"Transcription: '{text}'")

    print("\n" + "=" * 60)
    print("Full model integration test PASSED!")
    print("=" * 60)

    print("""
Next steps to test with real audio:
    1. Load a real audio file (e.g., with librosa or soundfile)
    2. Ensure it's 16kHz mono
    3. Normalize to [-1, 1]
    4. Call model.transcribe(audio)

Example:
    import soundfile as sf
    audio, sr = sf.read("audio.wav")
    text = model.transcribe(audio, sample_rate=sr)
    print(text)
""")


if __name__ == "__main__":
    main()
