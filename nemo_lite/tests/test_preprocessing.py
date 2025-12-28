"""Unit tests for AudioPreprocessor.

Uses librosa as reference (which is what NeMo uses internally) to verify
our implementation produces equivalent results.

Run with:
    python -m unittest nemo_lite.tests.test_preprocessing
"""

import unittest

import numpy as np
import torch
import librosa

from nemo_lite.preprocessing import AudioPreprocessor


class TestMelFilterbank(unittest.TestCase):
    """Test that our mel filterbank matches librosa's."""

    def test_filterbank_matches_librosa(self):
        """Verify our mel filterbank matches librosa's (which NeMo uses)."""
        n_fft = 512
        n_mels = 128
        sample_rate = 16000

        preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
        )
        our_fbank = preprocessor.mel_filterbank.numpy()

        librosa_fbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate / 2.0,
            norm="slaney",
        )

        self.assertEqual(our_fbank.shape, librosa_fbank.shape)

        max_diff = np.abs(librosa_fbank - our_fbank).max()
        self.assertLess(max_diff, 1e-6, f"Filterbanks differ by {max_diff:.6e}")

        # No zero rows
        our_zero_rows = (our_fbank.sum(axis=1) == 0).sum()
        self.assertEqual(our_zero_rows, 0, "Filterbank has zero rows")


class TestSTFT(unittest.TestCase):
    """Test that PyTorch STFT matches librosa's."""

    def test_stft_matches_librosa(self):
        """Verify PyTorch STFT matches librosa's within tolerance."""
        np.random.seed(42)

        n_fft = 512
        hop_length = 160
        win_length = 400
        sample_rate = 16000

        audio_np = np.random.randn(sample_rate * 2).astype(np.float32)
        audio_torch = torch.from_numpy(audio_np)

        # PyTorch STFT
        window = torch.hann_window(win_length)
        stft_torch = torch.stft(
            audio_torch,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power_torch = stft_torch.abs().pow(2).numpy()

        # librosa STFT
        stft_librosa = librosa.stft(
            audio_np,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )
        power_librosa = np.abs(stft_librosa) ** 2

        self.assertEqual(power_torch.shape, power_librosa.shape)

        max_diff = np.abs(power_librosa - power_torch).max()
        self.assertLess(max_diff, 1e-3, f"STFT differs by {max_diff:.6e}")


class TestFullPipeline(unittest.TestCase):
    """Test full mel spectrogram pipeline against librosa reference."""

    def test_pipeline_matches_librosa(self):
        """Test complete pipeline: pre-emphasis, STFT, mel, log."""
        np.random.seed(42)
        torch.manual_seed(42)

        sample_rate = 16000
        n_fft = 512
        hop_length = 160
        win_length = 400
        n_mels = 128
        preemph = 0.97
        log_guard = 2**-24

        audio_np = np.random.randn(sample_rate * 2).astype(np.float32)

        # Our implementation (without normalization)
        preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            preemph=preemph,
            log_zero_guard=log_guard,
            normalize=None,
        )

        audio_torch = torch.from_numpy(audio_np).unsqueeze(0)
        audio_lengths = torch.tensor([len(audio_np)])

        with torch.no_grad():
            our_mel, our_lengths = preprocessor(audio_torch, audio_lengths)
        our_mel = our_mel[0].numpy()

        # librosa reference
        audio_preemph = np.concatenate(
            [[audio_np[0]], audio_np[1:] - preemph * audio_np[:-1]]
        )
        mel_librosa = librosa.feature.melspectrogram(
            y=audio_preemph,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="constant",
            power=2.0,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate / 2.0,
            norm="slaney",
        )
        log_mel_librosa = np.log(mel_librosa + log_guard)

        # Compare only valid frames
        valid_frames = our_lengths.item()
        our_mel_valid = our_mel[:, :valid_frames]
        librosa_mel_valid = log_mel_librosa[:, :valid_frames]

        max_diff = np.abs(our_mel_valid - librosa_mel_valid).max()
        self.assertLess(max_diff, 1e-3, f"Pipeline differs by {max_diff:.6e}")


class TestNormalization(unittest.TestCase):
    """Test per-feature normalization."""

    def test_normalization_statistics(self):
        """Test that per-feature normalization produces mean~0, std~1."""
        torch.manual_seed(42)

        preprocessor = AudioPreprocessor(normalize="per_feature")

        audio = torch.randn(1, 16000 * 5)  # 5 seconds
        audio_lengths = torch.tensor([16000 * 5])

        with torch.no_grad():
            mel, mel_lengths = preprocessor(audio, audio_lengths)

        valid_len = mel_lengths[0].item()
        mel_valid = mel[0, :, :valid_len]

        mean_per_feature = mel_valid.mean(dim=1)
        std_per_feature = mel_valid.std(dim=1, unbiased=True)

        mean_abs_max = mean_per_feature.abs().max().item()
        std_deviation = (std_per_feature - 1.0).abs().max().item()

        self.assertLess(mean_abs_max, 1e-4, f"Mean not ~0: {mean_abs_max}")
        self.assertLess(std_deviation, 1e-4, f"Std not ~1: {std_deviation}")


class TestOutputShape(unittest.TestCase):
    """Test output shapes and dimensions."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        preprocessor = AudioPreprocessor()

        batch_size = 4
        num_samples = 16000 * 2

        audio = torch.randn(batch_size, num_samples)
        audio_lengths = torch.full((batch_size,), num_samples)

        with torch.no_grad():
            mel, mel_lengths = preprocessor(audio, audio_lengths)

        self.assertEqual(mel.shape[0], batch_size)
        self.assertEqual(mel.shape[1], 128)
        self.assertEqual(len(mel_lengths), batch_size)

    def test_variable_length_batch(self):
        """Test handling of variable-length audio in a batch."""
        torch.manual_seed(42)

        preprocessor = AudioPreprocessor()

        batch_size = 3
        max_samples = 16000 * 3
        audio = torch.randn(batch_size, max_samples)
        audio_lengths = torch.tensor([48000, 32000, 16000])

        for i in range(batch_size):
            audio[i, audio_lengths[i] :] = 0.0

        with torch.no_grad():
            mel, mel_lengths = preprocessor(audio, audio_lengths)

        expected_lengths = (audio_lengths + 160 - 1) // 160

        self.assertTrue(
            torch.equal(mel_lengths, expected_lengths),
            f"Mel lengths {mel_lengths.tolist()} != expected {expected_lengths.tolist()}",
        )


class TestDeviceHandling(unittest.TestCase):
    """Test device placement."""

    def test_to_device(self):
        """Test that .to() moves all buffers."""
        preprocessor = AudioPreprocessor()

        # Check buffers are on CPU by default
        self.assertEqual(preprocessor.window.device.type, "cpu")
        self.assertEqual(preprocessor.mel_filterbank.device.type, "cpu")

        # Skip CUDA test if not available
        if torch.cuda.is_available():
            preprocessor = preprocessor.to("cuda")
            self.assertEqual(preprocessor.window.device.type, "cuda")
            self.assertEqual(preprocessor.mel_filterbank.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
