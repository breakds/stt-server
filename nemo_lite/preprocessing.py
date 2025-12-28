"""Audio preprocessing for Canary ASR model (inference only).

Converts raw audio waveforms to mel spectrograms matching NeMo's
AudioToMelSpectrogramPreprocessor output exactly.

Key parameters (from Canary config):
- sample_rate: 16000 Hz
- n_mels: 128
- n_fft: 512
- window: 25ms Hann (400 samples)
- stride: 10ms (160 samples)
- normalize: per-feature (z-score per mel bin)
"""

import torch
import torch.nn as nn
import librosa


class AudioPreprocessor(nn.Module):
    """Mel spectrogram preprocessor matching NeMo's implementation (inference only).

    Pipeline:
        1. Pre-emphasis filter (y[n] = x[n] - 0.97 * x[n-1])
        2. STFT with Hann window
        3. Power spectrum (magnitude squared)
        4. Mel filterbank projection
        5. Log scaling with guard (log(x + 2^-24))
        6. Per-feature normalization (z-score per mel bin)

    Args:
        sample_rate: Expected input sample rate. Default: 16000.
        n_mels: Number of mel filterbank channels. Default: 128.
        n_fft: FFT size. Default: 512.
        win_length: Window length in samples. Default: 400 (25ms at 16kHz).
        hop_length: Hop length in samples. Default: 160 (10ms at 16kHz).
        preemph: Pre-emphasis coefficient. Default: 0.97.
        log_zero_guard: Guard value for log. Default: 2^-24.
        normalize: Normalization type. "per_feature" or None. Default: "per_feature".
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemph: float = 0.97,
        log_zero_guard: float = 2**-24,
        normalize: str | None = "per_feature",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemph = preemph
        self.log_zero_guard = log_zero_guard
        self.normalize = normalize

        # Create Hann window (same as NeMo's default)
        self.register_buffer("window", torch.hann_window(win_length))

        # Create mel filterbank using librosa (exactly what NeMo uses)
        mel_filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate / 2.0,
            norm="slaney",
        )
        # Shape: (n_mels, n_freqs) - ready for matmul
        self.register_buffer("mel_filterbank", torch.from_numpy(mel_filterbank).float())

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert audio waveforms to mel spectrograms.

        Args:
            audio: Audio waveforms of shape (batch, time).
            audio_lengths: Length of each audio in samples. If None, assumes
                all audios have the same length as the tensor dimension.

        Returns:
            Tuple of:
                - mel: Mel spectrograms of shape (batch, n_mels, time_frames).
                - mel_lengths: Number of valid frames for each sample.
        """
        batch_size = audio.shape[0]
        device = audio.device

        if audio_lengths is None:
            audio_lengths = torch.full(
                (batch_size,), audio.shape[1], dtype=torch.long, device=device
            )

        # Pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
        audio = torch.cat(
            [audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]], dim=1
        )

        # Mask out samples beyond audio_lengths
        time_mask = (
            torch.arange(audio.shape[1], device=device).unsqueeze(0)
            < audio_lengths.unsqueeze(1)
        )
        audio = audio.masked_fill(~time_mask, 0.0)

        # STFT (center=True pads the signal symmetrically)
        stft_out = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        # Shape: (batch, n_fft//2 + 1, time_frames)

        # Power spectrum (magnitude squared)
        power_spectrum = stft_out.abs().pow(2)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_filterbank, power_spectrum)
        # Shape: (batch, n_mels, time_frames)

        # Log scaling with guard
        mel = torch.log(mel + self.log_zero_guard)

        # Calculate output lengths (ceil division)
        mel_lengths = (audio_lengths + self.hop_length - 1) // self.hop_length

        # Per-feature normalization
        if self.normalize == "per_feature":
            mel = self._normalize_per_feature(mel, mel_lengths)

        # Mask out frames beyond mel_lengths
        max_mel_len = mel.shape[2]
        frame_mask = (
            torch.arange(max_mel_len, device=device).unsqueeze(0)
            < mel_lengths.unsqueeze(1)
        )
        mel = mel.masked_fill(~frame_mask.unsqueeze(1), 0.0)

        return mel, mel_lengths

    def _normalize_per_feature(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Normalize each mel bin independently (z-score normalization).

        NeMo computes mean and std only over valid frames for each sample.
        Uses unbiased std (N-1 denominator).

        Args:
            mel: Mel spectrogram of shape (batch, n_mels, time_frames).
            mel_lengths: Number of valid frames for each sample.

        Returns:
            Normalized mel spectrogram.
        """
        batch_size, n_mels, max_len = mel.shape
        device = mel.device

        # Create mask for valid frames: (batch, 1, time_frames)
        frame_mask = (
            torch.arange(max_len, device=device).unsqueeze(0)
            < mel_lengths.unsqueeze(1)
        ).unsqueeze(1)

        # Mask out invalid frames for statistics computation
        mel_masked = mel.masked_fill(~frame_mask, 0.0)

        # Compute mean per feature (batch, n_mels)
        # Sum over time, divide by number of valid frames
        mel_sum = mel_masked.sum(dim=2)  # (batch, n_mels)
        mel_mean = mel_sum / mel_lengths.unsqueeze(1).float()

        # Compute variance per feature (unbiased, N-1 denominator)
        # (x - mean)^2, summed over valid frames
        diff = mel_masked - mel_mean.unsqueeze(2)
        diff = diff.masked_fill(~frame_mask, 0.0)
        variance = (diff.pow(2).sum(dim=2)) / (mel_lengths.unsqueeze(1).float() - 1)
        mel_std = torch.sqrt(variance)

        # Normalize: (x - mean) / (std + eps)
        # NeMo uses eps = 1e-5
        eps = 1e-5
        mel_normalized = (mel - mel_mean.unsqueeze(2)) / (mel_std.unsqueeze(2) + eps)

        return mel_normalized
