"""Canary-Qwen-2.5B ASR model.

Complete speech-to-text pipeline that combines:
1. AudioPreprocessor: audio waveform → mel spectrogram
2. FastConformerEncoder: mel spectrogram → audio features
3. AudioProjection: audio features → LLM embedding space
4. QwenWrapper: audio embeddings → transcription text

Usage:
    from nemo_lite.model import CanaryQwen

    model = CanaryQwen(device="cuda")

    # Transcribe audio (numpy array or torch tensor)
    text = model.transcribe(audio, sample_rate=16000)
"""

import torch
import torch.nn as nn
import numpy as np

from nemo_lite.preprocessing import AudioPreprocessor
from nemo_lite.conformer_lite import FastConformerEncoder
from nemo_lite.projection import AudioProjection
from nemo_lite.qwen import QwenWrapper
from nemo_lite.weights import (
    get_encoder_config,
    load_encoder_weights,
    load_projection_weights,
    load_llm_weights,
)


class CanaryQwen(nn.Module):
    """Canary-Qwen-2.5B speech-to-text model.

    This model combines a FastConformer encoder with a Qwen LLM for
    speech recognition. Audio is processed through mel spectrogram
    extraction, encoded by the Conformer, projected to LLM space,
    and decoded by the Qwen model with LoRA adapters.

    Args:
        device: Device to load model on ("cpu", "cuda", "cuda:0", etc.).
        dtype: Model dtype. Default: torch.float16.
        load_weights: Whether to load pretrained weights. Default: True.
        cache_dir: Directory to cache downloaded models. If None, uses HuggingFace default.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        load_weights: bool = True,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Create components
        self.preprocessor = AudioPreprocessor()

        encoder_config = get_encoder_config()
        self.encoder = FastConformerEncoder(**encoder_config)

        self.projection = AudioProjection(
            encoder_dim=encoder_config["d_model"],  # 1024
            llm_dim=2048,
        )

        self.llm = QwenWrapper(device=device, dtype=dtype, cache_dir=cache_dir)

        # Move encoder components to device and dtype
        self.preprocessor = self.preprocessor.to(device)
        self.encoder = self.encoder.to(device=device, dtype=dtype)
        self.projection = self.projection.to(device=device, dtype=dtype)

        # Load pretrained weights
        if load_weights:
            self._load_weights()

        # Set to eval mode
        self.eval()

    def _load_weights(self):
        """Load pretrained weights from nvidia/canary-qwen-2.5b."""
        source = "nvidia/canary-qwen-2.5b"

        # Load encoder weights (FastConformer)
        missing, unexpected = load_encoder_weights(
            self.encoder, source, device=self.device, cache_dir=self.cache_dir
        )
        if missing:
            print(f"Warning: Encoder missing keys: {missing}")
        if unexpected:
            print(f"Warning: Encoder unexpected keys: {unexpected}")

        # Load projection weights
        missing, unexpected = load_projection_weights(
            self.projection, source, device=self.device, cache_dir=self.cache_dir
        )
        if missing:
            print(f"Warning: Projection missing keys: {missing}")
        if unexpected:
            print(f"Warning: Projection unexpected keys: {unexpected}")

        # Load LLM weights (Qwen + LoRA)
        missing, unexpected = load_llm_weights(
            self.llm, source, device=self.device, cache_dir=self.cache_dir
        )
        # Note: embed_tokens and lm_head are expected to be missing
        # as they come from the base Qwen model
        if unexpected:
            print(f"Warning: LLM unexpected keys: {unexpected}")

    @torch.inference_mode()
    def transcribe(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        prompt: str | None = None,
        max_new_tokens: int = 448,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform as numpy array or torch tensor.
                   Shape: (T,) for single audio or (B, T) for batch.
                   Values should be in range [-1, 1] (normalized).
            sample_rate: Sample rate of audio. Default: 16000.
                        Audio will be resampled if different.
            prompt: Optional custom prompt for the LLM.
            max_new_tokens: Maximum tokens to generate. Default: 448.

        Returns:
            Transcribed text.

        Note:
            Currently only supports batch_size=1.
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure 2D (batch, time)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if audio.shape[0] != 1:
            raise NotImplementedError("Batch transcription not yet supported")

        # Move to device
        audio = audio.to(self.device)

        # Resample if needed
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        # 1. Preprocess: audio → mel spectrogram
        mel, mel_lengths = self.preprocessor(audio)
        # mel: (B, 128, T_frames)

        # Convert to model dtype for encoder
        mel = mel.to(self.dtype)

        # 2. Encode: mel → audio features
        # Encoder expects (B, feat_in, T), which is what preprocessor outputs
        audio_features, feature_lengths = self.encoder(mel, mel_lengths)
        # audio_features: (B, T/8, 1024)

        # 3. Project: audio features → LLM embedding space
        audio_embeds = self.projection(audio_features)
        # audio_embeds: (B, T/8, 2048)

        # Convert to LLM dtype
        audio_embeds = audio_embeds.to(self.dtype)

        # 4. Generate: audio embeddings → text
        text = self.llm.generate(
            audio_embeds,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        return text

    def _resample(
        self,
        audio: torch.Tensor,
        orig_sr: int,
        target_sr: int,
    ) -> torch.Tensor:
        """Resample audio to target sample rate.

        Args:
            audio: Audio tensor of shape (B, T).
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio tensor.
        """
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=target_sr,
            ).to(audio.device)
            return resampler(audio)
        except ImportError:
            raise ImportError(
                "torchaudio is required for resampling. "
                "Install with: pip install torchaudio"
            )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> str:
        """Forward pass for transcription.

        This is an alias for transcribe() for nn.Module compatibility.

        Args:
            audio: Audio waveform of shape (1, T).
            audio_lengths: Optional lengths tensor (unused, for API compat).

        Returns:
            Transcribed text.
        """
        return self.transcribe(audio)
