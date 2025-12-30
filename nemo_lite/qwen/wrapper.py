"""Qwen LLM wrapper for Canary-Qwen-2.5B model.

Wraps Qwen3-1.7B with LoRA adapters for speech-to-text generation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


# Audio placeholder token used by Canary
AUDIO_PLACEHOLDER = "<|audioplaceholder|>"


class QwenWrapper(nn.Module):
    """Qwen3-1.7B wrapper with LoRA for audio-to-text generation.

    This wrapper:
    1. Loads Qwen3-1.7B base model
    2. Adds LoRA adapters (matching Canary-Qwen config)
    3. Adds audio placeholder token
    4. Provides methods for embedding injection and generation

    Args:
        device: Device to load model on ("cpu", "cuda", etc.)
        dtype: Model dtype (torch.float16, torch.bfloat16, etc.)
        cache_dir: Directory to cache downloaded models. If None, uses HuggingFace default.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-1.7B",
            cache_dir=cache_dir,
        )

        # Add audio placeholder token
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [AUDIO_PLACEHOLDER]
        })
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            dtype=dtype,
            device_map=device,
            cache_dir=cache_dir,
        )

        # Resize embeddings for new token
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Add LoRA adapters (matching Canary-Qwen config)
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.01,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.eval()

        # Clear sampling-related generation config to avoid warnings when using greedy decoding
        # The base Qwen model sets these, but we use do_sample=False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @property
    def embed_tokens(self) -> nn.Embedding:
        """Get the token embedding layer.

        Uses get_input_embeddings() for stability across HuggingFace/peft versions,
        rather than accessing the internal path (model.model.model.embed_tokens)
        which can change between library versions.
        """
        return self.model.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        """Get the hidden dimension."""
        return self.model.config.hidden_size

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (B, T).

        Returns:
            Embeddings of shape (B, T, hidden_size).
        """
        return self.embed_tokens(input_ids)

    def inject_audio_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace placeholder token with audio embeddings.

        Args:
            input_ids: Token IDs of shape (B, T_text).
            text_embeds: Text embeddings of shape (B, T_text, hidden).
            audio_embeds: Audio embeddings of shape (B, T_audio, hidden).

        Returns:
            Tuple of:
                - combined_embeds: (B, T_text - 1 + T_audio, hidden)
                - attention_mask: (B, T_text - 1 + T_audio)
        """
        batch_size = input_ids.shape[0]

        # Find placeholder positions
        placeholder_mask = (input_ids == self.placeholder_id)

        # For now, assume batch_size=1 and single placeholder per sequence
        # TODO: Support batched inference with multiple placeholders
        if batch_size != 1:
            raise NotImplementedError("Batched inference not yet supported")

        placeholder_pos = placeholder_mask.nonzero(as_tuple=True)[1]
        if len(placeholder_pos) != 1:
            raise ValueError(
                f"Expected exactly 1 placeholder, found {len(placeholder_pos)}"
            )

        pos = placeholder_pos[0].item()

        # Split text embeddings at placeholder position
        before = text_embeds[:, :pos, :]      # Before placeholder
        after = text_embeds[:, pos + 1:, :]   # After placeholder

        # Concatenate: [before] + [audio] + [after]
        combined = torch.cat([before, audio_embeds, after], dim=1)

        # Create attention mask (all 1s)
        attention_mask = torch.ones(
            batch_size, combined.shape[1],
            dtype=torch.long,
            device=combined.device,
        )

        return combined, attention_mask

    def generate(
        self,
        audio_embeds: torch.Tensor,
        prompt: str | None = None,
        max_new_tokens: int = 448,
        **generate_kwargs,
    ) -> str:
        """Generate transcription from audio embeddings.

        Args:
            audio_embeds: Audio embeddings of shape (1, T_audio, hidden).
            prompt: Optional custom prompt. If None, uses default transcription prompt.
            max_new_tokens: Maximum tokens to generate.
            **generate_kwargs: Additional arguments for model.generate().

        Returns:
            Generated transcription text.
        """
        if prompt is None:
            prompt = self.default_prompt()

        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(audio_embeds.device)

        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.get_text_embeddings(input_ids)

        # Inject audio embeddings
        combined_embeds, attention_mask = self.inject_audio_embeddings(
            input_ids, text_embeds, audio_embeds
        )

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Decode output
        # NOTE: When using inputs_embeds (not input_ids), HuggingFace generate()
        # returns ONLY the newly generated tokens, not the input. This is because
        # the input was embeddings, not token IDs, so there are no input IDs to
        # prepend to the output. Therefore, no prompt stripping is needed here.
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

    def default_prompt(self) -> str:
        """Get the default transcription prompt."""
        return f"""<|im_start|>user
Transcribe the following audio:{AUDIO_PLACEHOLDER}<|im_end|>
<|im_start|>assistant
"""

    def forward(
        self,
        audio_embeds: torch.Tensor,
        prompt: str | None = None,
    ) -> str:
        """Forward pass for transcription.

        Args:
            audio_embeds: Audio embeddings of shape (1, T_audio, hidden).
            prompt: Optional custom prompt.

        Returns:
            Transcription text.
        """
        return self.generate(audio_embeds, prompt)
