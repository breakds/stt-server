"""Weight loading utilities for Canary-Qwen model.

Loads and maps weights from Canary-Qwen checkpoints to our implementation.

The Canary-Qwen-2.5B weights are available on HuggingFace Hub at:
    nvidia/canary-qwen-2.5b

Weight format: safetensors (4.77 GB total)

The checkpoint uses these prefixes:
    - perception.encoder.* (FastConformer encoder)
    - perception.proj.* (audio projection layer)
    - perception.preprocessor.* (mel filterbank, not loaded)
    - llm.* (Qwen LLM with LoRA)

Usage:
    from nemo_lite.weights import load_encoder_weights, load_projection_weights
    from nemo_lite.conformer_lite import FastConformerEncoder
    from nemo_lite.projection import AudioProjection

    encoder = FastConformerEncoder(**get_encoder_config())
    load_encoder_weights(encoder, "nvidia/canary-qwen-2.5b")

    projection = AudioProjection()
    load_projection_weights(projection, "nvidia/canary-qwen-2.5b")
"""

import re

import torch
from safetensors import safe_open


# Prefix for encoder weights in the checkpoint
_ENCODER_PREFIX = "perception.encoder."

# Mapping from checkpoint subsampling indices to our module names
# Checkpoint uses nn.Sequential with indices; we use named modules
_SUBSAMPLING_MAP: dict[str, str] = {
    "conv.0": "conv1",  # Regular Conv2d
    "conv.2": "dwconv2",  # Depthwise Conv2d (layer 2)
    "conv.3": "pwconv2",  # Pointwise Conv2d (layer 2)
    "conv.5": "dwconv3",  # Depthwise Conv2d (layer 3)
    "conv.6": "pwconv3",  # Pointwise Conv2d (layer 3)
    # "out" stays the same
}


def _map_subsampling_key(key: str) -> str:
    """Map subsampling weight key to our key.

    Example:
        "pre_encode.conv.0.weight" -> "pre_encode.conv1.weight"
    """
    # Map conv indices to named modules
    for ckpt_name, our_name in _SUBSAMPLING_MAP.items():
        pattern = f"pre_encode.{ckpt_name}."
        if pattern in key:
            return key.replace(pattern, f"pre_encode.{our_name}.")

    return key


def _map_layer_key(key: str) -> str:
    """Map conformer layer weight key to our key.

    The checkpoint uses "conv" for the convolution module, but our
    implementation uses "conv_module" to match the NeMo naming convention.

    Example:
        "layers.0.conv.batch_norm.weight" -> "layers.0.conv_module.batch_norm.weight"
        "layers.0.self_attn.linear_q.weight" -> "layers.0.self_attn.linear_q.weight"
    """
    # Map "conv." to "conv_module." for the convolution module in each layer
    # Be careful not to match "conv" in other contexts
    # Pattern: layers.{i}.conv. -> layers.{i}.conv_module.
    key = re.sub(r"(layers\.\d+)\.conv\.", r"\1.conv_module.", key)
    return key


def map_weight_key(ckpt_key: str) -> str | None:
    """Map a checkpoint weight key to our implementation's key.

    Args:
        ckpt_key: Weight key from checkpoint.

    Returns:
        Mapped key for our implementation, or None if not an encoder weight.
    """
    if not ckpt_key.startswith(_ENCODER_PREFIX):
        return None

    # Remove the "perception.encoder." prefix
    key = ckpt_key[len(_ENCODER_PREFIX):]

    if "pre_encode" in key:
        return _map_subsampling_key(key)

    if "layers" in key:
        return _map_layer_key(key)

    if "pos_enc" in key:
        # Positional encoding has no learnable parameters
        # (sinusoidal embeddings are buffers, not loaded here)
        return key

    return None


def load_encoder_state_dict(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load encoder weights from a safetensors checkpoint.

    Args:
        checkpoint_path: Path to safetensors file.
        device: Device to load weights to.

    Returns:
        State dict with mapped keys for FastConformerEncoder.
    """
    state_dict: dict[str, torch.Tensor] = {}

    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for nemo_key in f.keys():
            our_key = map_weight_key(nemo_key)
            if our_key is not None:
                state_dict[our_key] = f.get_tensor(nemo_key)

    return state_dict


def load_encoder_state_dict_from_hub(
    repo_id: str = "nvidia/canary-qwen-2.5b",
    device: str = "cpu",
    token: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load encoder weights from HuggingFace Hub.

    Downloads the safetensors file and extracts encoder weights.

    Args:
        repo_id: HuggingFace Hub repository ID.
        device: Device to load weights to.
        token: Optional HuggingFace token for private repos.

    Returns:
        State dict with mapped keys for FastConformerEncoder.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download weights from the Hub. "
            "Install with: pip install huggingface_hub"
        ) from e

    # The model uses safetensors format with sharded files
    # We need to find and download all shard files
    try:
        from huggingface_hub import list_repo_files
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    files = list_repo_files(repo_id, token=token)
    safetensor_files = [f for f in files if f.endswith(".safetensors")]

    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {repo_id}")

    state_dict: dict[str, torch.Tensor] = {}

    for filename in safetensor_files:
        local_path = hf_hub_download(
            repo_id,
            filename,
            token=token,
        )
        with safe_open(local_path, framework="pt", device=device) as f:
            for nemo_key in f.keys():
                our_key = map_weight_key(nemo_key)
                if our_key is not None:
                    state_dict[our_key] = f.get_tensor(nemo_key)

    return state_dict


def load_encoder_weights(
    encoder: torch.nn.Module,
    source: str,
    device: str = "cpu",
    token: str | None = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load weights into a FastConformerEncoder.

    Args:
        encoder: FastConformerEncoder instance.
        source: Either a local path to safetensors file, or a HuggingFace Hub repo ID.
        device: Device to load weights to.
        token: Optional HuggingFace token for private repos.
        strict: Whether to require all weights to match.

    Returns:
        Tuple of (missing_keys, unexpected_keys) from load_state_dict.

    Raises:
        RuntimeError: If strict=True and there are missing/unexpected keys.
    """
    if source.endswith(".safetensors"):
        # Local file path
        state_dict = load_encoder_state_dict(source, device=device)
    else:
        # HuggingFace Hub repo ID
        state_dict = load_encoder_state_dict_from_hub(
            source, device=device, token=token
        )

    result = encoder.load_state_dict(state_dict, strict=strict)
    return result.missing_keys, result.unexpected_keys


def get_encoder_config(repo_id: str = "nvidia/canary-qwen-2.5b") -> dict:
    """Get encoder configuration from model config.

    Returns a dict with the hyperparameters needed to instantiate
    FastConformerEncoder with the correct architecture.

    Args:
        repo_id: HuggingFace Hub repository ID.

    Returns:
        Configuration dict with keys:
            feat_in, n_layers, d_model, d_ff, n_heads,
            conv_kernel_size, subsampling_factor, subsampling_conv_channels,
            dropout_rate, dropout_att
    """
    # Hardcoded for Canary-Qwen-2.5B
    # These values are from the model's config file
    return {
        "feat_in": 128,
        "n_layers": 32,
        "d_model": 1024,
        "d_ff": 4096,
        "n_heads": 8,
        "conv_kernel_size": 9,
        "subsampling_factor": 8,
        "subsampling_conv_channels": 256,
        "dropout_rate": 0.1,
        "dropout_att": 0.1,
    }


# =============================================================================
# Projection Layer Weight Loading
# =============================================================================

_PROJECTION_PREFIX = "perception.proj."


def map_projection_weight_key(ckpt_key: str) -> str | None:
    """Map a checkpoint projection weight key to our implementation's key.

    Args:
        ckpt_key: Weight key from checkpoint.

    Returns:
        Mapped key for our implementation, or None if not a projection weight.
    """
    if not ckpt_key.startswith(_PROJECTION_PREFIX):
        return None

    # perception.proj.weight -> proj.weight
    # perception.proj.bias -> proj.bias
    return ckpt_key[len("perception."):]


def load_projection_state_dict(
    checkpoint_path: str,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load projection layer weights from a safetensors checkpoint.

    Args:
        checkpoint_path: Path to safetensors file.
        device: Device to load weights to.

    Returns:
        State dict with mapped keys for AudioProjection.
    """
    state_dict: dict[str, torch.Tensor] = {}

    with safe_open(checkpoint_path, framework="pt", device=device) as f:
        for ckpt_key in f.keys():
            our_key = map_projection_weight_key(ckpt_key)
            if our_key is not None:
                state_dict[our_key] = f.get_tensor(ckpt_key)

    return state_dict


def load_projection_state_dict_from_hub(
    repo_id: str = "nvidia/canary-qwen-2.5b",
    device: str = "cpu",
    token: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load projection layer weights from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub repository ID.
        device: Device to load weights to.
        token: Optional HuggingFace token for private repos.

    Returns:
        State dict with mapped keys for AudioProjection.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    files = list_repo_files(repo_id, token=token)
    safetensor_files = [f for f in files if f.endswith(".safetensors")]

    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {repo_id}")

    state_dict: dict[str, torch.Tensor] = {}

    for filename in safetensor_files:
        local_path = hf_hub_download(repo_id, filename, token=token)
        with safe_open(local_path, framework="pt", device=device) as f:
            for ckpt_key in f.keys():
                our_key = map_projection_weight_key(ckpt_key)
                if our_key is not None:
                    state_dict[our_key] = f.get_tensor(ckpt_key)

    return state_dict


def load_projection_weights(
    projection: torch.nn.Module,
    source: str,
    device: str = "cpu",
    token: str | None = None,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load weights into an AudioProjection module.

    Args:
        projection: AudioProjection instance.
        source: Either a local path to safetensors file, or a HuggingFace Hub repo ID.
        device: Device to load weights to.
        token: Optional HuggingFace token for private repos.
        strict: Whether to require all weights to match.

    Returns:
        Tuple of (missing_keys, unexpected_keys) from load_state_dict.
    """
    if source.endswith(".safetensors"):
        state_dict = load_projection_state_dict(source, device=device)
    else:
        state_dict = load_projection_state_dict_from_hub(
            source, device=device, token=token
        )

    result = projection.load_state_dict(state_dict, strict=strict)
    return result.missing_keys, result.unexpected_keys
