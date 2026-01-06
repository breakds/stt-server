from pydantic import Base64Bytes, BaseModel, ConfigDict


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(word.capitalize() for word in components[1:])


# Shared configuration for camelCase JSON serialization
CAMEL_CASE_CONFIG = ConfigDict(
    alias_generator=snake_to_camel,
    serialize_by_alias=True,
    populate_by_name=True,
)


class Protocol(BaseModel):
    """Base class for all protocol messages with camelCase JSON serialization."""

    model_config = CAMEL_CASE_CONFIG


# =============================================================================
# Client -> Server Messages
# =============================================================================


class AudioFrame(Protocol):
    """Audio frame sent from client to server.

    Contains a chunk of PCM audio data. The client sends these continuously
    in a streaming fashion.

    Attributes:
        samples: Raw PCM audio bytes (16-bit signed, base64-encoded in JSON).
        sample_rate: Sample rate in Hz (e.g., 16000).
        channels: Number of audio channels (typically 1 for mono).
    """

    samples: Base64Bytes
    sample_rate: int
    channels: int


# =============================================================================
# Server -> Client Messages
# =============================================================================


class TranscriptionSegment(Protocol):
    """Transcription segment streamed from server to client.

    Segments come in two types:
    - Tentative (is_final=False): Partial result that may change as more audio
      arrives. Tentative segments accumulate until a final segment arrives.
    - Final (is_final=True): Confirmed result that won't change. Replaces all
      accumulated tentative segments.

    Attributes:
        text: The transcribed text for this segment.
        is_final: Whether this segment is final (True) or tentative (False).
        is_end_of_turn: Whether this segment marks the end of a speaking turn.
            Signals the conversational agent that it's time to respond.
    """

    text: str
    is_final: bool = False
    is_end_of_turn: bool = False


class ErrorResponse(Protocol):
    """Error response from server to client.

    Attributes:
        error: Human-readable error message.
        code: Machine-readable error code.
    """

    error: str
    code: str = "TRANSCRIPTION_ERROR"


# =============================================================================
# Inter-Stage Types (internal pipeline communication)
# =============================================================================


class AudioChunk(BaseModel):
    """Accumulated audio buffer passed from VAD to ASR stage.

    Contains a chunk of audio that has been segmented by the VAD stage
    based on silence gaps.

    Attributes:
        samples: Raw PCM audio bytes (16-bit signed, mono).
        sample_rate: Sample rate in Hz.
    """

    samples: bytes
    sample_rate: int


class EndOfTurnSignal(BaseModel):
    """Signal indicating end of a speaking turn.

    Emitted by VAD stage when a large silence gap is detected,
    signaling that the speaker has finished their turn.
    """

    pass

