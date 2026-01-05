"""Type stubs for strops - Rust extension module for string operations."""

def merge_transcripts(prev: list[str], new: list[str]) -> list[str]:
    """Merge two word sequences using semi-global alignment.

    Given a previous transcript and a new transcript (from overlapped audio),
    find the optimal alignment and merge them by keeping prev's non-overlapping
    prefix and all of new.

    Args:
        prev: The previous transcript as a list of words.
        new: The new transcript as a list of words.

    Returns:
        The merged transcript as a list of words.

    Example:
        >>> merge_transcripts(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
        ["The", "quick", "brown", "fox", "jumps"]
    """
    ...
