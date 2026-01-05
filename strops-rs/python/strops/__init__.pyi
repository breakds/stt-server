"""Type stubs for strops - Rust extension module for string operations."""

def merge_by_overlap(prev: list[str], new: list[str]) -> list[str]:
    """Merge two token sequences by finding their overlap.

    Uses semi-global alignment to find where the suffix of `prev` overlaps
    with the prefix of `new`, then merges them by keeping prev's non-overlapping
    prefix followed by all of new.

    Args:
        prev: The previous sequence of tokens.
        new: The new sequence of tokens.

    Returns:
        The merged sequence. If no overlap is found, returns prev + new (concatenation).

    Example:
        >>> merge_by_overlap(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
        ["The", "quick", "brown", "fox", "jumps"]
    """
    ...
