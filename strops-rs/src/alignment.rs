/// Result of semi-global alignment between two word sequences.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The index in `prev` where the overlap starts.
    /// prev[:overlap_start] is the non-overlapping prefix to keep.
    pub overlap_start: usize,
    /// The alignment score.
    pub score: i32,
}

/// Perform semi-global alignment to find where a suffix of `prev` aligns with a prefix of `new`.
///
/// This implements the DP-based semi-global alignment algorithm described in DESIGN.md:
/// - Allows free gaps at the start of `prev` (we only care about matching a suffix)
/// - Penalizes gaps elsewhere (insertions/deletions in the overlap region)
/// - Finds the optimal prefix of `new` that aligns with a suffix of `prev`
///
/// # Arguments
/// * `prev` - The previous transcript as a slice of words
/// * `new` - The new transcript as a slice of words
///
/// # Returns
/// An `AlignmentResult` containing the overlap boundary and score.
pub fn semi_global_align(prev: &[String], new: &[String]) -> AlignmentResult {
    // TODO: Implement the semi-global alignment algorithm
    // For now, return a placeholder that assumes no overlap
    AlignmentResult {
        overlap_start: prev.len(),
        score: 0,
    }
}

/// Merge two word sequences using semi-global alignment.
///
/// Given a previous transcript and a new transcript (from overlapped audio),
/// find the optimal alignment and merge them by keeping prev's non-overlapping
/// prefix and all of new.
///
/// # Arguments
/// * `prev` - The previous transcript as a slice of words
/// * `new` - The new transcript as a slice of words
///
/// # Returns
/// The merged transcript as a vector of words.
///
/// # Example
/// ```
/// use strops::alignment::merge_transcripts;
///
/// let prev = vec!["The", "quick", "brown", "fox", "jumps"]
///     .into_iter().map(String::from).collect::<Vec<_>>();
/// let new = vec!["brown", "fox", "jumps", "over", "the", "lazy", "dog"]
///     .into_iter().map(String::from).collect::<Vec<_>>();
///
/// let merged = merge_transcripts(&prev, &new);
/// // Expected: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
/// ```
pub fn merge_transcripts(prev: &[String], new: &[String]) -> Vec<String> {
    // TODO: Implement using semi_global_align
    // For now, return a placeholder that just concatenates
    let alignment = semi_global_align(prev, new);
    let mut result: Vec<String> = prev[..alignment.overlap_start].to_vec();
    result.extend(new.iter().cloned());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_no_overlap() {
        let prev: Vec<String> = vec!["hello".to_string(), "world".to_string()];
        let new: Vec<String> = vec!["foo".to_string(), "bar".to_string()];
        let result = merge_transcripts(&prev, &new);
        // With no overlap, should concatenate
        assert_eq!(result, vec!["hello", "world", "foo", "bar"]);
    }

    // TODO: Add more tests once the algorithm is implemented
}
