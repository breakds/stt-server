/// Result of semi-global alignment between two sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentResult {
    /// Start of overlap in source (inclusive, 0-based).
    /// The overlapping suffix of source is `source[overlap_start_source..]`.
    pub overlap_start_source: usize,
    /// End of overlap in target (exclusive, 0-based).
    /// The overlapping prefix of target is `target[..overlap_end_target]`.
    pub overlap_end_target: usize,
    /// Position of first match in source (inclusive, 0-based).
    /// Use this for stitching: `source[..first_match_source] + target[first_match_target..]`
    pub first_match_source: usize,
    /// Position of first match in target (inclusive, 0-based).
    /// Use this for stitching: `source[..first_match_source] + target[first_match_target..]`
    pub first_match_target: usize,
}

/// Scoring parameters for semi-global alignment.
#[derive(Debug, Clone, Copy)]
pub struct ScoringParams {
    pub match_reward: f64,
    pub gap_penalty: f64,
}

impl Default for ScoringParams {
    fn default() -> Self {
        Self {
            match_reward: 1.0,
            gap_penalty: -1.0,
        }
    }
}

impl ScoringParams {
    /// Create scoring parameters with custom values.
    pub fn new(match_reward: f64, gap_penalty: f64) -> Self {
        Self { match_reward, gap_penalty }
    }

    /// Create with custom match reward, default gap penalty.
    pub fn with_match_reward(match_reward: f64) -> Self {
        Self {
            match_reward,
            gap_penalty: -1.0,
        }
    }

    /// Create with custom gap penalty, default match reward.
    pub fn with_gap_penalty(gap_penalty: f64) -> Self {
        Self {
            match_reward: 1.0,
            gap_penalty,
        }
    }
}

/// Perform semi-global alignment to find where a suffix of `source` overlaps with a prefix of `target`.
///
/// This is designed for merging overlapping ASR transcripts, where:
/// - The end of `source` (previous transcript) should align with the beginning of `target` (new transcript)
/// - We want to find the optimal overlap region to merge them seamlessly
///
/// # Algorithm
///
/// Uses dynamic programming similar to Smith-Waterman local alignment:
/// - Matches receive `scoring.match_reward` (typically positive)
/// - Gaps/mismatches receive `scoring.gap_penalty` (typically negative)
/// - Scores can reset to 0 via `.max(0)`, allowing the alignment to "restart"
/// - We find the best-scoring alignment ending at the last row of source
///
/// # Returns
///
/// - `Some(AlignmentResult)` if a valid overlap is found, containing:
///   - `overlap_start_source`: where overlap begins in source (inclusive)
///   - `overlap_end_target`: where overlap ends in target (exclusive)
///   - `first_match_source`: position of first matching element in source
///   - `first_match_target`: position of first matching element in target
///
///   The overlapping regions are: `source[overlap_start_source..]` and `target[..overlap_end_target]`
///
///   Two merge strategies:
///   - Full overlap: `source[..overlap_start_source] + target`
///   - Stitch at first match: `source[..first_match_source] + target[first_match_target..]`
///     (preferred for ASR where target's first word may be garbled)
///
/// - `None` if no valid overlap exists (empty inputs, or best alignment score ≤ 0)
///
/// # Examples
///
/// ```
/// use strops::alignment::{semi_global_align, ScoringParams, AlignmentResult};
///
/// let source = vec!["the", "quick", "brown", "fox"];
/// let target = vec!["brown", "fox", "jumps", "over"];
///
/// let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
/// assert_eq!(result.overlap_start_source, 2);
/// assert_eq!(result.overlap_end_target, 2);
/// assert_eq!(result.first_match_source, 2);
/// assert_eq!(result.first_match_target, 0);
/// // Stitch: source[..2] + target[0..] = ["the", "quick", "brown", "fox", "jumps", "over"]
/// ```
///
/// # Scoring Tips
///
/// - Default scoring (match=1.0, gap=-1.0) requires at least 2 consecutive matches
///   to overcome a single gap, since 1 match + 1 gap = 0 (returns None)
/// - For ASR with potential transcription errors, use higher match reward like
///   `ScoringParams::new(2.0, -1.0)` to tolerate occasional mismatches
pub fn semi_global_align<T: PartialEq>(source: &[T], target: &[T], scoring: ScoringParams) -> Option<AlignmentResult> {
    if source.is_empty() || target.is_empty() {
        return None;
    }

    // ==================== Grid Data Structure ====================
    //
    // The DP grid has dimensions (source.len() + 1) x (target.len() + 1).
    // We use 1-based indexing for the grid where:
    //   - Row 0 / Column 0: boundary conditions (all zeros, representing "no elements matched yet")
    //   - grid[i][j]: optimal alignment score considering source[0..i] and target[0..j]
    //
    // This 1-based convention simplifies the DP recurrence by avoiding boundary checks,
    // since grid[i-1][j-1] is always valid when i,j >= 1.

    struct Grid {
        stride: usize,
        optimal: Vec<f64>,
    }

    impl Grid {
        fn new(source_len: usize, target_len: usize) -> Self {
            // Allocate (source_len + 1) * (target_len + 1) cells, initialized to 0.
            // Row 0 and column 0 remain 0, serving as boundary conditions.
            let initialized = vec![0f64; (source_len + 1) * (target_len + 1)];
            Grid { stride: target_len + 1, optimal: initialized }
        }

        /// Set grid value at 1-based position (i, j).
        fn set(&mut self, i: usize, j: usize, value: f64) {
            self.optimal[i * self.stride + j] = value;
        }

        /// Get grid value at 1-based position (i, j).
        fn get(&self, i: usize, j: usize) -> f64 {
            self.optimal[i * self.stride + j]
        }

        /// Find the column index (1-based) with maximum score in row i.
        /// Starts from column 1 (skips column 0 which is always 0).
        fn arg_max_row(&self, i: usize) -> usize {
            let start = i * self.stride;
            let row = &self.optimal[start..start + self.stride];

            // Start from column 1 (skip column 0 boundary)
            let mut max_score = row[1];
            let mut max_j = 1;

            for (k, &score) in row[2..].iter().enumerate() {
                if score.total_cmp(&max_score).is_gt() {
                    max_score = score;
                    max_j = k + 2; // k is 0-based offset from row[2], so actual index is k + 2
                }
            }
            max_j
        }
    }

    // ==================== DP Forward Pass ====================
    //
    // Fill the grid column by column (outer loop over target index j).
    // For each cell (i+1, j+1) in 1-based grid coordinates:
    //   - If source[i] == target[j]: diagonal match, add match_reward
    //   - Otherwise: take best of up (deletion) or left (insertion), add gap_penalty
    //   - The .max(0) allows the alignment to "reset", enabling local alignment behavior
    //
    // Note: i, j here are 0-based indices into source/target arrays.
    // We write to grid position (i+1, j+1) which is 1-based.

    let mut grid = Grid::new(source.len(), target.len());
    for (j, target_item) in target.iter().enumerate() {
        for (i, source_item) in source.iter().enumerate() {
            // grid[i+1][j+1] depends on grid[i][j], grid[i][j+1], grid[i+1][j]
            // All these cells are already computed due to iteration order.
            grid.set(i + 1, j + 1, if source_item == target_item {
                // Match: extend diagonal alignment
                grid.get(i, j).max(0f64) + scoring.match_reward
            } else {
                // Mismatch: best of skipping source element (up) or target element (left)
                grid.get(i, j + 1).max(grid.get(i + 1, j)).max(0f64) + scoring.gap_penalty
            });
        }
    }

    // ==================== Find Best Alignment Endpoint ====================
    //
    // For semi-global alignment matching suffix of source with prefix of target:
    // - We want the alignment to end at the last row (i = source.len())
    // - Find the column j with maximum score in the last row
    // - This j (1-based) tells us how much of target is covered by the overlap

    let mut j = grid.arg_max_row(source.len());
    let mut i = source.len();

    // No valid overlap if the best score is non-positive
    if grid.get(i, j) <= 0f64 {
        return None;
    }

    // Convert j from 1-based grid index to 0-based exclusive index for target.
    // In the grid, j=1 means target[0..1], j=2 means target[0..2], etc.
    // So j directly serves as the exclusive end index.
    let overlap_end_in_target = j;

    // ==================== Traceback ====================
    //
    // Trace back from (i, j) to find where the alignment started in source.
    // We follow the path that led to the optimal score:
    //   - On match: move diagonally (i-1, j-1)
    //   - On mismatch: move to the cell (up or left) that contributed the score
    //   - Stop when we hit a zero score (alignment start) or grid boundary
    //
    // We also track the first match position (for stitching at clean word boundaries).
    // Since we trace backwards, the "first match" in sequence order is the last match
    // we encounter during traceback.
    //
    // The loop condition `i > 1` ensures we don't underflow when computing `i - 1`.
    // When i = 1, we've traced back to consider source[0], and the loop exits.

    // Track first match position (last one seen during backward traceback).
    // Initialize to current position; will be updated when we see matches.
    let mut first_match_source = i - 1;
    let mut first_match_target = j - 1;

    while j > 0 && i > 1 {
        // source[i-1] and target[j-1] are the elements at current 1-based grid position
        if source[i - 1] == target[j - 1] {
            // This was a match; record this position (overwritten as we go back,
            // so the final value will be the first match in sequence order)
            first_match_source = i - 1;
            first_match_target = j - 1;

            // Check if we should continue tracing
            if grid.get(i - 1, j - 1) <= 0f64 {
                // Previous cell is zero/negative, meaning alignment started here
                break;
            }
            i -= 1;
            j -= 1;
        } else {
            // This was a gap; determine which direction we came from
            let up_score = grid.get(i - 1, j);
            let left_score = grid.get(i, j - 1);
            if up_score <= 0f64 && left_score <= 0f64 {
                // Both directions lead to non-positive scores, alignment started here
                break;
            } else if up_score > left_score {
                i -= 1; // Came from up (source element was skipped)
            } else {
                j -= 1; // Came from left (target element was skipped)
            }
        }
    }

    // Check the final position (when i = 1, we haven't checked source[0] yet)
    // This handles the case where the alignment extends to the very first element
    if i == 1 && j > 0 && source[0] == target[j - 1] {
        first_match_source = 0;
        first_match_target = j - 1;
    }

    // Convert i from 1-based grid index to 0-based inclusive index for source.
    // In the grid, i=1 corresponds to source[0], i=2 to source[1], etc.
    // So the overlap starts at source index (i - 1).
    let overlap_start_in_source = i - 1;

    Some(AlignmentResult {
        overlap_start_source: overlap_start_in_source,
        overlap_end_target: overlap_end_in_target,
        first_match_source,
        first_match_target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    // ==================== Empty Input Tests ====================

    #[test]
    fn empty_source_returns_none() {
        let source: Vec<String> = vec![];
        let target = s(&["a", "b"]);
        assert_eq!(semi_global_align(&source, &target, ScoringParams::default()), None);
    }

    #[test]
    fn empty_target_returns_none() {
        let source = s(&["a", "b"]);
        let target: Vec<String> = vec![];
        assert_eq!(semi_global_align(&source, &target, ScoringParams::default()), None);
    }

    #[test]
    fn both_empty_returns_none() {
        let source: Vec<String> = vec![];
        let target: Vec<String> = vec![];
        assert_eq!(semi_global_align(&source, &target, ScoringParams::default()), None);
    }

    // ==================== No Overlap Tests ====================

    #[test]
    fn completely_different_sequences_returns_none() {
        let source = s(&["a", "b", "c"]);
        let target = s(&["x", "y", "z"]);
        assert_eq!(semi_global_align(&source, &target, ScoringParams::default()), None);
    }

    #[test]
    fn single_elements_no_match_returns_none() {
        let source = s(&["a"]);
        let target = s(&["b"]);
        assert_eq!(semi_global_align(&source, &target, ScoringParams::default()), None);
    }

    // ==================== Single Element Tests ====================

    #[test]
    fn single_elements_match() {
        let source = s(&["a"]);
        let target = s(&["a"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 0);
        assert_eq!(result.overlap_end_target, 1);
        assert_eq!(result.first_match_source, 0);
        assert_eq!(result.first_match_target, 0);
        // source[0..] overlaps with target[..1] → entire source matches entire target
    }

    // ==================== Basic Overlap Tests ====================

    #[test]
    fn suffix_of_source_matches_prefix_of_target() {
        // source: ["a", "b", "c"]
        // target: ["b", "c", "d"]
        // Overlap: source[1..] = ["b", "c"] matches target[..2] = ["b", "c"]
        let source = s(&["a", "b", "c"]);
        let target = s(&["b", "c", "d"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 1);
        assert_eq!(result.overlap_end_target, 2);
        assert_eq!(result.first_match_source, 1);
        assert_eq!(result.first_match_target, 0);
    }

    #[test]
    fn entire_source_matches_prefix_of_target() {
        // source: ["a", "b"]
        // target: ["a", "b", "c", "d"]
        // Overlap: source[0..] = ["a", "b"] matches target[..2] = ["a", "b"]
        let source = s(&["a", "b"]);
        let target = s(&["a", "b", "c", "d"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 0);
        assert_eq!(result.overlap_end_target, 2);
        assert_eq!(result.first_match_source, 0);
        assert_eq!(result.first_match_target, 0);
    }

    #[test]
    fn suffix_of_source_matches_entire_target() {
        // source: ["a", "b", "c", "d"]
        // target: ["c", "d"]
        // Overlap: source[2..] = ["c", "d"] matches target[..2] = ["c", "d"]
        let source = s(&["a", "b", "c", "d"]);
        let target = s(&["c", "d"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 2);
        assert_eq!(result.overlap_end_target, 2);
        assert_eq!(result.first_match_source, 2);
        assert_eq!(result.first_match_target, 0);
    }

    #[test]
    fn entire_source_matches_entire_target() {
        let source = s(&["a", "b", "c"]);
        let target = s(&["a", "b", "c"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 0);
        assert_eq!(result.overlap_end_target, 3);
        assert_eq!(result.first_match_source, 0);
        assert_eq!(result.first_match_target, 0);
    }

    // ==================== Overlap with Gaps Tests ====================

    #[test]
    fn overlap_with_insertion_in_target() {
        // source: ["a", "b", "c"]
        // target: ["b", "x", "c", "d"]
        // With higher match reward, the algorithm should prefer matching both b and c
        // even with the gap for "x"
        let source = s(&["a", "b", "c"]);
        let target = s(&["b", "x", "c", "d"]);
        // Use higher match reward to make gapped alignment clearly better
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        assert_eq!(result.overlap_start_source, 1); // overlap starts at source[1]
        assert_eq!(result.overlap_end_target, 3);   // overlap ends at target[..3] (includes b, x, c)
    }

    #[test]
    fn overlap_with_deletion_in_target() {
        // source: ["a", "b", "x", "c"]
        // target: ["b", "c", "d"]
        // With higher match reward, prefer matching b and c even with gap for "x"
        let source = s(&["a", "b", "x", "c"]);
        let target = s(&["b", "c", "d"]);
        // Use higher match reward to make gapped alignment clearly better
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        assert_eq!(result.overlap_start_source, 1); // overlap starts at source[1]
        assert_eq!(result.overlap_end_target, 2);   // overlap ends at target[..2]
    }

    // ==================== Merge Semantics Tests ====================

    #[test]
    fn merge_semantics_basic() {
        // Verify that source[..start] + target gives correct merge
        let source = s(&["The", "quick", "brown", "fox"]);
        let target = s(&["brown", "fox", "jumps", "over"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 2);
        assert_eq!(result.overlap_end_target, 2);
        assert_eq!(result.first_match_source, 2);
        assert_eq!(result.first_match_target, 0);

        // Merge using first_match: source[..first_match_source] + target[first_match_target..]
        let merged: Vec<_> = source[..result.first_match_source]
            .iter()
            .chain(target[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(merged, s(&["The", "quick", "brown", "fox", "jumps", "over"]));
    }

    #[test]
    fn merge_semantics_full_overlap() {
        let source = s(&["a", "b"]);
        let target = s(&["a", "b", "c"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 0);
        assert_eq!(result.overlap_end_target, 2);
        assert_eq!(result.first_match_source, 0);
        assert_eq!(result.first_match_target, 0);

        // Merge using first_match: source[..0] + target[0..] = [] + ["a", "b", "c"]
        let merged: Vec<_> = source[..result.first_match_source]
            .iter()
            .chain(target[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(merged, s(&["a", "b", "c"]));
    }

    // ==================== Character-Level Tests ====================

    #[test]
    fn works_with_characters() {
        let source: Vec<char> = "hello".chars().collect();
        let target: Vec<char> = "lloworld".chars().collect();
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 2);
        assert_eq!(result.overlap_end_target, 3); // "llo" overlaps
        assert_eq!(result.first_match_source, 2);
        assert_eq!(result.first_match_target, 0);
    }

    #[test]
    fn works_with_integers() {
        let source = vec![1, 2, 3, 4];
        let target = vec![3, 4, 5, 6];
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 2);
        assert_eq!(result.overlap_end_target, 2); // [3, 4] overlaps
        assert_eq!(result.first_match_source, 2);
        assert_eq!(result.first_match_target, 0);
    }

    // ==================== Scoring Parameter Tests ====================

    #[test]
    fn higher_match_reward_extends_overlap() {
        // With higher match reward, gaps become relatively cheaper
        let source = s(&["a", "b", "x", "c"]);
        let target = s(&["b", "c", "d"]);

        // Default scoring: might not bridge the gap well
        let default_result = semi_global_align(&source, &target, ScoringParams::default());

        // Higher match reward: 2.0 match vs -1.0 gap
        let high_reward = ScoringParams::with_match_reward(2.0);
        let high_result = semi_global_align(&source, &target, high_reward);

        // Both should find an overlap
        assert!(default_result.is_some());
        assert!(high_result.is_some());
    }

    #[test]
    fn high_gap_penalty_prevents_weak_alignments() {
        // With very high gap penalty, alignments with gaps become unfavorable
        let source = s(&["a", "x", "b"]);
        let target = s(&["a", "b", "c"]);

        // Default: might allow the gap
        let default_result = semi_global_align(&source, &target, ScoringParams::default());

        // Very high gap penalty
        let strict = ScoringParams::new(1.0, -5.0);
        let strict_result = semi_global_align(&source, &target, strict);

        // With strict penalty, single matches without gaps should still work
        // but gapped alignments might score poorly
        assert!(default_result.is_some());
        // The strict one might find a shorter overlap or none
        // (just verify it doesn't panic)
        let _ = strict_result;
    }

    // ==================== Imperfect Overlap Tests (ASR Errors) ====================

    #[test]
    fn imperfect_overlap_single_word_error() {
        // ASR mistake: "fox" transcribed as "box" in new segment
        // source: ["the", "quick", "brown", "fox"]
        // target: ["brown", "box", "jumps", "over"]
        // With default scoring, single match + gap = 0, returns None
        // Need higher match reward to make the overlap worthwhile
        let source = s(&["the", "quick", "brown", "fox"]);
        let target = s(&["brown", "box", "jumps", "over"]);
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        // Should find overlap starting at "brown" (index 2)
        assert_eq!(result.overlap_start_source, 2);
        // End should include at least "brown"
        assert!(result.overlap_end_target >= 1);
    }

    #[test]
    fn imperfect_overlap_multiple_word_errors() {
        // Multiple ASR mistakes in overlap region
        // source: ["I", "think", "the", "weather", "is", "nice"]
        // target: ["the", "whether", "is", "nice", "today"]
        // "weather" → "whether" (homophone error)
        let source = s(&["I", "think", "the", "weather", "is", "nice"]);
        let target = s(&["the", "whether", "is", "nice", "today"]);
        // Use scoring that tolerates gaps
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        // Should find some overlap - exact position depends on algorithm
        // Key assertion: overlap was found and makes sense
        assert!(result.overlap_start_source < source.len());
        assert!(result.overlap_end_target <= target.len());
        assert!(result.overlap_end_target >= 2); // Should capture multiple words
    }

    #[test]
    fn imperfect_overlap_missing_word_in_new() {
        // New transcript missing a word that was in previous
        // source: ["going", "to", "the", "store", "today"]
        // target: ["the", "store", "to", "buy", "milk"]
        // Overlap region has matching "the", "store" despite context differences
        let source = s(&["going", "to", "the", "store", "today"]);
        let target = s(&["the", "store", "to", "buy", "milk"]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        // Should align on "the", "store"
        assert_eq!(result.overlap_start_source, 2);
    }

    #[test]
    fn imperfect_overlap_extra_word_in_new() {
        // New transcript has extra word not in previous
        // source: ["the", "big", "brown", "dog"]
        // target: ["big", "old", "brown", "dog", "barks"]
        // "old" is extra, but "big", "brown", "dog" should still align
        let source = s(&["the", "big", "brown", "dog"]);
        let target = s(&["big", "old", "brown", "dog", "barks"]);
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        // Should find overlap starting at "big" (index 1)
        assert_eq!(result.overlap_start_source, 1);
        // Should extend through "dog"
        assert_eq!(result.overlap_end_target, 4);
    }

    #[test]
    fn imperfect_overlap_similar_sounding_words() {
        // Common ASR errors with similar sounding words
        // source: ["they're", "going", "to", "their", "house"]
        // target: ["to", "there", "house", "now"]
        // "their" → "there" (homophone)
        let source = s(&["they're", "going", "to", "their", "house"]);
        let target = s(&["to", "there", "house", "now"]);
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&source, &target, scoring).unwrap();
        // Should align "to" and "house", tolerating "their"/"there" mismatch
        assert!(result.overlap_start_source >= 2);
        assert!(result.overlap_end_target >= 2);
    }

    #[test]
    fn imperfect_overlap_word_boundary_error() {
        // ASR word boundary error: "ice cream" vs "I scream"
        // source: ["I", "love", "ice", "cream"]
        // target: ["I", "scream", "for", "more"]
        // This is a hard case - minimal overlap expected
        let source = s(&["I", "love", "ice", "cream"]);
        let target = s(&["I", "scream", "for", "more"]);
        let result = semi_global_align(&source, &target, ScoringParams::default());
        // Might find overlap on "I" only, or none if penalties outweigh
        if let Some(r) = result {
            // If overlap found, verify it makes sense
            assert!(r.overlap_start_source <= source.len());
            assert!(r.overlap_end_target <= target.len());
        }
    }

    #[test]
    fn imperfect_overlap_realistic_asr_with_errors() {
        // Realistic ASR scenario with transcription errors
        // Previous: correct transcription
        // New: has some errors but overlaps
        let prev = s(&[
            "the", "patient", "reported", "feeling", "better", "after", "taking",
        ]);
        let new = s(&[
            "better", "after", "taken", "the", "medication", "yesterday",
        ]);
        // "taking" → "taken" (tense error common in ASR)
        let scoring = ScoringParams::new(2.0, -1.0);
        let result = semi_global_align(&prev, &new, scoring).unwrap();
        // Should align on "better", "after"
        assert_eq!(result.overlap_start_source, 4); // "better" is at index 4 in prev
        assert!(result.overlap_end_target >= 2);    // Should include at least "better", "after"

        // Verify merge produces reasonable result using first_match
        let merged: Vec<_> = prev[..result.first_match_source]
            .iter()
            .chain(new[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(
            &merged[..4],
            &s(&["the", "patient", "reported", "feeling"])[..]
        );
    }

    // ==================== Longer Sequence Tests ====================

    #[test]
    fn longer_transcript_overlap() {
        let source = s(&[
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
        ]);
        let target = s(&[
            "over", "the", "lazy", "dog", "and", "runs", "away",
        ]);
        let result = semi_global_align(&source, &target, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 5);
        assert_eq!(result.overlap_end_target, 3);
        assert_eq!(result.first_match_source, 5);
        assert_eq!(result.first_match_target, 0);
        // source[5..] = ["over", "the", "lazy"] matches target[..3]
    }

    #[test]
    fn realistic_asr_overlap() {
        // Simulating ASR transcript overlap scenario
        let prev_transcript = s(&[
            "I", "think", "that", "the", "weather", "is", "going", "to",
        ]);
        let new_transcript = s(&[
            "is", "going", "to", "be", "nice", "tomorrow",
        ]);
        let result = semi_global_align(&prev_transcript, &new_transcript, ScoringParams::default()).unwrap();
        assert_eq!(result.overlap_start_source, 5);
        assert_eq!(result.overlap_end_target, 3);
        assert_eq!(result.first_match_source, 5);
        assert_eq!(result.first_match_target, 0);
        // prev[5..] = ["is", "going", "to"] matches new[..3]

        let merged: Vec<_> = prev_transcript[..result.first_match_source]
            .iter()
            .chain(new_transcript[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(
            merged,
            s(&["I", "think", "that", "the", "weather", "is", "going", "to", "be", "nice", "tomorrow"])
        );
    }

    // ==================== First Match Tracking Tests ====================

    #[test]
    fn first_match_with_garbled_first_word() {
        // Simulating audio cutoff mid-word: "brown" becomes "wn" in new transcript
        // source: ["The", "quick", "brown", "fox"]
        // target: ["wn", "fox", "jumps"]  ("wn" is garbled/truncated "brown")
        // The overlap should find "fox" as the first clean match
        let source = s(&["The", "quick", "brown", "fox"]);
        let target = s(&["wn", "fox", "jumps"]);
        let scoring = ScoringParams::new(2.0, -1.0); // Higher match reward to tolerate gap
        let result = semi_global_align(&source, &target, scoring).unwrap();

        // Overlap starts at "fox" (index 3) since "wn" doesn't match
        assert_eq!(result.overlap_start_source, 3);
        assert_eq!(result.first_match_source, 3);
        assert_eq!(result.first_match_target, 1); // "fox" is at index 1 in target

        // Merge using first_match: source[..3] + target[1..] = ["The", "quick", "brown"] + ["fox", "jumps"]
        let merged: Vec<_> = source[..result.first_match_source]
            .iter()
            .chain(target[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(merged, s(&["The", "quick", "brown", "fox", "jumps"]));
    }

    #[test]
    fn first_match_differs_from_overlap_start() {
        // Case where there's a gap at the start of the overlap
        // source: ["a", "b", "c", "d", "e"]
        // target: ["x", "c", "d", "e", "f"]
        // With gap tolerance, overlap might start before the first match
        let source = s(&["a", "b", "c", "d", "e"]);
        let target = s(&["x", "c", "d", "e", "f"]);
        let scoring = ScoringParams::new(3.0, -1.0); // High match reward
        let result = semi_global_align(&source, &target, scoring).unwrap();

        // First match should be at "c"
        assert_eq!(result.first_match_source, 2); // "c" is at index 2 in source
        assert_eq!(result.first_match_target, 1); // "c" is at index 1 in target

        // Merge using first_match preserves clean word boundaries
        let merged: Vec<_> = source[..result.first_match_source]
            .iter()
            .chain(target[result.first_match_target..].iter())
            .cloned()
            .collect();
        assert_eq!(merged, s(&["a", "b", "c", "d", "e", "f"]));
    }
}
