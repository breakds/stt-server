use pyo3::prelude::*;

pub mod alignment;

use alignment::{semi_global_align, ScoringParams};

/// Merge two token sequences by finding their overlap.
///
/// Uses semi-global alignment to find where the suffix of `prev` overlaps
/// with the prefix of `new`. Merges by stitching at the first matching token,
/// which handles cases where the first token in `new` may be garbled due to
/// audio cutoff.
///
/// Args:
///     prev: The previous sequence of tokens.
///     new: The new sequence of tokens.
///
/// Returns:
///     The merged sequence. If no overlap is found, returns prev + new (concatenation).
///
/// Example:
///     >>> merge_by_overlap(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
///     ["The", "quick", "brown", "fox", "jumps"]
#[pyfunction]
#[pyo3(text_signature = "(prev, new)")]
fn merge_by_overlap(prev: Vec<String>, new: Vec<String>) -> PyResult<Vec<String>> {
    match semi_global_align(&prev, &new, ScoringParams::with_match_reward(3.0)) {
        Some(result) => {
            // Stitch at first match: take prev up to first match, then new from first match onward.
            // This handles garbled first tokens in new (e.g., from audio cutoff mid-word).
            let mut merged: Vec<String> = prev[..result.first_match_source].to_vec();
            merged.extend(new[result.first_match_target..].iter().cloned());
            Ok(merged)
        }
        None => {
            // No overlap found, concatenate
            let mut result = prev;
            result.extend(new);
            Ok(result)
        }
    }
}

#[pymodule]
fn strops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(merge_by_overlap, m)?)?;
    Ok(())
}
