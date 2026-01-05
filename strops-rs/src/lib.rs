use pyo3::prelude::*;

pub mod alignment;

use alignment::{semi_global_align, ScoringParams};

/// Merge two token sequences by finding their overlap.
///
/// Uses semi-global alignment to find where the suffix of `prev` overlaps
/// with the prefix of `new`, then merges them by keeping prev's non-overlapping
/// prefix followed by all of new.
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
    match semi_global_align(&prev, &new, ScoringParams::default().with_match_reward(3.0)) {
        Some((overlap_start, _)) => {
            let mut result: Vec<String> = prev[..overlap_start].to_vec();
            result.extend(new);
            Ok(result)
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
