use pyo3::prelude::*;

pub mod alignment;

use alignment::{semi_global_align, ScoringParams};

/// Merge two word sequences using semi-global alignment.
///
/// Given a previous transcript and a new transcript (from overlapped audio),
/// find the optimal alignment and merge them by keeping prev's non-overlapping
/// prefix and all of new.
///
/// Args:
///     prev: The previous transcript as a list of words.
///     new: The new transcript as a list of words.
///
/// Returns:
///     The merged transcript as a list of words. If no overlap is found,
///     returns prev + new (concatenation).
///
/// Example:
///     >>> merge_transcripts(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
///     ["The", "quick", "brown", "fox", "jumps"]
#[pyfunction]
#[pyo3(text_signature = "(prev, new)")]
fn merge_transcripts(prev: Vec<String>, new: Vec<String>) -> PyResult<Vec<String>> {
    match semi_global_align(&prev, &new, ScoringParams::default()) {
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
    m.add_function(wrap_pyfunction!(merge_transcripts, m)?)?;
    Ok(())
}
