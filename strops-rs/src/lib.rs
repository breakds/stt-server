use pyo3::prelude::*;

pub mod alignment;

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
///     The merged transcript as a list of words.
///
/// Example:
///     >>> merge_transcripts(["The", "quick", "brown", "fox"], ["brown", "fox", "jumps"])
///     ["The", "quick", "brown", "fox", "jumps"]
#[pyfunction]
#[pyo3(text_signature = "(prev, new)")]
fn merge_transcripts(prev: Vec<String>, new: Vec<String>) -> PyResult<Vec<String>> {
    Ok(alignment::merge_transcripts(&prev, &new))
}

#[pymodule]
fn strops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(merge_transcripts, m)?)?;
    Ok(())
}
