//! The selection gate: how much of the winning score is selection itself.
//!
//! Filled in below. A search evaluates thousands of candidate subsets and keeps
//! the best; the best of many noisy trials is upward biased whether or not the
//! data holds a signal. This module records every distinct trial the forecast
//! search scores and converts the winning score into a probability against the
//! null that the trials were exchangeable noise.

/// Placeholder; the ledger lands here.
pub struct Ledger {
    /// Trial scores, one per entry.
    pub scores: Vec<f64>,
}

impl Ledger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self { scores: Vec::new() }
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}
