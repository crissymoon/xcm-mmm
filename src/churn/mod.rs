//! Customer churn as an MMM outcome, with the memo text read beside it.
//!
//! The source is a telecom customer table from the industry-changes work: 7,043
//! customers, 1,869 of them churned, 1,869 free-text churn reasons. Three files
//! come out of it:
//!
//! * `panel.csv`, one row per customer with the label and the risk features,
//!   carrying no text, because the reason only exists for customers who left;
//! * `series.csv`, the discrete churn hazard by tenure month with right
//!   censoring, which is the shape an MMM outcome takes;
//! * `reasons.csv` and `reasons.json`, the churn reasons read on the eight
//!   emotion axes from [`emotion`](crate::emotion), grouped by the category the
//!   service team already uses.
//!
//! [`fit`](fit::build) then fits a fourteen-feature logistic with folds assigned
//! by customer id, and reports the held-out AUC beside the AUC of tenure and
//! monthly charge alone. That comparison is the point: it says how much of the
//! churn this table explains before any campaign is credited with retention.
//!
//! `churn check` rebuilds the three files from the source and compares them
//! byte for byte, so a hand-edited panel is a fault rather than a note.

pub mod card;
pub mod cli;
pub mod contract;
pub mod csvx;
pub mod fit;
pub mod load;
pub mod panel;

pub use cli::{run, ChurnArgs};
pub use panel::{Report, Summary};

/// The folder this module ships, relative to the repository root.
pub const FOLDER: &str = "data/churn";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_folder_names_its_files_once() {
        assert_eq!(panel::PANEL, "panel.csv");
        assert_eq!(panel::SERIES, "series.csv");
        assert_eq!(panel::REASONS, "reasons.csv");
        assert_eq!(fit::MODEL, "model.json");
        assert!(FOLDER.starts_with("data/"));
    }
}
