//! One text, every reading this crate can give it, in one report.
//!
//! The three fitted layers were built to answer separate questions and each
//! has its own command: [`detect`](crate::detect) asks what register and which
//! cues a text carries, [`emotion`](crate::emotion) says what it feels like
//! word by word, and [`segment`](crate::segment) groups advertisers by what
//! they did. A person pasting copy into a box wants one answer, not three
//! commands and a reconciliation, so this module fuses them without inventing
//! a fourth number: every score in the report is a shipped model's own score,
//! and anything the models cannot see is listed rather than filled in.
//!
//! The report has four parts:
//!
//! * [`report::Shape`] and [`report::Semantics`], the measurable part: length,
//!   sentence and syllable rates, Flesch ease and grade, second person share,
//!   and the surface readings the two learned detector axes carry (register,
//!   nominal, agentless, citation, authority).
//! * [`report::Emotion`] and [`report::Trigger`], the reading of each word on
//!   eight Plutchik axes, with the words that carried the mass and the tier
//!   that answered for each.
//! * [`language::LanguageView`], the copy as marketing: direct address, punch,
//!   specificity, emotional density and the two shingle readings that stand in
//!   for cross-ad reuse and concentration.
//! * [`report::Axis`], the seven detectors with their score, level, fired
//!   cues and, for a cue index, the component densities behind the score.
//! * [`segment::SegmentView`], a fit score against every behavioural segment on
//!   the four dimensions a text can supply, weighted by how far apart the
//!   segment centres actually sit on each. Volume, cadence, spend and
//!   burstiness cannot be read from one text, so they are named as unobserved
//!   instead of guessed, and `reach` states how much of the model the copy
//!   can see.
//!
//! `xcm-mmm read "some copy"` renders the report and `--json` prints the same
//! struct. Nothing in the render path recomputes a number; it prints the
//! fields [`report::build`] filled.

pub mod cli;
pub mod context;
pub mod language;
pub mod render;
pub mod report;
pub mod segment;

pub use cli::{run, ReadArgs};
pub use context::Loaded;
pub use language::LanguageView;
pub use report::{build, Options, Report};
pub use segment::SegmentView;

/// Shipped models this report reads, relative to the repository root.
pub const DETECTORS: &str = "data/detectors/model.json";
pub const EMOTION: &str = "data/emotion/model.json";
pub const SEGMENTS: &str = "data/segments/model.json";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_paths_point_at_the_folders_the_modules_write() {
        assert!(DETECTORS.starts_with(crate::detect::FOLDER));
        assert!(EMOTION.starts_with(crate::emotion::FOLDER));
        assert!(SEGMENTS.starts_with(crate::segment::FOLDER));
    }
}
