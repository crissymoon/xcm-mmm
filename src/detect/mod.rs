//! Seven text detectors, fitted in Rust, over advertising copy.
//!
//! Two of the axes are learned from a labelled bank, because a suitable corpus
//! exists: legal register looks like a Federal Register abstract, abstraction
//! looks like a grant synopsis, and advertisers' own copy is the negative side
//! of both. Five have no labelled corpus of advertising at all, so they ship as
//! declared cue indices behind a gate, and are audited on whether their own
//! disjoint cue halves agree rather than on an accuracy nobody could measure.
//!
//! The folder this crate ships is `data/detectors/`:
//!
//! * `bank.jsonl`, the labelled sentence bank, with the source of every row;
//! * `ads.jsonl`, ad copy at document grain with its source split;
//! * `cues.json`, the declared cue table, exported from the module that produced
//!   the published readings rather than retyped here;
//! * `model.json`, the artifact, and `manifest.json` and `card.md` beside it.
//!
//! `detect train` re-fits all of it, `detect score` reads the artifact, and
//! `detect check` rebuilds from the shipped inputs and reports every way a copy
//! disagrees with its own account.

use serde::{Deserialize, Serialize};

pub mod bank;
pub mod card;
pub mod eval;
pub mod feature;
pub mod index;
pub mod lexicon;
pub mod logistic;
pub mod model;
pub mod token;
pub mod train;

pub use feature::{FeatureSpec, Features};
pub use index::{Check, Index, IndexSpec};
pub use lexicon::{FitConfig, Lexicon};
pub use logistic::{FitParams, Logistic};
pub use model::{Artifact, Axis, Detector, Manifest, Reading};
pub use train::{build, check, fit};

/// The folder this crate ships, relative to the repository root.
pub const FOLDER: &str = "data/detectors";
pub const BANK: &str = "bank.jsonl";
pub const ADS: &str = "ads.jsonl";
pub const CUES: &str = "cues.json";
pub const MODEL: &str = "model.json";
pub const MANIFEST: &str = "manifest.json";
pub const CARD: &str = "card.md";

/// How a fit is run. Carried into the artifact and the manifest, so a rebuild
/// from a folder uses the settings that folder was built with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    pub name: String,
    pub folds: usize,
    /// Smallest number of rows either side of an axis may hold.
    pub min_rows: usize,
    /// Grid points in a component quantile table.
    pub bins: usize,
    pub lexicon: FitConfig,
    pub logistic: FitParams,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            name: "marketlab-ad-detectors".to_string(),
            folds: 5,
            min_rows: 40,
            bins: 21,
            lexicon: FitConfig::default(),
            logistic: FitParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_folder_names_its_files_once() {
        assert_eq!(MODEL, "model.json");
        assert_eq!(BANK, "bank.jsonl");
        assert_eq!(ADS, "ads.jsonl");
        assert_eq!(CUES, "cues.json");
        assert!(FOLDER.starts_with("data/"));
    }

    #[test]
    fn the_default_config_is_a_refit_of_the_shipped_one() {
        let cfg = TrainConfig::default();
        assert!(cfg.folds >= 2);
        assert!(cfg.bins >= 11);
        assert!(cfg.min_rows > 0);
    }
}
