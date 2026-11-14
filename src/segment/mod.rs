//! Behavioural segmentation of advertisers, fitted in Rust.
//!
//! The seven detectors describe what an ad says. This module describes what an
//! advertiser does: how often it starts an ad, how far apart those starts fall,
//! how uneven the spacing is, what it pays at the published band midpoint, how
//! long its copy runs, how many terms it buys against, how much of its copy is
//! reused and how far one fingerprint dominates it.
//!
//! There are no labels here, so nothing in the folder is an accuracy. What can
//! be measured is measured: whether the chosen partition beats the same protocol
//! run on columns with their pairing destroyed, whether it survives resampling
//! of the entities, and whether every group holds enough of the panel to be a
//! group rather than an accident. A fit that cannot answer those is refused.
//!
//! The folder this crate ships is `data/segments/`:
//!
//! * `features.jsonl`, one behavioural row per advertiser, in plain units;
//! * `model.json`, the feature table, standardisation, centroids, profiles,
//!   assignments and both audits;
//! * `manifest.json` with a digest per file, and `card.md` beside it.
//!
//! `segment extract` folds the ad panel into entities, `segment fit` writes the
//! folder, `segment assign` places an entity against it and `segment check`
//! rebuilds from the shipped entities and reports every disagreement.
//!
//! Two design choices are worth stating. The search starts at three groups: a
//! two-way split of an advertising panel is a high and low score, not a
//! segmentation, and it wins on silhouette for that reason alone. And the z is
//! clipped at three standard deviations, because an advertiser with an almost
//! constant reading would otherwise form a group of one and drag a centroid
//! with it.

use serde::{Deserialize, Serialize};

pub mod ad;
pub mod card;
pub mod cli;
pub mod feature;
pub mod kmeans;
pub mod model;
pub mod stability;
pub mod train;

pub use ad::{AdRow, Day, Entity};
pub use feature::{FeatureSpec, Matrix, Scaling, Transform};
pub use kmeans::{Fit, Sweep};
pub use model::{Artifact, Manifest, Metrics, ProfileRow, Segment};
pub use train::{build, check, extract, fit};

/// The folder this crate ships, relative to the repository root.
pub const FOLDER: &str = "data/segments";
pub const FEATURES: &str = "features.jsonl";
pub const MODEL: &str = "model.json";
pub const MANIFEST: &str = "manifest.json";
pub const CARD: &str = "card.md";

/// How a fit is run. Carried into the artifact and the manifest, so a rebuild
/// from a folder uses the settings that folder was built with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentConfig {
    pub name: String,
    /// Smallest number of groups the search will consider.
    pub k_min: usize,
    /// Largest number of groups the search will consider.
    pub k_max: usize,
    /// Smallest share of the panel any one group may hold.
    pub min_share: f64,
    /// Smallest panel the fit will touch.
    pub min_entities: usize,
    /// Floor on the chosen silhouette.
    pub min_silhouette: f64,
    /// How far above the shuffled control the silhouette has to stand.
    pub margin: f64,
    /// Floor on the mean agreement between resampled fits.
    pub min_stability: f64,
    pub restarts: usize,
    pub iterations: usize,
    pub bootstraps: usize,
    pub controls: usize,
    /// Standard deviations at which a reading is clipped.
    pub clip: f64,
    pub seed: u64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            name: "marketlab-behaviour-segments".to_string(),
            k_min: 3,
            k_max: 6,
            min_share: 0.05,
            min_entities: 60,
            min_silhouette: 0.15,
            margin: 0.08,
            min_stability: 0.60,
            restarts: 32,
            iterations: 200,
            bootstraps: 50,
            controls: 20,
            clip: 3.0,
            seed: 20261114,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_folder_names_its_files_once() {
        assert_eq!(FEATURES, "features.jsonl");
        assert_eq!(MODEL, "model.json");
        assert!(FOLDER.starts_with("data/"));
    }

    #[test]
    fn the_defaults_are_a_refit_of_the_shipped_folder() {
        let cfg = SegmentConfig::default();
        assert!(cfg.k_min >= 3 && cfg.k_max > cfg.k_min);
        assert!(cfg.min_share > 0.0 && cfg.min_share < 0.5);
        assert!(cfg.margin > 0.0 && cfg.min_stability > 0.0);
        assert!(cfg.restarts >= 8 && cfg.bootstraps >= 10 && cfg.controls >= 5);
        assert!(cfg.clip >= 1.0);
        assert!(!cfg.name.is_empty());
    }
}
