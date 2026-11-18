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
//! Three design choices are worth stating. The search starts at three groups: a
//! two-way split of an advertising panel is a high and low score, not a
//! segmentation, and it wins on silhouette for that reason alone. The z is
//! clipped at three standard deviations, because an advertiser with an almost
//! constant reading would otherwise form a group of one and drag a centroid
//! with it. And the shape of a group is chosen per panel rather than assumed:
//! `mixture` states what a spherical, a diagonal and a full covariance each
//! believe, and `select` picks between them on held-out likelihood with a BIC
//! choice of k and a stated margin, so the answer a two-way split can never give
//! is written down instead of inherited from k-means.

use serde::{Deserialize, Serialize};

pub mod ad;
pub mod card;
pub mod cli;
pub mod concomitant;
pub mod feature;
pub mod gate;
pub mod kmeans;
pub mod mixture;
pub mod model;
pub mod pattern;
pub mod pipeline;
pub mod select;
pub mod sharp;
pub mod stability;
pub mod train;

pub use ad::{AdRow, Day, Entity};
pub use concomitant::{Audit as ConcomitantAudit, Covariate};
pub use feature::{FeatureSpec, Matrix, Scaling, Transform};
pub use gate::{Gate, GateAudit, GateSpec};
pub use kmeans::{Fit, Sweep};
pub use mixture::{Component, Family, Mixture};
pub use model::{Artifact, Manifest, Metrics, ProfileRow, Segment};
pub use pattern::{Pattern, Verdict};
pub use pipeline::fit;
pub use select::{Baseline, Candidate, Choice, Dependence, Pair, Shuffle};
pub use sharp::{Group as SharpGroup, Separation, Sharpness};
pub use train::{build, check, extract};

fn default_folds() -> usize {
    5
}

fn default_gain() -> f64 {
    0.02
}

fn default_restarts() -> usize {
    4
}

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
    /// Folds the held-out likelihood is read on.
    #[serde(default = "default_folds")]
    pub folds: usize,
    /// Nats per entity a freer family must add before it is preferred.
    #[serde(default = "default_gain")]
    pub min_gain: f64,
    /// Starts per expectation maximisation fit; half are soft, half k-means.
    #[serde(default = "default_restarts")]
    pub em_restarts: usize,
    /// Share of a declared edge a reading needs before it is read as a gate.
    #[serde(default = "default_gate_share")]
    pub gate_share: f64,
    /// A family held fixed for this run, or `None` to let the panel choose.
    #[serde(default)]
    pub family: Option<Family>,
    /// How the pattern mixture pools a group's inside slope toward the panel's.
    #[serde(default)]
    pub inside_pooling: pattern::Pool,
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
            folds: 5,
            min_gain: 0.02,
            em_restarts: 4,
            gate_share: 0.20,
            family: None,
            inside_pooling: pattern::Pool::default(),
            seed: 20261114,
        }
    }
}

fn default_gate_share() -> f64 {
    0.20
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
        assert!(cfg.folds >= 2 && cfg.min_gain > 0.0 && cfg.em_restarts >= 2);
        assert!(cfg.family.is_none());
        assert!(!cfg.name.is_empty());
    }
}
