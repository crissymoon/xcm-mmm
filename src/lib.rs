//! xcm-mmm: differential evolution and MOEA/D in Rust.
//!
//! The crate has four layers:
//!
//! * [`problem`] defines the search space and the score, including Deb's
//!   feasibility rules.
//! * [`de`] solves single objective problems with SHADE, jDE or classic DE.
//! * [`moead`] solves multi-objective problems with MOEA/D-DE.
//! * [`metrics`] measures a front with hypervolume, IGD and spacing.
//!
//! [`forecast`] builds a walk-forward cross-validation problem on top of the
//! same traits, so feature selection and its accuracy against model size can be
//! solved with either engine.
//!
//! [`text`], [`pairs`] and [`dataset`] sit beside the optimiser rather than inside it: they
//! clean a corpus, deduplicate it, select a relevant subset, and build the
//! training pairs that one model can be fitted on. [`detect`] fits the seven
//! text detectors over advertising copy: two lexicons learned from a labelled
//! bank, five declared cue indices, and the artifact that ships with them.
//! [`segment`] groups advertisers by what they did in an ad panel, with a
//! silhouette selection, a shuffled control and a resampling audit in place of
//! an accuracy no label could supply. [`emotion`] reads eight Plutchik axes
//! from any English word, and [`churn`] turns a customer table into a retention
//! outcome with its memo text read on those axes. [`profile`] puts one text
//! through the detectors, the emotion scanner and the segment centres and
//! prints every score beside the evidence for it.
//!
//! Four modules sit around the models rather than inside them. [`learn`] puts
//! ridge and a gradient boosted tree ensemble behind the same fold interface.
//! [`uncertainty`] supplies bootstrap intervals, calibration and split
//! conformal sets. [`causal`] estimates uplift from a table with an observed
//! treatment, with cross-fitting, IPW, Qini, deciles and a placebo test.
//! [`monitor`] measures PSI and KS drift, tracks coverage over batches and keeps
//! a versioned artifact with a rollback path. [`support`] reports events per
//! feature and a learning curve so a small bank is visible as a slope rather
//! than a caveat.

pub mod benchmarks;
pub mod causal;
pub mod churn;
pub mod data;
pub mod dataset;
pub mod de;
pub mod deflate;
pub mod detect;
pub mod digest;
pub mod emotion;
pub mod eval;
pub mod forecast;
pub mod learn;
pub mod linalg;
pub mod metrics;
pub mod moead;
pub mod monitor;
pub mod pairs;
pub mod problem;
pub mod profile;
pub mod rng;
pub mod segment;
pub mod stats;
pub mod support;
pub mod text;
pub mod transformer;
pub mod uncertainty;

pub use de::{minimize as minimize_de, DeConfig, DeOutcome};
pub use moead::{minimize as minimize_moead, MoeadConfig, MoeadOutcome};
pub use problem::{Boundary, Bounds, Problem, Score};
