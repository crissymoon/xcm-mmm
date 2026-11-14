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

pub mod benchmarks;
pub mod data;
pub mod dataset;
pub mod de;
pub mod detect;
pub mod eval;
pub mod forecast;
pub mod linalg;
pub mod metrics;
pub mod moead;
pub mod pairs;
pub mod problem;
pub mod rng;
pub mod text;
pub mod transformer;

pub use de::{minimize as minimize_de, DeConfig, DeOutcome};
pub use moead::{minimize as minimize_moead, MoeadConfig, MoeadOutcome};
pub use problem::{Boundary, Bounds, Problem, Score};
