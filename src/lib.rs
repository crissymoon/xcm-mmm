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

pub mod benchmarks;
pub mod data;
pub mod de;
pub mod eval;
pub mod forecast;
pub mod linalg;
pub mod metrics;
pub mod moead;
pub mod problem;
pub mod rng;

pub use de::{minimize as minimize_de, DeConfig, DeOutcome};
pub use moead::{minimize as minimize_moead, MoeadConfig, MoeadOutcome};
pub use problem::{Boundary, Bounds, Problem, Score};
