//! Drift, coverage and rollback for a model that is already in service.
//!
//! The repository builds artifacts and checks that they rebuild. That is a
//! property of the build, not of a deployment. This module supplies the
//! operational pieces a deployed model needs and a card can report:
//!
//! * `drift` compares a reference panel with a current one, feature by
//!   feature, with the population stability index and the Kolmogorov-Smirnov
//!   distance.
//! * `gate` turns those readings into a decision that knows the sample size
//!   it was taken at: a simulated null for the index, Cramer-von Mises scored
//!   against its asymptotic law, Benjamini-Hochberg across features, and a
//!   kernel two-sample reading on the whole standardised vector.
//! * [`coverage_series`] tracks conformal coverage over batches and raises an
//!   alarm when a rolling window falls under the nominal rate.
//! * [`Registry`] keeps versioned copies of an artifact, with the audit trail a
//!   model risk review asks for on each one, and can move the pointer back to
//!   the previous version, so a rollback is a command rather than a rebuild.
//!
//! Nothing here watches a live system by itself. The module is the measurement
//! and the rollback path; a scheduler still has to call it. That boundary is
//! stated on the card rather than hidden.

mod drift;
mod gate;
mod registry;

pub use drift::{
    compare, counts, coverage_series, edges, ks, paired_columns, psi, psi_counts, psi_shares,
    CoveragePoint, DriftReport, FeatureDrift, PairedColumns,
};
pub use gate::{
    assess, bh, cvm, CvmNull, GateConfig, GateFeature, GateReport, MmdReport, PsiNull, GATE_SEED,
};
pub use registry::{Audit, Event, Registry, Version};
