//! Whether the labels can carry the claim, and whether the sample can see one.
//!
//! Two numbers are missing from a card that prints an accuracy. The first is a
//! ceiling: a labelled bank is not a source of truth, and if five percent of its
//! rows are wrong then a three point gain sits inside the noise the labels
//! contribute. [`confident::audit`] estimates that noise without a clean test
//! set, from the out-of-fold probabilities the fold loop already computes.
//!
//! The second is resolution. An AUC has a standard error, and a difference
//! smaller than a few of them is not a finding. [`power::detectable_auc`] and
//! [`power::detectable_share`] turn a bank shape into the smallest gain it can
//! resolve at a stated power, beside the floor the project pre-registered, so a
//! card can say whether its own claim clears its own bar.
//!
//! Both readings are descriptions of a bank at a size, not properties of a
//! construct. They travel with the sample and must be recomputed on a new one.

pub mod confident;
pub mod power;

pub use confident::{audit, AuditConfig, LabelAudit, Suspect};
pub use power::{detectable_auc, detectable_share, hanley_mcneil, Power};

#[cfg(test)]
mod tests;
