//! Corpus cleaning and pair mining.
//!
//! The optimiser in this crate answers questions about a model. This layer
//! answers the question in front of it: which text is worth keeping, and what
//! pairs should a single model be trained on.
//!
//! Five stages, each a module, each runnable on its own:
//!
//! * [`normalize`] folds characters, removes markup, links and boilerplate, and
//!   reports every removal so the cost is visible rather than implied.
//! * [`tokenize`] gives words and sentences without a tokenizer dependency, and
//!   [`hash`] turns them into shingles, MinHash signatures and SimHash bits.
//! * [`quality`] scores a document on length, lexical content, diversity and
//!   repetition, with Gopher and C4 style rules on top and an optional logistic
//!   model fitted from labelled examples.
//! * [`dedupe`] clusters near duplicates with MinHash bands and keeps one
//!   canonical member per cluster, by quality with a deterministic tie break.
//! * [`select`] chooses a budget worth of documents by submodular coverage with
//!   a redundancy penalty and a per source cap, which is the "most relevant
//!   data" step rather than the "least broken data" step.
//!
//! [`doc`] carries the record shape and the readers and writers, including
//! gzip through the system `gzip` binary so the crate keeps four dependencies.
//! [`run`] is the glue the command line calls.

pub mod dedupe;
pub mod doc;
pub mod hash;
pub mod normalize;
pub mod quality;
pub mod run;
pub mod select;
pub mod tokenize;

pub use doc::{Sink, TextRow};
pub use quality::{Flag, QualityModel, Rules, Stats};
