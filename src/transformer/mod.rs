//! A small transformer trained on the pair corpus.
//!
//! The cleaning and pair stages decide what a single model should see. This
//! module is that model: one shared encoder, one head per task, trained on the
//! mixed pair file the mining and packing stages produce. It is deliberately
//! small and dependency free, because the point is a model that runs where the
//! data processor runs rather than one that needs a GPU image.
//!
//! The pieces:
//!
//! * [`tokenizer`] learns a byte level byte pair vocabulary from the training
//!   split. Unknown words fall back to bytes, so nothing is dropped.
//! * [`tensor`] is the linear algebra, with a finite difference test on every
//!   backward pass that is easy to get wrong.
//! * [`model`] is the encoder: pre norm transformer blocks, mean pooling, a
//!   final normalisation and a linear head per task. Gradients are written by
//!   hand and checked against finite differences.
//! * [`loss`] turns a task pair into a scalar and two score gradients. Ranking,
//!   preference and contrast pairs use the Bradley-Terry form; a labelled row
//!   uses a pointwise logistic loss.
//! * [`data`] reads the pair schema back and encodes both sides.
//! * [`train`] is AdamW, a warmup and cosine schedule, gradient clipping, an
//!   epoch schedule with early stopping, per task validation, and a parallel
//!   gradient accumulation that reproduces the serial update to the bit.
//! * [`checkpoint`] is the mid-run state, so a stopped training resumes from the
//!   epoch it reached with its optimizer moments intact.
//!
//! The result is one file: tokenizer, config and weights. A document is scored
//! by [`model::Model::score`], which is what a downstream ranker calls.

pub mod checkpoint;
pub mod data;
pub mod grads;
pub mod loss;
pub mod model;
pub mod run;
pub mod store;
pub mod tensor;
pub mod tokenizer;
pub mod train;

pub use model::{Cache, Config, Model, Params};
pub use tokenizer::Tokenizer;
pub use train::{Report, TaskMetric, TrainConfig};
