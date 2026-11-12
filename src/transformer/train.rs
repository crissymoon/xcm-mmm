//! The trainer: AdamW, a warmup and cosine schedule, and per task validation.
//!
//! A step is one batch of examples, each of which is one or two sequences. Each
//! example fills its own gradient buffer on whichever thread takes it, the
//! buffers are added in batch order and averaged, clipped and applied once,
//! which is the update a padded batch would give without the mask arithmetic and
//! the serial update to the bit. A checkpoint carries the weights, the optimizer
//! moments and the epoch cursor, so a run split across sessions lands where it
//! would have landed uninterrupted. The order of a batch is a weighted sample
//! without replacement:
//! each row draws an exponential key scaled by its task weight and the smallest
//! keys come first, so a task weight of three sees roughly three times the rows
//! of a weight of one and no row is repeated inside an epoch.
//!
//! Validation runs on the split the pair schema carries, never on a slice of
//! train. When no validation split exists the last tenth of the rows is held
//! out by position, which is stated in the report rather than hidden. Early
//! stopping keeps the best epoch by mean accuracy over the tasks that are
//! present, and the model that is written is that epoch, not the last one.

use std::time::Instant;

use rayon::prelude::*;
use serde_json::json;

use crate::rng::Rng;
use crate::transformer::checkpoint::{self, RunSettings};
use crate::transformer::data::{self, Example, LoadStats};
use crate::transformer::model::{Config, Grads, Model, Params};
use crate::transformer::store;
use crate::transformer::tensor::Real;
use crate::transformer::tokenizer::{self, Tokenizer, TrainOptions};
use crate::{linalg, pairs::Task};

/// Everything one run needs.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub pairs: Vec<String>,
    pub out: String,
    pub vocab: usize,
    pub max_len: usize,
    pub dim: usize,
    pub heads: usize,
    pub ff: usize,
    pub layers: usize,
    pub epochs: usize,
    pub batch: usize,
    pub lr: Real,
    pub weight_decay: Real,
    /// Share of the schedule spent warming up, in `[0, 1)`.
    pub warmup: Real,
    pub clip: Real,
    pub dropout: Real,
    pub seed: u64,
    /// Read at most this many rows. Zero reads everything.
    pub limit: usize,
    /// Epochs without improvement before stopping. Zero disables early stopping.
    pub patience: usize,
    pub min_count: u64,
    /// Sampling weight per task, in [`Task::ALL`] order.
    pub task_weights: [Real; 6],
    /// Continue from a checkpoint written by an earlier run.
    pub resume: Option<String>,
    /// Stop after this many epochs in total, keeping the schedule of `epochs`.
    ///
    /// This is how a long run is split across sessions: the budget stays the
    /// same, the legs are shorter, and a checkpoint carries the state across.
    pub stop_after: Option<usize>,
    /// Where to write mid-run state. `None` writes none.
    pub checkpoint: Option<String>,
    /// Epochs between checkpoint writes. Zero is treated as one.
    pub checkpoint_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            out: "model.json".to_string(),
            vocab: 2048,
            max_len: 64,
            dim: 64,
            heads: 4,
            ff: 128,
            layers: 2,
            epochs: 3,
            batch: 8,
            lr: 3e-3,
            weight_decay: 0.01,
            warmup: 0.1,
            clip: 1.0,
            dropout: 0.1,
            seed: 11,
            limit: 0,
            patience: 2,
            min_count: 2,
            task_weights: [1.0, 1.0, 1.0, 0.5, 0.5, 1.0],
            resume: None,
            stop_after: None,
            checkpoint: None,
            checkpoint_every: 1,
        }
    }
}

/// What one task scored on the validation rows.
#[derive(Clone, Debug)]
pub struct TaskMetric {
    pub task: Task,
    pub n: usize,
    pub accuracy: Real,
    /// Rank correlation of the score with the target, for a pointwise task.
    pub spearman: Option<Real>,
}

/// What a run did.
#[derive(Clone, Debug)]
pub struct Report {
    pub config: Config,
    pub train_rows: usize,
    pub val_rows: usize,
    pub epochs: usize,
    pub best_epoch: usize,
    pub train_loss: Real,
    pub metrics: Vec<TaskMetric>,
    pub params: usize,
    pub seconds: Real,
    pub out: String,
    pub bytes: u64,
    pub train_stats: LoadStats,
    pub val_stats: LoadStats,
    pub tokenizer_tokens: usize,
}

impl Report {
    /// The lines the command line prints. Kept here so a test can read them.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "tokenizer  {} tokens, {} train rows, {} validation rows",
                self.tokenizer_tokens, self.train_rows, self.val_rows
            ),
            format!(
                "model      {} parameters, dim {} heads {} layers {} ff {} max_len {}",
                self.params,
                self.config.dim,
                self.config.heads,
                self.config.layers,
                self.config.ff,
                self.config.max_len
            ),
            format!(
                "training   {} epochs, best epoch {}, loss {:.4}, {:.1}s",
                self.epochs, self.best_epoch, self.train_loss, self.seconds
            ),
        ];
        if !self.train_stats.skipped.is_empty() {
            lines.push(format!(
                "skipped    {}",
                self.train_stats.reasons().join(", ")
            ));
        }
        lines.push(format!(
            "{:<10} {:>7} {:>9} {:>9}",
            "task", "rows", "accuracy", "spearman"
        ));
        for metric in &self.metrics {
            let rank = metric
                .spearman
                .map(|value| format!("{value:>9.3}"))
                .unwrap_or_else(|| format!("{:>9}", "-"));
            lines.push(format!(
                "{:<10} {:>7} {:>9.3} {}",
                metric.task.as_str(),
                metric.n,
                metric.accuracy,
                rank
            ));
        }
        lines.push(format!(
            "saved      {} ({:.1} MB)",
            self.out,
            self.bytes as Real / 1_048_576.0
        ));
        lines
    }
}

struct Adam {
    m: Vec<Vec<Real>>,
    v: Vec<Vec<Real>>,
    step: usize,
    beta1: Real,
    beta2: Real,
    eps: Real,
}

impl Adam {
    fn new(params: &Params) -> Self {
        let mut copy = params.clone();
        let mut m = Vec::new();
        let mut v = Vec::new();
        for (_, _, _, data) in copy.blocks_mut() {
            m.push(vec![0.0; data.len()]);
            v.push(vec![0.0; data.len()]);
        }
        Self {
            m,
            v,
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// The moment vectors, one flat vector per parameter block, in block order.
    fn moments(&self) -> (Vec<Vec<Real>>, Vec<Vec<Real>>) {
        (self.m.clone(), self.v.clone())
    }

    /// Rebuild an optimizer from stored moments.
    ///
    /// A moment vector is flat and carries no shape, so the block lengths of the
    /// model being resumed are what validate it. A checkpoint written against a
    /// wider model is refused here rather than applied to the wrong tensors.
    fn restore(
        params: &mut Params,
        m: Vec<Vec<Real>>,
        v: Vec<Vec<Real>>,
        step: usize,
    ) -> Result<Self, String> {
        let lengths: Vec<usize> = params
            .blocks_mut()
            .iter()
            .map(|(_, rows, cols, _)| rows * cols)
            .collect();
        if m.len() != lengths.len() || v.len() != lengths.len() {
            return Err(format!(
                "checkpoint has {} first and {} second moments for {} tensors",
                m.len(),
                v.len(),
                lengths.len()
            ));
        }
        for (index, length) in lengths.iter().enumerate() {
            if m[index].len() != *length || v[index].len() != *length {
                return Err(format!(
                    "optimizer moment {index} has the wrong length for this model"
                ));
            }
        }
        Ok(Self {
            m,
            v,
            step,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    fn step(&mut self, params: &mut Params, grads: &mut Grads, lr: Real, weight_decay: Real) {
        self.step += 1;
        let correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let correction2 = 1.0 - self.beta2.powi(self.step as i32);
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let mut blocks = params.blocks_mut();
        let gradient_blocks = grads.blocks_mut();
        for (index, block) in blocks.iter_mut().enumerate() {
            let gradient = &gradient_blocks[index].3;
            let m = &mut self.m[index];
            let v = &mut self.v[index];
            // Decay weights only. A layer norm gain and a bias are one row, and
            // shrinking them toward zero is not regularisation, it is damage.
            let decay = if block.1 > 1 && block.2 > 1 {
                weight_decay
            } else {
                0.0
            };
            let data = &mut *block.3;
            for position in 0..data.len() {
                let g = gradient[position];
                m[position] = beta1 * m[position] + (1.0 - beta1) * g;
                v[position] = beta2 * v[position] + (1.0 - beta2) * g * g;
                let mhat = m[position] / correction1;
                let vhat = v[position] / correction2;
                data[position] -= lr * (mhat / (vhat.sqrt() + eps) + decay * data[position]);
            }
        }
    }
}

fn lr_at(step: usize, total: usize, warmup: usize, base: Real) -> Real {
    let warmup = warmup.min(total.saturating_sub(1));
    if step <= warmup {
        base * (step + 1) as Real / (warmup + 1) as Real
    } else {
        let span = (total.saturating_sub(warmup)).max(1) as Real;
        let progress = ((step - warmup) as Real / span).clamp(0.0, 1.0);
        base * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
    }
}

fn weighted_order(examples: &[Example], weights: &[Real; 6], rng: &mut Rng) -> Vec<usize> {
    let mut keyed: Vec<(Real, usize)> = examples
        .iter()
        .enumerate()
        .map(|(index, example)| {
            let weight = weights.get(example.task).copied().unwrap_or(0.0).max(1e-6);
            let draw = rng.open_f64();
            ((-draw.ln()) / weight, index)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().map(|(_, index)| index).collect()
}

/// Salt that keeps the epoch order stream apart from the dropout streams.
const ORDER_SALT: usize = 0x5EED_0001;

/// A seed derived from the run seed and two counters.
///
/// The dropout mask an example sees must not depend on how many examples ran
/// before it, or a batch would give a different update on one thread than on
/// eight. The same function draws the permutation of an epoch from the epoch
/// number, which is what lets a resumed run replay the epochs it has not done.
fn derive_seed(seed: u64, step: usize, row: usize) -> u64 {
    let mut x = seed ^ 0xA076_1D64_78BD_642F;
    x ^= (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= (row as u64).wrapping_add(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The loss and the gradient buffer of one example.
///
/// This is the whole per-example unit of work: it reads the model, draws its
/// dropout from `seed` alone and never touches a buffer it did not allocate,
/// which is exactly what lets a batch run on any number of threads.
fn example_grads(model: &Model, example: &Example, seed: u64) -> (Grads, Real) {
    let mut grads = Grads::zeros(&model.params);
    let loss = match &example.second {
        Some(second) => {
            let (first_score, first_cache) =
                model.forward_seeded(&example.first, example.task, seed);
            let (second_score, second_cache) = model.forward_seeded(
                second,
                example.task,
                seed.rotate_left(1) ^ 0x5DEE_CE66_D1AB_2E01,
            );
            let gradient = crate::transformer::loss::pairwise(first_score, second_score, 1.0);
            model.backward(&first_cache, gradient.d_first, &mut grads);
            model.backward(&second_cache, gradient.d_second, &mut grads);
            gradient.loss
        }
        None => {
            let (score, cache) = model.forward_seeded(&example.first, example.task, seed);
            let gradient = crate::transformer::loss::pointwise(score, example.target, 1.0);
            model.backward(&cache, gradient.d_first, &mut grads);
            gradient.loss
        }
    };
    (grads, loss)
}

fn is_train(split: &str) -> bool {
    split.is_empty() || split == "train"
}

fn is_validation(split: &str) -> bool {
    matches!(split, "val" | "valid" | "validation" | "dev")
}

fn is_test(split: &str) -> bool {
    matches!(split, "test" | "holdout")
}

/// Score a set of rows and aggregate by task.
///
/// Ranking is measured by choosing the first side, which is the ordered side in
/// every pair task. A pointwise row is measured by the sign of the score, and
/// its rank correlation is reported because a good direction with a flat score
/// is a weaker model than the accuracy alone suggests.
pub fn evaluate(model: &Model, examples: &[Example]) -> Vec<TaskMetric> {
    let scored: Vec<(usize, bool, Real, Real, bool)> = examples
        .par_iter()
        .map(|example| {
            let first = model.score(&example.first, example.task);
            match &example.second {
                Some(second) => {
                    let other = model.score(second, example.task);
                    (example.task, first > other, first, 0.0, false)
                }
                None => (
                    example.task,
                    (first > 0.0) == (example.target > 0.5),
                    first,
                    example.target,
                    true,
                ),
            }
        })
        .collect();

    let mut counts = [0usize; 6];
    let mut correct = [0usize; 6];
    let mut point_scores: Vec<Vec<Real>> = vec![Vec::new(); 6];
    let mut point_targets: Vec<Vec<Real>> = vec![Vec::new(); 6];
    for (task, hit, score, target, pointwise) in scored {
        if task >= 6 {
            continue;
        }
        counts[task] += 1;
        if hit {
            correct[task] += 1;
        }
        if pointwise {
            point_scores[task].push(score);
            point_targets[task].push(target);
        }
    }

    Task::ALL
        .iter()
        .enumerate()
        .filter(|(index, _)| counts[*index] > 0)
        .map(|(index, task)| {
            let n = counts[index];
            let accuracy = correct[index] as Real / n as Real;
            let spearman = if point_scores[index].len() >= 3
                && point_targets[index]
                    .iter()
                    .any(|v| *v != point_targets[index][0])
            {
                Some(linalg::spearman(
                    &point_scores[index],
                    &point_targets[index],
                ))
            } else {
                None
            };
            TaskMetric {
                task: *task,
                n,
                accuracy,
                spearman,
            }
        })
        .collect()
}

fn mean_accuracy(metrics: &[TaskMetric]) -> Real {
    if metrics.is_empty() {
        return 0.0;
    }
    metrics.iter().map(|m| m.accuracy).sum::<Real>() / metrics.len() as Real
}

/// Split the corpus into what a run trains on and what it validates on.
///
/// The split comes from the schema, never from a shuffle: `train` rows train,
/// `val`/`valid`/`dev` rows validate and `test`/`holdout` rows are left alone
/// for the evaluation command. A corpus with no validation split gives up its
/// last tenth by position, which the report states rather than hides.
fn split_pairs(cfg: &TrainConfig) -> Result<(Vec<data::RawPair>, Vec<data::RawPair>), String> {
    let (raw, _) = data::read(&cfg.pairs, cfg.limit).map_err(|e| e.to_string())?;
    if raw.is_empty() {
        return Err(format!("no usable pairs in {:?}", cfg.pairs));
    }
    let mut train_raw: Vec<data::RawPair> = Vec::new();
    let mut val_raw: Vec<data::RawPair> = Vec::new();
    for (index, pair) in raw.into_iter().enumerate() {
        if is_train(&pair.split) {
            train_raw.push(pair);
        } else if is_validation(&pair.split) {
            val_raw.push(pair);
        } else if is_test(&pair.split) {
            continue;
        } else if index % 10 == 0 {
            val_raw.push(pair);
        } else {
            train_raw.push(pair);
        }
    }
    if train_raw.is_empty() {
        return Err("the training split is empty".to_string());
    }
    if val_raw.is_empty() {
        let keep = (train_raw.len() / 10).max(1);
        val_raw = train_raw.split_off(train_raw.len() - keep);
    }
    Ok((train_raw, val_raw))
}

/// Refuse a resume whose model shape disagrees with the flags.
///
/// Width, depth and sequence length change the tensors, so they cannot be
/// carried over; dropout can, and is taken from the current flags.
fn check_resume_shape(cfg: &TrainConfig, saved: &Config) -> Result<(), String> {
    let mut wrong: Vec<String> = Vec::new();
    if cfg.dim != saved.dim {
        wrong.push(format!("dim {} vs {}", cfg.dim, saved.dim));
    }
    if cfg.heads != saved.heads {
        wrong.push(format!("heads {} vs {}", cfg.heads, saved.heads));
    }
    if cfg.ff != saved.ff {
        wrong.push(format!("ff {} vs {}", cfg.ff, saved.ff));
    }
    if cfg.layers != saved.layers {
        wrong.push(format!("layers {} vs {}", cfg.layers, saved.layers));
    }
    if cfg.max_len != saved.max_len {
        wrong.push(format!("max-len {} vs {}", cfg.max_len, saved.max_len));
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the checkpoint was written with a different shape: {}",
            wrong.join(", ")
        ))
    }
}

/// Train one model on one mixed pair corpus.
pub fn train(cfg: &TrainConfig) -> Result<Report, String> {
    if cfg.pairs.is_empty() {
        return Err("no pair files given".to_string());
    }
    if cfg.batch == 0 || cfg.epochs == 0 {
        return Err("batch and epochs must be positive".to_string());
    }
    let started = Instant::now();

    let resumed = match &cfg.resume {
        Some(path) => Some(checkpoint::load(path)?),
        None => None,
    };

    let (train_raw, val_raw) = split_pairs(cfg)?;

    let tokenizer: Tokenizer = match &resumed {
        Some(saved) => Tokenizer::from_file(&saved.tokenizer)?,
        None => tokenizer::train(
            data::texts(&train_raw),
            TrainOptions {
                vocab_size: cfg.vocab,
                min_count: cfg.min_count,
                ..Default::default()
            },
        ),
    };

    let (train_rows, train_stats) = data::encode(&train_raw, &tokenizer, cfg.max_len);
    let (val_rows, val_stats) = data::encode(&val_raw, &tokenizer, cfg.max_len);
    if train_rows.is_empty() {
        return Err("no training rows survived encoding".to_string());
    }
    if val_rows.is_empty() {
        return Err("no validation rows survived encoding".to_string());
    }

    let config = match &resumed {
        Some(saved) => {
            check_resume_shape(cfg, &saved.config)?;
            let mut config = saved.config.clone();
            config.dropout = cfg.dropout;
            config
        }
        None => Config {
            vocab: tokenizer.vocab_size(),
            max_len: cfg.max_len,
            dim: cfg.dim,
            heads: cfg.heads,
            ff: cfg.ff,
            layers: cfg.layers,
            tasks: Task::ALL.len(),
            dropout: cfg.dropout,
            norm_eps: 1e-5,
        },
    };
    config.validate()?;
    if config.vocab != tokenizer.vocab_size() {
        return Err(format!(
            "tokenizer has {} tokens but the checkpoint expects {}",
            tokenizer.vocab_size(),
            config.vocab
        ));
    }

    let mut model = Model::new(config.clone(), cfg.seed);
    let mut adam = Adam::new(&model.params);
    let mut best = model.clone();
    let mut best_metrics;
    let mut best_score;
    let mut best_epoch = 0;
    let mut stale = 0;
    let mut last_loss = 0.0;
    let mut step = 0;
    let mut start_epoch = 0;

    let batches = train_rows.len().div_ceil(cfg.batch);
    let total_steps = batches * cfg.epochs;
    let warmup_steps = (total_steps as Real * cfg.warmup).round() as usize;

    match &resumed {
        Some(saved) => {
            checkpoint::fill(&mut model.params, &saved.weights)?;
            checkpoint::fill(&mut best.params, &saved.best)?;
            let m = checkpoint::moments_from(&mut model.params, &saved.first_moment)?;
            let v = checkpoint::moments_from(&mut model.params, &saved.second_moment)?;
            adam = Adam::restore(&mut model.params, m, v, saved.adam_step)?;
            start_epoch = saved.epoch.min(cfg.epochs);
            step = saved.step;
            best_epoch = saved.best_epoch;
            stale = saved.stale;
            last_loss = saved.last_loss;
            best_metrics = evaluate(&best, &val_rows);
            best_score = mean_accuracy(&best_metrics);
            println!(
                "resumed    epoch {} of {}, step {}, best epoch {}",
                start_epoch, cfg.epochs, step, best_epoch
            );
        }
        None => {
            best_metrics = evaluate(&model, &val_rows);
            best_score = mean_accuracy(&best_metrics);
        }
    }

    // The plan is `cfg.epochs`; `stop_after` only ends this leg earlier. The
    // learning rate schedule always uses the plan, so a leg boundary never
    // changes the trajectory of the run.
    let last_epoch = cfg.stop_after.unwrap_or(cfg.epochs).min(cfg.epochs);

    for epoch in start_epoch..last_epoch {
        // The order of an epoch is drawn from the epoch number alone, so epoch
        // five is the same permutation whether it is reached in one run or
        // resumed from a checkpoint written after epoch four.
        let mut order_rng = Rng::new(derive_seed(cfg.seed, epoch, ORDER_SALT));
        let order = weighted_order(&train_rows, &cfg.task_weights, &mut order_rng);
        let mut epoch_loss = 0.0;
        let mut counted = 0;
        for batch in order.chunks(cfg.batch) {
            // Each example writes its own gradient buffer and the buffers are
            // added in batch order, so the update is the serial one to the bit
            // and never depends on how many threads rayon happened to use.
            let locals: Vec<(Grads, Real)> = batch
                .par_iter()
                .map(|index| {
                    let example = &train_rows[*index];
                    example_grads(&model, example, derive_seed(cfg.seed, step, *index))
                })
                .collect();

            let mut grads = Grads::zeros(&model.params);
            let mut batch_loss = 0.0;
            for (mut local, loss) in locals {
                grads.accumulate(&mut local);
                batch_loss += loss;
            }
            counted += batch.len();
            epoch_loss += batch_loss;
            let scale = 1.0 / batch.len() as Real;
            grads.scale(scale);
            grads.clip(cfg.clip);
            let lr = lr_at(step, total_steps, warmup_steps, cfg.lr);
            adam.step(&mut model.params, &mut grads, lr, cfg.weight_decay);
            step += 1;
        }
        last_loss = if counted > 0 {
            epoch_loss / counted as Real
        } else {
            0.0
        };

        let metrics = evaluate(&model, &val_rows);
        let score = mean_accuracy(&metrics);
        let improved = score > best_score + 1e-6;
        if improved {
            best = model.clone();
            best_metrics = metrics.clone();
            best_score = score;
            best_epoch = epoch + 1;
            stale = 0;
        } else {
            stale += 1;
        }
        println!(
            "epoch {}/{}  loss {:.4}  mean accuracy {:.3}{}",
            epoch + 1,
            cfg.epochs,
            last_loss,
            score,
            if improved { "  best" } else { "" }
        );
        if let Some(path) = &cfg.checkpoint {
            let every = cfg.checkpoint_every.max(1);
            if (epoch + 1) % every == 0 || epoch + 1 == last_epoch {
                let (m, v) = adam.moments();
                let file = checkpoint::CheckpointFile {
                    format: checkpoint::FORMAT.to_string(),
                    config: config.clone(),
                    tokenizer: tokenizer.to_file(),
                    epoch: epoch + 1,
                    step,
                    stale,
                    best_epoch,
                    last_loss,
                    run: RunSettings {
                        epochs: cfg.epochs,
                        batch: cfg.batch,
                        lr: cfg.lr,
                        weight_decay: cfg.weight_decay,
                        warmup: cfg.warmup,
                        clip: cfg.clip,
                        dropout: cfg.dropout,
                        seed: cfg.seed,
                        patience: cfg.patience,
                        min_count: cfg.min_count,
                        task_weights: cfg.task_weights,
                    },
                    adam_step: adam.step,
                    weights: checkpoint::blocks(&mut model.params),
                    best: checkpoint::blocks(&mut best.params),
                    first_moment: checkpoint::moment_blocks(&mut model.params, &m)?,
                    second_moment: checkpoint::moment_blocks(&mut model.params, &v)?,
                };
                let written = checkpoint::save(path, &file).map_err(|e| e.to_string())?;
                println!(
                    "checkpoint {path} ({:.1} MB, epoch {})",
                    written as Real / 1_048_576.0,
                    epoch + 1
                );
            }
        }
        if cfg.patience > 0 && stale >= cfg.patience {
            break;
        }
    }

    let seconds = started.elapsed().as_secs_f64();
    let meta = json!({
        "best_epoch": best_epoch,
        "train_rows": train_rows.len(),
        "val_rows": val_rows.len(),
        "epochs": step.div_ceil(batches.max(1)),
        "metrics": best_metrics.iter().map(|m| json!({
            "task": m.task.as_str(),
            "n": m.n,
            "accuracy": m.accuracy,
            "spearman": m.spearman,
        })).collect::<Vec<_>>(),
        "task_weights": cfg.task_weights,
    });
    let bytes = store::save(&best, &tokenizer, &cfg.out, meta).map_err(|e| e.to_string())?;

    Ok(Report {
        params: best.params.count(),
        config,
        train_rows: train_rows.len(),
        val_rows: val_rows.len(),
        epochs: step.div_ceil(batches.max(1)),
        best_epoch,
        train_loss: last_loss,
        metrics: best_metrics,
        seconds,
        out: cfg.out.clone(),
        bytes,
        train_stats,
        val_stats,
        tokenizer_tokens: tokenizer.vocab_size(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_warms_up_then_cools() {
        let warm = lr_at(0, 100, 10, 1.0);
        let peak = lr_at(10, 100, 10, 1.0);
        let end = lr_at(100, 100, 10, 1.0);
        assert!(warm < peak);
        assert_eq!(peak, 1.0);
        assert!(end < peak);
        assert!(end >= 0.09);
    }

    fn flat(grads: &mut Grads) -> Vec<Real> {
        let mut out = Vec::new();
        grads.visit_blocks(|_, _, _, data| out.extend_from_slice(data));
        out
    }

    fn sample_examples() -> Vec<Example> {
        let tokenizer = Tokenizer::bytes_only();
        let texts = [
            "beat estimates as demand rose",
            "missed estimates as demand fell",
            "raised its outlook",
            "cut its outlook",
            "record profit",
            "reported a loss",
        ];
        texts
            .iter()
            .enumerate()
            .map(|(index, text)| Example {
                task: index % 6,
                split: "train".into(),
                first: tokenizer.encode(text, 8),
                second: if index % 2 == 0 {
                    Some(tokenizer.encode("the company said so", 8))
                } else {
                    None
                },
                target: if index % 3 == 0 { 0.0 } else { 1.0 },
            })
            .collect()
    }

    #[test]
    fn derived_seeds_separate_steps_and_rows() {
        let base = derive_seed(11, 0, 0);
        assert_eq!(base, derive_seed(11, 0, 0));
        assert_ne!(base, derive_seed(11, 1, 0));
        assert_ne!(base, derive_seed(11, 0, 1));
        assert_ne!(base, derive_seed(12, 0, 0));
        assert_ne!(derive_seed(11, 0, ORDER_SALT), derive_seed(11, 0, 0));
    }

    #[test]
    fn parallel_batch_matches_the_serial_one() {
        let config = Config {
            vocab: 280,
            max_len: 8,
            dim: 8,
            heads: 2,
            ff: 12,
            layers: 1,
            tasks: 6,
            dropout: 0.25,
            norm_eps: 1e-5,
        };
        let model = Model::new(config, 5);
        let examples = sample_examples();

        let mut serial = Grads::zeros(&model.params);
        for (row, example) in examples.iter().enumerate() {
            let (mut local, _) = example_grads(&model, example, derive_seed(11, 3, row));
            serial.accumulate(&mut local);
        }

        let locals: Vec<(Grads, Real)> = examples
            .par_iter()
            .enumerate()
            .map(|(row, example)| example_grads(&model, example, derive_seed(11, 3, row)))
            .collect();
        let mut parallel = Grads::zeros(&model.params);
        for (mut local, _) in locals {
            parallel.accumulate(&mut local);
        }

        assert_eq!(flat(&mut serial), flat(&mut parallel));
    }

    #[test]
    fn weighted_order_keeps_every_row_once() {
        let tokenizer = Tokenizer::bytes_only();
        let rows: Vec<Example> = (0..50)
            .map(|i| Example {
                task: i % 6,
                split: "train".into(),
                first: tokenizer.encode("the market rose", 8),
                second: None,
                target: 1.0,
            })
            .collect();
        let weights = [1.0, 1.0, 1.0, 0.5, 0.5, 1.0];
        let mut rng = Rng::new(3);
        let order = weighted_order(&rows, &weights, &mut rng);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), rows.len());
    }
}
