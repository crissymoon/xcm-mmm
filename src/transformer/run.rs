//! The command line side of the transformer: train, evaluate, inspect, encode.
//!
//! The printing lives here rather than in the binary so a test can call the
//! same path, and so the binary stays a set of argument structs. Every command
//! that reads a model goes through [`store::load`], which means a corrupt or
//! stale file fails with a sentence about the tensor that disagreed.

use std::error::Error;
use std::io;

use serde_json::json;

use crate::pairs::Task;
use crate::text::doc;
use crate::transformer::data;
use crate::transformer::store;
use crate::transformer::tensor::Real;
use crate::transformer::train::{self, Report, TaskMetric, TrainConfig};

/// Train a model and return its report.
pub fn train_model(config: &TrainConfig) -> Result<Report, Box<dyn Error>> {
    let report = train::train(config)?;
    for line in report.lines() {
        println!("{line}");
    }
    Ok(report)
}

/// What an evaluation run found.
#[derive(Clone, Debug)]
pub struct EvalReport {
    pub rows: usize,
    pub metrics: Vec<TaskMetric>,
}

/// Score a pair corpus with a saved model.
pub fn eval_model(
    model_path: &str,
    pairs: &[String],
    limit: usize,
) -> Result<EvalReport, Box<dyn Error>> {
    let (model, tokenizer, _meta) = store::load(model_path)?;
    let (raw, read_stats) = data::read(pairs, limit)?;
    let (examples, stats) = data::encode(&raw, &tokenizer, model.config.max_len);
    if examples.is_empty() {
        return Err("no rows survived encoding".into());
    }
    let metrics = train::evaluate(&model, &examples);
    println!(
        "rows      {} read, {} scored{}",
        read_stats.rows,
        stats.kept,
        if stats.skipped.is_empty() {
            String::new()
        } else {
            format!(" (skipped {})", stats.reasons().join(", "))
        }
    );
    println!(
        "{:<10} {:>7} {:>9} {:>9}",
        "task", "rows", "accuracy", "spearman"
    );
    for metric in &metrics {
        println!(
            "{:<10} {:>7} {:>9.3} {:>9}",
            metric.task.as_str(),
            metric.n,
            metric.accuracy,
            metric
                .spearman
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string())
        );
    }
    Ok(EvalReport {
        rows: examples.len(),
        metrics,
    })
}

/// Print the shape and the metadata of a saved model.
pub fn info_model(path: &str) -> Result<(), Box<dyn Error>> {
    let (model, tokenizer, meta) = store::load(path)?;
    let config = &model.config;
    println!(
        "config    dim {} heads {} layers {} ff {} max_len {} vocab {}",
        config.dim, config.heads, config.layers, config.ff, config.max_len, config.vocab
    );
    let mut params = model.params.clone();
    println!(
        "params    {} tensors, {} scalars",
        params.blocks_mut().len(),
        model.params.count()
    );
    println!(
        "tokenizer {} tokens, {} merges",
        tokenizer.vocab_size(),
        tokenizer.merge_count()
    );
    if !meta.is_null() {
        let compact = serde_json::to_string(&meta)?;
        println!("meta      {compact}");
    }
    Ok(())
}

/// Encode a text and score it with every task head.
pub fn encode_text(path: &str, text: &str) -> Result<(), Box<dyn Error>> {
    let (model, tokenizer, _meta) = store::load(path)?;
    let ids = tokenizer.encode(text, model.config.max_len);
    if ids.is_empty() {
        return Err("the text encoded to nothing".into());
    }
    println!("tokens    {}", ids.len());
    let pieces: Vec<String> = ids
        .iter()
        .map(|id| String::from_utf8_lossy(tokenizer.token_bytes(*id)).to_string())
        .collect();
    println!("pieces    {}", pieces.join(" | "));
    println!("decoded   {}", tokenizer.decode(&ids));
    println!("{:<10} {:>10}", "task", "score");
    for task in Task::ALL {
        let score = model.score(&ids, data::task_index(task));
        println!("{:<10} {score:>10.4}", task.as_str());
    }
    Ok(())
}

/// Score every text row in a corpus with a saved model.
///
/// The row reader is the same one the cleaning stages use, so a corpus that
/// already went through `clean` is scored from the fields it carries. Without
/// `task` the line holds every head, which is what a downstream ranker wants
/// when it does not know yet which objective the pair corpus favours; with a
/// task the line is one number.
pub fn score_rows(
    model_path: &str,
    rows_path: &str,
    out_path: &str,
    task: Option<Task>,
) -> Result<usize, Box<dyn Error>> {
    let (model, tokenizer, _meta) = store::load(model_path)?;
    let mut sink = doc::Sink::create(out_path)?;
    let mut written = 0usize;
    let mut skipped = 0usize;
    let mut failure: Option<io::Error> = None;
    doc::for_each_line(rows_path, |line| {
        if failure.is_some() || line.trim().is_empty() {
            return;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            skipped += 1;
            return;
        };
        let Some(row) = doc::TextRow::from_json(&value) else {
            skipped += 1;
            return;
        };
        let ids = tokenizer.encode(&row.text, model.config.max_len);
        if ids.is_empty() {
            skipped += 1;
            return;
        }
        let payload = match task {
            Some(name) => json!({
                "id": row.id,
                "task": name.as_str(),
                "score": model.score(&ids, data::task_index(name)),
            }),
            None => {
                let mut scores = serde_json::Map::new();
                for name in Task::ALL {
                    scores.insert(
                        name.as_str().to_string(),
                        json!(model.score(&ids, data::task_index(name))),
                    );
                }
                json!({"id": row.id, "tokens": ids.len(), "scores": scores})
            }
        };
        if let Err(error) = sink.line(&payload.to_string()) {
            failure = Some(error);
            return;
        }
        written += 1;
    })?;
    sink.finish()?;
    if let Some(error) = failure {
        return Err(error.into());
    }
    println!("rows      {written} written, {skipped} skipped");
    println!("out       {out_path}");
    Ok(written)
}

/// Parse `rank=3,label=0.5` into the six sampling weights.
///
/// A task that is not named keeps its default. Negative weights are refused
/// because a negative weight makes the exponential race meaningless.
pub fn parse_task_weights(spec: &str) -> Result<[Real; 6], String> {
    let mut weights = TrainConfig::default().task_weights;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, value) = part
            .split_once('=')
            .ok_or_else(|| format!("task weight {part:?} is not name=value"))?;
        let task: Task = name
            .trim()
            .parse()
            .map_err(|_| format!("unknown task {:?}", name.trim()))?;
        let weight: Real = value
            .trim()
            .parse()
            .map_err(|_| format!("weight {:?} is not a number", value.trim()))?;
        if weight < 0.0 {
            return Err(format!("weight for {task} is negative"));
        }
        weights[data::task_index(task)] = weight;
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_parse_and_default() {
        let weights = parse_task_weights("rank=3,label=0.5").expect("parse");
        assert_eq!(weights[data::task_index(Task::Rank)], 3.0);
        assert_eq!(weights[data::task_index(Task::Label)], 0.5);
        assert_eq!(weights[data::task_index(Task::Contrast)], 1.0);
    }

    #[test]
    fn a_bad_weight_is_refused() {
        assert!(parse_task_weights("nope=1").is_err());
        assert!(parse_task_weights("rank=-1").is_err());
        assert!(parse_task_weights("rank").is_err());
    }
}
