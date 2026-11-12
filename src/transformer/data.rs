//! Turning the pair schema back into encoded rows.
//!
//! Two passes, because the tokenizer must be fitted on the training split
//! before anything can be encoded: [`read`] collects the rows and keeps the
//! text, [`encode`] turns each side into token ids. The split field travels
//! with the row rather than being inferred, so a pair mined from a validation
//! session stays in validation here.
//!
//! Which side is which is a task decision and it lives in one place, so a
//! change to the miner's convention is one edit rather than a hunt:
//!
//! * `rank` and `preference` carry the better outcome in `a`; a negative margin
//!   means the file disagrees, so the sides are swapped.
//! * `contrast` is an anchor in `a` and the negative in `b`, whatever the
//!   outcomes say.
//! * `correction` is dirty in `a` and clean in `b`, so the model is trained to
//!   prefer the cleaned text.
//! * `retrieval` is query in `a` and document in `b`.
//! * `label` is pointwise: the target is the direction of `y_a`, not the value,
//!   because a score head that tries to predict a raw return is a regression in
//!   disguise and this crate already has a regression forecaster.
//!
//! A row that cannot be used is counted with its reason rather than dropped
//! silently, and the count is printed by the trainer.

use std::collections::BTreeMap;
use std::io;

use crate::pairs::{self, Task};
use crate::text::doc;
use crate::transformer::tensor::Real;
use crate::transformer::tokenizer::Tokenizer;

/// A pair as read, before encoding.
#[derive(Clone, Debug)]
pub struct RawPair {
    pub task: Task,
    pub split: String,
    pub first: String,
    pub second: Option<String>,
    pub target: Real,
}

/// A pair after encoding.
#[derive(Clone, Debug)]
pub struct Example {
    pub task: usize,
    pub split: String,
    pub first: Vec<u32>,
    pub second: Option<Vec<u32>>,
    pub target: Real,
}

/// How many rows were read and why the rest were dropped.
#[derive(Clone, Debug, Default)]
pub struct LoadStats {
    pub rows: usize,
    pub kept: usize,
    pub skipped: BTreeMap<String, usize>,
}

impl LoadStats {
    fn skip(&mut self, reason: &str) {
        *self.skipped.entry(reason.to_string()).or_insert(0) += 1;
    }

    /// One line per reason, for the report.
    pub fn reasons(&self) -> Vec<String> {
        self.skipped
            .iter()
            .map(|(reason, count)| format!("{reason} {count}"))
            .collect()
    }
}

/// Where a task sits in the head table.
pub fn task_index(task: Task) -> usize {
    Task::ALL
        .iter()
        .position(|candidate| *candidate == task)
        .unwrap_or(0)
}

fn non_empty(text: &str) -> bool {
    !text.trim().is_empty()
}

fn convert(pair: &pairs::Pair) -> Result<RawPair, &'static str> {
    let split = if pair.split.is_empty() {
        "train".to_string()
    } else {
        pair.split.clone()
    };
    let task = pair.task;
    match task {
        Task::Label => {
            let outcome = pair.y_a.or(pair.y_b).ok_or("label without an outcome")?;
            if !non_empty(&pair.text_a) {
                return Err("empty text");
            }
            Ok(RawPair {
                task,
                split,
                first: pair.text_a.clone(),
                second: None,
                target: if outcome > 0.0 { 1.0 } else { 0.0 },
            })
        }
        Task::Correction => {
            let clean = pair
                .text_b
                .as_deref()
                .ok_or("correction without a clean side")?;
            if !non_empty(clean) || !non_empty(&pair.text_a) {
                return Err("empty text");
            }
            Ok(RawPair {
                task,
                split,
                first: clean.to_string(),
                second: Some(pair.text_a.clone()),
                target: 1.0,
            })
        }
        _ => {
            let mut second = pair.text_b.as_deref().ok_or("pair without a second side")?;
            let mut first = pair.text_a.as_str();
            if matches!(task, Task::Rank | Task::Preference)
                && pair.margin.map(|m| m < 0.0).unwrap_or(false)
            {
                std::mem::swap(&mut first, &mut second);
            }
            if !non_empty(first) || !non_empty(second) {
                return Err("empty text");
            }
            Ok(RawPair {
                task,
                split,
                first: first.to_string(),
                second: Some(second.to_string()),
                target: 1.0,
            })
        }
    }
}

/// Read pair files, up to `limit` rows in total. Zero reads everything.
pub fn read(paths: &[String], limit: usize) -> io::Result<(Vec<RawPair>, LoadStats)> {
    let mut stats = LoadStats::default();
    let mut out: Vec<RawPair> = Vec::new();
    for path in paths {
        doc::for_each_line(path, |line| {
            if limit > 0 && out.len() >= limit {
                return;
            }
            if line.trim().is_empty() {
                return;
            }
            stats.rows += 1;
            let Some(pair) = pairs::from_line(line) else {
                stats.skip("unparsed");
                return;
            };
            match convert(&pair) {
                Ok(raw) => out.push(raw),
                Err(reason) => stats.skip(reason),
            }
        })?;
    }
    stats.kept = out.len();
    Ok((out, stats))
}

/// Encode both sides of every row. Rows that encode to nothing are dropped.
pub fn encode(
    pairs: &[RawPair],
    tokenizer: &Tokenizer,
    max_len: usize,
) -> (Vec<Example>, LoadStats) {
    let mut stats = LoadStats {
        rows: pairs.len(),
        ..Default::default()
    };
    let mut out = Vec::with_capacity(pairs.len());
    for raw in pairs {
        let first = tokenizer.encode(&raw.first, max_len);
        if first.is_empty() {
            stats.skip("empty encoding");
            continue;
        }
        let second = match &raw.second {
            Some(text) => {
                let ids = tokenizer.encode(text, max_len);
                if ids.is_empty() {
                    stats.skip("empty encoding");
                    continue;
                }
                Some(ids)
            }
            None => None,
        };
        out.push(Example {
            task: task_index(raw.task),
            split: raw.split.clone(),
            first,
            second,
            target: raw.target,
        });
    }
    stats.kept = out.len();
    (out, stats)
}

/// Every text on both sides of a set of rows, for fitting the tokenizer.
pub fn texts(pairs: &[RawPair]) -> Vec<&str> {
    let mut out = Vec::with_capacity(pairs.len() * 2);
    for pair in pairs {
        out.push(pair.first.as_str());
        if let Some(second) = &pair.second {
            out.push(second.as_str());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::Task;

    fn pair(task: Task) -> pairs::Pair {
        let mut p = pairs::Pair::new(task, "train");
        p.a = "a".into();
        p.b = Some("b".into());
        p.text_a = "the market rose".into();
        p.text_b = Some("the market fell".into());
        p
    }

    #[test]
    fn a_negative_margin_swaps_the_sides() {
        let mut p = pair(Task::Rank);
        p.margin = Some(-0.2);
        let raw = convert(&p).expect("convert");
        assert_eq!(raw.first, "the market fell");
        assert_eq!(raw.second.as_deref(), Some("the market rose"));
    }

    #[test]
    fn correction_prefers_the_clean_side() {
        let p = pair(Task::Correction);
        let raw = convert(&p).expect("convert");
        assert_eq!(raw.first, "the market fell");
        assert_eq!(raw.second.as_deref(), Some("the market rose"));
    }

    #[test]
    fn label_carries_a_direction_not_a_value() {
        let mut p = pair(Task::Label);
        p.text_b = None;
        p.y_a = Some(0.42);
        let raw = convert(&p).expect("convert");
        assert_eq!(raw.target, 1.0);
        assert!(raw.second.is_none());
    }

    #[test]
    fn read_and_encode_round_trip() {
        let mut p = pair(Task::Preference);
        p.stamp();
        let path = std::env::temp_dir().join(format!("xcm-mmm-pairs-{}.jsonl", std::process::id()));
        std::fs::write(&path, pairs::to_line(&p) + "\n").unwrap();
        let (raw, stats) = read(&[path.to_string_lossy().to_string()], 0).unwrap();
        assert_eq!(stats.kept, 1);
        let tokenizer = Tokenizer::bytes_only();
        let (examples, encoded) = encode(&raw, &tokenizer, 16);
        assert_eq!(encoded.kept, 1);
        assert!(!examples[0].first.is_empty());
        assert!(examples[0].second.is_some());
        std::fs::remove_file(&path).unwrap();
    }
}
