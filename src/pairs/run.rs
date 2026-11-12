//! The pair stages wired into one pass, for the command line.
//!
//! `mine` reads labelled rows and writes the banks, `pack` merges files into one
//! corpus with a task mix, and `audit` reads a corpus and says whether it is
//! safe to train on. All three write a manifest next to the output so a run can
//! be described without re-running it.

use std::io;

use serde_json::{json, Value};

use crate::pairs::audit::{self, Config as AuditConfig};
use crate::pairs::mine::{self, Config as MineConfig, Row};
use crate::pairs::pack::{self, Plan};
use crate::pairs::{self, Pair};
use crate::text::doc::{self, Sink};
use crate::text::normalize::{self, Options};

/// Read labelled rows. A row needs a text and an outcome; the outcome is `y`,
/// `y_a` or `label`, and a row without one is skipped rather than paired with a
/// zero.
pub fn read_labelled(path: &str) -> io::Result<(Vec<Row>, usize)> {
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    doc::for_each_line(path, |line| {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            skipped += 1;
            return;
        };
        let text = ["text", "text_a", "body"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .map(str::to_string);
        let y = ["y", "y_a", "label", "ret"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_f64));
        let (Some(text), Some(y)) = (text, y) else {
            skipped += 1;
            return;
        };
        let cleaned = normalize::clean(&text, &Options::default()).0;
        if cleaned.is_empty() {
            skipped += 1;
            return;
        }
        let field = |keys: &[&str], default: &str| -> String {
            keys.iter()
                .find_map(|k| value.get(*k).and_then(Value::as_str))
                .unwrap_or(default)
                .to_string()
        };
        let group = ["group", "a_ticker", "publisher", "source"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .map(str::to_string)
            .or_else(|| {
                value
                    .get("tickers")
                    .and_then(Value::as_array)
                    .and_then(|list| list.first())
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string());
        let id = ["id", "doc", "a"]
            .iter()
            .find_map(|k| value.get(*k).and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| crate::text::hash::hex64(crate::text::hash::text_hash(&cleaned)));
        rows.push(Row {
            id,
            text: cleaned,
            group,
            source: field(&["source", "publisher", "a_ticker"], "unknown"),
            day: field(&["day", "ts"], ""),
            split: field(&["split"], "train"),
            label_end: field(&["label_end"], ""),
            y,
        });
    })?;
    Ok((rows, skipped))
}

/// Mine the banks and write them, returning the report and the rows written.
pub fn run_mine(input: &str, out: &str, cfg: &MineConfig) -> io::Result<(mine::Report, usize)> {
    let (rows, skipped) = read_labelled(input)?;
    let (pairs, mut report) = mine::mine(&rows, cfg);
    report.rows = rows.len().max(report.rows);
    let mut sink = Sink::create(out)?;
    for pair in &pairs {
        sink.line(&pairs::to_line(pair))?;
    }
    sink.finish()?;
    let _ = skipped;
    Ok((report, pairs.len()))
}

/// Merge files and write the corpus plus its manifest.
pub fn run_pack(inputs: &[String], out: &str, plan: &Plan) -> io::Result<(pack::Report, String)> {
    let (rows, report) = pack::merge(inputs, plan)?;
    let mut sink = Sink::create(out)?;
    for pair in &rows {
        sink.line(&pairs::to_line(pair))?;
    }
    sink.finish()?;
    let manifest = manifest_json(&report, plan, inputs);
    Ok((report, manifest))
}

fn manifest_json(report: &pack::Report, plan: &Plan, inputs: &[String]) -> String {
    let tasks: Vec<Value> = report
        .by_task
        .iter()
        .map(|(task, count)| {
            json!({
                "task": task.as_str(),
                "rows": count,
                "realised": report
                    .realised
                    .iter()
                    .find(|(t, _)| t == task)
                    .map(|(_, share)| (share * 1e4).round() / 1e4),
                "weight": plan.weight_of(*task),
            })
        })
        .collect();
    let splits: Vec<Value> = report
        .by_split
        .iter()
        .map(|(split, count)| json!({"split": split, "rows": count}))
        .collect();
    json!({
        "inputs": inputs,
        "requested_total": plan.total,
        "seed": plan.seed,
        "max_similarity": plan.max_similarity,
        "read": report.read,
        "malformed": report.malformed,
        "duplicate_pairs": report.duplicate_pairs,
        "trivial": report.trivial,
        "kept": report.kept,
        "tasks": tasks,
        "splits": splits,
    })
    .to_string()
}

/// Audit a corpus against optional evaluation texts.
pub fn run_audit(input: &str, eval: Option<&str>, cfg: &AuditConfig) -> io::Result<audit::Report> {
    let mut rows: Vec<Pair> = Vec::new();
    let mut bad = 0usize;
    doc::for_each_line(input, |line| match pairs::from_line(line) {
        Some(pair) => rows.push(pair),
        None => bad += 1,
    })?;
    let eval_texts: Vec<String> = match eval {
        Some(path) => {
            let mut out = Vec::new();
            doc::for_each_line(path, |line| {
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    if let Some(text) = value.get("text").and_then(Value::as_str) {
                        out.push(text.to_string());
                    }
                } else {
                    out.push(line.to_string());
                }
            })?;
            out
        }
        None => Vec::new(),
    };
    let mut report = audit::audit(&rows, &eval_texts, cfg);
    if bad > 0 {
        report.problems.push(format!("{bad} lines are not pairs"));
    }
    Ok(report)
}
