//! One pair schema for every task, and the stages that build it.
//!
//! A single model trained on one file needs one row shape. The shape here holds
//! six kinds of pair, distinguished by [`Task`], and leaves the fields a task
//! does not need absent rather than empty:
//!
//! * `rank` pairs two documents from the same session whose forward returns
//!   landed in different quartiles. `y_a` and `y_b` carry both returns and
//!   `margin` is their gap.
//! * `contrast` pairs an anchor with a positive (same issuer, earlier session)
//!   and a negative (same session, different issuer). The negative is the hard
//!   case and `difficulty` records how close it came. It carries both outcomes
//!   but no margin, because the label is temporal and cross group rather than an
//!   ordering, and a margin that is not the label would be a false promise.
//! * `preference` pairs two documents on the same issuer where one outcome was
//!   clearly better. It is the rank bank with the group held fixed.
//! * `label` is a single document and its outcome, for a pointwise head.
//! * `correction` pairs text that failed a rule with the same text cleaned, so a
//!   model can learn the cleaning itself.
//! * `retrieval` pairs a query shaped from a document with the document, for a
//!   contrastive encoder.
//!
//! [`mine`] builds the pairs from labelled rows, [`pack`] merges files into one
//! corpus, and [`audit`] answers the questions that have to be answered before
//! training: is anything leaked across splits, is any pair trivial, and is the
//! task mix the one that was asked for.

pub mod audit;
pub mod mine;
pub mod pack;
pub mod run;

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::text::hash;

/// What a pair is for.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    #[default]
    Rank,
    Contrast,
    Preference,
    Label,
    Correction,
    Retrieval,
}

impl Task {
    /// Every task, in a stable order.
    pub const ALL: [Task; 6] = [
        Task::Rank,
        Task::Contrast,
        Task::Preference,
        Task::Label,
        Task::Correction,
        Task::Retrieval,
    ];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Rank => "rank",
            Task::Contrast => "contrast",
            Task::Preference => "preference",
            Task::Label => "label",
            Task::Correction => "correction",
            Task::Retrieval => "retrieval",
        }
    }

    /// Whether the task pairs two documents rather than scoring one.
    pub fn is_pairwise(self) -> bool {
        !matches!(self, Task::Label | Task::Correction)
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rank" => Ok(Task::Rank),
            "contrast" => Ok(Task::Contrast),
            "preference" | "dpo" => Ok(Task::Preference),
            "label" | "pointwise" => Ok(Task::Label),
            "correction" | "clean" => Ok(Task::Correction),
            "retrieval" | "query" => Ok(Task::Retrieval),
            other => Err(format!("unknown task {other:?}")),
        }
    }
}

/// How the negative side of a pair was chosen. A model is only as good as its
/// negatives, so the provenance travels with the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NegSource {
    /// Drawn from the other quartile of the same session.
    Rank,
    /// Drawn by feature similarity.
    Similarity,
    /// Drawn from an earlier or later session.
    Temporal,
    /// Drawn without looking at the text.
    Random,
}

impl NegSource {
    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            NegSource::Rank => "rank",
            NegSource::Similarity => "similarity",
            NegSource::Temporal => "temporal",
            NegSource::Random => "random",
        }
    }
}

fn is_zero(value: &Option<f64>) -> bool {
    value.map(|v| v == 0.0).unwrap_or(true)
}

fn is_zero_count(value: &Option<usize>) -> bool {
    value.map(|v| v == 0).unwrap_or(true)
}

/// One training pair, whatever its task.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Pair {
    pub id: String,
    pub task: Task,
    pub split: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub day: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub label_end: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub a: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub b: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub pos: Option<String>,
    pub text_a: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text_b: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text_pos: Option<String>,
    #[serde(skip_serializing_if = "is_zero", default)]
    pub y_a: Option<f64>,
    #[serde(skip_serializing_if = "is_zero", default)]
    pub y_b: Option<f64>,
    /// Gap in outcome between the two sides, positive when `a` is better.
    #[serde(skip_serializing_if = "is_zero", default)]
    pub margin: Option<f64>,
    /// How close the two sides were in feature space, in [0, 1]. A high value
    /// means the pair is hard; a value near zero means one side is obvious.
    #[serde(skip_serializing_if = "is_zero", default)]
    pub difficulty: Option<f64>,
    #[serde(skip_serializing_if = "is_zero", default)]
    pub quality_a: Option<f64>,
    #[serde(skip_serializing_if = "is_zero", default)]
    pub quality_b: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub neg_source: Option<NegSource>,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub source_a: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub source_b: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub group_a: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub group_b: String,
    #[serde(skip_serializing_if = "is_zero_count", default)]
    pub words_a: Option<usize>,
    #[serde(skip_serializing_if = "is_zero_count", default)]
    pub words_b: Option<usize>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Pair {
    /// A pair with the digest id filled in and the task set.
    pub fn new(task: Task, split: &str) -> Self {
        Self {
            task,
            split: split.to_string(),
            ..Self::default()
        }
    }

    /// Recompute the id from the task and the two document ids. Two runs on the
    /// same rows produce the same ids, which is what makes the merge step
    /// idempotent.
    pub fn stamp(&mut self) {
        let key = format!(
            "{}|{}|{}|{}",
            self.task,
            self.a,
            self.b.clone().unwrap_or_default(),
            self.pos.clone().unwrap_or_default()
        );
        self.id = hash::hex64(hash::fnv1a(key.as_bytes()));
    }

    /// Every text this pair carries, for the audits that need it.
    pub fn texts(&self) -> Vec<&str> {
        let mut out = vec![self.text_a.as_str()];
        if let Some(t) = &self.text_b {
            out.push(t);
        }
        if let Some(t) = &self.text_pos {
            out.push(t);
        }
        out
    }

    /// Document ids this pair carries.
    pub fn doc_ids(&self) -> Vec<&str> {
        let mut out = vec![self.a.as_str()];
        if let Some(b) = &self.b {
            out.push(b);
        }
        if let Some(p) = &self.pos {
            out.push(p);
        }
        out.retain(|id| !id.is_empty());
        out
    }
}

/// Serialise a pair to one line, without a trailing newline.
pub fn to_line(pair: &Pair) -> String {
    serde_json::to_string(pair).unwrap_or_else(|_| "{}".to_string())
}

/// Read a pair from one line.
pub fn from_line(line: &str) -> Option<Pair> {
    serde_json::from_str(line).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_is_stable_and_distinct() {
        let mut a = Pair::new(Task::Rank, "train");
        a.a = "d1".into();
        a.b = Some("d2".into());
        a.text_a = "one".into();
        let mut b = a.clone();
        let mut c = a.clone();
        a.stamp();
        b.stamp();
        c.b = Some("d3".into());
        c.stamp();
        assert_eq!(a.id, b.id);
        assert_ne!(a.id, c.id);
        assert_eq!(a.id.len(), 16);
    }

    #[test]
    fn round_trip_drops_absent_fields() {
        let mut p = Pair::new(Task::Label, "test");
        p.a = "x".into();
        p.text_a = "hello".into();
        p.y_a = Some(0.5);
        p.stamp();
        let line = to_line(&p);
        assert!(!line.contains("text_b"));
        assert!(!line.contains("margin"));
        let back = from_line(&line).expect("round trip");
        assert_eq!(back.task, Task::Label);
        assert_eq!(back.y_a, Some(0.5));
        assert_eq!(back.texts(), vec!["hello"]);
    }

    #[test]
    fn task_names_parse_both_ways() {
        for task in Task::ALL {
            assert_eq!(Task::from_str(task.as_str()).unwrap(), task);
        }
        assert_eq!(Task::from_str("dpo").unwrap(), Task::Preference);
        assert!(Task::from_str("nope").is_err());
    }
}
