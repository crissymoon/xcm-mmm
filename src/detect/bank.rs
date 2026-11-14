//! The three files a detector folder carries, and the faults that stop a fit.
//!
//! `bank.jsonl` is the labelled sentence bank, `ads.jsonl` is ad copy at
//! document grain with its source split, and `cues.json` is the declared cue
//! table for the index axes. Each reader returns what it could parse plus the
//! lines it could not, so a truncated file is a refusal with a count rather than
//! a silently smaller bank.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::text::hash::{fnv1a, hex64};

use super::index::IndexSpec;

/// One labelled sentence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BankRow {
    pub axis: String,
    pub label: u8,
    pub source: String,
    pub text: String,
}

/// One ad, as a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdRow {
    pub id: String,
    pub split: String,
    pub text: String,
}

/// The cue table for the index axes, and the feature list for the learned ones.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CuesFile {
    pub version: u32,
    pub axes: BTreeMap<String, IndexSpec>,
    /// Readings for the lexicon axes: the learned density plus the surface
    /// counts. An axis absent here is read on its density alone.
    #[serde(default)]
    pub features: BTreeMap<String, crate::detect::feature::FeatureSpec>,
}

/// Rows that parsed, and the lines that did not.
pub struct Read<T> {
    pub rows: Vec<T>,
    pub bad: Vec<String>,
    pub bytes: Vec<u8>,
}

impl<T> Read<T> {
    pub fn digest(&self) -> String {
        hex64(fnv1a(&self.bytes))
    }
}

fn read_lines<T: DeserializeOwned>(path: &str) -> io::Result<Read<T>> {
    let bytes = fs::read(path)?;
    let mut rows = Vec::new();
    let mut bad = Vec::new();
    for (number, line) in BufReader::new(bytes.as_slice()).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<T>(&line) {
            Ok(row) => rows.push(row),
            Err(e) => bad.push(format!("{}:{}: {e}", file_name(path), number + 1)),
        }
    }
    Ok(Read { rows, bad, bytes })
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

pub fn read_bank(path: &str) -> io::Result<Read<BankRow>> {
    read_lines(path)
}

pub fn read_ads(path: &str) -> io::Result<Read<AdRow>> {
    read_lines(path)
}

pub fn read_cues(path: &str) -> io::Result<Read<CuesFile>> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut rows = Vec::new();
    let mut bad = Vec::new();
    match serde_json::from_str::<CuesFile>(&text) {
        Ok(file) => rows.push(file),
        Err(e) => bad.push(format!("{}: {e}", file_name(path))),
    }
    Ok(Read { rows, bad, bytes })
}

/// The axis name the shared negative side of a bank carries.
///
/// One corpus of plain copy is the negative side of several axes: legal
/// register and abstraction are both fitted against the same advertisers' text.
/// The convention is one axis named `plain` holding label 0 rows, and an axis
/// with no negatives of its own takes that bank as its other side.
pub const SHARED_NEGATIVE: &str = "plain";

/// Faults that make a bank unfit to fit on, before any model is built.
///
/// Every rule here is a mistake that would otherwise be discovered as a strange
/// number instead of as a refusal: no positive side, no negative side to share,
/// an axis with too few rows to say anything, a text labelled both ways.
pub fn validate(bank: &[BankRow], min_rows: usize) -> Vec<String> {
    let mut faults = Vec::new();
    let mut positives: BTreeMap<&str, usize> = BTreeMap::new();
    let mut own_negatives: BTreeMap<&str, usize> = BTreeMap::new();
    let mut shared = 0usize;
    let mut labels: BTreeMap<(&str, &str), u8> = BTreeMap::new();
    for row in bank {
        if row.text.trim().is_empty() {
            faults.push(format!("axis {} holds an empty text", row.axis));
            continue;
        }
        if row.label > 1 {
            faults.push(format!("axis {} holds label {}", row.axis, row.label));
            continue;
        }
        if row.axis == SHARED_NEGATIVE {
            if row.label == 1 {
                faults.push(format!(
                    "the shared negative axis {SHARED_NEGATIVE} holds a positive row"
                ));
            } else {
                shared += 1;
            }
        } else if row.label == 1 {
            *positives.entry(row.axis.as_str()).or_insert(0) += 1;
        } else {
            *own_negatives.entry(row.axis.as_str()).or_insert(0) += 1;
        }
        match labels.insert((row.axis.as_str(), row.text.as_str()), row.label) {
            Some(previous) if previous != row.label => faults.push(format!(
                "axis {} labels one text both ways: {}",
                row.axis,
                shorten(&row.text)
            )),
            _ => {}
        }
    }
    if positives.is_empty() {
        faults.push("the bank holds no positive rows, so it fits no axis".to_string());
    }
    let needs_shared = positives
        .keys()
        .any(|axis| own_negatives.get(axis).copied().unwrap_or(0) < min_rows);
    if needs_shared && shared < min_rows {
        faults.push(format!(
            "the shared negative axis {SHARED_NEGATIVE} holds {shared} rows, under {min_rows}"
        ));
    }
    for (axis, count) in &positives {
        if *count < min_rows {
            faults.push(format!(
                "axis {axis} holds {count} positives, under {min_rows}"
            ));
        }
    }
    for (axis, count) in &own_negatives {
        if !positives.contains_key(axis) {
            faults.push(format!(
                "axis {axis} holds {count} negatives and no positives"
            ));
        }
    }
    faults
}

/// Faults in the ad corpus the index axes are calibrated on.
pub fn validate_ads(ads: &[AdRow], min_rows: usize) -> Vec<String> {
    let mut faults = Vec::new();
    if ads.len() < min_rows {
        faults.push(format!("{} ads, under {min_rows}", ads.len()));
    }
    for ad in ads {
        if ad.text.trim().is_empty() {
            faults.push(format!("ad {} holds no text", ad.id));
        }
    }
    let splits: BTreeSet<&str> = ads.iter().map(|a| a.split.as_str()).collect();
    if !splits.contains("train") {
        faults.push("no ad carries the train split, so nothing can be calibrated".to_string());
    }
    faults
}

fn shorten(text: &str) -> String {
    text.chars().take(48).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(axis: &str, label: u8, text: &str) -> BankRow {
        BankRow {
            axis: axis.into(),
            label,
            source: "test".into(),
            text: text.into(),
        }
    }

    #[test]
    fn a_bank_with_one_class_is_refused() {
        let bank = vec![row("legal", 1, "a"), row("legal", 1, "b")];
        let faults = validate(&bank, 1);
        assert!(faults.iter().any(|f| f.contains("shared")), "{faults:?}");
    }

    #[test]
    fn one_text_labelled_both_ways_is_refused() {
        let bank = vec![
            row("legal", 1, "the same words"),
            row("legal", 0, "the same words"),
        ];
        let faults = validate(&bank, 1);
        assert!(faults.iter().any(|f| f.contains("both ways")), "{faults:?}");
    }

    #[test]
    fn an_empty_text_is_refused() {
        let bank = vec![row("legal", 1, " "), row("plain", 0, "text")];
        let faults = validate(&bank, 1);
        assert!(faults.iter().any(|f| f.contains("empty")), "{faults:?}");
    }

    #[test]
    fn one_plain_bank_serves_several_positive_axes() {
        let bank = vec![
            row("legal", 1, "shall"),
            row("jargon", 1, "utilization"),
            row("plain", 0, "join"),
            row("plain", 0, "call"),
        ];
        assert!(validate(&bank, 1).is_empty(), "{:?}", validate(&bank, 1));
    }

    #[test]
    fn an_axis_with_negatives_and_no_positives_is_refused() {
        let bank = vec![row("legal", 1, "shall"), row("asides", 0, "join")];
        let faults = validate(&bank, 1);
        assert!(
            faults.iter().any(|f| f.contains("no positives")),
            "{faults:?}"
        );
    }
}
