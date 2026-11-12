//! The stages wired into one pass, for the command line.
//!
//! The order is the one that spends the least work per document: normalise and
//! screen first, because a rejected document then costs nothing downstream;
//! score next, because the score is what deduplication uses to pick a survivor
//! and what selection uses to rank; deduplicate next, because duplicate
//! documents waste an inverted index slot each; select last, because it is the
//! only stage whose cost depends on how much survives.

use std::collections::BTreeMap;
use std::io;

use serde_json::json;

use crate::text::dedupe;
use crate::text::doc::{self, Sink, TextRow};
use crate::text::hash;
use crate::text::normalize::{self, Options};
use crate::text::quality::{self, Flag, Rules, Stats, Weights};
use crate::text::select::{self};
use crate::text::tokenize;

/// Everything the pass needs.
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub clean: Options,
    pub rules: Rules,
    pub weights: Weights,
    /// Merge near duplicates. The threshold is the Jaccard gate.
    pub dedupe: Option<f64>,
    /// Keep at most this many documents. Zero keeps every survivor.
    pub budget: usize,
    pub floor: f64,
    pub theta: f64,
    pub group_cap: f64,
    /// A curated target set, as a path to JSON Lines or plain text. When it is
    /// set, selection switches from greedy coverage to target-aware importance
    /// resampling, which is the only one of the two that knows what "relevant"
    /// means for this corpus.
    pub target: Option<String>,
    /// Buckets for the target and proposal densities.
    pub target_bins: usize,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            clean: Options::default(),
            rules: Rules::default(),
            weights: Weights::default(),
            dedupe: Some(0.80),
            budget: 0,
            floor: 0.0,
            theta: 0.85,
            group_cap: 0.10,
            target: None,
            target_bins: 10_000,
        }
    }
}

/// Counts from one run.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub read: usize,
    pub malformed: usize,
    pub kept: usize,
    pub dropped_rules: usize,
    pub dropped_duplicate: usize,
    pub selected: usize,
    pub flags: BTreeMap<String, usize>,
    pub mean_quality: f64,
    pub mean_words: f64,
    pub duplicates_merged: usize,
    pub largest_cluster: usize,
    pub redundant: usize,
    pub over_cap: usize,
    /// Rows the second pass took after the group cap had refused them.
    pub relaxed: usize,
    pub feature_coverage: f64,
    pub groups: usize,
    /// Whether the target-aware rule chose the survivors.
    pub target_aware: bool,
    /// Rows in the target set.
    pub target_rows: usize,
    pub chars_in: usize,
    pub chars_out: usize,
}

/// A row on its way out, with the measurements that justify keeping it.
fn row_json(row: &TextRow, text: &str, stats: &Stats, score: f64, flags: &[Flag]) -> String {
    let names: Vec<&'static str> = flags.iter().map(|f| f.reason()).collect();
    json!({
        "id": row.id,
        "text": text,
        "source": row.source,
        "group": row.group,
        "split": row.split,
        "day": row.day,
        "chars": stats.chars,
        "words": stats.words,
        "quality": (score * 1e6).round() / 1e6,
        "flags": names,
    })
    .to_string()
}

/// Run the pass and write the surviving rows.
pub fn run(input: &str, out: &str, p: &Pipeline) -> io::Result<Summary> {
    let (rows, malformed) = doc::read_rows(input)?;
    let mut summary = Summary {
        read: rows.len(),
        malformed,
        ..Summary::default()
    };
    let mut kept: Vec<(TextRow, String, f64, usize)> = Vec::with_capacity(rows.len());
    let mut flag_counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        let (text, report) = normalize::clean(&row.text, &p.clean);
        summary.chars_in += report.chars_in;
        summary.chars_out += report.chars_out;
        if text.is_empty() {
            summary.dropped_rules += 1;
            continue;
        }
        let stats = Stats::of(&text);
        let flags = p.rules.flags(&stats);
        if !flags.is_empty() {
            summary.dropped_rules += 1;
            for flag in &flags {
                *flag_counts.entry(format!("{flag:?}")).or_insert(0) += 1;
            }
            continue;
        }
        let score = quality::score(&stats, &p.weights).total;
        kept.push((row, text, score, stats.words));
    }
    summary.flags = flag_counts;

    if let Some(threshold) = p.dedupe {
        let texts: Vec<&str> = kept.iter().map(|(_, t, _, _)| t.as_str()).collect();
        let cfg = dedupe::Config {
            threshold,
            ..dedupe::Config::default()
        };
        let sigs = dedupe::signatures(&texts, &cfg);
        let exact: Vec<u64> = texts.iter().map(|t| hash::text_hash(t)).collect();
        let scores: Vec<f64> = kept.iter().map(|(_, _, s, _)| *s).collect();
        let (survivors, report) = dedupe::cluster(&sigs, &exact, &scores, &cfg);
        summary.duplicates_merged = report.duplicates;
        summary.largest_cluster = report.largest_cluster;
        let mut next = Vec::with_capacity(survivors.len());
        for index in survivors {
            next.push(kept[index].clone());
        }
        summary.dropped_duplicate = kept.len() - next.len();
        kept = next;
    }
    summary.kept = kept.len();

    if let Some(path) = &p.target {
        let target = read_texts(path)?;
        let texts: Vec<&str> = kept.iter().map(|(_, t, _, _)| t.as_str()).collect();
        let bins = p.target_bins.max(64).min(texts.len().max(64));
        let target_refs: Vec<&str> = target.iter().map(|t| t.as_str()).collect();
        let dsir = select::Dsir::fit(&target_refs, &texts, bins);
        let weights = dsir.log_weights(&texts);
        let mut rng = crate::rng::Rng::new(0x5eed);
        let k = if p.budget > 0 {
            p.budget.min(texts.len())
        } else {
            texts.len()
        };
        let chosen = dsir.resample(&weights, k, &mut rng);
        summary.selected = chosen.len();
        summary.target_aware = true;
        summary.target_rows = target.len();
        summary.groups = kept
            .iter()
            .map(|(r, _, _, _)| r.group.as_str())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let mut next = Vec::with_capacity(chosen.len());
        for index in chosen {
            next.push(kept[index].clone());
        }
        kept = next;
    } else if p.budget > 0 || p.floor > 0.0 || p.theta < 1.0 {
        let texts: Vec<&str> = kept.iter().map(|(_, t, _, _)| t.as_str()).collect();
        let df = select::document_frequency(&texts);
        let vectors: Vec<Vec<(u32, f32)>> = texts
            .iter()
            .map(|t| select::weigh(&select::counts(t), &df, texts.len()))
            .collect();
        let scores: Vec<f64> = kept.iter().map(|(_, _, s, _)| *s).collect();
        let groups: Vec<String> = kept.iter().map(|(r, _, _, _)| r.group.clone()).collect();
        let words: Vec<usize> = kept.iter().map(|(_, _, _, w)| *w).collect();
        let cfg = select::Config {
            budget: p.budget,
            floor: p.floor,
            theta: p.theta,
            group_cap: p.group_cap,
        };
        let (chosen, report) = select::select(&vectors, &scores, &groups, &words, &cfg);
        summary.selected = report.selected;
        summary.redundant = report.redundant;
        summary.over_cap = report.over_cap;
        summary.relaxed = report.relaxed;
        summary.feature_coverage = report.feature_coverage;
        summary.groups = report.groups;
        let mut next = Vec::with_capacity(chosen.len());
        for index in chosen {
            next.push(kept[index].clone());
        }
        kept = next;
    } else {
        summary.selected = kept.len();
    }

    let count = kept.len().max(1);
    summary.mean_quality = kept.iter().map(|(_, _, s, _)| *s).sum::<f64>() / count as f64;
    summary.mean_words = kept.iter().map(|(_, _, _, w)| *w as f64).sum::<f64>() / count as f64;

    let mut sink = Sink::create(out)?;
    for (row, text, score, _) in &kept {
        let stats = Stats::of(text);
        sink.line(&row_json(row, text, &stats, *score, &[]))?;
    }
    sink.finish()?;
    Ok(summary)
}

/// Read a target set: JSON Lines with a `text` field, or one document per line.
pub fn read_texts(path: &str) -> io::Result<Vec<String>> {
    let mut out = Vec::new();
    doc::for_each_line(path, |line| {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
                out.push(text.to_string());
                return;
            }
        }
        out.push(line.to_string());
    })?;
    Ok(out)
}

/// Fit a quality model from two files of documents, one kept and one dropped.
pub fn fit(
    pos: &str,
    neg: &str,
    epochs: usize,
    lr: f64,
    l2: f64,
) -> io::Result<Option<quality::QualityModel>> {
    let load = |path: &str| -> io::Result<Vec<Stats>> {
        let (rows, _) = doc::read_rows(path)?;
        Ok(rows
            .iter()
            .map(|r| Stats::of(&normalize::clean(&r.text, &Options::default()).0))
            .collect())
    };
    let (positive, negative) = (load(pos)?, load(neg)?);
    Ok(quality::QualityModel::fit(
        &negative, &positive, epochs, lr, l2,
    ))
}

/// Truncate every row of a pair of files to a word boundary, which is what the
/// pair packer does to the texts it inlines.
pub fn truncate_field(input: &str, out: &str, max_chars: usize) -> io::Result<usize> {
    let mut sink = Sink::create(out)?;
    let mut n = 0;
    doc::for_each_line(input, |line| {
        if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = value
                .get_mut("text")
                .and_then(|t| t.as_str())
                .map(str::to_string)
            {
                value["text"] = json!(tokenize::truncate(&text, max_chars));
                sink.line(&value.to_string()).ok();
                n += 1;
            }
        }
    })?;
    sink.finish()?;
    Ok(n)
}
