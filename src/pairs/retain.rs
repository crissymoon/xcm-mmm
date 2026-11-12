//! Retention: one bank of the best pairs seen so far.
//!
//! Data arrives a day at a time and an old document comes back with a new
//! label, so a corpus rebuilt from scratch every morning throws away the work
//! of the day before and drifts in composition. This module keeps one file that
//! only ever improves. Each run scores the new candidates, admits the ones that
//! clear a floor, and lets a better pair push out a worse one.
//!
//! A pair is worth keeping when four things hold at once:
//!
//! * the outcome gap is real, so the label is not noise;
//! * the two sides are hard to tell apart, because an easy pair teaches
//!   nothing, but not so hard that the label is a coin flip;
//! * the anchor reads like prose, by the same composite score the corpus
//!   builder uses;
//! * the anchor is not a near duplicate of one already in the bank.
//!
//! Composition is capped as well. Without a cap per day or per issuer, one busy
//! Tuesday or one heavily covered ticker fills the bank and the model learns
//! that ticker instead of the signal.
//!
//! The index is a separate small file, so a second run reads thousands of
//! stamps rather than parsing the whole corpus again. Novelty is an exact
//! Hamming search up to seven bits over eight byte stripes: if two 64 bit
//! signatures are within seven bits they share at least one stripe, so scanning
//! the eight stripe buckets of one key finds every such neighbour.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pairs::{self, Pair, Task};
use crate::text::doc::{self, Sink};
use crate::text::{hash, quality, tokenize};

/// Bytes per stripe of the near duplicate index. Eight stripes of eight bits
/// make the Hamming screen exact for radii up to seven.
const STRIPES: usize = 8;
const STRIPE_BITS: u32 = 8;

/// How much each part of a pair counts. The gap leads, because a pair without
/// a real gap is not a training row at all.
const W_MARGIN: f64 = 0.40;
const W_DIFFICULTY: f64 = 0.25;
const W_QUALITY: f64 = 0.20;
const W_LENGTH: f64 = 0.15;

/// Word count at which the length term peaks, which is one lead paragraph.
const LENGTH_PEAK: f64 = 60.0;

/// The retention rules. Every field has a default, so an index written by an
/// older build still loads.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Policy {
    /// Rows the bank may hold. Better rows push out worse ones. Zero is no cap.
    pub budget: usize,
    /// Most rows the bank may hold from one label day. Zero is no cap.
    pub per_day: usize,
    /// Most rows the bank may hold from one group. Zero is no cap.
    pub per_group: usize,
    /// Most rows the bank may hold that anchor on one document. Zero is no cap.
    pub per_anchor: usize,
    /// Most rows the bank may hold per task. Zero is no cap.
    pub per_task: usize,
    /// Smallest score a new row may carry.
    pub min_score: f64,
    /// Difficulty the score prefers, in [0, 1].
    pub target_difficulty: f64,
    /// Half width of the difficulty band, in [0, 1].
    pub difficulty_width: f64,
    /// Gap at which the margin term reaches 63 percent.
    pub margin_scale: f64,
    /// Smallest Hamming distance a new anchor needs from every retained anchor.
    /// Values above seven are treated as seven, the exact radius of the index.
    pub novelty_bits: u32,
    /// Shortest anchor, in words.
    pub min_words: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            budget: 60_000,
            per_day: 4_000,
            per_group: 500,
            per_anchor: 3,
            per_task: 12_000,
            min_score: 0.45,
            target_difficulty: 0.5,
            difficulty_width: 0.55,
            margin_scale: 0.25,
            novelty_bits: 4,
            min_words: 5,
        }
    }
}

/// The four parts of a pair score, kept so a rejection can be explained.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Score {
    pub total: f64,
    pub margin: f64,
    pub difficulty: f64,
    pub quality: f64,
    pub length: f64,
}

/// One retained row, without the text. This is what the index holds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub task: Task,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub day: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub group: String,
    /// Document id the row is scored against, normally `a`.
    pub anchor: String,
    pub score: f64,
    /// SimHash of the anchor text, for the near duplicate screen.
    pub sig: u64,
    pub words: usize,
}

/// Counters and bucket tables rebuilt from [`Bank::entries`]. They are not
/// serialised because they are derivable, and a stale copy would be worse than
/// none.
#[derive(Clone, Debug, Default)]
struct Stamps {
    ids: HashSet<String>,
    by_day: HashMap<String, usize>,
    by_group: HashMap<String, usize>,
    by_anchor: HashMap<String, usize>,
    by_task: HashMap<String, usize>,
    edges: Vec<HashMap<u64, Vec<usize>>>,
}

/// The retained corpus index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bank {
    pub version: u32,
    pub policy: Policy,
    pub entries: Vec<Entry>,
    #[serde(skip)]
    stamps: Stamps,
}

impl Default for Bank {
    fn default() -> Self {
        Self::new(Policy::default())
    }
}

/// What one run did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub candidates: usize,
    pub malformed: usize,
    pub known: usize,
    pub short: usize,
    pub below_floor: usize,
    pub admitted: usize,
    pub evicted: usize,
    pub rejected_day: usize,
    pub rejected_group: usize,
    pub rejected_anchor: usize,
    pub rejected_task: usize,
    pub rejected_novelty: usize,
    /// Rows whose text was missing from the corpus and the candidates, so the
    /// index dropped them rather than claiming a row it cannot write.
    pub missing: usize,
    pub bank_rows: usize,
    pub mean_score: f64,
    pub top_score: f64,
    pub by_task: Vec<(String, usize)>,
    pub by_day: Vec<(String, usize)>,
    pub groups: usize,
    pub anchors: usize,
}

impl Bank {
    pub fn new(policy: Policy) -> Self {
        let mut bank = Self {
            version: 1,
            policy,
            entries: Vec::new(),
            stamps: Stamps::default(),
        };
        bank.rebuild();
        bank
    }

    /// Load an index, or start empty when the file is absent.
    pub fn load(path: &str, policy: Policy) -> io::Result<Self> {
        let mut bank = match Self::open(path)? {
            Some(bank) => bank,
            None => Self::new(policy.clone()),
        };
        bank.policy = policy;
        bank.rebuild();
        Ok(bank)
    }

    /// Load an index with the policy it was written with, for inspection.
    pub fn open(path: &str) -> io::Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut bank: Bank = serde_json::from_str(&text)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        bank.rebuild();
        Ok(Some(bank))
    }

    /// Composition of the bank without running a step.
    pub fn summary(&self) -> Report {
        let mut report = Report::default();
        self.summarise(&mut report);
        report
    }

    /// The best `count` rows, which are also the first rows of the corpus.
    pub fn top(&self, count: usize) -> &[Entry] {
        &self.entries[..count.min(self.entries.len())]
    }

    /// Write the index beside the corpus. The write is atomic, so a crash
    /// halfway through a week leaves the previous index intact.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let text = serde_json::to_string(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let target = Path::new(path);
        let tmp = temporary(target);
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, target)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.stamps.ids.contains(id)
    }

    /// Count of retained rows for one anchor document.
    pub fn anchor_count(&self, anchor: &str) -> usize {
        self.stamps.by_anchor.get(anchor).copied().unwrap_or(0)
    }

    /// Hamming distance from `sig` to the nearest retained anchor, capped at
    /// 64. Exact for every distance up to seven.
    pub fn nearest(&self, sig: u64) -> u32 {
        let mut best = 64u32;
        for (stripe, table) in self.stamps.edges.iter().enumerate() {
            let key = stripe_key(sig, stripe);
            if let Some(bucket) = table.get(&key) {
                for &index in bucket {
                    let distance = hash::hamming(sig, self.entries[index].sig);
                    if distance < best {
                        best = distance;
                        if best == 0 {
                            return 0;
                        }
                    }
                }
            }
        }
        best
    }

    /// Admit the candidates worth keeping, then enforce the budget.
    pub fn admit(&mut self, candidates: &[Pair], policy: &Policy) -> Report {
        let mut report = Report {
            candidates: candidates.len(),
            ..Report::default()
        };
        let verdicts: Vec<Verdict> = candidates
            .par_iter()
            .map(|pair| assess(pair, policy))
            .collect();
        let mut order: Vec<(f64, usize)> = Vec::with_capacity(candidates.len());
        for (index, verdict) in verdicts.iter().enumerate() {
            match verdict {
                Verdict::Keep(scored) => order.push((scored.total, index)),
                Verdict::Short => report.short += 1,
                Verdict::Floor => report.below_floor += 1,
            }
        }
        order.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| candidates[a.1].id.cmp(&candidates[b.1].id))
        });

        for (_, index) in order {
            let pair = &candidates[index];
            let Verdict::Keep(scored) = &verdicts[index] else {
                continue;
            };
            if pair.id.is_empty() || self.contains(&pair.id) {
                report.known += 1;
                continue;
            }
            if !self.within_caps(pair, policy, &mut report) {
                continue;
            }
            if self.nearest(scored.sig) <= policy.novelty_bits.min(STRIPES as u32 - 1) {
                report.rejected_novelty += 1;
                continue;
            }
            self.push(Entry {
                id: pair.id.clone(),
                task: pair.task,
                day: pair.day.clone(),
                group: pair.group_a.clone(),
                anchor: anchor_of(pair),
                score: scored.total,
                sig: scored.sig,
                words: scored.words,
            });
            report.admitted += 1;
        }

        report.evicted = self.prune(policy.budget);
        self.sort();
        self.summarise(&mut report);
        report
    }

    fn within_caps(&self, pair: &Pair, policy: &Policy, report: &mut Report) -> bool {
        let day = bucket(&pair.day);
        if over(policy.per_day, self.stamps.by_day.get(day)) {
            report.rejected_day += 1;
            return false;
        }
        let group = bucket(&pair.group_a);
        if over(policy.per_group, self.stamps.by_group.get(group)) {
            report.rejected_group += 1;
            return false;
        }
        let anchor = anchor_of(pair);
        if over(policy.per_anchor, self.stamps.by_anchor.get(&anchor)) {
            report.rejected_anchor += 1;
            return false;
        }
        if over(policy.per_task, self.stamps.by_task.get(pair.task.as_str())) {
            report.rejected_task += 1;
            return false;
        }
        true
    }

    fn push(&mut self, entry: Entry) {
        let index = self.entries.len();
        self.stamps.ids.insert(entry.id.clone());
        *self
            .stamps
            .by_day
            .entry(bucket(&entry.day).to_string())
            .or_insert(0) += 1;
        *self
            .stamps
            .by_group
            .entry(bucket(&entry.group).to_string())
            .or_insert(0) += 1;
        *self
            .stamps
            .by_anchor
            .entry(entry.anchor.clone())
            .or_insert(0) += 1;
        *self
            .stamps
            .by_task
            .entry(entry.task.as_str().to_string())
            .or_insert(0) += 1;
        for stripe in 0..STRIPES {
            self.stamps.edges[stripe]
                .entry(stripe_key(entry.sig, stripe))
                .or_default()
                .push(index);
        }
        self.entries.push(entry);
    }

    /// Drop the weakest rows until the bank fits `budget`, returning how many
    /// left.
    pub fn prune(&mut self, budget: usize) -> usize {
        if budget == 0 || self.entries.len() <= budget {
            return 0;
        }
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by(|&x, &y| {
            let a = &self.entries[x];
            let b = &self.entries[y];
            a.score.total_cmp(&b.score).then_with(|| b.id.cmp(&a.id))
        });
        let doomed: HashSet<usize> = order[..self.entries.len() - budget]
            .iter()
            .copied()
            .collect();
        let mut index = 0usize;
        self.entries.retain(|_| {
            let keep = !doomed.contains(&index);
            index += 1;
            keep
        });
        let evicted = doomed.len();
        self.rebuild();
        evicted
    }

    /// Sort entries best first, so the corpus and the index share one order and
    /// a rerun produces the same bytes.
    fn sort(&mut self) {
        self.entries
            .sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        self.rebuild();
    }

    fn rebuild(&mut self) {
        let mut stamps = Stamps {
            edges: (0..STRIPES).map(|_| HashMap::new()).collect(),
            ..Stamps::default()
        };
        for (index, entry) in self.entries.iter().enumerate() {
            stamps.ids.insert(entry.id.clone());
            *stamps
                .by_day
                .entry(bucket(&entry.day).to_string())
                .or_insert(0) += 1;
            *stamps
                .by_group
                .entry(bucket(&entry.group).to_string())
                .or_insert(0) += 1;
            *stamps.by_anchor.entry(entry.anchor.clone()).or_insert(0) += 1;
            *stamps
                .by_task
                .entry(entry.task.as_str().to_string())
                .or_insert(0) += 1;
            for stripe in 0..STRIPES {
                stamps.edges[stripe]
                    .entry(stripe_key(entry.sig, stripe))
                    .or_default()
                    .push(index);
            }
        }
        self.stamps = stamps;
    }

    fn summarise(&self, report: &mut Report) {
        report.bank_rows = self.entries.len();
        report.groups = self.stamps.by_group.len();
        report.anchors = self.stamps.by_anchor.len();
        if self.entries.is_empty() {
            return;
        }
        let sum: f64 = self.entries.iter().map(|e| e.score).sum();
        report.mean_score = sum / self.entries.len() as f64;
        report.top_score = self.entries[0].score;
        report.by_task = ranked(&self.stamps.by_task);
        report.by_day = ranked(&self.stamps.by_day);
    }
}

enum Verdict {
    Keep(Scored),
    Short,
    Floor,
}

struct Scored {
    total: f64,
    sig: u64,
    words: usize,
}

/// Score one pair and apply the hard floors.
fn assess(pair: &Pair, policy: &Policy) -> Verdict {
    if pair.text_a.trim().is_empty() {
        return Verdict::Short;
    }
    let words = match pair.words_a {
        Some(words) if words > 0 => words,
        _ => tokenize::count_words(&pair.text_a),
    };
    if words < policy.min_words {
        return Verdict::Short;
    }
    let score = score_pair(pair, policy, words);
    if score.total < policy.min_score {
        return Verdict::Floor;
    }
    let hashes: Vec<u64> = tokenize::words(&pair.text_a).map(hash::word_hash).collect();
    Verdict::Keep(Scored {
        total: score.total,
        sig: hash::simhash(&hashes),
        words,
    })
}

/// The four parts of the score and their weighted sum, each in [0, 1].
pub fn score_pair(pair: &Pair, policy: &Policy, words: usize) -> Score {
    let gap = margin_of(pair);
    let margin = gap.map(|m| 1.0 - (-m.abs() / policy.margin_scale.max(1e-9)).exp());
    let difficulty = pair
        .difficulty
        .map(|d| {
            let width = policy.difficulty_width.max(1e-9);
            (1.0 - (d.clamp(0.0, 1.0) - policy.target_difficulty).abs() / width).clamp(0.0, 1.0)
        })
        .unwrap_or(0.7);
    let quality = quality_of(pair);
    let length = length_term(words);
    let margin_part = margin.unwrap_or(0.5);
    let total = W_MARGIN * margin_part
        + W_DIFFICULTY * difficulty
        + W_QUALITY * quality
        + W_LENGTH * length;
    Score {
        total,
        margin: margin_part,
        difficulty,
        quality,
        length,
    }
}

fn margin_of(pair: &Pair) -> Option<f64> {
    if let Some(margin) = pair.margin {
        if margin != 0.0 {
            return Some(margin);
        }
    }
    match (pair.y_a, pair.y_b) {
        (Some(a), Some(b)) if a != b => Some(a - b),
        _ => None,
    }
}

fn quality_of(pair: &Pair) -> f64 {
    let given: Vec<f64> = [pair.quality_a, pair.quality_b]
        .into_iter()
        .flatten()
        .filter(|value| *value > 0.0)
        .collect();
    if !given.is_empty() {
        return (given.iter().sum::<f64>() / given.len() as f64).clamp(0.0, 1.0);
    }
    let stats = quality::Stats::of(&pair.text_a);
    quality::score(&stats, &quality::Weights::default())
        .total
        .clamp(0.0, 1.0)
}

fn length_term(words: usize) -> f64 {
    let ratio = (words as f64 + 1.0) / LENGTH_PEAK;
    (1.0 - ratio.ln().abs() / LENGTH_PEAK.ln()).clamp(0.0, 1.0)
}

fn anchor_of(pair: &Pair) -> String {
    if !pair.a.is_empty() {
        return pair.a.clone();
    }
    hash::hex64(hash::text_hash(&pair.text_a))
}

fn bucket(value: &str) -> &str {
    if value.is_empty() {
        "unknown"
    } else {
        value
    }
}

fn over(cap: usize, count: Option<&usize>) -> bool {
    cap > 0 && count.copied().unwrap_or(0) >= cap
}

fn stripe_key(sig: u64, stripe: usize) -> u64 {
    let shift = stripe as u32 * STRIPE_BITS;
    ((stripe as u64) << 32) | ((sig >> shift) & 0xff)
}

fn ranked(counts: &HashMap<String, usize>) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = counts
        .iter()
        .map(|(key, count)| (key.clone(), *count))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn temporary(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Where the index for a corpus lives.
pub fn index_path(out: &str) -> String {
    format!("{out}.idx.json")
}

/// Where the run ledger for a corpus lives.
pub fn ledger_path(out: &str) -> String {
    format!("{out}.ledger.jsonl")
}

/// Read candidates from one or more JSON Lines pair files.
pub fn read_candidates(inputs: &[String]) -> io::Result<(Vec<Pair>, usize)> {
    let mut pairs = Vec::new();
    let mut malformed = 0usize;
    for path in inputs {
        doc::for_each_line(path, |line| {
            if line.trim().is_empty() {
                return;
            }
            match pairs::from_line(line) {
                Some(pair) => pairs.push(pair),
                None => malformed += 1,
            }
        })?;
    }
    Ok((pairs, malformed))
}

/// One weekly step: score new candidates, admit the best, rewrite the corpus
/// and the index, and append a ledger line.
pub fn update(inputs: &[String], out: &str, policy: Policy) -> io::Result<Report> {
    let index = index_path(out);
    let index_exists = Path::new(&index).exists();
    if !index_exists && Path::new(out).exists() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{out} exists without {index}; refusing to rewrite a corpus this bank did not build"),
        ));
    }
    let mut bank = Bank::load(&index, policy.clone())?;
    let (candidates, malformed) = read_candidates(inputs)?;
    let mut report = bank.admit(&candidates, &policy);
    report.malformed = malformed;

    let held: HashSet<String> = bank.entries.iter().map(|e| e.id.clone()).collect();
    let mut corpus: HashMap<String, Pair> = HashMap::with_capacity(bank.len());
    if index_exists {
        doc::for_each_line(out, |line| {
            if let Some(pair) = pairs::from_line(line) {
                if held.contains(&pair.id) {
                    corpus.insert(pair.id.clone(), pair);
                }
            }
        })?;
    }
    for pair in candidates {
        if held.contains(&pair.id) {
            corpus.entry(pair.id.clone()).or_insert(pair);
        }
    }

    let before = bank.len();
    bank.entries.retain(|entry| corpus.contains_key(&entry.id));
    report.missing = before - bank.len();
    if report.missing > 0 {
        bank.rebuild();
        bank.summarise(&mut report);
    }

    let mut sink = Sink::create(out)?;
    for entry in &bank.entries {
        if let Some(pair) = corpus.get(&entry.id) {
            sink.line(&pairs::to_line(pair))?;
        }
    }
    sink.finish()?;
    report.bank_rows = bank.len();

    bank.save(&index)?;
    let run = ledger_line(&report, &policy);
    append(&ledger_path(out), &run)?;
    Ok(report)
}

fn ledger_line(report: &Report, policy: &Policy) -> String {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut value = serde_json::to_value(report).unwrap_or_default();
    if let Some(map) = value.as_object_mut() {
        map.insert("stamp".into(), serde_json::json!(stamp));
        map.insert("budget".into(), serde_json::json!(policy.budget));
        map.insert("min_score".into(), serde_json::json!(policy.min_score));
    }
    value.to_string()
}

fn append(path: &str, line: &str) -> io::Result<()> {
    use std::fs::OpenOptions;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(id: &str, text: &str, margin: f64, difficulty: f64) -> Pair {
        let mut p = Pair::new(Task::Rank, "train");
        p.a = format!("doc-{id}");
        p.b = Some(format!("neg-{id}"));
        p.text_a = text.to_string();
        p.text_b = Some(format!("{text} and other words here"));
        p.y_a = Some(margin);
        p.y_b = Some(0.0);
        p.margin = Some(margin);
        p.difficulty = Some(difficulty);
        p.day = "2026-11-09".into();
        p.group_a = "AAA".into();
        p.quality_a = Some(0.8);
        p.quality_b = Some(0.7);
        p.words_a = Some(tokenize::count_words(text));
        p.stamp();
        p.id = id.to_string();
        p
    }

    fn prose() -> &'static str {
        "The issuer reported a wider margin than the prior quarter and raised guidance for the \
         full year, which the desk read as a clean beat against a low bar and a strong backlog."
    }

    #[test]
    fn score_prefers_a_real_gap() {
        let policy = Policy::default();
        let thin = score_pair(&pair("t", prose(), 0.02, 0.8), &policy, 20);
        let wide = score_pair(&pair("w", prose(), 0.60, 0.5), &policy, 20);
        assert!(wide.total > thin.total, "{wide:?} {thin:?}");
    }

    #[test]
    fn score_prefers_hard_but_not_ambiguous() {
        let policy = Policy::default();
        let easy = score_pair(&pair("e", prose(), 0.4, 0.0), &policy, 20);
        let hard = score_pair(&pair("h", prose(), 0.4, 0.5), &policy, 20);
        assert!(hard.total > easy.total);
    }

    #[test]
    fn admits_once_and_is_idempotent() {
        let policy = Policy {
            min_score: 0.0,
            novelty_bits: 0,
            ..Policy::default()
        };
        let rows = vec![
            pair("a", prose(), 0.5, 0.5),
            pair(
                "b",
                &format!(
                    "{} A second, different report on the same session.",
                    prose()
                ),
                0.2,
                0.6,
            ),
        ];
        let mut bank = Bank::new(policy.clone());
        let first = bank.admit(&rows, &policy);
        assert_eq!(first.admitted, 2);
        let second = bank.admit(&rows, &policy);
        assert_eq!(second.admitted, 0);
        assert_eq!(second.known, 2);
        assert_eq!(bank.len(), 2);
    }

    #[test]
    fn caps_one_anchor() {
        let policy = Policy {
            per_anchor: 2,
            min_score: 0.0,
            novelty_bits: 0,
            ..Policy::default()
        };
        let mut bank = Bank::new(policy.clone());
        let texts = [
            "First headline about the issuer beat and the raised guidance for the year ahead.",
            "Second headline about the issuer beat and the raised guidance for the year ahead.",
            "Third headline about the issuer beat and the raised guidance for the year ahead.",
            "Fourth headline about the issuer beat and the raised guidance for the year ahead.",
        ];
        let rows: Vec<Pair> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut p = pair(&format!("c{i}"), text, 0.5, 0.5);
                p.a = "same-doc".into();
                p
            })
            .collect();
        let report = bank.admit(&rows, &policy);
        assert_eq!(report.admitted, 2);
        assert_eq!(report.rejected_anchor, 2);
        assert_eq!(bank.anchor_count("same-doc"), 2);
    }

    #[test]
    fn rejects_a_near_duplicate_anchor() {
        let policy = Policy {
            per_anchor: 0,
            novelty_bits: 6,
            min_score: 0.0,
            ..Policy::default()
        };
        let mut bank = Bank::new(policy.clone());
        let first = pair("one", prose(), 0.5, 0.5);
        let mut second = pair("two", prose(), 0.6, 0.5);
        second.a = "doc-two".into();
        let report = bank.admit(&[first, second], &policy);
        assert_eq!(report.admitted, 1);
        assert_eq!(report.rejected_novelty, 1);
    }

    #[test]
    fn budget_keeps_the_best_rows() {
        let policy = Policy {
            budget: 2,
            min_score: 0.0,
            novelty_bits: 0,
            ..Policy::default()
        };
        let mut bank = Bank::new(policy.clone());
        let rows = vec![
            pair("low", &format!("{} low", prose()), 0.05, 0.9),
            pair("high", &format!("{} high", prose()), 0.90, 0.5),
            pair("mid", &format!("{} mid", prose()), 0.30, 0.5),
        ];
        let report = bank.admit(&rows, &policy);
        assert_eq!(report.admitted, 3);
        assert_eq!(report.evicted, 1);
        assert_eq!(bank.len(), 2);
        assert!(bank.entries.iter().all(|e| e.id != "low"));
        assert!(bank.entries[0].score >= bank.entries[1].score);
    }

    #[test]
    fn index_round_trips() {
        let policy = Policy {
            min_score: 0.0,
            novelty_bits: 0,
            ..Policy::default()
        };
        let mut bank = Bank::new(policy.clone());
        bank.admit(&[pair("a", prose(), 0.5, 0.5)], &policy);
        let dir = std::env::temp_dir().join(format!("xcm-retain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bank.idx.json");
        let path_str = path.to_str().unwrap();
        bank.save(path_str).unwrap();
        let loaded = Bank::load(path_str, policy).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains(&bank.entries[0].id));
        assert_eq!(loaded.nearest(bank.entries[0].sig), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn policy_loads_with_missing_fields() {
        let policy: Policy = serde_json::from_str("{\"budget\": 10}").unwrap();
        assert_eq!(policy.budget, 10);
        assert_eq!(policy.min_words, Policy::default().min_words);
    }

    #[test]
    fn short_rows_are_skipped() {
        let policy = Policy {
            min_score: 0.0,
            novelty_bits: 0,
            ..Policy::default()
        };
        let mut bank = Bank::new(policy.clone());
        let mut tiny = pair("tiny", "two words", 0.9, 0.5);
        tiny.words_a = Some(2);
        let report = bank.admit(&[tiny], &policy);
        assert_eq!(report.short, 1);
        assert_eq!(report.admitted, 0);
    }
}
