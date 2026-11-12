//! The checks a pair corpus has to pass before anyone trains on it.
//!
//! Six questions, each one cheap and each one able to stop a run:
//!
//! * does any document appear under two splits, which is leakage;
//! * does any pair carry the same text on both sides, which teaches nothing;
//! * does any pair carry two texts that are nearly the same, which teaches a
//!   model to key on one word;
//! * is the outcome gap real, or are the margins so small that the label noise
//!   is larger than the signal;
//! * is one side systematically longer, which lets a model win on length alone;
//! * does any pair overlap the evaluation texts.
//!
//! The last check builds a shingle index over the evaluation texts and screens
//! every pair text against it. A pair that shares 80 percent of an evaluation
//! document's shingles is contaminated, and the count is reported rather than
//! quietly removed, because contamination is a fact about the corpus, not a
//! filter to hide behind.

use std::collections::{HashMap, HashSet};

use crate::pairs::{Pair, Task};
use crate::text::hash;
use crate::text::tokenize;

/// Thresholds of the audit.
#[derive(Clone, Debug)]
pub struct Config {
    /// Similarity at or above which two sides are the same text for our
    /// purposes.
    pub max_similarity: f64,
    /// Share of an evaluation document a pair side must match to count as
    /// contamination.
    pub contamination: f64,
    /// Smallest margin that counts as signal.
    pub min_margin: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_similarity: 0.98,
            contamination: 0.80,
            min_margin: 0.01,
        }
    }
}

/// What the audit found. Every count is a count of rows, not a rate, so the
/// caller can decide what a rate means for its corpus size.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub pairs: usize,
    pub by_task: Vec<(Task, usize)>,
    pub by_split: Vec<(String, usize)>,
    pub duplicate_ids: usize,
    pub identical_sides: usize,
    pub near_identical: usize,
    pub leaked_docs: usize,
    pub leaked_texts: usize,
    pub contaminated: usize,
    /// Pairs sharing a verbatim run of 13 words with an evaluation text, the
    /// overlap rule GPT-3 used (Brown et al 2020, arXiv:2005.14165).
    pub verbatim_overlap: usize,
    pub thin_margin: usize,
    pub mean_margin: f64,
    pub min_margin: f64,
    pub mean_difficulty: f64,
    pub mean_words_a: f64,
    pub mean_words_b: f64,
    /// Share of pairwise rows whose first side is the longer one.
    pub longer_side_a: f64,
    pub problems: Vec<String>,
}

impl Report {
    /// Whether nothing was found that should stop a training run.
    pub fn ok(&self) -> bool {
        self.leaked_docs == 0
            && self.leaked_texts == 0
            && self.identical_sides == 0
            && self.contaminated == 0
            && self.verbatim_overlap == 0
            && self.duplicate_ids == 0
    }

    /// One line per finding, for a log.
    pub fn summary(&self) -> Vec<String> {
        let mut out = self.problems.clone();
        if out.is_empty() {
            out.push("no leakage, no trivial pair, no contamination".to_string());
        }
        out
    }
}

fn shingle_set(text: &str) -> Vec<u64> {
    let words: Vec<u64> = tokenize::words(text).map(hash::word_hash).collect();
    hash::shingles(&words, 3)
}

/// Hashes of every 13 word run, the verbatim overlap rule.
fn verbatim_set(text: &str) -> Vec<u64> {
    let words: Vec<u64> = tokenize::words(text).map(hash::word_hash).collect();
    if words.len() < 13 {
        return Vec::new();
    }
    hash::shingles(&words, 13)
}

/// Screen pair texts against evaluation texts. Returns the pairs caught by the
/// shingle share rule and the pairs caught by the verbatim run rule.
fn contamination(pairs: &[Pair], eval: &[String], threshold: f64) -> (usize, usize) {
    if eval.is_empty() {
        return (0, 0);
    }
    let verbatim: HashSet<u64> = eval.iter().flat_map(|t| verbatim_set(t)).collect();
    let sets: Vec<Vec<u64>> = eval.iter().map(|t| shingle_set(t)).collect();
    let mut index: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, set) in sets.iter().enumerate() {
        for &shingle in set {
            index.entry(shingle).or_default().push(i as u32);
        }
    }
    let mut acc = vec![0u32; eval.len()];
    let mut touched: Vec<u32> = Vec::new();
    let mut hits = 0usize;
    let mut verbatim_hits = 0usize;
    for pair in pairs {
        if pair
            .texts()
            .iter()
            .any(|text| verbatim_set(text).iter().any(|g| verbatim.contains(g)))
        {
            verbatim_hits += 1;
        }
        for text in pair.texts() {
            let set = shingle_set(text);
            if set.is_empty() {
                continue;
            }
            for &shingle in &set {
                if let Some(list) = index.get(&shingle) {
                    for &doc in list {
                        if acc[doc as usize] == 0 {
                            touched.push(doc);
                        }
                        acc[doc as usize] += 1;
                    }
                }
            }
            let matched = touched.iter().any(|&doc| {
                acc[doc as usize] as f64 / sets[doc as usize].len() as f64 >= threshold
            });
            for &doc in &touched {
                acc[doc as usize] = 0;
            }
            touched.clear();
            if matched {
                hits += 1;
                break;
            }
        }
    }
    (hits, verbatim_hits)
}

/// Run every check.
pub fn audit(pairs: &[Pair], eval: &[String], cfg: &Config) -> Report {
    let mut report = Report {
        pairs: pairs.len(),
        ..Report::default()
    };
    if pairs.is_empty() {
        report.problems.push("the corpus has no rows".to_string());
        return report;
    }
    let mut ids: HashSet<&str> = HashSet::with_capacity(pairs.len());
    let mut doc_splits: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut text_splits: HashMap<u64, HashSet<&str>> = HashMap::new();
    let mut task_counts: HashMap<Task, usize> = HashMap::new();
    let mut split_counts: HashMap<String, usize> = HashMap::new();
    let (mut margins, mut difficulties) = (Vec::new(), Vec::new());
    let (mut wa, mut wb) = (0.0f64, 0.0f64);
    let (mut na, mut nb, mut longer_a) = (0u64, 0u64, 0u64);

    for pair in pairs {
        if !ids.insert(pair.id.as_str()) {
            report.duplicate_ids += 1;
        }
        *task_counts.entry(pair.task).or_insert(0) += 1;
        *split_counts.entry(pair.split.clone()).or_insert(0) += 1;
        for id in pair.doc_ids() {
            doc_splits
                .entry(id)
                .or_default()
                .insert(pair.split.as_str());
        }
        for text in pair.texts() {
            let digest = hash::text_hash(text);
            text_splits
                .entry(digest)
                .or_default()
                .insert(pair.split.as_str());
        }
        if let Some(b) = &pair.text_b {
            if pair.text_a.eq_ignore_ascii_case(b) {
                report.identical_sides += 1;
            } else if hash::jaccard(&shingle_set(&pair.text_a), &shingle_set(b))
                >= cfg.max_similarity
            {
                report.near_identical += 1;
            }
            let la = pair.text_a.chars().count() as f64;
            let lb = b.chars().count() as f64;
            wa += la;
            wb += lb;
            if la > lb {
                longer_a += 1;
            }
            nb += 1;
        } else {
            na += pair.text_a.chars().count() as u64;
        }
        if let Some(m) = pair.margin {
            if m.abs() < cfg.min_margin {
                report.thin_margin += 1;
            }
            margins.push(m.abs());
        }
        if let Some(d) = pair.difficulty {
            difficulties.push(d);
        }
    }

    report.by_task = Task::ALL
        .iter()
        .map(|t| (*t, task_counts.get(t).copied().unwrap_or(0)))
        .filter(|(_, n)| *n > 0)
        .collect();
    let mut splits: Vec<(String, usize)> = split_counts.into_iter().collect();
    splits.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    report.by_split = splits;

    report.leaked_docs = doc_splits.values().filter(|s| s.len() > 1).count();
    report.leaked_texts = text_splits.values().filter(|s| s.len() > 1).count();
    let (share_hits, verbatim_hits) = contamination(pairs, eval, cfg.contamination);
    report.contaminated = share_hits;
    report.verbatim_overlap = verbatim_hits;

    if !margins.is_empty() {
        margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        report.mean_margin = margins.iter().sum::<f64>() / margins.len() as f64;
        report.min_margin = margins[0];
    }
    if !difficulties.is_empty() {
        report.mean_difficulty = difficulties.iter().sum::<f64>() / difficulties.len() as f64;
    }
    report.mean_words_a = if nb > 0 {
        wa / nb as f64
    } else {
        na as f64 / pairs.len() as f64
    };
    report.mean_words_b = if nb > 0 { wb / nb as f64 } else { 0.0 };
    report.longer_side_a = if nb > 0 {
        longer_a as f64 / nb as f64
    } else {
        0.0
    };

    if report.leaked_docs > 0 {
        report.problems.push(format!(
            "{} documents appear under more than one split",
            report.leaked_docs
        ));
    }
    if report.leaked_texts > 0 {
        report.problems.push(format!(
            "{} texts appear under more than one split",
            report.leaked_texts
        ));
    }
    if report.identical_sides > 0 {
        report.problems.push(format!(
            "{} pairs carry the same text on both sides",
            report.identical_sides
        ));
    }
    if report.near_identical > 0 {
        report.problems.push(format!(
            "{} pairs are near identical at {:.2}",
            report.near_identical, cfg.max_similarity
        ));
    }
    if report.contaminated > 0 {
        report.problems.push(format!(
            "{} pairs overlap the evaluation texts at {:.2}",
            report.contaminated, cfg.contamination
        ));
    }
    if report.verbatim_overlap > 0 {
        report.problems.push(format!(
            "{} pairs share a 13 word run with an evaluation text",
            report.verbatim_overlap
        ));
    }
    if report.thin_margin > 0 {
        report.problems.push(format!(
            "{} pairs carry a margin below {:.3}",
            report.thin_margin, cfg.min_margin
        ));
    }
    let skew = report.longer_side_a;
    if nb >= 200 && (skew - 0.5).abs() > 0.15 {
        report.problems.push(format!(
            "the first side is longer in {:.0} percent of pairs",
            skew * 100.0
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(task: Task, a: &str, ta: &str, tb: Option<&str>, split: &str) -> Pair {
        let mut p = Pair::new(task, split);
        p.a = a.into();
        p.text_a = ta.into();
        p.text_b = tb.map(str::to_string);
        p.margin = Some(0.05);
        p.stamp();
        p
    }

    #[test]
    fn leakage_and_triviality_are_both_caught() {
        let pairs = vec![
            pair(
                Task::Rank,
                "d1",
                "Fed holds rates steady as inflation cools",
                Some("Airline cancels flights as storm closes airports"),
                "train",
            ),
            pair(
                Task::Label,
                "d1",
                "Fed holds rates steady as inflation cools",
                None,
                "test",
            ),
            pair(
                Task::Rank,
                "d2",
                "Chip maker lifts guidance on demand",
                Some("Chip maker lifts guidance on demand"),
                "train",
            ),
        ];
        let report = audit(&pairs, &[], &Config::default());
        assert_eq!(report.leaked_docs, 1);
        assert_eq!(report.identical_sides, 1);
        assert!(!report.ok());
        assert!(report.problems.len() >= 2);
    }

    #[test]
    fn a_clean_corpus_passes() {
        let pairs = vec![
            pair(
                Task::Rank,
                "d1",
                "Fed holds rates steady as inflation cools",
                Some("Airline cancels flights as storm closes airports"),
                "train",
            ),
            pair(
                Task::Rank,
                "d2",
                "Chip maker lifts guidance on demand",
                Some("Retailer cuts outlook after weak trading"),
                "test",
            ),
        ];
        let report = audit(&pairs, &[], &Config::default());
        assert!(report.ok(), "{:?}", report.problems);
        assert_eq!(report.by_task.len(), 1);
        assert_eq!(report.pairs, 2);
    }

    #[test]
    fn an_eval_overlap_is_reported() {
        let eval = vec!["Fed holds rates steady as inflation cools and hiring slows".to_string()];
        let pairs = vec![pair(
            Task::Rank,
            "d1",
            "Fed holds rates steady as inflation cools and hiring slows",
            Some("Airline cancels flights as a storm closes northern airports"),
            "train",
        )];
        let report = audit(&pairs, &eval, &Config::default());
        assert_eq!(report.contaminated, 1);
        assert_eq!(report.verbatim_overlap, 0);
    }
}
