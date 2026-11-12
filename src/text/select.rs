//! Selection: keep a budget worth of the most relevant documents.
//!
//! Relevance here has three parts, and all three are visible in the report.
//!
//! * a floor on the quality score, because a relevant document that is broken
//!   is not usable;
//! * a cap per group, so one publisher or one ticker cannot fill the budget;
//! * a redundancy threshold, which is maximum marginal relevance in its
//!   thresholded form: walk the documents in score order and accept a document
//!   only when its cosine similarity to every accepted document is below
//!   `theta`. The cosine runs over hashed word and bigram features with an
//!   inverse document frequency weight, and the check stops at the first
//!   accepted neighbour above `theta`, so the pass is near linear on real text.
//!
//! The cost is that the result is greedy, not the optimum of a submodular
//! objective. The gain is an auditable budget: every accepted document cleared
//! the same three tests, and the report says how many failed each one.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::text::hash::{self, mix};
use crate::text::quality::STOPWORDS;
use crate::text::tokenize;

/// Number of hashed feature bins. 65,536 keeps collisions rare on a million
/// distinct bigrams and keeps the inverted index small enough to cache.
pub const BINS: u32 = 1 << 16;

/// Configuration of the selection pass.
#[derive(Clone, Debug)]
pub struct Config {
    /// How many documents to keep. Zero means "keep every document that passes
    /// the floor and the redundancy test".
    pub budget: usize,
    /// Minimum composite quality score.
    pub floor: f64,
    /// Cosine similarity above which a candidate is a near neighbour of an
    /// accepted document and is dropped.
    pub theta: f64,
    /// Maximum share of the budget one group may take.
    pub group_cap: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            budget: 0,
            floor: 0.0,
            theta: 0.85,
            group_cap: 0.10,
        }
    }
}

/// Hashed count features of a text: unigrams and bigrams, stopwords removed,
/// sorted by feature id with their counts.
pub fn counts(text: &str) -> Vec<(u32, u32)> {
    let words: Vec<u64> = tokenize::words(text)
        .map(tokenize::lowercase_word)
        .filter(|w| !STOPWORDS.contains(&w.as_str()) && w.chars().any(char::is_alphanumeric))
        .map(|w| hash::word_hash(&w))
        .collect();
    let mut map: HashMap<u32, u32> = HashMap::with_capacity(words.len() * 2);
    for &w in &words {
        *map.entry((w % BINS as u64) as u32).or_insert(0) += 1;
    }
    for pair in words.windows(2) {
        let id = mix(pair[0] ^ pair[1].rotate_left(32)) % BINS as u64;
        *map.entry(id as u32).or_insert(0) += 1;
    }
    let mut out: Vec<(u32, u32)> = map.into_iter().collect();
    out.sort_unstable_by_key(|&(id, _)| id);
    out
}

/// Document frequency of every bin over a batch of texts, in parallel.
pub fn document_frequency(texts: &[&str]) -> Vec<u32> {
    let partial: Vec<Vec<u32>> = texts
        .par_chunks(2048)
        .map(|chunk| {
            let mut local = vec![0u32; BINS as usize];
            for text in chunk {
                for (id, _) in counts(text) {
                    local[id as usize] += 1;
                }
            }
            local
        })
        .collect();
    let mut df = vec![0u32; BINS as usize];
    for local in partial {
        for (slot, value) in df.iter_mut().zip(local) {
            *slot += value;
        }
    }
    df
}

/// Turn counts into a normalised tf-idf vector.
pub fn weigh(counts: &[(u32, u32)], df: &[u32], n_docs: usize) -> Vec<(u32, f32)> {
    let n = n_docs.max(1) as f64;
    let mut out = Vec::with_capacity(counts.len());
    for &(id, tf) in counts {
        let d = df.get(id as usize).copied().unwrap_or(0) as f64;
        let idf = ((n + 1.0) / (d + 1.0)).ln() + 1.0;
        out.push((id, ((1.0 + (tf as f64).ln()) * idf) as f32));
    }
    let norm = out
        .iter()
        .map(|&(_, v)| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for (_, v) in &mut out {
            *v = (*v as f64 / norm) as f32;
        }
    }
    out
}

/// What a selection pass did, with the reason for every rejection.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub candidates: usize,
    pub below_floor: usize,
    pub redundant: usize,
    pub over_cap: usize,
    pub selected: usize,
    /// Rows added by the second pass, after the group cap had refused them and
    /// the budget was still not full.
    pub relaxed: usize,
    pub groups: usize,
    pub mean_quality: f64,
    pub mean_words: f64,
    /// Share of distinct hashed features the selection covers.
    pub feature_coverage: f64,
}

/// Whether a candidate clears the group cap. The cap is strict: a group that
/// has spent its share is refused even when the budget is not yet full, because
/// the alternative is a budget that quietly belongs to one publisher. The report
/// counts the refusals so a caller can see when the cap bound.
fn cap_ok(counts: &HashMap<String, usize>, group: &str, budget: usize, cap: f64) -> bool {
    if budget == 0 || cap >= 1.0 {
        return true;
    }
    let allowed = ((budget as f64 * cap).ceil() as usize).max(1);
    counts.get(group).copied().unwrap_or(0) < allowed
}

/// Select documents. `vectors` must be the output of [`weigh`] for each index,
/// `quality` the composite score, and `groups` the cap unit.
///
/// Returns the accepted indices in score order and the report.
pub fn select(
    vectors: &[Vec<(u32, f32)>],
    quality: &[f64],
    groups: &[String],
    words: &[usize],
    cfg: &Config,
) -> (Vec<usize>, Report) {
    let n = vectors.len();
    let mut report = Report {
        candidates: n,
        ..Report::default()
    };
    if n == 0 {
        return (Vec::new(), report);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&a, &b| {
        quality[b]
            .partial_cmp(&quality[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // Inverted index over the accepted documents. Every value in a vector is
    // positive, so a partial dot product is a lower bound on the final one and
    // the redundancy test can stop the moment it crosses theta.
    let mut index: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
    let mut accepted: Vec<u32> = Vec::with_capacity(cfg.budget.min(n));
    let mut kept_set: HashSet<u32> = HashSet::with_capacity(cfg.budget.min(n));
    let mut acc = vec![0.0f32; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut per_group: HashMap<String, usize> = HashMap::new();
    let budget = if cfg.budget == 0 {
        n
    } else {
        cfg.budget.min(n)
    };
    let theta = cfg.theta as f32;

    let mut decided = vec![false; n];
    for phase in 0..2 {
        for &doc in &order {
            if accepted.len() >= budget {
                break;
            }
            if decided[doc] {
                continue;
            }
            if quality[doc] < cfg.floor {
                decided[doc] = true;
                if phase == 0 {
                    report.below_floor += 1;
                }
                continue;
            }
            if phase == 0 && !cap_ok(&per_group, &groups[doc], budget, cfg.group_cap) {
                // Left undecided: the second pass offers the same row again
                // once every capped source has been walked over.
                report.over_cap += 1;
                continue;
            }
            let mut early = false;
            'features: for &(id, value) in &vectors[doc] {
                let Some(postings) = index.get(&id) else {
                    continue;
                };
                for &(other, other_value) in postings {
                    let slot = other as usize;
                    if acc[slot] == 0.0 {
                        touched.push(other);
                    }
                    acc[slot] += value * other_value;
                    if acc[slot] >= theta {
                        early = true;
                        break 'features;
                    }
                }
            }
            for &other in &touched {
                acc[other as usize] = 0.0;
            }
            touched.clear();
            if early {
                decided[doc] = true;
                if phase == 0 {
                    report.redundant += 1;
                }
                continue;
            }
            accepted.push(doc as u32);
            kept_set.insert(doc as u32);
            decided[doc] = true;
            if phase == 1 {
                report.relaxed += 1;
            }
            *per_group.entry(groups[doc].clone()).or_insert(0) += 1;
            for &(id, value) in &vectors[doc] {
                index.entry(id).or_default().push((doc as u32, value));
            }
        }
    }

    let sum_q: f64 = accepted.iter().map(|&i| quality[i as usize]).sum();
    let sum_w: usize = accepted.iter().map(|&i| words[i as usize]).sum();
    report.selected = accepted.len();
    report.groups = per_group.len();
    report.mean_quality = if accepted.is_empty() {
        0.0
    } else {
        sum_q / accepted.len() as f64
    };
    report.mean_words = if accepted.is_empty() {
        0.0
    } else {
        sum_w as f64 / accepted.len() as f64
    };
    let mut all_features: HashSet<u32> = HashSet::new();
    let mut kept_features: HashSet<u32> = HashSet::new();
    for (i, vector) in vectors.iter().enumerate() {
        let kept = kept_set.contains(&(i as u32));
        for &(id, _) in vector {
            all_features.insert(id);
            if kept {
                kept_features.insert(id);
            }
        }
    }
    report.feature_coverage = if all_features.is_empty() {
        0.0
    } else {
        kept_features.len() as f64 / all_features.len() as f64
    };
    (accepted.into_iter().map(|i| i as usize).collect(), report)
}

/// Target-aware importance resampling, the DSIR rule from Xie et al 2023
/// (arXiv:2302.03169).
///
/// The idea is one line: a document is worth more when its hashed n-grams are
/// common in a small curated target set and rare in the raw corpus. Both
/// densities are estimated as normalised counts over `bins` hashed unigram and
/// bigram features, and the log weight of a document is
///
/// `sum_f count_f * (ln p_f - ln q_f)`
///
/// with a `1e-8` floor inside each logarithm. Selection then draws `k` documents
/// without replacement with probability proportional to `exp(log w)`, which is
/// implemented as a Gumbel top-k so it stays one pass and one sort.
///
/// No weight clipping is applied, because the paper applies none; the 1e-8 floor
/// is what keeps a feature absent from the target from producing an infinite
/// penalty. Unlike the greedy coverage pass in [`select`], this rule answers
/// "relevant to what": give it a target and it tilts the corpus towards that
/// target, give it the corpus itself and every weight is one.
#[derive(Clone, Debug)]
pub struct Dsir {
    pub bins: usize,
    pub target: Vec<f64>,
    pub proposal: Vec<f64>,
}

impl Dsir {
    /// Fit the two densities. `bins` is 10,000 in the paper; a corpus smaller
    /// than that should lower it so the histograms are not mostly empty.
    pub fn fit(target: &[&str], proposal: &[&str], bins: usize) -> Self {
        let bins = bins.max(16);
        Self {
            bins,
            target: density(target, bins),
            proposal: density(proposal, bins),
        }
    }

    /// Log weight of every document.
    pub fn log_weights(&self, docs: &[&str]) -> Vec<f64> {
        let mut floor = vec![1e-8f64; self.bins];
        for (f, p) in self.target.iter().enumerate() {
            floor[f] = p.max(1e-8);
        }
        docs.iter()
            .map(|text| {
                let mut sum = 0.0;
                for (id, tf) in counts(text) {
                    let f = id as usize % self.bins;
                    let p = floor[f];
                    let q = self.proposal[f].max(1e-8);
                    sum += tf as f64 * (p.ln() - q.ln());
                }
                sum
            })
            .collect()
    }

    /// Draw `k` indices without replacement, proportional to `exp(weight)`.
    pub fn resample(&self, log_weights: &[f64], k: usize, rng: &mut crate::rng::Rng) -> Vec<usize> {
        let k = k.min(log_weights.len());
        let mut scored: Vec<(f64, usize)> = log_weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                // Gumbel noise: -ln(-ln(u)), with u kept out of the ends.
                let u = rng.open_f64().clamp(1e-12, 1.0 - 1e-12);
                (w + -(-u.ln()).ln(), i)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }
}

fn density(texts: &[&str], bins: usize) -> Vec<f64> {
    let mut hist = vec![0.0f64; bins];
    for text in texts {
        for (id, tf) in counts(text) {
            hist[id as usize % bins] += tf as f64;
        }
    }
    let total: f64 = hist.iter().sum();
    if total > 0.0 {
        for h in &mut hist {
            *h /= total;
        }
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(texts: &[&str]) -> Vec<Vec<(u32, f32)>> {
        let df = document_frequency(texts);
        texts
            .iter()
            .map(|t| weigh(&counts(t), &df, texts.len()))
            .collect()
    }

    #[test]
    fn counts_are_sorted_and_positive() {
        let c = counts("Fed holds rates steady as inflation cools");
        assert!(!c.is_empty());
        assert!(c.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(c.iter().all(|&(_, tf)| tf > 0));
    }

    #[test]
    fn redundant_copies_are_dropped_and_distinct_texts_survive() {
        let texts = [
            "Fed holds rates steady as inflation cools and hiring slows",
            "Fed holds rates steady as inflation cools and hiring slows",
            "Airline cancels flights as storm closes northern airports",
        ];
        let vectors = build(&texts);
        let quality = vec![0.8, 0.8, 0.7];
        let groups = vec!["wire".to_string(), "wire".to_string(), "paper".to_string()];
        let words = vec![10, 10, 9];
        let cfg = Config {
            theta: 0.9,
            group_cap: 1.0,
            ..Config::default()
        };
        let (kept, report) = select(&vectors, &quality, &groups, &words, &cfg);
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert_eq!(report.redundant, 1);
        assert!(kept.contains(&2));
    }

    #[test]
    fn dsir_tilts_towards_the_target() {
        let target = [
            "Fed holds rates steady as inflation cools",
            "Rate decision leaves policy unchanged for the quarter",
            "Inflation and rates dominate the central bank meeting",
        ];
        let proposal = [
            "Fed holds rates steady as inflation cools",
            "Airline cancels flights as storm closes airports",
            "Chip maker lifts guidance on data centre demand",
            "Retailer cuts outlook after weak spring trading",
        ];
        let dsir = Dsir::fit(&target, &proposal, 4096);
        let weights = dsir.log_weights(&proposal);
        assert!(weights[0] > weights[1], "{weights:?}");
        let mut rng = crate::rng::Rng::new(3);
        let chosen = dsir.resample(&weights, 2, &mut rng);
        assert_eq!(chosen.len(), 2);
        assert!(chosen[0] == 0 || chosen[0] == 1, "{chosen:?}");
    }

    #[test]
    fn the_second_pass_fills_a_budget_the_cap_refused() {
        let vectors = vec![vec![(1, 1.0)], vec![(2, 1.0)], vec![(3, 1.0)]];
        let quality = vec![0.9, 0.8, 0.7];
        let groups = vec!["wire".to_string(), "wire".to_string(), "wire".to_string()];
        let words = vec![10, 10, 10];
        let cfg = Config {
            budget: 2,
            theta: 1.1,
            group_cap: 0.5,
            ..Config::default()
        };
        let (kept, report) = select(&vectors, &quality, &groups, &words, &cfg);
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert_eq!(report.relaxed, 1);
        assert_eq!(report.over_cap, 2);
        assert_eq!(report.groups, 1);
    }

    #[test]
    fn group_cap_spreads_the_budget() {
        let text_a = "Fed holds rates steady as inflation cools and hiring slows";
        let text_b = "Airline cancels flights as storm closes northern airports";
        let text_c = "Chip maker lifts guidance on data centre demand for the year";
        let texts = [text_a, text_a, text_b, text_c];
        // Distinct texts, identical scores, one dominant group.
        let vectors: Vec<Vec<(u32, f32)>> = texts
            .iter()
            .map(|t| {
                if *t == text_a {
                    vec![(1, 1.0), (2, 1.0)]
                } else if *t == text_b {
                    vec![(3, 1.0)]
                } else {
                    vec![(4, 1.0)]
                }
            })
            .collect();
        let quality = vec![0.9, 0.9, 0.9, 0.9];
        let groups = vec![
            "wire".to_string(),
            "wire".to_string(),
            "paper".to_string(),
            "paper".to_string(),
        ];
        let words = vec![10, 10, 10, 10];
        let cfg = Config {
            budget: 2,
            theta: 1.1,
            group_cap: 0.5,
            ..Config::default()
        };
        let (kept, report) = select(&vectors, &quality, &groups, &words, &cfg);
        assert_eq!(kept.len(), 2);
        assert_eq!(report.groups, 2);
    }
}
