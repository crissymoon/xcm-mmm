//! Near-duplicate clustering with MinHash bands.
//!
//! The index is the usual banded LSH: a signature of `perms` positions is cut
//! into `bands` bands of `rows` positions, and two documents that agree on every
//! position of one band become candidates. The probability that a pair with
//! Jaccard `s` becomes a candidate is `1 - (1 - s^rows)^bands`.
//!
//! The default is 14 bands of 8 rows over 112 permutations, which is the
//! configuration the datatrove MinHash step released with FineWeb (Penedo et al
//! 2024, arXiv:2406.17557). Its crossing point is `(1/14)^(1/8) = 0.72`, so a
//! pair at 0.8 is caught with probability 0.92 and a pair at 0.5 with 0.02. A
//! corpus that wants the Lee et al 2021 setting (arXiv:2107.06499) of 450 bands
//! of 20 rows needs 9000 permutations and should raise `perms` and `bands`
//! together; [`Config::recall`] prints the curve for either.
//!
//! Candidates are then verified with the signature estimate, which is unbiased
//! for Jaccard and costs nothing extra. Exact duplicates are merged first on a
//! hash of the cleaned text, because the cheapest duplicate to catch is the one
//! that is byte for byte identical.
//!
//! One canonical member survives per cluster: the highest quality score, then
//! the lowest index, so two runs on the same input keep the same rows.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::text::hash::{self, Perms};

/// Parameters of the deduplication pass.
#[derive(Clone, Debug)]
pub struct Config {
    /// Words per shingle. The published web pipelines use five (datatrove) or
    /// five (Lee et al); three is the default here because a headline of six
    /// words would otherwise produce two shingles, and a near-duplicate headline
    /// is exactly the case this pass has to catch. A long body corpus should
    /// raise this to five.
    pub shingle: usize,
    /// Signature length, `bands * rows`.
    pub perms: usize,
    /// Number of bands.
    pub bands: usize,
    /// Jaccard threshold a pair must reach to be merged.
    pub threshold: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            shingle: 3,
            perms: 112,
            bands: 14,
            threshold: 0.80,
        }
    }
}

impl Config {
    /// Positions per band.
    pub fn rows(&self) -> usize {
        (self.perms / self.bands.max(1)).max(1)
    }

    /// Probability that a pair at `jaccard` shares a band.
    pub fn recall(&self, jaccard: f64) -> f64 {
        let rows = self.rows() as f64;
        1.0 - (1.0 - jaccard.powf(rows)).powi(self.bands as i32)
    }
}

/// Signatures for a batch of texts, computed in parallel.
pub fn signatures(texts: &[&str], cfg: &Config) -> Vec<Vec<u32>> {
    let perms = Perms::new(cfg.perms, 0x5eed);
    texts
        .par_iter()
        .map(|text| {
            let words: Vec<u64> = crate::text::tokenize::words(text)
                .map(hash::word_hash)
                .collect();
            let shingles = hash::shingles(&words, cfg.shingle);
            hash::signature(&shingles, &perms)
        })
        .collect()
}

/// Union-find over document indices.
struct Sets {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Sets {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grand = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grand;
            x = grand;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
        true
    }
}

/// What a deduplication pass did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub docs: usize,
    pub clusters: usize,
    pub duplicates: usize,
    pub largest_cluster: usize,
    pub exact_duplicates: usize,
    /// Fraction of documents that are not canonical members.
    pub drop_rate: f64,
}

fn largest(members: &HashMap<u32, Vec<usize>>) -> usize {
    members.values().map(Vec::len).max().unwrap_or(0)
}

/// Group indices into near-duplicate clusters and pick one survivor per group.
///
/// `exact` is the text hash of each document; identical hashes are merged
/// without looking at the signature. `quality` breaks ties in favour of the
/// better document and never changes the group.
pub fn cluster(
    sigs: &[Vec<u32>],
    exact: &[u64],
    quality: &[f64],
    cfg: &Config,
) -> (Vec<usize>, Report) {
    let n = sigs.len();
    let mut report = Report {
        docs: n,
        ..Report::default()
    };
    if n == 0 {
        return (Vec::new(), report);
    }
    let mut sets = Sets::new(n);
    let rows = cfg.rows();
    let mut buckets: HashMap<u64, Vec<u32>> = HashMap::with_capacity(n * cfg.bands);

    let mut by_hash: HashMap<u64, u32> = HashMap::with_capacity(n);
    for (i, &h) in exact.iter().enumerate() {
        match by_hash.entry(h) {
            std::collections::hash_map::Entry::Occupied(slot) => {
                if sets.union(*slot.get(), i as u32) {
                    report.exact_duplicates += 1;
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(i as u32);
            }
        }
    }

    for band in 0..cfg.bands {
        buckets.clear();
        for (i, sig) in sigs.iter().enumerate() {
            if let Some(key) = hash::band_key(sig, band, rows) {
                buckets.entry(key).or_default().push(i as u32);
            }
        }
        for members in buckets.values() {
            if members.len() < 2 {
                continue;
            }
            for (a, &i) in members.iter().enumerate() {
                for &j in &members[a + 1..] {
                    if sets.find(i) == sets.find(j) {
                        continue;
                    }
                    let sim = hash::signature_similarity(&sigs[i as usize], &sigs[j as usize]);
                    if sim >= cfg.threshold {
                        sets.union(i, j);
                    }
                }
            }
        }
    }

    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = sets.find(i as u32);
        groups.entry(root).or_default().push(i);
    }
    let mut kept = Vec::with_capacity(groups.len());
    for members in groups.values() {
        let best = members
            .iter()
            .copied()
            .max_by(|&a, &b| {
                quality[a]
                    .partial_cmp(&quality[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    // Explicit index tie break keeps the choice deterministic.
                    .then(b.cmp(&a))
            })
            .unwrap_or(0);
        kept.push(best);
    }
    kept.sort_unstable();
    report.clusters = groups.len();
    report.largest_cluster = largest(&groups);
    report.duplicates = n - groups.len();
    report.drop_rate = report.duplicates as f64 / n as f64;
    (kept, report)
}

/// Cluster count and duplicate rate for a set of signatures, without choosing
/// survivors. Useful when a caller only wants the audit numbers.
pub fn audit(sigs: &[Vec<u32>], exact: &[u64], cfg: &Config) -> Report {
    let zeros = vec![0.0; sigs.len()];
    let (_, report) = cluster(sigs, exact, &zeros, cfg);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(texts: &[&str]) -> Vec<u64> {
        texts.iter().map(|t| hash::text_hash(t)).collect()
    }

    #[test]
    fn recall_curve_matches_the_formula() {
        let cfg = Config::default();
        assert!(cfg.recall(0.95) > 0.99);
        assert!(cfg.recall(0.80) > 0.9);
        assert!(cfg.recall(0.30) < 0.05);
        assert_eq!(cfg.perms, cfg.rows() * cfg.bands);
    }

    #[test]
    fn near_duplicates_merge_and_exact_duplicates_count() {
        let texts = [
            "Fed holds rates steady as inflation cools and hiring slows",
            "Fed holds rates steady as inflation cools and hiring slows down",
            "Fed holds rates steady as inflation cools and hiring slows",
            "Nvidia lifts guidance on data centre demand for the year",
        ];
        let sigs = signatures(&texts, &Config::default());
        let exact = hashes(&texts);
        let quality = vec![0.1, 0.9, 0.5, 0.5];
        let (kept, report) = cluster(&sigs, &exact, &quality, &Config::default());
        assert_eq!(report.exact_duplicates, 1);
        assert_eq!(kept.len(), 2, "{kept:?}");
        assert!(kept.contains(&1), "the highest quality member survives");
        assert!(kept.contains(&3));
        assert!((report.drop_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn unrelated_texts_stay_apart() {
        let texts = [
            "Fed holds rates steady as inflation cools and hiring slows",
            "Airline cancels flights as storm closes northern airports",
            "Chip maker lifts guidance on data centre demand",
        ];
        let sigs = signatures(&texts, &Config::default());
        let exact = hashes(&texts);
        let zeros = vec![0.0; 3];
        let (kept, report) = cluster(&sigs, &exact, &zeros, &Config::default());
        assert_eq!(kept.len(), 3);
        assert_eq!(report.duplicates, 0);
    }
}
