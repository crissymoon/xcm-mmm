//! Building pairs from labelled rows.
//!
//! Three rules shape every bank, and each one is a published lesson rather than
//! a preference:
//!
//! * the negative has to be hard, so it is drawn by similarity, not at random;
//! * the two sides must be farther apart in outcome than `min_margin`, because a
//!   pair with no signal teaches a model to guess;
//! * the two sides must not be nearly the same text, because a pair that differs
//!   by one word teaches a model to key on that word.
//!
//! The negative is also drawn from a band rather than the single hardest
//! candidate. The hardest neighbour in a small corpus is usually a document that
//! agrees with the anchor and happens to have a different label, which is the
//! false negative that ANCE, RocketQA and Robinson all warn about. Sampling
//! among the hardest `neighbours` is the cheap defence.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::pairs::{NegSource, Pair, Task};
use crate::rng::Rng;
use crate::text::hash::{self, mix};
use crate::text::{quality, select, tokenize};

/// One labelled row to mine from.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: String,
    pub text: String,
    pub group: String,
    pub source: String,
    pub day: String,
    pub split: String,
    pub label_end: String,
    pub y: f64,
}

/// Limits and knobs.
#[derive(Clone, Debug)]
pub struct Config {
    /// Smallest outcome gap a pair may carry.
    pub min_margin: f64,
    /// Largest cosine similarity the two sides may share.
    pub max_similarity: f64,
    /// How many of the hardest negatives to consider before sampling one.
    pub neighbours: usize,
    /// Documents shorter than this are never used.
    pub min_words: usize,
    /// Cap per task, zero for no cap.
    pub max_per_task: usize,
    /// Longest text copied into a pair, in characters.
    pub max_chars: usize,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_margin: 0.01,
            max_similarity: 0.95,
            neighbours: 5,
            min_words: 4,
            max_per_task: 0,
            max_chars: 700,
            seed: 11,
        }
    }
}

/// What mining produced, and what it refused.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub rows: usize,
    pub days: usize,
    pub groups: usize,
    pub rank: usize,
    pub preference: usize,
    pub contrast: usize,
    pub dropped_margin: usize,
    pub dropped_trivial: usize,
    pub dropped_short: usize,
    pub dropped_duplicate: usize,
}

struct Prepared {
    quality: f64,
    words: usize,
    vector: Vec<(u32, f32)>,
}

/// Build the pair banks from labelled rows.
pub fn mine(rows: &[Row], cfg: &Config) -> (Vec<Pair>, Report) {
    let mut report = Report {
        rows: rows.len(),
        ..Report::default()
    };
    if rows.is_empty() {
        return (Vec::new(), report);
    }
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let df = select::document_frequency(&texts);
    let prepared: Vec<Prepared> = texts
        .par_iter()
        .map(|text| Prepared {
            quality: quality::score(&quality::Stats::of(text), &quality::Weights::default()).total,
            words: tokenize::count_words(text),
            vector: select::weigh(&select::counts(text), &df, texts.len()),
        })
        .collect();

    let mut by_day: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_group: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, row) in rows.iter().enumerate() {
        by_day.entry(row.day.as_str()).or_default().push(i);
        by_group.entry(row.group.as_str()).or_default().push(i);
    }
    report.days = by_day.len();
    report.groups = by_group.len();

    let mut out: Vec<Pair> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut rng = Rng::new(cfg.seed);

    // Rank bank: within one session and one split, a top quartile document
    // against a bottom quartile document from a different group.
    let mut days: Vec<&&str> = by_day.keys().collect();
    days.sort_unstable();
    for day in days {
        let members = &by_day[*day];
        if members.len() < 3 {
            continue;
        }
        let mut sorted = members.clone();
        sorted.sort_by(|&a, &b| {
            rows[a]
                .y
                .partial_cmp(&rows[b].y)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let low = sorted[sorted.len() / 4];
        let high = sorted[sorted.len() * 3 / 4];
        let cut_low = rows[low].y;
        let cut_high = rows[high].y;
        for &i in members {
            if rows[i].y < cut_high {
                continue;
            }
            let mut candidates: Vec<(f32, usize)> = Vec::new();
            for &j in members {
                if rows[j].y > cut_low
                    || rows[j].group == rows[i].group
                    || rows[j].split != rows[i].split
                    || prepared[j].words < cfg.min_words
                    || prepared[i].words < cfg.min_words
                {
                    continue;
                }
                let sim = hash::cosine(&prepared[i].vector, &prepared[j].vector);
                if sim >= cfg.max_similarity as f32 {
                    report.dropped_trivial += 1;
                    continue;
                }
                candidates.push((sim, j));
            }
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let shortlist = &candidates[..candidates.len().min(cfg.neighbours.max(1))];
            if shortlist.is_empty() {
                continue;
            }
            let (sim, j) = shortlist[rng.below(shortlist.len())];
            let margin = rows[i].y - rows[j].y;
            if margin < cfg.min_margin {
                report.dropped_margin += 1;
                continue;
            }
            let gap = Gap {
                similarity: sim,
                margin,
            };
            let mut pair = build(
                Task::Rank,
                &rows[i],
                &rows[j],
                Some(&prepared[i]),
                Some(&prepared[j]),
                gap,
                cfg,
            );
            pair.neg_source = Some(NegSource::Rank);
            if seen.insert(pair.id.clone()) {
                report.rank += 1;
                out.push(pair);
            } else {
                report.dropped_duplicate += 1;
            }
        }
    }

    // Preference bank: same group, a clear gap in outcome, the harder of the two
    // candidate orders is kept and the easier one is discarded. Comparisons are
    // limited to a window of neighbouring rows in the group so a group with a
    // long history does not turn mining into a quadratic pass.
    const WINDOW: usize = 60;
    const MAX_LAG: i64 = 30;
    for group in by_group.values() {
        let mut order = group.clone();
        order.sort_by(|&a, &b| rows[a].day.cmp(&rows[b].day).then(a.cmp(&b)));
        for (position, &i) in order.iter().enumerate() {
            if prepared[i].words < cfg.min_words {
                report.dropped_short += 1;
                continue;
            }
            let mut best: Option<(f32, usize)> = None;
            for &j in order.iter().skip(position + 1).take(WINDOW) {
                if rows[j].split != rows[i].split
                    || prepared[j].words < cfg.min_words
                    || (rows[j].y - rows[i].y).abs() < cfg.min_margin
                    || day_distance(&rows[i].day, &rows[j].day) > MAX_LAG
                {
                    continue;
                }
                let sim = hash::cosine(&prepared[i].vector, &prepared[j].vector);
                if sim >= cfg.max_similarity as f32 {
                    report.dropped_trivial += 1;
                    continue;
                }
                if best.map(|(s, _)| sim > s).unwrap_or(true) {
                    best = Some((sim, j));
                }
            }
            let Some((sim, j)) = best else {
                continue;
            };
            let (winner, loser) = if rows[i].y >= rows[j].y {
                (i, j)
            } else {
                (j, i)
            };
            let margin = rows[winner].y - rows[loser].y;
            let gap = Gap {
                similarity: sim,
                margin,
            };
            let mut pair = build(
                Task::Preference,
                &rows[winner],
                &rows[loser],
                Some(&prepared[winner]),
                Some(&prepared[loser]),
                gap,
                cfg,
            );
            pair.neg_source = Some(NegSource::Similarity);
            if seen.insert(pair.id.clone()) {
                report.preference += 1;
                out.push(pair);
            } else {
                report.dropped_duplicate += 1;
            }
        }
    }

    // Contrast bank: anchor, a positive from the same group on an earlier day,
    // and a negative from the same day in another group.
    for (i, row) in rows.iter().enumerate() {
        if prepared[i].words < cfg.min_words || row.day.is_empty() {
            continue;
        }
        let mut positive: Option<(f32, usize, i64)> = None;
        let mut negative: Option<(f32, usize)> = None;
        let mut dropped_for_similarity = false;
        for &j in by_group.get(row.group.as_str()).into_iter().flatten() {
            if j == i || rows[j].split != row.split || rows[j].day >= row.day {
                continue;
            }
            if prepared[j].words < cfg.min_words {
                continue;
            }
            let lag = day_distance(&rows[j].day, &row.day);
            if lag > 5 {
                continue;
            }
            let sim = hash::cosine(&prepared[i].vector, &prepared[j].vector);
            if sim >= cfg.max_similarity as f32 {
                dropped_for_similarity = true;
                continue;
            }
            // Most similar first, then closest in time, so the positive is the
            // document the anchor is most likely to be confused with.
            let better = match positive {
                None => true,
                Some((s, _, l)) => sim > s || (sim == s && lag < l),
            };
            if better {
                positive = Some((sim, j, lag));
            }
        }
        if dropped_for_similarity {
            report.dropped_trivial += 1;
        }
        for &j in by_day.get(row.day.as_str()).into_iter().flatten() {
            if j == i || rows[j].group == row.group || rows[j].split != row.split {
                continue;
            }
            if prepared[j].words < cfg.min_words {
                continue;
            }
            let sim = hash::cosine(&prepared[i].vector, &prepared[j].vector);
            if sim >= cfg.max_similarity as f32 {
                continue;
            }
            if negative.map(|(s, _)| sim > s).unwrap_or(true) {
                negative = Some((sim, j));
            }
        }
        let (Some((_, p, _)), Some((sim, n))) = (positive, negative) else {
            continue;
        };
        // A contrast row is a retrieval label: same issuer earlier, another
        // issuer on the same day. The outcome gap is not the label, so no margin
        // is claimed; both outcomes are still recorded.
        let gap = Gap {
            similarity: sim,
            margin: 0.0,
        };
        let mut pair = build(
            Task::Contrast,
            row,
            &rows[n],
            Some(&prepared[i]),
            Some(&prepared[n]),
            gap,
            cfg,
        );
        pair.margin = None;
        pair.pos = Some(rows[p].id.clone());
        pair.text_pos = Some(tokenize::truncate(&rows[p].text, cfg.max_chars));
        pair.neg_source = Some(NegSource::Temporal);
        pair.stamp();
        if seen.insert(pair.id.clone()) {
            report.contrast += 1;
            out.push(pair);
        } else {
            report.dropped_duplicate += 1;
        }
    }

    if cfg.max_per_task > 0 {
        out = cap_per_task(out, cfg, &mut rng, &mut report);
    }
    (out, report)
}

/// How the two sides relate: how similar they are, and how far apart their
/// outcomes were.
#[derive(Clone, Copy, Debug)]
struct Gap {
    similarity: f32,
    margin: f64,
}

fn build(
    task: Task,
    a: &Row,
    b: &Row,
    pa: Option<&Prepared>,
    pb: Option<&Prepared>,
    gap: Gap,
    cfg: &Config,
) -> Pair {
    let mut pair = Pair::new(task, &a.split);
    pair.day = a.day.clone();
    pair.label_end = a.label_end.clone();
    pair.a = a.id.clone();
    pair.b = Some(b.id.clone());
    pair.text_a = tokenize::truncate(&a.text, cfg.max_chars);
    pair.text_b = Some(tokenize::truncate(&b.text, cfg.max_chars));
    pair.y_a = Some(a.y);
    pair.y_b = Some(b.y);
    pair.margin = Some(gap.margin);
    pair.difficulty = Some(gap.similarity as f64);
    pair.quality_a = pa.map(|p| p.quality);
    pair.quality_b = pb.map(|p| p.quality);
    pair.source_a = a.source.clone();
    pair.source_b = b.source.clone();
    pair.group_a = a.group.clone();
    pair.group_b = b.group.clone();
    pair.words_a = pa.map(|p| p.words);
    pair.words_b = pb.map(|p| p.words);
    pair.stamp();
    pair
}

fn day_distance(earlier: &str, later: &str) -> i64 {
    // Dates arrive as YYYY-MM-DD and compare lexically, so the distance is only
    // needed as a coarse gate; parse the day of month when the prefix matches.
    if earlier.len() != 10 || later.len() != 10 || earlier[..7] != later[..7] {
        return 6;
    }
    let a = earlier[8..].parse::<i64>().unwrap_or(0);
    let b = later[8..].parse::<i64>().unwrap_or(0);
    (b - a).abs()
}

fn cap_per_task(pairs: Vec<Pair>, cfg: &Config, rng: &mut Rng, report: &mut Report) -> Vec<Pair> {
    let mut by_task: HashMap<Task, Vec<Pair>> = HashMap::new();
    for pair in pairs {
        by_task.entry(pair.task).or_default().push(pair);
    }
    let mut out = Vec::new();
    for (task, mut list) in by_task {
        if list.len() > cfg.max_per_task {
            rng.shuffle(&mut list);
            list.truncate(cfg.max_per_task);
            list.sort_by(|a, b| a.id.cmp(&b.id));
            let _ = task;
        }
        match task {
            Task::Rank => report.rank = list.len(),
            Task::Preference => report.preference = list.len(),
            Task::Contrast => report.contrast = list.len(),
            _ => {}
        }
        out.extend(list);
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Feature digest of a text, used as the merge key in [`crate::pairs::pack`].
pub fn text_digest(text: &str) -> String {
    hash::hex64(mix(hash::text_hash(text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, text: &str, group: &str, day: &str, y: f64) -> Row {
        Row {
            id: id.into(),
            text: text.into(),
            group: group.into(),
            source: group.into(),
            day: day.into(),
            split: "train".into(),
            label_end: day.into(),
            y,
        }
    }

    #[test]
    fn rank_and_preference_pairs_carry_a_margin() {
        let rows = vec![
            row(
                "a",
                "Chip maker lifts guidance on data centre demand for the year",
                "AAA",
                "2024-05-01",
                0.09,
            ),
            row(
                "b",
                "Airline cancels flights as storm closes northern airports",
                "BBB",
                "2024-05-01",
                -0.04,
            ),
            row(
                "c",
                "Retailer cuts outlook after weak spring trading",
                "CCC",
                "2024-05-01",
                -0.07,
            ),
            row(
                "d",
                "Bank raises dividend after stress test clears",
                "AAA",
                "2024-05-03",
                0.05,
            ),
        ];
        let (pairs, report) = mine(&rows, &Config::default());
        assert!(report.rank > 0, "{report:?}");
        assert!(pairs.iter().all(|p| p.margin.unwrap_or(0.0) >= 0.01));
        assert!(pairs
            .iter()
            .all(|p| p.task != Task::Rank || p.y_a.unwrap() >= p.y_b.unwrap()));
    }

    #[test]
    fn a_pair_never_crosses_a_split() {
        let mut rows = vec![
            row(
                "a",
                "Chip maker lifts guidance on data centre demand for the year",
                "AAA",
                "2024-05-01",
                0.09,
            ),
            row(
                "b",
                "Airline cancels flights as storm closes northern airports",
                "BBB",
                "2024-05-01",
                -0.04,
            ),
            row(
                "c",
                "Retailer cuts outlook after weak spring trading",
                "CCC",
                "2024-05-01",
                -0.07,
            ),
        ];
        rows[2].split = "test".into();
        let (pairs, _) = mine(&rows, &Config::default());
        for pair in &pairs {
            let splits: Vec<&str> = pair
                .doc_ids()
                .iter()
                .map(|id| {
                    rows.iter()
                        .find(|r| &r.id == id)
                        .map(|r| r.split.as_str())
                        .unwrap_or("")
                })
                .collect();
            assert!(splits.windows(2).all(|w| w[0] == w[1]), "{pair:?}");
        }
    }

    #[test]
    fn near_identical_documents_are_not_paired() {
        let same = "Fed holds rates steady as inflation cools and hiring slows";
        let rows = vec![
            row("a", same, "AAA", "2024-05-01", 0.09),
            row("b", same, "BBB", "2024-05-01", -0.08),
            row(
                "c",
                "Airline cancels flights as storm closes northern airports",
                "CCC",
                "2024-05-01",
                -0.07,
            ),
            row(
                "d",
                "Retailer cuts outlook after weak spring trading",
                "DDD",
                "2024-05-01",
                -0.06,
            ),
        ];
        let (pairs, report) = mine(&rows, &Config::default());
        assert_eq!(report.dropped_trivial, 1, "{report:?}");
        assert!(pairs
            .iter()
            .all(|p| p.text_b.as_deref() != Some(p.text_a.as_str())));
    }
}
