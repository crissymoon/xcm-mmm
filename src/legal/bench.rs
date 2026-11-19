//! Scoring the translator against the pairs it ships beside.
//!
//! `pairs.jsonl` holds a legal passage and a plain passage that is known to say
//! the same thing, either published as a before-and-after by a plain language
//! project or edited here from the same source. Until now nothing scored the
//! translator on them, so "the rewrite helps" was a claim with no number under
//! it.
//!
//! Three measurements per pair, and a win needs all three:
//!
//! * `delta` is the drop in token distance to the plain side. Tokens are
//!   lowercased words, the distance is an edit distance over them, and it is
//!   normalised by the longer side so a long pair is not worth more than a
//!   short one.
//! * `retention` is the share of the source's words still present in the
//!   output. Without it a rewrite that deletes half the document would look
//!   like a large improvement, because the survivors are all plain.
//! * `avoidable` is the term count the instrument flags after the rewrite. A
//!   rule that trades one flagged phrase for another has not helped, whatever
//!   the distance says.
//!
//! The pair is only evidence of direction, never of quality: a rewrite that
//! lands closer to the reference may still be wrong about what the section
//! requires, and the module says so rather than counting a distance as an
//! answer. `identity` is reported beside every number for the same reason. If
//! doing nothing already wins a pair, the corpus contains a reference that the
//! source matches, and a win rate has to be read against that count.
//!
//! The aggregate is a share with a Wilson interval, and the mean delta carries
//! a percentile bootstrap interval from [`crate::uncertainty`]. Rows are
//! resampled, not scores, so the pairing survives: a case that improves by
//! twenty points is not allowed to reappear beside one that worsens by one.
//!
//! Two numbers are reported rather than one, because they answer different
//! questions. `all` asks whether the instrument helps on the corpus as shipped.
//! `edited` asks whether an edit it did make helped, over the pairs a rule
//! actually touched. When the first is low and the second is not, the table is
//! missing wording, not skill, and the fix is `registers::Coverage`, not a
//! larger replacement table. This is the same distinction the paired
//! comparisons in LegalBench and the contract tasks of CUAD draw between
//! coverage and correctness; a benchmark that reports only the mixed number
//! cannot tell a caller which of the two to work on.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};

use serde::{Deserialize, Serialize};

use crate::num::round4;
use crate::uncertainty::{bootstrap, wilson};

use super::translate::Legal;

/// A share of the source words below this is treated as a truncation, not a
/// rewrite. Chosen at two thirds because the published pairs drop the padding
/// and keep the content, and nothing honest drops a third of the words.
pub const MIN_RETENTION: f64 = 0.6;

/// One row of `pairs.jsonl`, narrowed to what scoring needs.
#[derive(Clone, Debug, Deserialize)]
pub struct Pair {
    pub id: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub text_a: String,
    pub text_b: String,
}

/// One pair, scored.
#[derive(Clone, Debug, Serialize)]
pub struct Case {
    pub id: String,
    pub task: String,
    pub split: String,
    pub words_in: usize,
    pub edits: usize,
    pub avoidable_in: usize,
    pub avoidable_out: usize,
    pub retention: f64,
    pub distance_before: f64,
    pub distance_after: f64,
    /// How far the rewrite moved the text towards the plain reference.
    pub delta: f64,
    pub won: bool,
    /// Why it did not win, empty when it did.
    pub refused: String,
}

impl Case {
    pub fn lost(&self) -> bool {
        !self.won
    }
}

/// A group of cases reduced to the numbers a reader scans first.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Slice {
    pub name: String,
    pub cases: usize,
    pub won: usize,
    /// Wins over cases, from zero to one.
    pub rate: f64,
    /// Wilson interval on `rate`, never a normal interval: the rate is often
    /// near one and a normal interval collapses there.
    pub low: f64,
    pub high: f64,
    pub mean_delta: f64,
    pub delta_low: f64,
    pub delta_high: f64,
    pub retention: f64,
    pub avoidable_drop: f64,
}

/// Exact McNemar test on paired direction of movement.
///
/// A win rate on its own does not say whether the rewrite moved anything or
/// whether the corpus happens to be easy. McNemar asks the narrower question
/// that a paired design can answer: among the pairs that moved at all, is the
/// split between closer and further distinguishable from a coin flip? The
/// reference is Dietterich (1998), *Approximate Statistical Tests for Comparing
/// Supervised Classification Learning Algorithms*, which is also the reason the
/// test is exact here rather than the chi-square approximation: the discordant
/// count is small on a corpus this size, and the approximation is worst there.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Mcnemar {
    pub closer: usize,
    pub further: usize,
    /// Two sided exact binomial probability under a fair coin.
    pub p: f64,
}

impl Mcnemar {
    /// Count the pairs that moved, then test the split.
    pub fn of(cases: &[Case]) -> Self {
        let closer = cases.iter().filter(|case| case.delta > 0.0).count();
        let further = cases.iter().filter(|case| case.delta < 0.0).count();
        Self {
            closer,
            further,
            p: round4(exact_two_sided(closer, further)),
        }
    }

    /// True when the movement is one sided at the usual five percent.
    pub fn significant(&self) -> bool {
        self.p < 0.05 && self.closer != self.further
    }
}

/// Two sided exact binomial test on `k` of `n`, both flips being all that a
/// discordant pair can be. The recursion is on the probability mass, so no
/// binomial coefficient is ever formed and a corpus of ten thousand pairs does
/// not overflow.
pub fn exact_two_sided(a: usize, b: usize) -> f64 {
    let n = a + b;
    if n == 0 {
        return 1.0;
    }
    let tail = a.min(b);
    let mut mass = 0.5f64.powi(n as i32);
    let mut sum = mass;
    for k in 1..=tail {
        mass *= (n - k + 1) as f64 / k as f64;
        sum += mass;
    }
    (2.0 * sum).min(1.0)
}

/// The whole corpus, scored.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Report {
    pub pairs: usize,
    /// Pairs that could not be scored, by id, with the reason.
    pub skipped: Vec<String>,
    pub all: Slice,
    /// The same share over the pairs a rule actually edited. A pair no rule
    /// touched is a gap in the table, and mixing the two hides which problem a
    /// low rate is.
    pub when_edited: Slice,
    /// One slice per `task`, in order of size.
    pub tasks: Vec<Slice>,
    /// One slice per `split`, so a held out share is read on its own.
    pub splits: Vec<Slice>,
    /// Pairs where doing nothing already won.
    pub identity: usize,
    /// Pairs where the rewrite moved nothing either way.
    pub unresolved: usize,
    /// Pairs the rule table edited at all. The other pairs are a coverage gap,
    /// not a rewrite that was tried and failed.
    pub edited: usize,
    /// Every pair, scored, so a report carries its own evidence.
    pub cases: Vec<Case>,
    pub level: f64,
    pub resamples: usize,
    pub seed: u64,
    /// The paired test on direction of movement, against doing nothing.
    pub test: Mcnemar,
}

/// Read `pairs.jsonl`. A malformed line is a refusal with its number, because
/// a silently dropped pair would change a denominator nobody would notice.
pub fn read_pairs(path: &str) -> Result<Vec<Pair>, String> {
    let file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut pairs = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{}: {e}", number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let pair: Pair =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{}: {e}", number + 1))?;
        pairs.push(pair);
    }
    Ok(pairs)
}

/// Score every pair. Pure: the instrument is read, never written.
pub fn score(legal: &Legal, pairs: &[Pair], seed: u64, resamples: usize) -> Report {
    let mut cases = Vec::new();
    let mut skipped = Vec::new();
    for pair in pairs {
        match one(legal, pair) {
            Ok(case) => cases.push(case),
            Err(why) => skipped.push(format!("{}: {why}", pair.id)),
        }
    }
    let tasks = group(&cases, |case| case.task.clone(), seed, resamples);
    let splits = group(&cases, |case| case.split.clone(), seed, resamples);
    let touched: Vec<Case> = cases
        .iter()
        .filter(|case| case.edits > 0)
        .cloned()
        .collect();
    Report {
        pairs: cases.len(),
        skipped,
        all: summarise("all", &cases, seed, resamples),
        when_edited: summarise("edited", &touched, seed, resamples),
        tasks,
        splits,
        identity: cases
            .iter()
            .filter(|case| case.distance_before == 0.0 && case.won)
            .count(),
        unresolved: cases.iter().filter(|case| case.delta == 0.0).count(),
        edited: cases.iter().filter(|case| case.edits > 0).count(),
        test: Mcnemar::of(&cases),
        cases,
        level: 0.95,
        resamples,
        seed,
    }
}

fn one(legal: &Legal, pair: &Pair) -> Result<Case, String> {
    let reading = legal.translate(&pair.text_a, false);
    let before = tokens(&pair.text_a);
    let after = tokens(&reading.text);
    let target = tokens(&pair.text_b);
    if before.is_empty() || target.is_empty() {
        return Err("empty side".into());
    }
    let distance_before = distance(&before, &target);
    let distance_after = distance(&after, &target);
    let delta = round4((distance_before - distance_after).max(-1.0));
    let retention = round4(retain(&before, &after));
    let avoided = reading.after.avoidable;
    let grew = avoided > reading.stats.avoidable;
    let truncated = retention < MIN_RETENTION;
    let won = delta > 0.0 && !grew && !truncated;
    let refused = if delta <= 0.0 {
        "no closer to the plain side".to_string()
    } else if grew {
        format!(
            "flagged terms rose from {} to {avoided}",
            reading.stats.avoidable
        )
    } else if truncated {
        format!("kept {:.0} percent of the words", retention * 100.0)
    } else {
        String::new()
    };
    Ok(Case {
        id: pair.id.clone(),
        task: if pair.task.is_empty() {
            "unlabelled".to_string()
        } else {
            pair.task.clone()
        },
        split: if pair.split.is_empty() {
            "unlabelled".to_string()
        } else {
            pair.split.clone()
        },
        words_in: before.len(),
        edits: reading.changes.len(),
        avoidable_in: reading.stats.avoidable,
        avoidable_out: avoided,
        retention,
        distance_before,
        distance_after,
        delta,
        won,
        refused,
    })
}

fn group(cases: &[Case], key: impl Fn(&Case) -> String, seed: u64, resamples: usize) -> Vec<Slice> {
    let mut buckets: BTreeMap<String, Vec<&Case>> = BTreeMap::new();
    for case in cases {
        buckets.entry(key(case)).or_default().push(case);
    }
    let mut slices: Vec<Slice> = buckets
        .into_iter()
        .map(|(name, rows)| {
            let owned: Vec<Case> = rows.into_iter().cloned().collect();
            summarise(&name, &owned, seed, resamples)
        })
        .collect();
    slices.sort_by(|a, b| b.cases.cmp(&a.cases).then(a.name.cmp(&b.name)));
    slices
}

/// Reduce cases to one row. A zero `resamples` gives a point estimate only,
/// which is what a task slice with three cases deserves.
pub fn summarise(name: &str, cases: &[Case], seed: u64, resamples: usize) -> Slice {
    let n = cases.len();
    if n == 0 {
        return Slice {
            name: name.to_string(),
            ..Default::default()
        };
    }
    let won = cases.iter().filter(|case| case.won).count();
    let rate = won as f64 / n as f64;
    let (low, high) = wilson(won, n, 0.95);
    let deltas: Vec<f64> = cases.iter().map(|case| case.delta).collect();
    let wins: Vec<bool> = cases.iter().map(|case| case.won).collect();
    let mean = deltas.iter().sum::<f64>() / n as f64;
    let (delta_low, delta_high) = if resamples >= 200 && n >= 4 {
        let interval = bootstrap(&deltas, &wins, resamples, 0.95, seed, |drawn, _| {
            drawn.iter().sum::<f64>() / drawn.len().max(1) as f64
        });
        (interval.low, interval.high)
    } else {
        (mean, mean)
    };
    Slice {
        name: name.to_string(),
        cases: n,
        won,
        rate: round4(rate),
        low: round4(low),
        high: round4(high),
        mean_delta: round4(mean),
        delta_low: round4(delta_low),
        delta_high: round4(delta_high),
        retention: round4(cases.iter().map(|c| c.retention).sum::<f64>() / n as f64),
        avoidable_drop: round4(
            cases
                .iter()
                .map(|c| c.avoidable_in as f64 - c.avoidable_out as f64)
                .sum::<f64>()
                / n as f64,
        ),
    }
}

/// Lowercased words with punctuation removed. Digits are kept, so a section
/// number counts as a token: deleting "30" from "30 days" is a content change
/// and should cost distance.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' || ch == '$' {
            word.extend(ch.to_lowercase());
        } else if !word.is_empty() {
            out.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

/// Word level edit distance, normalised by the longer side.
///
/// One row of the matrix is kept rather than all of it. A hundred word pair is
/// ten thousand cells either way; a document length pair is the reason for the
/// single row, not a benchmark of three sentences.
pub fn distance(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, left) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, right) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(left != right);
            let deletion = previous[j + 1] + 1;
            let insertion = current[j] + 1;
            current[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    round4(previous[b.len()] as f64 / a.len().max(b.len()) as f64)
}

/// Share of `source` words that survive into `output`, as a multiset.
pub fn retain(source: &[String], output: &[String]) -> f64 {
    if source.is_empty() {
        return 1.0;
    }
    let mut bag: BTreeMap<&str, usize> = BTreeMap::new();
    for token in output {
        *bag.entry(token.as_str()).or_default() += 1;
    }
    let mut kept = 0usize;
    for token in source {
        if let Some(count) = bag.get_mut(token.as_str()) {
            if *count > 0 {
                *count -= 1;
                kept += 1;
            }
        }
    }
    round4(kept as f64 / source.len() as f64)
}

impl Report {
    /// The report as a reader scans it: one line per slice, the interval beside
    /// the rate, and the pairs that lost named with their reason.
    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{} pairs scored, {} skipped, {} edited by a rule, {} identity, {} unmoved",
            self.pairs,
            self.skipped.len(),
            self.edited,
            self.identity,
            self.unresolved
        ));
        lines.push(format!(
            "{:<12} {:>5} {:>5} {:>7} {:>15} {:>9} {:>15} {:>7} {:>6}",
            "slice", "n", "won", "rate", "95% interval", "delta", "delta interval", "kept", "drop"
        ));
        for slice in std::iter::once(&self.all)
            .chain(std::iter::once(&self.when_edited))
            .chain(self.tasks.iter())
            .chain(self.splits.iter())
        {
            lines.push(format!(
                "{:<12} {:>5} {:>5} {:>7.3} {:>7.3} to {:<6.3} {:>9.3} {:>6.3} to {:<6.3} {:>7.3} {:>6.2}",
                slice.name,
                slice.cases,
                slice.won,
                slice.rate,
                slice.low,
                slice.high,
                slice.mean_delta,
                slice.delta_low,
                slice.delta_high,
                slice.retention,
                slice.avoidable_drop
            ));
        }
        if self.resamples > 0 {
            lines.push(format!(
                "intervals: Wilson on the rate, {} bootstrap resamples on the delta, seed {}",
                self.resamples, self.seed
            ));
        }
        lines.push(format!(
            "closer on {}, further on {}, unchanged on {}: McNemar exact p = {:.4}{}",
            self.test.closer,
            self.test.further,
            self.unresolved,
            self.test.p,
            if self.test.significant() {
                ", significant at five percent"
            } else {
                ""
            }
        ));
        lines.push(
            "the plain side was written by a person, so a rule pass cannot reach 1.0: read the rate against the edited count, not against one"
                .to_string(),
        );
        for case in self.cases.iter().filter(|case| case.lost()).take(12) {
            lines.push(format!(
                "  {} {}: {} (delta {:+.3})",
                case.split, case.id, case.refused, case.delta
            ));
        }
        if self.identity > 0 {
            lines.push(format!(
                "{} pairs were already plain on the source side; doing nothing won those",
                self.identity
            ));
        }
        for row in &self.skipped {
            lines.push(format!("skipped {row}"));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(text: &str) -> Vec<String> {
        tokens(text)
    }

    #[test]
    fn identical_token_lists_are_zero_apart() {
        assert_eq!(distance(&toks("a b c"), &toks("a b c")), 0.0);
    }

    #[test]
    fn a_missing_word_costs_one_edit_over_the_longer_side() {
        assert_eq!(distance(&toks("a b c"), &toks("a c")), round4(1.0 / 3.0));
    }

    #[test]
    fn punctuation_and_case_do_not_count_as_edits() {
        assert_eq!(
            distance(&toks("Shall, and until."), &toks("shall and until")),
            0.0
        );
    }

    #[test]
    fn a_deleted_document_retains_nothing() {
        assert_eq!(
            retain(&toks("one two three"), &toks("one")),
            round4(1.0 / 3.0)
        );
        assert_eq!(retain(&toks("one one"), &toks("one one")), 1.0);
    }

    #[test]
    fn a_repeated_word_is_not_credited_twice() {
        assert_eq!(retain(&toks("one one"), &toks("one")), 0.5);
    }

    fn case(delta: f64, won: bool) -> Case {
        Case {
            id: format!("case-{delta}"),
            task: "correction".into(),
            split: "test".into(),
            words_in: 10,
            edits: 1,
            avoidable_in: 2,
            avoidable_out: 1,
            retention: 1.0,
            distance_before: 0.5,
            distance_after: round4(0.5 - delta),
            delta,
            won,
            refused: String::new(),
        }
    }

    #[test]
    fn a_slice_that_wins_every_case_has_a_lower_bound_below_one() {
        let cases = vec![
            case(0.1, true),
            case(0.2, true),
            case(0.1, true),
            case(0.3, true),
        ];
        let slice = summarise("all", &cases, 7, 200);
        assert_eq!(slice.rate, 1.0);
        assert!(slice.low < 1.0 && slice.low > 0.5, "{}", slice.low);
        assert!(slice.delta_low <= slice.mean_delta);
        assert!(slice.delta_high >= slice.mean_delta);
    }

    #[test]
    fn a_slice_with_no_resamples_reports_its_point_estimate_only() {
        let cases = vec![case(0.1, true), case(-0.1, false)];
        let slice = summarise("thin", &cases, 7, 0);
        assert_eq!(slice.mean_delta, 0.0);
        assert_eq!(slice.delta_low, 0.0);
        assert_eq!(slice.delta_high, 0.0);
    }

    #[test]
    fn a_split_that_moves_nowhere_is_not_significant() {
        let test = Mcnemar::of(&[case(0.0, false), case(0.0, false)]);
        assert_eq!(test.closer, 0);
        assert_eq!(test.further, 0);
        assert_eq!(test.p, 1.0);
        assert!(!test.significant());
    }

    #[test]
    fn nineteen_closer_and_three_further_is_a_small_p() {
        let p = exact_two_sided(19, 3);
        assert!(p < 0.01, "{p}");
        assert!(p > 0.0);
    }

    #[test]
    fn an_even_split_is_not_significant() {
        assert_eq!(exact_two_sided(4, 4), 1.0);
    }

    #[test]
    fn the_test_is_two_sided_and_never_exceeds_one() {
        for (a, b) in [(1, 0), (0, 5), (10, 9), (30, 1)] {
            let p = exact_two_sided(a, b);
            assert!((0.0..=1.0).contains(&p), "{a} {b} {p}");
            assert_eq!(p, exact_two_sided(b, a));
        }
    }

    #[test]
    fn an_empty_slice_is_zero_everywhere_rather_than_a_division() {
        let slice = summarise("none", &[], 7, 200);
        assert_eq!(slice.cases, 0);
        assert_eq!(slice.rate, 0.0);
        assert_eq!(slice.mean_delta, 0.0);
    }
}
