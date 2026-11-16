//! Combinatorial purged cross-validation and the probability of backtest
//! overfitting.
//!
//! Walk-forward gives one path: every candidate is scored on the same sequence
//! of test blocks, so the run reports one history, not a distribution over
//! histories. Combinatorial purged cross-validation cuts the panel into `N`
//! contiguous groups, tests every choice of `k` of them, and reconstructs the
//! `N / k` complete paths that those splits imply (Lopez de Prado, *Advances in
//! Financial Machine Learning*, 2018, ch. 12; Bailey, Borwein, Lopez de Prado
//! and Zhu, *Journal of Computational Finance* 20(4), 2017). `N = 6, k = 2`
//! is fifteen fits per candidate and five paths.
//!
//! Three decisions are worth stating, because a split is only honest when they
//! are made explicitly.
//!
//! **Purge, not just embargo.** A label computed over the next `horizon` rows
//! overlaps the rows on either side of a test block. Every training row within
//! `purge` rows of a test row is dropped, so the gap is symmetric. A
//! one-directional embargo built for a walk-forward split is not enough here
//! because a test group can sit in the middle of the panel.
//!
//! **Paths, not splits, for the decision reading.** A split scores one
//! candidate on two groups. A path scores the chosen candidate across all `N`
//! groups once, which is the closest analogue to "would this procedure have
//! been right on the whole history". The path spread is printed unflattered.
//!
//! **PBO is a rank statement, not a probability about the future.** For each
//! split, the candidate with the best training score is looked up in the
//! out-of-sample ordering, its relative rank is turned into a logit, and PBO is
//! the share of splits whose logit is at or below zero. It answers "how often
//! does the winner of the search land in the better half out of sample", which
//! is what an overfit selection looks like. It is bounded below by the number
//! of candidates: with two candidates no split can do better than a half.

use crate::data::Dataset;
use crate::learn::Learner;

/// How the panel is cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpcvConfig {
    /// Contiguous groups the panel is cut into.
    pub groups: usize,
    /// Groups withheld together as the test set, per split.
    pub test_groups: usize,
    /// Rows dropped on each side of a test run.
    pub purge: usize,
}

impl Default for CpcvConfig {
    fn default() -> Self {
        Self {
            groups: 6,
            test_groups: 2,
            purge: 1,
        }
    }
}

impl CpcvConfig {
    /// How many fits one candidate costs.
    pub fn splits(&self) -> usize {
        combinations(self.groups, self.test_groups)
    }

    /// How many complete paths the splits imply, `C(N, k) * k / N`.
    ///
    /// Zero when a path cannot exist, which happens only when `k` is not a
    /// divisor of the group count in the sense the construction needs.
    pub fn paths(&self) -> usize {
        if self.groups == 0 || self.test_groups == 0 {
            return 0;
        }
        self.splits() * self.test_groups / self.groups
    }
}

/// One train and test selection of rows.
///
/// The row lists are index vectors rather than ranges because a training set
/// here is every group except the tested ones, which is a union of separated
/// windows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpcvSplit {
    /// Group ids tested by this split, ascending.
    pub groups: Vec<usize>,
    /// Training row indices.
    pub train: Vec<usize>,
    /// Test row indices.
    pub test: Vec<usize>,
}

/// The half-open row spans of `rows` cut into `groups` blocks.
///
/// Any remainder goes to the last group, so the blocks stay contiguous and the
/// last one is at most `groups - 1` rows longer.
pub fn block_bounds(rows: usize, groups: usize) -> Vec<(usize, usize)> {
    if groups == 0 {
        return Vec::new();
    }
    let block = rows / groups;
    if block == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(groups);
    for g in 0..groups {
        let start = g * block;
        let end = if g + 1 == groups {
            rows
        } else {
            (g + 1) * block
        };
        out.push((start, end));
    }
    out
}

/// Every split the configuration implies.
///
/// A split whose training set would fall below twelve rows is dropped instead
/// of fitted on almost nothing; the returned count is what `run` reports.
pub fn splits(rows: usize, config: &CpcvConfig) -> Result<Vec<CpcvSplit>, String> {
    if config.groups < 2 {
        return Err("combinatorial splits need at least two groups".to_string());
    }
    if config.test_groups == 0 || config.test_groups >= config.groups {
        return Err(format!(
            "test_groups {} must be between 1 and {}",
            config.test_groups,
            config.groups - 1
        ));
    }
    let bounds = block_bounds(rows, config.groups);
    if bounds.is_empty() {
        return Err(format!(
            "{rows} rows is too few for {} groups of at least one row",
            config.groups
        ));
    }
    let mut out = Vec::new();
    for choice in subsets(config.groups, config.test_groups) {
        let mut held = vec![false; rows];
        let mut test = Vec::new();
        for &group in &choice {
            let (start, end) = bounds[group];
            for (row, flag) in held.iter_mut().enumerate().take(end).skip(start) {
                *flag = true;
                test.push(row);
            }
        }
        // A purge is symmetric: the rows on both sides of a test run are
        // dropped, because a label can reach either way across the boundary.
        for (first, last) in runs(&choice) {
            let start = bounds[first].0;
            let end = bounds[last].1;
            let from = start.saturating_sub(config.purge);
            let to = (end + config.purge).min(rows);
            for flag in held.iter_mut().take(to).skip(from) {
                *flag = true;
            }
        }
        let train: Vec<usize> = (0..rows).filter(|row| !held[*row]).collect();
        if train.len() < 12 || test.is_empty() {
            continue;
        }
        out.push(CpcvSplit {
            groups: choice,
            train,
            test,
        });
    }
    if out.len() < 2 {
        return Err(format!(
            "{rows} rows leaves {} usable splits, too few to compare",
            out.len()
        ));
    }
    Ok(out)
}

/// Group the split combinations into complete paths.
///
/// Each path is a list of split indices whose test groups are disjoint and
/// together cover every group exactly once, so a path scores one candidate
/// across the whole panel. The search is a depth-first walk over ascending split
/// indices, which finds a cover when one exists and never enumerates the
/// permutations of the same cover.
pub fn paths(splits: &[CpcvSplit], groups: usize) -> Vec<Vec<usize>> {
    let candidate = match splits.first() {
        Some(split) if !split.groups.is_empty() => split.groups.len(),
        _ => return Vec::new(),
    };
    // A path scores every group once, so it can only exist when the tested
    // group count divides the panel's group count: N / k splits per path.
    if groups == 0 || candidate == 0 || groups % candidate != 0 {
        return Vec::new();
    }
    let per_path = groups / candidate;
    let mut remaining: Vec<usize> = (0..splits.len()).collect();
    let mut out = Vec::new();
    while !remaining.is_empty() {
        let mut used = vec![false; groups];
        let mut path = Vec::with_capacity(per_path);
        if !cover(splits, &remaining, &mut used, &mut path, per_path) {
            break;
        }
        for &taken in &path {
            remaining.retain(|&index| index != taken);
        }
        out.push(path);
    }
    out
}

fn cover(
    splits: &[CpcvSplit],
    remaining: &[usize],
    used: &mut Vec<bool>,
    path: &mut Vec<usize>,
    per_path: usize,
) -> bool {
    if path.len() == per_path {
        return used.iter().all(|taken| *taken);
    }
    for &index in remaining {
        if path.last().is_some_and(|last| index <= *last) {
            continue;
        }
        if splits[index].groups.iter().any(|group| used[*group]) {
            continue;
        }
        for &group in &splits[index].groups {
            used[group] = true;
        }
        path.push(index);
        if cover(splits, remaining, used, path, per_path) {
            return true;
        }
        path.pop();
        for &group in &splits[index].groups {
            used[group] = false;
        }
    }
    false
}

/// Probability of backtest overfitting from paired training and test errors.
///
/// Both arguments are indexed `[split][candidate]` and hold mean squared
/// error, so lower is better. Returns `(pbo, mean logit)`. The logit is
/// `ln(omega / (1 - omega))` for the relative out-of-sample rank `omega` of the
/// candidate that won on the training rows, so a negative mean logit means the
/// search picked a candidate in the bottom half on average.
pub fn pbo(train_mse: &[Vec<f64>], test_mse: &[Vec<f64>]) -> (f64, f64) {
    let rows = train_mse.len().min(test_mse.len());
    let mut logits = Vec::with_capacity(rows);
    for row in 0..rows {
        let inside = &train_mse[row];
        let outside = &test_mse[row];
        if inside.len() < 2 || inside.len() != outside.len() {
            continue;
        }
        let mut winner = 0usize;
        for (index, value) in inside.iter().enumerate() {
            if value.total_cmp(&inside[winner]).is_lt() {
                winner = index;
            }
        }
        if !outside[winner].is_finite() {
            continue;
        }
        let mut behind = 0.0;
        let mut ties = 0.0;
        for (index, value) in outside.iter().enumerate() {
            if index == winner {
                continue;
            }
            match value.total_cmp(&outside[winner]) {
                // A candidate with a higher error is ranked behind the winner,
                // so `omega` is a share of the field beaten, not of the field
                // that beat it.
                std::cmp::Ordering::Greater => behind += 1.0,
                std::cmp::Ordering::Equal => ties += 1.0,
                std::cmp::Ordering::Less => {}
            }
        }
        let omega = (1.0 + behind + 0.5 * ties) / (outside.len() as f64 + 1.0);
        let logit = (omega / (1.0 - omega)).ln();
        if logit.is_finite() {
            logits.push(logit);
        }
    }
    if logits.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let share = logits.iter().filter(|value| **value <= 0.0).count() as f64 / logits.len() as f64;
    let mean = logits.iter().sum::<f64>() / logits.len() as f64;
    (share, mean)
}

/// A column subset the search proposed.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// Column indices.
    pub columns: Vec<usize>,
    /// Names of those columns.
    pub names: Vec<String>,
}

/// Per split detail.
#[derive(Clone, Debug)]
pub struct SplitScore {
    /// Groups tested by this split.
    pub groups: Vec<usize>,
    /// Training rows.
    pub train_rows: usize,
    /// Test rows.
    pub test_rows: usize,
    /// Training error per candidate.
    pub train_mse: Vec<f64>,
    /// Test error per candidate.
    pub test_mse: Vec<f64>,
}

/// The audit of a candidate set.
#[derive(Clone, Debug)]
pub struct CpcvResult {
    /// Learner name.
    pub learner: String,
    /// Configuration used.
    pub config: CpcvConfig,
    /// Candidates, in the order given.
    pub candidates: Vec<Candidate>,
    /// Per split detail.
    pub splits: Vec<SplitScore>,
    /// Mean training error per candidate.
    pub train_mse: Vec<f64>,
    /// Mean test error per candidate.
    pub test_mse: Vec<f64>,
    /// Share of splits whose training winner landed in the bottom half.
    pub pbo: f64,
    /// Mean logit of the winning candidate's relative test rank.
    pub mean_logit: f64,
    /// One score per path.
    pub path_scores: Vec<f64>,
    /// Mean of the path scores.
    pub path_mean: f64,
    /// Standard deviation across paths.
    pub path_spread: f64,
    /// Candidate chosen by each split on its training rows.
    pub selected: Vec<usize>,
}

impl CpcvResult {
    /// One line for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "cpcv {} splits {} paths  pbo {:.2}  path mse {:.6} +/- {:.6}",
            self.splits.len(),
            self.path_scores.len(),
            self.pbo,
            self.path_mean,
            self.path_spread
        )
    }

    /// The plain reading of the probability, including its floor.
    pub fn verdict(&self) -> String {
        let floor = 1.0 / (self.candidates.len() as f64 + 1.0);
        if self.candidates.len() < 2 {
            return "pbo undefined: one candidate, nothing to rank".to_string();
        }
        if !self.pbo.is_finite() {
            return "pbo undefined: fewer than two usable splits".to_string();
        }
        if self.pbo >= 0.5 {
            format!(
                "the training winner lands in the better half on {:.0}% of splits, so selection is no better than a coin at this candidate count",
                (1.0 - self.pbo) * 100.0
            )
        } else if self.pbo > floor {
            format!(
                "the training winner is out of the top half on {:.0}% of splits, above the {:.0}% a coin would give at {} candidates",
                self.pbo * 100.0,
                floor * 100.0,
                self.candidates.len()
            )
        } else {
            format!(
                "the training winner ranking is at the chance floor of {:.0}%: the candidate set is too alike for this test to separate",
                floor * 100.0
            )
        }
    }
}

/// Score every candidate on every split and audit the selection.
///
/// The dataset must already be standardised with statistics the caller is
/// willing to defend. Scaling is refitted nowhere, so the numbers are comparable
/// with the walk-forward report beside them.
pub fn run(
    dataset: &Dataset,
    config: &CpcvConfig,
    candidates: &[Candidate],
    learner: &Learner,
) -> Result<CpcvResult, String> {
    if candidates.is_empty() {
        return Err("cpcv needs at least one candidate".to_string());
    }
    if candidates.iter().any(|c| c.columns.is_empty()) {
        return Err("every candidate must name a feature".to_string());
    }
    let splits = splits(dataset.len(), config)?;
    let mut scored = Vec::with_capacity(splits.len());
    for split in &splits {
        let train = dataset.gather(&split.train);
        let test = dataset.gather(&split.test);
        let mut train_mse = Vec::with_capacity(candidates.len());
        let mut test_mse = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let model = learner.fit(&train, 0..train.len(), &candidate.columns);
            let (inside, outside): (f64, f64) = match &model {
                Some(fitted) => (error(&train, fitted), error(&test, fitted)),
                None => (f64::INFINITY, f64::INFINITY),
            };
            train_mse.push(inside);
            test_mse.push(outside);
        }
        scored.push(SplitScore {
            groups: split.groups.clone(),
            train_rows: split.train.len(),
            test_rows: split.test.len(),
            train_mse,
            test_mse,
        });
    }
    let width = candidates.len();
    let mut train_mean = vec![0.0; width];
    let mut test_mean = vec![0.0; width];
    let mut counts = vec![0usize; width];
    for split in &scored {
        for index in 0..width {
            if split.test_mse[index].is_finite() && split.train_mse[index].is_finite() {
                train_mean[index] += split.train_mse[index];
                test_mean[index] += split.test_mse[index];
                counts[index] += 1;
            }
        }
    }
    for index in 0..width {
        if counts[index] == 0 {
            train_mean[index] = f64::NAN;
            test_mean[index] = f64::NAN;
        } else {
            train_mean[index] /= counts[index] as f64;
            test_mean[index] /= counts[index] as f64;
        }
    }
    let inside: Vec<Vec<f64>> = scored.iter().map(|s| s.train_mse.clone()).collect();
    let outside: Vec<Vec<f64>> = scored.iter().map(|s| s.test_mse.clone()).collect();
    let (pbo_value, mean_logit) = pbo(&inside, &outside);
    let selected: Vec<usize> = scored
        .iter()
        .map(|s| trained_winner(&s.train_mse))
        .collect();
    let groups = paths(&splits, config.groups);
    let mut path_scores = Vec::with_capacity(groups.len());
    for path in &groups {
        let per_path = scored[path[0]].train_mse.len();
        let mut best = 0usize;
        let mut best_train = f64::INFINITY;
        for chosen in 0..per_path {
            let mean = path
                .iter()
                .map(|i| scored[*i].train_mse[chosen])
                .sum::<f64>()
                / path.len() as f64;
            if mean < best_train {
                best_train = mean;
                best = chosen;
            }
        }
        let mean = path.iter().map(|i| scored[*i].test_mse[best]).sum::<f64>() / path.len() as f64;
        path_scores.push(mean);
    }
    let path_mean = if path_scores.is_empty() {
        f64::NAN
    } else {
        path_scores.iter().sum::<f64>() / path_scores.len() as f64
    };
    let path_spread = if path_scores.len() < 2 {
        0.0
    } else {
        let variance = path_scores
            .iter()
            .map(|value| (value - path_mean) * (value - path_mean))
            .sum::<f64>()
            / (path_scores.len() - 1) as f64;
        variance.sqrt()
    };
    Ok(CpcvResult {
        learner: learner.name().to_string(),
        config: config.clone(),
        candidates: candidates.to_vec(),
        splits: scored,
        train_mse: train_mean,
        test_mse: test_mean,
        pbo: pbo_value,
        mean_logit,
        path_scores,
        path_mean,
        path_spread,
        selected,
    })
}

fn trained_winner(values: &[f64]) -> usize {
    let mut winner = 0usize;
    for (index, value) in values.iter().enumerate() {
        if value.total_cmp(&values[winner]).is_lt() {
            winner = index;
        }
    }
    winner
}

fn error(data: &Dataset, model: &crate::learn::Fitted) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for (row, target) in data.features.iter().zip(&data.target) {
        let residual = target - model.predict(row);
        sum += residual * residual;
    }
    sum / data.len() as f64
}

/// The `first..first+len` contiguous group ids, merged into runs.
fn runs(choice: &[usize]) -> Vec<(usize, usize)> {
    let mut sorted = choice.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for group in sorted {
        match out.last_mut() {
            Some((_, last)) if *last + 1 == group => *last = group,
            _ => out.push((group, group)),
        }
    }
    out
}

/// All `k` element subsets of `0..n`, lexicographic.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut index: Vec<usize> = (0..k).collect();
    loop {
        out.push(index.clone());
        let mut at = k;
        loop {
            if at == 0 {
                return out;
            }
            at -= 1;
            if index[at] < n - k + at {
                break;
            }
        }
        index[at] += 1;
        for next in at + 1..k {
            index[next] = index[next - 1] + 1;
        }
    }
}

/// `C(n, k)`, or zero when the request is impossible.
fn combinations(n: usize, k: usize) -> usize {
    if k == 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut value = 1usize;
    for step in 0..k {
        value = value * (n - step) / (step + 1);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_never_trains_on_a_row_within_the_purge() {
        let config = CpcvConfig {
            groups: 6,
            test_groups: 2,
            purge: 3,
        };
        let splits = splits(240, &config).expect("splits");
        for split in &splits {
            for &row in &split.train {
                for &test in &split.test {
                    assert!(
                        row.abs_diff(test) > 3,
                        "row {row} sits within the purge of test row {test}"
                    );
                }
            }
        }
    }

    #[test]
    fn every_group_is_tested_once_per_path() {
        let config = CpcvConfig::default();
        let splits = splits(600, &config).expect("splits");
        assert_eq!(splits.len(), 15);
        let paths = paths(&splits, config.groups);
        assert_eq!(paths.len(), 5);
        for path in &paths {
            let mut seen = vec![0usize; config.groups];
            for &index in path {
                for &group in &splits[index].groups {
                    seen[group] += 1;
                }
            }
            assert!(seen.iter().all(|count| *count == 1), "{seen:?}");
        }
    }

    #[test]
    fn the_pbo_floor_follows_the_candidate_count() {
        // Three candidates and a test ordering that reverses the training one
        // gives the worst possible rank for the training winner.
        let inside = vec![vec![1.0, 2.0, 3.0]; 4];
        let outside = vec![vec![3.0, 2.0, 1.0]; 4];
        let (pbo_value, logit) = pbo(&inside, &outside);
        assert!((pbo_value - 1.0).abs() < 1e-12);
        assert!(logit < 0.0);
        // The same ordering keeps the winner first.
        let (clean, _) = pbo(&inside, &inside);
        assert_eq!(clean, 0.0);
    }

    #[test]
    fn a_signal_is_found_and_noise_is_not() {
        let mut data = Dataset::synthetic(600, 8, 3, 21);
        data.standardize();
        let informative = Candidate {
            columns: vec![0, 1, 2],
            names: vec!["a".into(), "b".into(), "c".into()],
        };
        let noise = Candidate {
            columns: vec![6, 7],
            names: vec!["g".into(), "h".into()],
        };
        let config = CpcvConfig::default();
        let result =
            run(&data, &config, &[informative, noise], &Learner::ridge(1.0)).expect("cpcv");
        assert_eq!(result.splits.len(), 15);
        assert_eq!(result.path_scores.len(), 5);
        assert!(result.test_mse[0] < result.test_mse[1]);
        assert_eq!(result.selected.iter().filter(|s| **s == 0).count(), 15);
        assert!(result.pbo <= 0.5);
        assert!(result.summary().contains("15 splits"));
        assert!(result.summary().contains("5 paths"));
    }

    #[test]
    fn the_path_spread_is_over_paths_not_splits() {
        let mut data = Dataset::synthetic(600, 6, 2, 4);
        data.standardize();
        let first = Candidate {
            columns: vec![0, 1],
            names: vec!["a".into(), "b".into()],
        };
        let second = Candidate {
            columns: vec![0, 1, 4, 5],
            names: vec!["a".into(), "b".into(), "e".into(), "f".into()],
        };
        let result = run(
            &data,
            &CpcvConfig::default(),
            &[first, second],
            &Learner::ridge(1.0),
        )
        .expect("cpcv");
        assert_eq!(result.path_scores.len(), 5);
        assert_eq!(result.splits.len(), 15);
        assert!(result.path_spread.is_finite());
        assert!(result.path_spread >= 0.0);
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        assert!(splits(
            600,
            &CpcvConfig {
                groups: 1,
                test_groups: 1,
                purge: 0
            }
        )
        .is_err());
        assert!(splits(
            600,
            &CpcvConfig {
                groups: 6,
                test_groups: 0,
                purge: 0
            }
        )
        .is_err());
        assert!(splits(
            600,
            &CpcvConfig {
                groups: 6,
                test_groups: 6,
                purge: 0
            }
        )
        .is_err());
        assert!(splits(
            10,
            &CpcvConfig {
                groups: 6,
                test_groups: 2,
                purge: 0
            }
        )
        .is_err());
    }

    #[test]
    fn the_binomial_counts_match_the_enumeration() {
        for n in 2..10 {
            for k in 1..n {
                assert_eq!(subsets(n, k).len(), combinations(n, k), "n={n} k={k}");
            }
        }
    }

    #[test]
    fn a_single_candidate_has_no_rank_to_report() {
        let mut data = Dataset::synthetic(300, 5, 2, 9);
        data.standardize();
        let only = Candidate {
            columns: vec![0, 1],
            names: vec!["a".into(), "b".into()],
        };
        let result =
            run(&data, &CpcvConfig::default(), &[only], &Learner::ridge(1.0)).expect("cpcv");
        assert!(result.verdict().contains("one candidate"));
        assert!(result.mean_logit.is_nan());
    }
}
