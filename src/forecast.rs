//! Walk-forward forecasting built on the same problem traits as the benchmark
//! suite.
//!
//! The question this module answers is narrow and practical: which features
//! predict the target out of sample, and how much accuracy does a smaller model
//! cost? Ridge regression is fitted on a training block and scored on the block
//! that follows, with an embargo so a label can never look at its own future.
//! Feature selection is then a problem with two objectives, cross-validated
//! error and model size, which [`crate::moead`] solves directly; the
//! single-objective wrapper turns it into a penalised objective for
//! [`crate::de`].
//!
//! Ridge is used rather than a gradient boosted model because it is closed form,
//! deterministic, fast enough to evaluate tens of thousands of times inside an
//! evolutionary loop, and honest about its own uncertainty through the ridge
//! penalty.

use std::ops::Range;
use std::sync::Arc;

use crate::data::{walk_forward, Dataset, Fold};
use crate::linalg::{mean, solve_spd, spearman};
use crate::problem::{Bounds, Problem};

/// Default ridge penalty. Features are standardised first, so one is a mild
/// shrinkage rather than a strong one.
pub const DEFAULT_LAMBDA: f64 = 1.0;

/// Cross-validated error used when a fit fails, large enough to lose.
pub const FAILED_FIT: f64 = 1.0e12;

/// A fitted ridge model over a chosen subset of columns.
#[derive(Clone, Debug)]
pub struct Ridge {
    /// Feature indices the model uses.
    pub columns: Vec<usize>,
    /// Coefficient per entry of `columns`.
    pub weights: Vec<f64>,
    /// Intercept in the original target units.
    pub intercept: f64,
    /// Penalty that was used.
    pub lambda: f64,
}

impl Ridge {
    /// Predict one row.
    pub fn predict(&self, row: &[f64]) -> f64 {
        let mut value = self.intercept;
        for (w, &c) in self.weights.iter().zip(&self.columns) {
            value += w * row[c];
        }
        value
    }

    /// Coefficients paired with their feature indices, largest magnitude first.
    pub fn ranked(&self) -> Vec<(usize, f64)> {
        let mut pairs: Vec<(usize, f64)> = self
            .columns
            .iter()
            .copied()
            .zip(&self.weights)
            .map(|(c, w)| (c, *w))
            .collect();
        pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        pairs
    }
}

/// Fit a ridge regression on `rows` for the given feature columns.
///
/// Returns `None` for an empty column set, a window shorter than two rows, or a
/// system that stays singular after the ridge.
// The normal equations are indexed by column on both sides, so the plain range
// keeps the formula legible.
#[allow(clippy::needless_range_loop)]
pub fn fit_ridge(
    dataset: &Dataset,
    rows: Range<usize>,
    columns: &[usize],
    lambda: f64,
) -> Option<Ridge> {
    let p = columns.len();
    let n = rows.len();
    if p == 0 || n < 3 || rows.end > dataset.len() {
        return None;
    }
    let y_mean = mean(&dataset.target[rows.clone()]);
    let mut x_means = vec![0.0; p];
    for (j, &c) in columns.iter().enumerate() {
        let total: f64 = dataset.features[rows.clone()].iter().map(|r| r[c]).sum();
        x_means[j] = total / n as f64;
    }

    let mut a = vec![vec![0.0; p]; p];
    let mut b = vec![0.0; p];
    for i in rows {
        let y = dataset.target[i] - y_mean;
        for j in 0..p {
            let xj = dataset.features[i][columns[j]] - x_means[j];
            b[j] += xj * y;
            for k in 0..=j {
                a[j][k] += xj * (dataset.features[i][columns[k]] - x_means[k]);
            }
        }
    }
    for j in 0..p {
        for k in 0..j {
            a[k][j] = a[j][k];
        }
    }
    let weights = solve_spd(a, b, lambda.max(0.0))?;
    let intercept = y_mean
        - weights
            .iter()
            .zip(&x_means)
            .map(|(w, m)| w * m)
            .sum::<f64>();
    if !intercept.is_finite() {
        return None;
    }
    Some(Ridge {
        columns: columns.to_vec(),
        weights,
        intercept,
        lambda,
    })
}

/// Mean squared error.
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() {
        return f64::NAN;
    }
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p) * (a - p))
        .sum::<f64>()
        / actual.len() as f64
}

/// Mean absolute error.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.is_empty() {
        return f64::NAN;
    }
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Coefficient of determination against the mean of `actual`.
pub fn r2(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() < 2 {
        return f64::NAN;
    }
    let m = mean(actual);
    let sse: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p) * (a - p))
        .sum();
    let sst: f64 = actual.iter().map(|a| (a - m) * (a - m)).sum();
    if sst <= 0.0 {
        0.0
    } else {
        1.0 - sse / sst
    }
}

/// Score of one fold.
#[derive(Clone, Debug)]
pub struct FoldScore {
    /// First row of the training window.
    pub train_start: usize,
    /// One past the last training row.
    pub train_end: usize,
    /// First test row.
    pub test_start: usize,
    /// One past the last test row.
    pub test_end: usize,
    /// Mean squared error on the test window.
    pub mse: f64,
    /// Coefficient of determination on the test window.
    pub r2: f64,
    /// Mean absolute error on the test window.
    pub mae: f64,
    /// Spearman rank correlation between prediction and outcome.
    pub ic: f64,
}

/// Out-of-sample score of one feature set.
#[derive(Clone, Debug)]
pub struct Report {
    /// Column indices used.
    pub columns: Vec<usize>,
    /// Names of those columns.
    pub names: Vec<String>,
    /// Pooled mean squared error.
    pub mse: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Pooled mean absolute error.
    pub mae: f64,
    /// Pooled coefficient of determination.
    pub r2: f64,
    /// Mean information coefficient across folds.
    pub ic: f64,
    /// Error of predicting the training mean, the honest baseline.
    pub baseline_mse: f64,
    /// Fraction of the baseline error removed, `1 - mse / baseline_mse`.
    pub skill: f64,
    /// Per fold detail.
    pub folds: Vec<FoldScore>,
}

impl Report {
    /// One line summary for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "{} features  mse {:.6}  r2 {:.4}  ic {:.3}  skill {:.1}%",
            self.columns.len(),
            self.mse,
            self.r2,
            self.ic,
            self.skill * 100.0
        )
    }
}

/// Feature selection as a two objective problem: out-of-sample error against the
/// fraction of features kept.
///
/// The decision vector is a mask in `[0, 1]`; a feature is used when its value
/// is above 0.5. At least one feature must survive, expressed as the constraint
/// `1 - count <= 0`.
#[derive(Clone, Debug)]
pub struct SelectionProblem {
    /// Data, shared with the caller.
    pub dataset: Arc<Dataset>,
    /// Walk-forward windows.
    pub folds: Vec<Fold>,
    /// Ridge penalty.
    pub lambda: f64,
    bounds: Bounds,
}

impl SelectionProblem {
    /// Build a problem with a walk-forward split.
    pub fn new(dataset: Arc<Dataset>, folds: usize, embargo: usize, lambda: f64) -> Self {
        let windows = walk_forward(dataset.len(), folds, embargo);
        let bounds = Bounds::unit(dataset.n_features());
        Self {
            dataset,
            folds: windows,
            lambda,
            bounds,
        }
    }

    /// Indices of the features a mask selects.
    pub fn decode(&self, mask: &[f64]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.5)
            .map(|(i, _)| i)
            .collect()
    }

    /// Mean cross-validated error of a column subset.
    pub fn cv_mse(&self, columns: &[usize]) -> f64 {
        if columns.is_empty() || self.folds.is_empty() {
            return FAILED_FIT;
        }
        let mut total = 0.0;
        for fold in &self.folds {
            let model = match fit_ridge(&self.dataset, fold.train(), columns, self.lambda) {
                Some(m) => m,
                None => return FAILED_FIT,
            };
            let mut sum = 0.0;
            let mut count = 0usize;
            for i in fold.test() {
                let error = self.dataset.target[i] - model.predict(&self.dataset.features[i]);
                sum += error * error;
                count += 1;
            }
            if count == 0 {
                return FAILED_FIT;
            }
            total += sum / count as f64;
        }
        total / self.folds.len() as f64
    }

    /// Full out-of-sample report for a column subset.
    pub fn report(&self, columns: &[usize]) -> Report {
        score_over(&self.dataset, &self.folds, columns, self.lambda)
    }
}

/// Score a column subset over an explicit set of folds.
///
/// Taking the folds as an argument is what lets a caller hold rows back: a
/// selection search can run on one window and the chosen subset be scored on a
/// window the search never saw.
pub fn score_over(dataset: &Dataset, folds: &[Fold], columns: &[usize], lambda: f64) -> Report {
    let mut predicted = Vec::new();
    let mut actual = Vec::new();
    let mut baseline = Vec::new();
    let mut scores = Vec::new();
    for fold in folds {
        let model = fit_ridge(dataset, fold.train(), columns, lambda);
        let train_mean = mean(&dataset.target[fold.train()]);
        let mut fold_actual = Vec::new();
        let mut fold_predicted = Vec::new();
        for i in fold.test() {
            let p = match &model {
                Some(m) => m.predict(&dataset.features[i]),
                None => train_mean,
            };
            fold_actual.push(dataset.target[i]);
            fold_predicted.push(p);
            baseline.push(train_mean);
        }
        predicted.extend_from_slice(&fold_predicted);
        actual.extend_from_slice(&fold_actual);
        scores.push(FoldScore {
            train_start: fold.train_start,
            train_end: fold.train_end,
            test_start: fold.test_start,
            test_end: fold.test_end,
            mse: mse(&fold_actual, &fold_predicted),
            r2: r2(&fold_actual, &fold_predicted),
            mae: mae(&fold_actual, &fold_predicted),
            ic: spearman(&fold_predicted, &fold_actual),
        });
    }
    let pooled_mse = mse(&actual, &predicted);
    let baseline_mse = mse(&actual, &baseline);
    let ic = if scores.is_empty() {
        f64::NAN
    } else {
        scores.iter().map(|s| s.ic).sum::<f64>() / scores.len() as f64
    };
    Report {
        columns: columns.to_vec(),
        names: columns.iter().map(|&c| dataset.names[c].clone()).collect(),
        mse: pooled_mse,
        rmse: pooled_mse.sqrt(),
        mae: mae(&actual, &predicted),
        r2: r2(&actual, &predicted),
        ic,
        baseline_mse,
        skill: if baseline_mse > 0.0 {
            1.0 - pooled_mse / baseline_mse
        } else {
            0.0
        },
        folds: scores,
    }
}

impl Problem for SelectionProblem {
    fn name(&self) -> &str {
        "feature-selection"
    }

    fn dim(&self) -> usize {
        self.dataset.n_features()
    }

    fn objectives(&self) -> usize {
        2
    }

    fn constraints(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        // The mask is decoded against a fixed 0.5 threshold, so the box is the
        // unit hypercube and never depends on the data.
        &self.bounds
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], g: &mut [f64]) {
        let columns = self.decode(x);
        if columns.is_empty() {
            f[0] = FAILED_FIT;
            f[1] = 0.0;
            g[0] = 1.0;
            return;
        }
        f[0] = self.cv_mse(&columns);
        f[1] = columns.len() as f64 / self.dim() as f64;
        // Satisfied when at least one feature survives.
        g[0] = 1.0 - columns.len() as f64;
    }
}

/// The same selection, collapsed to one objective with a sparsity penalty.
///
/// This is what [`crate::de`] minimises. The penalty trades error against model
/// size on one axis, where [`SelectionProblem`] gives the whole trade-off curve.
#[derive(Clone, Debug)]
pub struct SparseAccuracy {
    /// The two objective problem underneath.
    pub inner: SelectionProblem,
    /// Error added per unit of feature fraction.
    pub sparsity_penalty: f64,
}

impl SparseAccuracy {
    /// Wrap a selection problem with a sparsity penalty.
    pub fn new(inner: SelectionProblem, sparsity_penalty: f64) -> Self {
        Self {
            inner,
            sparsity_penalty,
        }
    }
}

impl Problem for SparseAccuracy {
    fn name(&self) -> &str {
        "sparse-accuracy"
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn bounds(&self) -> &Bounds {
        self.inner.bounds()
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], _g: &mut [f64]) {
        let columns = self.inner.decode(x);
        if columns.is_empty() {
            // No feature at all is not a model, so it loses by construction.
            f[0] = FAILED_FIT;
            return;
        }
        let fraction = columns.len() as f64 / self.inner.dim() as f64;
        f[0] = self.inner.cv_mse(&columns) + self.sparsity_penalty * fraction;
    }
}

/// Pick the knee of a two objective front, the point furthest below the chord
/// between the two extremes.
///
/// Returns the index of the smallest error when the front has fewer than three
/// points, and `None` for an empty front. Both objectives are normalised first,
/// so the result does not depend on their units.
pub fn knee(front: &[Vec<f64>]) -> Option<usize> {
    if front.is_empty() {
        return None;
    }
    if front.len() < 3 {
        return Some(
            front
                .iter()
                .enumerate()
                .min_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
        );
    }
    let m = front[0].len();
    let mut lo = vec![f64::INFINITY; m];
    let mut hi = vec![f64::NEG_INFINITY; m];
    for p in front {
        for i in 0..m {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    let normalized: Vec<Vec<f64>> = front
        .iter()
        .map(|p| {
            (0..m)
                .map(|i| {
                    let span = hi[i] - lo[i];
                    if span > 0.0 {
                        (p[i] - lo[i]) / span
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let first = &normalized[0];
    let last = &normalized[normalized.len() - 1];
    let chord: Vec<f64> = (0..m).map(|i| last[i] - first[i]).collect();
    let chord_len = chord.iter().map(|v| v * v).sum::<f64>().sqrt();
    if chord_len <= 0.0 {
        return Some(0);
    }
    let mut best = 0usize;
    let mut best_distance = -1.0;
    for (idx, p) in normalized.iter().enumerate() {
        let offset: Vec<f64> = (0..m).map(|i| p[i] - first[i]).collect();
        // Perpendicular distance to the chord in the plane it spans.
        let along = offset.iter().zip(&chord).map(|(o, c)| o * c).sum::<f64>() / chord_len;
        let perp2 = offset
            .iter()
            .zip(&chord)
            .map(|(o, c)| {
                let component = along * c / chord_len;
                (o - component) * (o - component)
            })
            .sum::<f64>();
        let distance = perp2.sqrt();
        if distance > best_distance {
            best_distance = distance;
            best = idx;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::de;

    fn standardised(rows: usize, p: usize, informative: usize, seed: u64) -> Arc<Dataset> {
        let mut d = Dataset::synthetic(rows, p, informative, seed);
        d.standardize();
        Arc::new(d)
    }

    #[test]
    fn ridge_recovers_an_exact_linear_relation() {
        let mut d = Dataset::synthetic(300, 4, 2, 11);
        // Overwrite the target with a known function of two features.
        for (row, y) in d.features.iter().zip(d.target.iter_mut()) {
            *y = 2.0 * row[0] - 1.5 * row[1];
        }
        let model = fit_ridge(&d, 0..300, &[0, 1], 1e-8).unwrap();
        assert!((model.weights[0] - 2.0).abs() < 1e-4, "{:?}", model.weights);
        assert!((model.weights[1] + 1.5).abs() < 1e-4, "{:?}", model.weights);
        assert!(model.intercept.abs() < 1e-6);
    }

    #[test]
    fn report_beats_the_mean_baseline_on_a_real_signal() {
        let dataset = standardised(400, 12, 4, 5);
        let problem = SelectionProblem::new(dataset, 4, 2, DEFAULT_LAMBDA);
        let report = problem.report(&[0, 1, 2, 3]);
        assert!(report.r2 > 0.3, "r2 {}", report.r2);
        assert!(report.skill > 0.2, "skill {}", report.skill);
        assert!(report.ic > 0.3, "ic {}", report.ic);
    }

    #[test]
    fn a_mask_round_trips_to_columns() {
        let dataset = standardised(120, 5, 2, 2);
        let problem = SelectionProblem::new(dataset, 3, 1, DEFAULT_LAMBDA);
        let mask = vec![0.9, 0.1, 0.6, 0.4, 0.51];
        assert_eq!(problem.decode(&mask), vec![0, 2, 4]);
    }

    #[test]
    fn the_constraint_rejects_an_empty_mask() {
        let dataset = standardised(120, 5, 2, 2);
        let problem = SelectionProblem::new(dataset, 3, 1, DEFAULT_LAMBDA);
        let mut f = vec![0.0; 2];
        let mut g = vec![0.0; 1];
        problem.evaluate(&[0.0; 5], &mut f, &mut g);
        assert!(g[0] > 0.0);
        assert_eq!(f[1], 0.0);
    }

    #[test]
    fn de_finds_a_working_subset() {
        let dataset = standardised(300, 10, 3, 21);
        let inner = SelectionProblem::new(dataset, 4, 2, DEFAULT_LAMBDA);
        let problem = SparseAccuracy::new(inner, 0.001);
        let cfg = de::DeConfig {
            budget: 4_000,
            population: 40,
            seed: 4,
            ..de::DeConfig::default()
        };
        let out = de::minimize(&problem, &cfg);
        assert!(out.best.is_feasible());
        let columns = problem.inner.decode(&out.best_x);
        assert!(!columns.is_empty());
        // The first three features carry the signal, so a good subset should
        // overlap them rather than chase noise.
        let overlap = columns.iter().filter(|c| **c < 3).count();
        assert!(overlap >= 2, "selected {columns:?}");
    }

    #[test]
    fn score_over_scores_a_window_the_search_never_used() {
        let dataset = standardised(400, 8, 3, 5);
        let fold = Fold {
            train_start: 0,
            train_end: 300,
            test_start: 310,
            test_end: 400,
        };
        let all: Vec<usize> = (0..8).collect();
        let report = score_over(&dataset, std::slice::from_ref(&fold), &all, DEFAULT_LAMBDA);
        assert_eq!(report.folds.len(), 1);
        assert_eq!(report.folds[0].test_start, 310);
        assert_eq!(report.folds[0].test_end, 400);
        assert!(report.mse.is_finite() && report.mse >= 0.0);
        assert!(report.r2.is_finite());
        assert_eq!(report.columns, all);
    }

    #[test]
    fn knee_picks_the_middle_of_a_curve() {
        let front = vec![
            vec![0.1, 0.9],
            vec![0.3, 0.4],
            vec![0.5, 0.2],
            vec![0.9, 0.1],
        ];
        let pick = knee(&front).unwrap();
        assert!(pick == 1 || pick == 2, "picked {pick}");
        assert_eq!(knee(&[]), None);
    }
}
