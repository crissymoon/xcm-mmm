//! Fitted regression models behind one interface.
//!
//! Feature selection should not be tied to one hypothesis class. This module
//! keeps the walk-forward folds and the metrics where they are and swaps the
//! learner underneath: ridge, which is closed form and fast enough for tens of
//! thousands of evaluations, or a small gradient boosted tree ensemble, which
//! can represent an interaction that ridge cannot unless the interaction was
//! constructed by hand before the search.
//!
//! The tree ensemble is deliberately modest: depth three by default, a minimum
//! leaf, squared error and shrinkage. It is not an XGBoost replacement. It is a
//! nonlinear learner that is deterministic, dependency free and cheap enough to
//! sit inside the same evolutionary loop, so the claim that a nonlinear model
//! cannot be substituted is no longer true of this crate.
//!
//! A model is fitted on an explicit row range and an explicit column subset, so
//! the embargo and the held-out window mean exactly what they mean for ridge.

use std::ops::Range;

use crate::data::Dataset;
use crate::forecast::{fit_ridge, Ridge};

/// A row index and the gradient it carries, which is what a split partitions.
type Sample = Vec<(usize, f64)>;

/// Which hypothesis class a selection run uses.
#[derive(Clone, Debug, PartialEq)]
pub enum Learner {
    /// Ridge regression with a fixed penalty.
    Ridge { lambda: f64 },
    /// Gradient boosted regression trees.
    Trees(TreeConfig),
}

impl Learner {
    /// Ridge with a penalty.
    pub fn ridge(lambda: f64) -> Self {
        Self::Ridge { lambda }
    }

    /// The default nonlinear learner.
    pub fn trees() -> Self {
        Self::Trees(TreeConfig::default())
    }

    /// A short name for a report.
    pub fn name(&self) -> &'static str {
        match self {
            Learner::Ridge { .. } => "ridge",
            Learner::Trees(_) => "trees",
        }
    }

    /// Fit on a row range and a column subset.
    pub fn fit(&self, dataset: &Dataset, rows: Range<usize>, columns: &[usize]) -> Option<Fitted> {
        match self {
            Learner::Ridge { lambda } => {
                fit_ridge(dataset, rows, columns, *lambda).map(Fitted::Ridge)
            }
            Learner::Trees(config) => {
                TreeEnsemble::fit(dataset, rows, columns, config).map(Fitted::Trees)
            }
        }
    }
}

/// A fitted model that predicts one row.
#[derive(Clone, Debug)]
pub enum Fitted {
    Ridge(Ridge),
    Trees(TreeEnsemble),
}

impl Fitted {
    /// Prediction for one row, in the target's units.
    pub fn predict(&self, row: &[f64]) -> f64 {
        match self {
            Fitted::Ridge(model) => model.predict(row),
            Fitted::Trees(model) => model.predict(row),
        }
    }
}

/// Shape and step size of the tree ensemble.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeConfig {
    /// Number of boosting rounds.
    pub estimators: usize,
    /// Maximum depth of each tree. One is a stump.
    pub depth: usize,
    /// Smallest number of rows a leaf may hold.
    pub min_leaf: usize,
    /// Shrinkage applied to every tree.
    pub learning_rate: f64,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            estimators: 48,
            depth: 3,
            min_leaf: 8,
            learning_rate: 0.08,
        }
    }
}

/// One regression tree. A leaf holds a value and no split.
#[derive(Clone, Debug)]
pub struct Tree {
    pub feature: Option<usize>,
    pub threshold: f64,
    pub value: f64,
    pub left: Option<Box<Tree>>,
    pub right: Option<Box<Tree>>,
}

impl Tree {
    fn leaf(value: f64) -> Self {
        Self {
            feature: None,
            threshold: 0.0,
            value,
            left: None,
            right: None,
        }
    }

    /// Prediction for one row.
    pub fn predict(&self, row: &[f64]) -> f64 {
        match self.feature {
            None => self.value,
            Some(column) => {
                let next = if row[column] <= self.threshold {
                    self.left.as_ref()
                } else {
                    self.right.as_ref()
                };
                next.map(|tree| tree.predict(row)).unwrap_or(self.value)
            }
        }
    }

    /// Number of leaves, for a report.
    pub fn leaves(&self) -> usize {
        match (&self.left, &self.right) {
            (Some(a), Some(b)) => a.leaves() + b.leaves(),
            _ => 1,
        }
    }
}

/// A boosted ensemble fitted on a column subset.
#[derive(Clone, Debug)]
pub struct TreeEnsemble {
    pub columns: Vec<usize>,
    pub base: f64,
    pub learning_rate: f64,
    pub trees: Vec<Tree>,
    pub config: TreeConfig,
}

impl TreeEnsemble {
    /// Fit by least squares boosting on the training rows.
    pub fn fit(
        dataset: &Dataset,
        rows: Range<usize>,
        columns: &[usize],
        config: &TreeConfig,
    ) -> Option<Self> {
        let start = rows.start.min(dataset.len());
        let end = rows.end.min(dataset.len());
        if columns.is_empty() || end <= start + 1 {
            return None;
        }
        let indices: Vec<usize> = (start..end).collect();
        let base = indices.iter().map(|&i| dataset.target[i]).sum::<f64>() / indices.len() as f64;
        if !base.is_finite() {
            return None;
        }
        let mut residual: Vec<(usize, f64)> = indices
            .iter()
            .map(|&i| (i, dataset.target[i] - base))
            .collect();
        let mut trees = Vec::with_capacity(config.estimators);
        for _ in 0..config.estimators.max(1) {
            let tree = build_tree(dataset, &residual, columns, config, config.depth);
            if tree.feature.is_none() {
                break;
            }
            for (index, value) in residual.iter_mut() {
                *value -= config.learning_rate * tree.predict(&dataset.features[*index]);
            }
            trees.push(tree);
        }
        // A constant target, or a target with no admissible split, is still a
        // model: the base rate. Refusing it would make selection treat a flat
        // window as a failed fit rather than a flat one.
        Some(Self {
            columns: columns.to_vec(),
            base,
            learning_rate: config.learning_rate,
            trees,
            config: config.clone(),
        })
    }

    /// Prediction for one row.
    pub fn predict(&self, row: &[f64]) -> f64 {
        let mut value = self.base;
        for tree in &self.trees {
            value += self.learning_rate * tree.predict(row);
        }
        value
    }

    /// Total leaves across the ensemble.
    pub fn leaves(&self) -> usize {
        self.trees.iter().map(Tree::leaves).sum()
    }
}

fn build_tree(
    dataset: &Dataset,
    samples: &[(usize, f64)],
    columns: &[usize],
    config: &TreeConfig,
    depth: usize,
) -> Tree {
    let mean = samples.iter().map(|(_, value)| value).sum::<f64>() / samples.len().max(1) as f64;
    if depth == 0 || samples.len() < 2 * config.min_leaf.max(1) {
        return Tree::leaf(mean);
    }
    let min_leaf = config.min_leaf.max(1);
    let total: f64 = samples.iter().map(|(_, value)| value).sum();
    let total_sq: f64 = samples.iter().map(|(_, value)| value * value).sum();
    let mut best_sse = total_sq;
    let mut best: Option<(usize, f64)> = None;

    for &column in columns {
        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by(|&a, &b| {
            dataset.features[samples[a].0][column]
                .total_cmp(&dataset.features[samples[b].0][column])
        });
        let mut left_sum = 0.0;
        let mut left_sq = 0.0;
        for position in 0..order.len().saturating_sub(1) {
            let index = order[position];
            let value = dataset.features[samples[index].0][column];
            let r = samples[index].1;
            left_sum += r;
            left_sq += r * r;
            let next = dataset.features[samples[order[position + 1]].0][column];
            if position + 1 < min_leaf || order.len() - position - 1 < min_leaf || next <= value {
                continue;
            }
            let right_sum = total - left_sum;
            let right_sq = total_sq - left_sq;
            let left_n = (position + 1) as f64;
            let right_n = (order.len() - position - 1) as f64;
            let sse = (left_sq - left_sum * left_sum / left_n)
                + (right_sq - right_sum * right_sum / right_n);
            if sse < best_sse - 1e-12 {
                best_sse = sse;
                best = Some((column, (value + next) / 2.0));
            }
        }
    }
    let Some((column, threshold)) = best else {
        return Tree::leaf(mean);
    };
    let (left, right): (Sample, Sample) = samples
        .iter()
        .copied()
        .partition(|(i, _)| dataset.features[*i][column] <= threshold);
    if left.is_empty() || right.is_empty() {
        return Tree::leaf(mean);
    }
    Tree {
        feature: Some(column),
        threshold,
        value: mean,
        left: Some(Box::new(build_tree(
            dataset,
            &left,
            columns,
            config,
            depth - 1,
        ))),
        right: Some(Box::new(build_tree(
            dataset,
            &right,
            columns,
            config,
            depth - 1,
        ))),
    }
}

/// Score predictions against actual values.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn nonlinear(rows: usize) -> Dataset {
        let mut data = Dataset::synthetic(rows, 4, 2, 21);
        for (row, y) in data.features.iter().zip(data.target.iter_mut()) {
            // A pure interaction plus a little noise from the synthetic target's
            // own linear part, which ridge can only partly reproduce.
            *y = 3.0 * row[0] * row[1] + 0.2 * row[2];
        }
        data
    }

    #[test]
    fn trees_capture_an_interaction_that_ridge_misses() {
        let data = nonlinear(500);
        let columns = vec![0, 1, 2];
        let trees = Learner::trees()
            .fit(&data, 0..400, &columns)
            .expect("a tree fit");
        let ridge = Learner::ridge(1.0)
            .fit(&data, 0..400, &columns)
            .expect("a ridge fit");
        let actual: Vec<f64> = data.target[400..].to_vec();
        let tree_pred: Vec<f64> = data.features[400..]
            .iter()
            .map(|row| trees.predict(row))
            .collect();
        let ridge_pred: Vec<f64> = data.features[400..]
            .iter()
            .map(|row| ridge.predict(row))
            .collect();
        let tree_mse = mse(&actual, &tree_pred);
        let ridge_mse = mse(&actual, &ridge_pred);
        assert!(
            tree_mse < ridge_mse * 0.5,
            "trees {tree_mse:.4} ridge {ridge_mse:.4}"
        );
    }

    #[test]
    fn a_tree_ensemble_is_deterministic() {
        let data = nonlinear(240);
        let a = Learner::trees().fit(&data, 0..200, &[0, 1, 2]).unwrap();
        let b = Learner::trees().fit(&data, 0..200, &[0, 1, 2]).unwrap();
        let row = &data.features[210];
        assert_eq!(a.predict(row), b.predict(row));
    }

    #[test]
    fn an_empty_column_set_is_refused() {
        let data = nonlinear(60);
        assert!(Learner::trees().fit(&data, 0..50, &[]).is_none());
    }

    #[test]
    fn a_constant_target_falls_back_to_a_leaf() {
        let mut data = Dataset::synthetic(60, 3, 1, 4);
        for y in data.target.iter_mut() {
            *y = 7.0;
        }
        let model = Learner::trees().fit(&data, 0..50, &[0, 1]).unwrap();
        assert!((model.predict(&data.features[0]) - 7.0).abs() < 1e-9);
    }
}
