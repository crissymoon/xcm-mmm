//! Nested walk-forward selection.
//!
//! The single holdout in the forecasting command repairs the headline number,
//! not the procedure: the subset is still chosen on a window that includes the
//! rows used to compare candidates. Nested selection closes that gap. An outer
//! walk-forward loop reserves a test block that the inner loop never sees; the
//! inner loop runs the whole evolutionary search on the prefix before that
//! block, and the outer loop scores the chosen subset once on the block.
//!
//! The cost is that the search is repeated once per outer fold. The benefit is
//! that the reported mean and spread are out-of-sample for the selection
//! procedure itself, not only for the final fitted model. This is the textbook
//! answer for a time series with a tuned model choice; a single tail split is
//! weaker protection and the card says so.

use std::sync::Arc;

use crate::data::{walk_forward, Dataset, Fold};
use crate::de::{self, DeConfig};
use crate::forecast::{score_over_with, SelectionProblem, SparseAccuracy};
use crate::learn::Learner;

/// How a nested run is configured.
#[derive(Clone, Debug)]
pub struct NestedConfig {
    /// Test blocks in the outer loop.
    pub outer_folds: usize,
    /// Validation blocks inside each outer training prefix.
    pub inner_folds: usize,
    /// Rows dropped between a training window and the window after it.
    pub embargo: usize,
    /// Evaluation budget for every inner search.
    pub budget: usize,
    /// Population size for every inner search.
    pub population: usize,
    /// Error added per unit of feature fraction.
    pub sparsity: f64,
    /// Base seed; each outer fold uses `seed + fold`.
    pub seed: u64,
    /// Hypothesis class fitted inside the folds.
    pub learner: Learner,
}

impl Default for NestedConfig {
    fn default() -> Self {
        Self {
            outer_folds: 4,
            inner_folds: 3,
            embargo: 1,
            budget: 2_000,
            population: 30,
            sparsity: 0.001,
            seed: 1,
            learner: Learner::ridge(1.0),
        }
    }
}

/// One outer fold's selected subset and its score on the reserved block.
#[derive(Clone, Debug)]
pub struct OuterScore {
    pub fold: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
    pub columns: Vec<usize>,
    pub names: Vec<String>,
    pub mse: f64,
    pub r2: f64,
    pub ic: f64,
    pub skill: f64,
}

/// The result of a nested selection run.
#[derive(Clone, Debug)]
pub struct NestedResult {
    pub learner: String,
    pub outer: Vec<OuterScore>,
    /// Mean skill across outer folds.
    pub mean_skill: f64,
    /// Standard deviation of skill across outer folds.
    pub spread: f64,
    /// Mean number of selected features.
    pub mean_features: f64,
    /// How often each column was chosen, most frequent first.
    pub selection: Vec<(usize, String, usize)>,
}

impl NestedResult {
    /// One line for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "nested {} outer {}  skill {:.1}% +/- {:.1}  features {:.1}",
            self.learner,
            self.outer.len(),
            self.mean_skill * 100.0,
            self.spread * 100.0,
            self.mean_features
        )
    }
}

/// Run nested walk-forward selection.
///
/// The dataset must already be standardised with statistics that a caller is
/// willing to defend; the outer test blocks are not standardised again.
pub fn run(dataset: &Dataset, config: &NestedConfig) -> Result<NestedResult, String> {
    if config.outer_folds == 0 || config.inner_folds == 0 {
        return Err("nested selection needs at least one outer and one inner fold".to_string());
    }
    let outer = walk_forward(dataset.len(), config.outer_folds, config.embargo);
    if outer.is_empty() {
        return Err(format!(
            "{} rows is too few for {} outer folds",
            dataset.len(),
            config.outer_folds
        ));
    }
    let mut scores = Vec::with_capacity(outer.len());
    let mut frequency = vec![0usize; dataset.n_features()];
    for (index, fold) in outer.iter().enumerate() {
        let prefix = dataset.slice(0..fold.train_end);
        if prefix.len() < 12 {
            return Err(format!(
                "outer fold {index} leaves {} rows, too few to select on",
                prefix.len()
            ));
        }
        let inner = SelectionProblem::with_learner(
            Arc::new(prefix),
            config.inner_folds,
            config.embargo,
            config.learner.clone(),
        );
        if inner.folds.is_empty() {
            return Err(format!(
                "outer fold {index} has too few rows for {} inner folds",
                config.inner_folds
            ));
        }
        let problem = SparseAccuracy::new(inner, config.sparsity);
        let de_config = DeConfig {
            budget: config.budget,
            population: config.population,
            seed: config.seed.wrapping_add(index as u64),
            ..DeConfig::default()
        };
        let outcome = de::minimize(&problem, &de_config);
        let columns = problem.inner.decode(&outcome.best_x);
        if columns.is_empty() {
            return Err(format!("outer fold {index} selected no feature"));
        }
        for &column in &columns {
            frequency[column] += 1;
        }
        let held = Fold {
            train_start: 0,
            train_end: fold.train_end,
            test_start: fold.test_start,
            test_end: fold.test_end,
        };
        let report = score_over_with(
            dataset,
            std::slice::from_ref(&held),
            &columns,
            &config.learner,
        );
        scores.push(OuterScore {
            fold: index,
            train_end: fold.train_end,
            test_start: fold.test_start,
            test_end: fold.test_end,
            columns,
            names: report.names.clone(),
            mse: report.mse,
            r2: report.r2,
            ic: report.ic,
            skill: report.skill,
        });
    }
    let n = scores.len() as f64;
    let mean_skill = scores.iter().map(|s| s.skill).sum::<f64>() / n;
    let variance = scores
        .iter()
        .map(|s| (s.skill - mean_skill) * (s.skill - mean_skill))
        .sum::<f64>()
        / n;
    let mean_features = scores.iter().map(|s| s.columns.len() as f64).sum::<f64>() / n;
    let mut selection: Vec<(usize, String, usize)> = frequency
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(column, count)| (column, dataset.names[column].clone(), *count))
        .collect();
    selection.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    Ok(NestedResult {
        learner: config.learner.name().to_string(),
        outer: scores,
        mean_skill,
        spread: variance.sqrt(),
        mean_features,
        selection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> Dataset {
        let mut data = Dataset::synthetic(400, 8, 3, 13);
        data.standardize();
        data
    }

    #[test]
    fn every_outer_fold_selects_and_scores_out_of_sample() {
        let data = panel();
        let config = NestedConfig {
            outer_folds: 3,
            inner_folds: 2,
            budget: 500,
            population: 20,
            seed: 5,
            ..NestedConfig::default()
        };
        let result = run(&data, &config).expect("nested run");
        assert_eq!(result.outer.len(), 3);
        assert!(result.mean_skill.is_finite());
        assert!(result.spread >= 0.0);
        assert!(result.mean_features >= 1.0);
        for score in &result.outer {
            assert!(!score.columns.is_empty());
            assert_eq!(score.names.len(), score.columns.len());
            assert!(score.test_end > score.test_start);
            assert!(score.test_start >= score.train_end);
        }
        let total: usize = result.selection.iter().map(|(_, _, count)| count).sum();
        assert!(total >= result.outer.len());
    }

    #[test]
    fn too_few_rows_is_a_refusal() {
        let data = panel().slice(0..20);
        let config = NestedConfig {
            outer_folds: 4,
            ..NestedConfig::default()
        };
        assert!(run(&data, &config).is_err());
    }
}
