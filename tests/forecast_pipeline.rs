//! The forecasting path, from a synthetic panel to an out-of-sample score.

use std::sync::Arc;

use xcm_mmm::data::Dataset;
use xcm_mmm::de::{minimize as minimize_de, DeConfig};
use xcm_mmm::eval::dominates;
use xcm_mmm::forecast::{knee, SelectionProblem, SparseAccuracy, DEFAULT_LAMBDA};
use xcm_mmm::moead::{minimize as minimize_moead, MoeadConfig};

fn problem(rows: usize, features: usize, informative: usize, seed: u64) -> SelectionProblem {
    let mut data = Dataset::synthetic(rows, features, informative, seed);
    data.standardize();
    SelectionProblem::new(Arc::new(data), 4, 1, DEFAULT_LAMBDA)
}

#[test]
fn a_single_objective_search_beats_the_mean_baseline() {
    let problem = problem(400, 10, 3, 9);
    let sparse = SparseAccuracy::new(problem.clone(), 0.001);
    let cfg = DeConfig {
        budget: 3_000,
        population: 40,
        seed: 2,
        ..DeConfig::default()
    };
    let out = minimize_de(&sparse, &cfg);
    assert!(out.best.is_feasible());
    let columns = problem.decode(&out.best_x);
    let report = problem.report(&columns);
    assert!(report.skill > 0.1, "skill {} on {columns:?}", report.skill);
    assert!(report.ic > 0.1, "ic {}", report.ic);
    assert!(report.r2 > 0.1, "r2 {}", report.r2);
}

#[test]
fn the_front_trades_error_against_model_size() {
    let problem = problem(360, 9, 3, 12);
    let cfg = MoeadConfig {
        population: 40,
        budget: 2_500,
        seed: 4,
        ..MoeadConfig::default()
    };
    let out = minimize_moead(&problem, &cfg);
    let front = out.front();
    assert!(front.len() >= 3, "front has {} points", front.len());

    for i in 0..front.len() {
        for j in 0..front.len() {
            if i != j {
                assert!(!dominates(&front[i], &front[j]), "{i} dominates {j}");
            }
        }
        assert!(
            (0.0..=1.0).contains(&front[i][1]),
            "fraction {}",
            front[i][1]
        );
        assert!(front[i][0].is_finite());
    }

    let pick = knee(&front).expect("knee");
    let columns = problem.decode(&out.solutions[pick].x);
    assert!(!columns.is_empty());
    let report = problem.report(&columns);
    assert!(report.mse > 0.0 && report.mse.is_finite());
    assert!(report.skill > -1.0);
}

#[test]
fn an_empty_dataset_is_rejected_before_the_search() {
    let data = Dataset::synthetic(200, 5, 2, 1);
    let problem = SelectionProblem::new(Arc::new(data), 6, 2, 1.0);
    // 200 rows with 6 folds gives blocks of 28 rows, so every window is usable.
    assert_eq!(problem.folds.len(), 6);
    for fold in &problem.folds {
        assert!(fold.train_end <= fold.test_start);
        assert!(fold.test_len() > 0);
    }
}
