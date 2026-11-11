//! Select features on a synthetic panel and compare against using all of them.
//!
//! cargo run --release --example forecast

use std::sync::Arc;

use xcm_mmm::data::Dataset;
use xcm_mmm::de::{minimize as minimize_de, DeConfig};
use xcm_mmm::forecast::{knee, SelectionProblem, SparseAccuracy, DEFAULT_LAMBDA};
use xcm_mmm::moead::{minimize as minimize_moead, MoeadConfig};

fn main() {
    let mut data = Dataset::synthetic(800, 12, 4, 7);
    data.standardize();
    let problem = SelectionProblem::new(Arc::new(data), 4, 1, DEFAULT_LAMBDA);
    let all: Vec<usize> = (0..problem.dataset.n_features()).collect();
    let baseline = problem.report(&all);

    let sparse = SparseAccuracy::new(problem.clone(), 0.01);
    let de = minimize_de(
        &sparse,
        &DeConfig {
            budget: 6_000,
            seed: 1,
            ..DeConfig::default()
        },
    );
    let selected = problem.decode(&de.best_x);
    let report = problem.report(&selected);

    let moead = minimize_moead(
        &problem,
        &MoeadConfig {
            population: 60,
            budget: 6_000,
            seed: 1,
            ..MoeadConfig::default()
        },
    );
    let front = moead.front();
    let pick = knee(&front).unwrap_or(0);
    let knee_report = problem.report(&problem.decode(&moead.solutions[pick].x));

    println!(
        "{:<8} {:>5} {:>10} {:>8} {:>7} {:>7}",
        "model", "feat", "mse", "r2", "ic", "skill"
    );
    for (label, r) in [("all", &baseline), ("de", &report), ("moead", &knee_report)] {
        println!(
            "{:<8} {:>5} {:>10.6} {:>8.4} {:>7.3} {:>6.1}%",
            label,
            r.columns.len(),
            r.mse,
            r.r2,
            r.ic,
            r.skill * 100.0
        );
    }
    println!("\nde keeps: {}", report.names.join(", "));
    println!("true signal: x1, x2, x3, x4");
    println!("front has {} points", front.len());
}
