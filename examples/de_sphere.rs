//! Solve a single objective benchmark and print the convergence tail.
//!
//! cargo run --release --example de_sphere

use xcm_mmm::benchmarks::Benchmark;
use xcm_mmm::de::{minimize, DeConfig};

fn main() {
    let problem = Benchmark::named("sphere", 30).unwrap();
    let config = DeConfig {
        budget: 100_000,
        seed: 7,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &config);
    println!("best       {:.6e}", out.best_objective());
    println!("evaluations {}", out.evaluations);
    println!("generations {}", out.generations);
    println!("population  {}", out.population);
    println!("diversity   {:.4}", out.diversity);
    let tail: Vec<String> = out
        .history
        .iter()
        .rev()
        .take(5)
        .rev()
        .map(|v| format!("{v:.3e}"))
        .collect();
    println!("last five generations: {}", tail.join(", "));
}
