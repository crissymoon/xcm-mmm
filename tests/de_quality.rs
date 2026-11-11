//! Convergence, determinism and constraint handling, checked end to end.

use rayon::ThreadPoolBuilder;

use xcm_mmm::benchmarks::Benchmark;
use xcm_mmm::de::{minimize, Adaptation, DeConfig, Strategy};
use xcm_mmm::problem::{Bounds, Problem};

fn sphere(dim: usize) -> Benchmark {
    Benchmark::named("sphere", dim).unwrap()
}

#[test]
fn shade_solves_sphere_to_machine_precision() {
    let problem = sphere(10);
    let cfg = DeConfig {
        budget: 40_000,
        seed: 7,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &cfg);
    assert!(
        out.best_objective() < 1e-10,
        "best {:.3e}",
        out.best_objective()
    );
    assert!(out.best.is_feasible());
}

#[test]
fn shade_follows_the_rosenbrock_valley() {
    let problem = Benchmark::named("rosenbrock", 8).unwrap();
    let cfg = DeConfig {
        budget: 120_000,
        seed: 11,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &cfg);
    assert!(
        out.best_objective() < 1e-2,
        "best {:.3e}",
        out.best_objective()
    );
}

#[test]
fn shade_escapes_the_rastrigin_lattice() {
    let problem = Benchmark::named("rastrigin", 10).unwrap();
    let cfg = DeConfig {
        budget: 120_000,
        seed: 13,
        stall_generations: 40,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &cfg);
    assert!(
        out.best_objective() < 1.0,
        "best {:.3e}",
        out.best_objective()
    );
}

#[test]
fn every_adaptation_reduces_a_simple_problem() {
    let problem = sphere(6);
    for adaptation in [Adaptation::Shade, Adaptation::Jde, Adaptation::Static] {
        let cfg = DeConfig {
            adaptation,
            budget: 20_000,
            seed: 3,
            ..DeConfig::default()
        };
        let out = minimize(&problem, &cfg);
        assert!(
            out.best_objective() < 1e-3,
            "{adaptation:?} stalled at {:.3e}",
            out.best_objective()
        );
    }
}

#[test]
fn classic_strategies_also_converge() {
    let problem = sphere(6);
    for strategy in [
        Strategy::Rand1Bin,
        Strategy::Best1Bin,
        Strategy::CurrentToPBest1,
    ] {
        let cfg = DeConfig {
            adaptation: Adaptation::Static,
            strategy,
            f: 0.5,
            cr: 0.9,
            budget: 30_000,
            seed: 5,
            ..DeConfig::default()
        };
        let out = minimize(&problem, &cfg);
        assert!(
            out.best_objective() < 1e-5,
            "{strategy:?} stalled at {:.3e}",
            out.best_objective()
        );
    }
}

#[test]
fn the_same_seed_gives_the_same_run() {
    let problem = sphere(8);
    let cfg = DeConfig {
        budget: 5_000,
        seed: 99,
        ..DeConfig::default()
    };
    let a = minimize(&problem, &cfg);
    let b = minimize(&problem, &cfg);
    assert_eq!(a.best_objective(), b.best_objective());
    assert_eq!(a.best_x, b.best_x);
    assert_eq!(a.history, b.history);
}

#[test]
fn thread_count_does_not_change_the_answer() {
    let problem = sphere(9);
    let cfg = DeConfig {
        budget: 12_000,
        seed: 21,
        ..DeConfig::default()
    };
    let serial = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| minimize(&problem, &cfg));
    let parallel = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| minimize(&problem, &cfg));
    assert_eq!(serial.best_objective(), parallel.best_objective());
    assert_eq!(serial.best_x, parallel.best_x);
    assert_eq!(serial.history, parallel.history);
}

#[test]
fn population_reduction_can_be_switched_off() {
    let problem = sphere(5);
    let cfg = DeConfig {
        budget: 6_000,
        reduce_population: false,
        seed: 2,
        ..DeConfig::default()
    };
    // A fixed population of 90 is larger than the automatic 18 per dimension,
    // so the outcome must report exactly what the caller asked for.
    let cfg = DeConfig {
        population: 90,
        ..cfg
    };
    let out = minimize(&problem, &cfg);
    assert_eq!(out.population, cfg.resolved_population(5));
}

/// Minimise `sum x^2` subject to `x0 >= 1`, whose optimum is at `x = (1, 0, ...)`
/// with value one.
struct Constrained {
    bounds: Bounds,
}

impl Problem for Constrained {
    fn name(&self) -> &str {
        "constrained-sphere"
    }

    fn dim(&self) -> usize {
        2
    }

    fn constraints(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], g: &mut [f64]) {
        f[0] = x.iter().map(|v| v * v).sum();
        // Satisfied when x0 is at least one, written in the g <= 0 form.
        g[0] = 1.0 - x[0];
    }
}

#[test]
fn constraints_are_respected_before_objective_quality() {
    let problem = Constrained {
        bounds: Bounds::symmetric(2, -5.0, 5.0),
    };
    let cfg = DeConfig {
        budget: 25_000,
        seed: 17,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &cfg);
    assert!(out.best.is_feasible(), "violation {}", out.best.violation());
    assert!(out.best_x[0] >= 1.0 - 1e-6, "x0 {}", out.best_x[0]);
    assert!(
        (out.best_objective() - 1.0).abs() < 1e-4,
        "best {:.6}",
        out.best_objective()
    );
}

#[test]
fn a_large_population_is_capped_by_the_budget() {
    let problem = sphere(4);
    let cfg = DeConfig {
        population: 10_000,
        budget: 500,
        seed: 1,
        ..DeConfig::default()
    };
    let out = minimize(&problem, &cfg);
    assert!(out.evaluations <= 1_000, "spent {}", out.evaluations);
    assert!(out.best_objective().is_finite());
}
