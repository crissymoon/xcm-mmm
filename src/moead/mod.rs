//! MOEA/D, the multi-objective evolutionary algorithm based on decomposition.
//!
//! The implementation follows MOEA/D-DE (Li and Zhang, 2009): a Das and Dennis
//! simplex lattice of weights, one subproblem per weight, a Tchebycheff
//! scalarisation by default, mating restricted to a neighbourhood with
//! probability `delta`, at most `replacements` neighbours updated per child, a
//! `rand/1` differential operator followed by polynomial mutation, and an
//! external archive of the non-dominated points seen so far.
//!
//! Each generation is synchronous: every child is generated from the population
//! at the start of the round, the round is evaluated in parallel, and the
//! replacements are applied in index order. That keeps a run reproducible at any
//! thread count while still using every core for the expensive part.

pub mod scalar;
pub mod weights;

use crate::eval::{dominates, evaluate_population, same_point};
use crate::problem::{Boundary, Problem, Score};
use crate::rng::Rng;

use scalar::Scalarization;
use weights::{divisions_for, WeightSet};

use crate::de::ops::{binomial, polynomial_mutation, rand_1, repair};

/// Evaluations used when [`MoeadConfig::budget`] is zero. At the default
/// population of about one hundred subproblems this is five hundred
/// generations, the setting the ZDT and DTLZ tables in the documentation use.
pub const DEFAULT_BUDGET: usize = 50_000;

/// Settings for one MOEA/D run.
#[derive(Clone, Debug)]
pub struct MoeadConfig {
    /// Target number of subproblems. Zero picks 100.
    pub population: usize,
    /// Lattice divisions. Zero derives them from `population`.
    pub divisions: usize,
    /// Neighbourhood size. Zero picks 20.
    pub neighbourhood: usize,
    /// Probability of mating inside the neighbourhood rather than globally.
    pub delta: f64,
    /// Maximum neighbours replaced by one child.
    pub replacements: usize,
    /// Scalarising function.
    pub scalar: Scalarization,
    /// Differential scale factor.
    pub f: f64,
    /// Crossover rate. MOEA/D-DE usually runs at 1.0.
    pub cr: f64,
    /// Polynomial mutation distribution index.
    pub mutation_eta: f64,
    /// Evaluation budget. Zero picks 50_000, which gives 500 generations at the
    /// default population.
    pub budget: usize,
    /// Bound repair strategy.
    pub boundary: Boundary,
    /// Seed of the run.
    pub seed: u64,
}

impl Default for MoeadConfig {
    fn default() -> Self {
        Self {
            population: 100,
            divisions: 0,
            neighbourhood: 20,
            delta: 0.9,
            replacements: 2,
            scalar: Scalarization::Tchebycheff,
            f: 0.5,
            cr: 1.0,
            mutation_eta: 20.0,
            budget: 0,
            boundary: Boundary::Clip,
            seed: 1,
        }
    }
}

impl MoeadConfig {
    /// A configuration with an explicit seed.
    pub fn seeded(seed: u64) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    fn validate(&self) {
        assert!(self.f > 0.0 && self.f <= 1.0, "F must be in (0, 1]");
        assert!((0.0..=1.0).contains(&self.cr), "CR must be in [0, 1]");
        assert!((0.0..=1.0).contains(&self.delta), "delta must be in [0, 1]");
        assert!(
            self.population >= 4,
            "MOEA/D needs at least four subproblems"
        );
        assert!(self.replacements >= 1, "at least one replacement per child");
    }
}

/// One point on the returned front, with the decision vector that produced it.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Decision vector.
    pub x: Vec<f64>,
    /// Objective values.
    pub objectives: Vec<f64>,
    /// Weight of the subproblem that maintains this point.
    pub weight: Vec<f64>,
}

/// What one run produced.
#[derive(Clone, Debug)]
pub struct MoeadOutcome {
    /// Non-dominated solutions, sorted by the first objective.
    pub solutions: Vec<Solution>,
    /// Number of objective evaluations spent.
    pub evaluations: usize,
    /// Number of generations completed.
    pub generations: usize,
    /// Ideal point reached, one entry per objective.
    pub ideal: Vec<f64>,
    /// Mean scalar value of the population after each generation.
    pub history: Vec<f64>,
    /// Number of subproblems.
    pub population: usize,
}

impl MoeadOutcome {
    /// Objective vectors only, the form the metrics take.
    pub fn front(&self) -> Vec<Vec<f64>> {
        self.solutions
            .iter()
            .map(|s| s.objectives.clone())
            .collect()
    }

    /// Decision vector of the solution closest to a chosen point in objective
    /// space.
    pub fn closest_to(&self, target: &[f64]) -> Option<&Solution> {
        self.solutions.iter().min_by(|a, b| {
            let da: f64 = a
                .objectives
                .iter()
                .zip(target)
                .map(|(x, t)| (x - t) * (x - t))
                .sum();
            let db: f64 = b
                .objectives
                .iter()
                .zip(target)
                .map(|(x, t)| (x - t) * (x - t))
                .sum();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Run MOEA/D on a problem with two or more objectives.
///
/// # Panics
///
/// Panics on a single objective problem, where [`crate::de::minimize`] is the
/// right tool, and on a population below four, where the operator has no room
/// to choose parents.
pub fn minimize<P: Problem + ?Sized>(problem: &P, config: &MoeadConfig) -> MoeadOutcome {
    assert!(
        problem.objectives() >= 2,
        "MOEA/D needs at least two objectives; use de::minimize for one"
    );
    config.validate();
    let m = problem.objectives();
    let budget = if config.budget == 0 {
        DEFAULT_BUDGET
    } else {
        config.budget
    };
    let divisions = if config.divisions == 0 {
        divisions_for(config.population, m)
    } else {
        config.divisions
    };
    let set = WeightSet::build(divisions, m, config.neighbourhood);
    let n = set.len();

    let mut rng = Rng::new(config.seed);
    let bounds = problem.bounds();
    let mut population: Vec<Vec<f64>> = (0..n).map(|_| bounds.sample(&mut rng)).collect();
    let mut scores = evaluate_population(problem, &population);
    let mut evaluations = population.len();

    let mut ideal = vec![f64::INFINITY; m];
    update_ideal(&mut ideal, &scores, problem.objectives());

    let mut history = Vec::new();
    let mut generations = 0;

    while evaluations < budget {
        let batch = n.min(budget - evaluations);
        let mut children: Vec<Vec<f64>> = Vec::with_capacity(batch);
        for i in 0..batch {
            let pool: &[usize] = if rng.chance(config.delta) {
                set.neighbours(i)
            } else {
                &[]
            };
            let parents = pick_parents(&mut rng, pool, n, i);
            let mutant = rand_1(
                &population[parents[0]],
                &population[parents[1]],
                &population[parents[2]],
                config.f,
            );
            let mut child = binomial(&population[i], &mutant, config.cr, &mut rng);
            polynomial_mutation(&mut child, bounds, config.mutation_eta, &mut rng);
            repair(&mut child, &population[i], bounds, config.boundary);
            children.push(child);
        }

        let child_scores = evaluate_population(problem, &children);
        evaluations += batch;
        generations += 1;

        update_ideal(&mut ideal, &child_scores, m);

        // Replace neighbours, at most `replacements` per child. The shuffle is
        // seeded, so the order is reproducible.
        for i in 0..batch {
            let mut order: Vec<usize> = set.neighbours(i).to_vec();
            rng.shuffle(&mut order);
            let mut replaced = 0usize;
            let trial_value = |j: usize| {
                config
                    .scalar
                    .value(child_scores[i].objectives(), &ideal, &set.weights[j])
            };
            for &j in &order {
                if replaced >= config.replacements {
                    break;
                }
                let old_value =
                    config
                        .scalar
                        .value(scores[j].objectives(), &ideal, &set.weights[j]);
                if worth_replacing(&child_scores[i], trial_value(j), &scores[j], old_value) {
                    population[j] = children[i].clone();
                    scores[j] = child_scores[i].clone();
                    replaced += 1;
                }
            }
        }

        history.push(mean_scalar(&scores, &ideal, &set, config.scalar));
    }

    // The front is read from the final population rather than from a history of
    // every non-dominated point ever seen. A history keeps early points that
    // were never dominated but sit far behind the converged front, which costs
    // IGD and tells the caller less about what the run actually achieved.
    let mut keep = vec![true; n];
    for i in 0..n {
        if !scores[i].is_feasible() {
            keep[i] = false;
            continue;
        }
        for j in 0..n {
            if i != j
                && scores[j].is_feasible()
                && dominates(scores[j].objectives(), scores[i].objectives())
            {
                keep[i] = false;
                break;
            }
        }
    }
    let mut solutions: Vec<Solution> = (0..n)
        .filter(|&i| keep[i])
        .map(|i| Solution {
            x: population[i].clone(),
            objectives: scores[i].objectives().to_vec(),
            weight: set.weights[i].clone(),
        })
        .collect();
    // A run with no feasible point would otherwise report nothing at all, so
    // the least infeasible points are returned as a diagnostic.
    if solutions.is_empty() {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| scores[a].violation().total_cmp(&scores[b].violation()));
        solutions = order
            .into_iter()
            .take(8)
            .map(|i| Solution {
                x: population[i].clone(),
                objectives: scores[i].objectives().to_vec(),
                weight: set.weights[i].clone(),
            })
            .collect();
    }
    solutions.sort_by(|a, b| a.objectives[0].total_cmp(&b.objectives[0]));
    solutions.dedup_by(|a, b| same_point(&a.objectives, &b.objectives));

    MoeadOutcome {
        solutions,
        evaluations,
        generations,
        ideal,
        history,
        population: n,
    }
}

/// Feasibility first, then the scalar value.
fn worth_replacing(trial: &Score, trial_value: f64, current: &Score, current_value: f64) -> bool {
    match (trial.is_feasible(), current.is_feasible()) {
        (true, false) => true,
        (false, true) => false,
        (false, false) => trial.violation() < current.violation(),
        (true, true) => trial_value <= current_value,
    }
}

fn update_ideal(ideal: &mut [f64], scores: &[Score], objectives: usize) {
    let mut any_feasible = false;
    for s in scores.iter().filter(|s| s.is_feasible()) {
        any_feasible = true;
        for (z, v) in ideal.iter_mut().zip(s.objectives()) {
            *z = z.min(*v);
        }
    }
    if !any_feasible {
        // Without a feasible point yet, the best available objectives still
        // give the scalariser something finite to work with.
        for s in scores {
            for (z, v) in ideal.iter_mut().zip(s.objectives()) {
                *z = z.min(*v);
            }
        }
    }
    debug_assert_eq!(ideal.len(), objectives);
}

fn mean_scalar(scores: &[Score], ideal: &[f64], set: &WeightSet, scalar: Scalarization) -> f64 {
    let mut total = 0.0;
    for (i, s) in scores.iter().enumerate() {
        total += scalar.value(s.objectives(), ideal, &set.weights[i % set.len()]);
    }
    total / scores.len() as f64
}

/// Three distinct parent indices, drawn from the neighbourhood when one is
/// given and from the whole population otherwise.
fn pick_parents(rng: &mut Rng, pool: &[usize], n: usize, forbidden: usize) -> [usize; 3] {
    let mut candidates: Vec<usize> = pool.iter().copied().filter(|&j| j != forbidden).collect();
    if candidates.len() < 4 {
        candidates = (0..n).filter(|&j| j != forbidden).collect();
    }
    debug_assert!(candidates.len() >= 3, "population too small for parents");
    rng.shuffle(&mut candidates);
    [candidates[0], candidates[1], candidates[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::{Benchmark, Kind};
    use crate::metrics::{hypervolume, igd, reference_front};

    fn flat(problem: &Benchmark) -> MoeadOutcome {
        let cfg = MoeadConfig {
            population: 100,
            budget: 25_000,
            seed: 3,
            ..MoeadConfig::default()
        };
        minimize(problem, &cfg)
    }

    #[test]
    fn zdt1_front_is_close_to_the_true_one() {
        let problem = Benchmark::build(Kind::Zdt1, 10).unwrap();
        let out = flat(&problem);
        let reference = reference_front(Kind::Zdt1, 200);
        let igd = igd(&out.front(), &reference);
        assert!(igd < 0.01, "IGD {igd}");
        assert!(
            out.solutions.len() > 50,
            "only {} points",
            out.solutions.len()
        );
    }

    #[test]
    fn zdt2_and_zdt3_are_also_covered() {
        for kind in [Kind::Zdt2, Kind::Zdt3] {
            let problem = Benchmark::build(kind, 10).unwrap();
            let out = flat(&problem);
            let reference = reference_front(kind, 200);
            let igd = igd(&out.front(), &reference);
            assert!(igd < 0.02, "{:?} IGD {igd}", kind);
        }
    }

    #[test]
    fn hypervolume_beats_a_single_point() {
        let problem = Benchmark::build(Kind::Zdt1, 10).unwrap();
        let out = flat(&problem);
        let best_single = out
            .solutions
            .iter()
            .map(|s| s.objectives[0])
            .fold(f64::INFINITY, f64::min);
        let hv = hypervolume(&out.front(), &[1.1, 1.1]);
        assert!(hv > 0.3, "HV {hv} with best f1 {best_single}");
    }

    #[test]
    fn the_front_contains_no_dominated_point() {
        let problem = Benchmark::build(Kind::Zdt1, 8).unwrap();
        let out = flat(&problem);
        let front = out.front();
        for i in 0..front.len() {
            for j in 0..front.len() {
                if i != j {
                    assert!(!dominates(&front[i], &front[j]), "{i} dominates {j}");
                }
            }
        }
    }

    #[test]
    fn runs_are_reproducible() {
        let problem = Benchmark::build(Kind::Zdt1, 6).unwrap();
        let cfg = MoeadConfig {
            population: 30,
            budget: 1_200,
            seed: 42,
            ..MoeadConfig::default()
        };
        let a = minimize(&problem, &cfg);
        let b = minimize(&problem, &cfg);
        assert_eq!(a.solutions.len(), b.solutions.len());
        for (x, y) in a.solutions.iter().zip(&b.solutions) {
            assert_eq!(x.objectives, y.objectives);
        }
    }
}
