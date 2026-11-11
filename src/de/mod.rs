//! Differential evolution for single objective, possibly constrained problems.
//!
//! Three layers live here. The operator layer in [`ops`] holds the mutations and
//! crossovers. The parameter memory in [`memory`] holds the SHADE success
//! history. This module runs the loop, and it offers three adaptations:
//!
//! * [`Adaptation::Shade`] is SHADE with the current-to-pbest mutation, a
//!   success-history memory, an external archive and, when asked, the linear
//!   population reduction of L-SHADE.
//! * [`Adaptation::Jde`] is self-adaptive DE, where every individual carries its
//!   own scale factor and crossover rate and mutates them with a small
//!   probability.
//! * [`Adaptation::Static`] is the classic Storn and Price algorithm, useful as
//!   a baseline.
//!
//! Only evaluation runs on the rayon pool. Mutation and selection stay on one
//! thread, so a run is reproducible at any thread count.

pub mod memory;
pub mod ops;

use crate::eval::evaluate_population;
use crate::problem::{Boundary, Problem, Score};
use crate::rng::Rng;

use memory::ParameterMemory;
use ops::{binomial, current_to_pbest, exponential, rand_1, repair};

/// Evaluations per dimension used when [`DeConfig::budget`] is zero, the CEC
/// convention for a fair comparison across dimensions.
pub const DEFAULT_BUDGET_PER_DIM: usize = 10_000;

/// Mutation operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// `v = x + F (x_pbest - x) + F (x_r1 - x_r2)`, the SHADE operator.
    CurrentToPBest1,
    /// `v = x_r1 + F (x_r2 - x_r3)`, the classic operator.
    Rand1Bin,
    /// `v = x_best + F (x_r1 - x_r2)`, greedy and quick to converge.
    Best1Bin,
}

/// How the scale factor and crossover rate are chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adaptation {
    /// Success history with an external archive.
    Shade,
    /// Per individual self adaptation with a mutation probability of 0.1.
    Jde,
    /// Fixed parameters from the configuration.
    Static,
}

/// Crossover operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossover {
    /// One forced coordinate plus independent draws.
    Binomial,
    /// A contiguous run of coordinates.
    Exponential,
}

/// Settings for one run.
#[derive(Clone, Debug)]
pub struct DeConfig {
    /// Population size. Zero picks `18 * dim`, clamped to 40 and 1000.
    pub population: usize,
    /// Evaluation budget. Zero picks `10_000 * dim`.
    pub budget: usize,
    /// Mutation operator, used unless the adaptation implies one.
    pub strategy: Strategy,
    /// Parameter adaptation scheme.
    pub adaptation: Adaptation,
    /// Crossover operator.
    pub crossover: Crossover,
    /// Fixed scale factor for [`Adaptation::Static`].
    pub f: f64,
    /// Fixed crossover rate for [`Adaptation::Static`].
    pub cr: f64,
    /// Fraction of the population that may serve as `x_pbest`.
    pub p_best: f64,
    /// Archive capacity as a multiple of the population size.
    pub archive_scale: f64,
    /// Number of success history slots.
    pub history: usize,
    /// Apply the L-SHADE linear population reduction.
    pub reduce_population: bool,
    /// Bound repair strategy.
    pub boundary: Boundary,
    /// Reinitialise the worst half after this many stalled generations. Zero
    /// disables the restart.
    pub stall_generations: usize,
    /// Seed of the run.
    pub seed: u64,
}

impl Default for DeConfig {
    fn default() -> Self {
        Self {
            population: 0,
            budget: 0,
            strategy: Strategy::CurrentToPBest1,
            adaptation: Adaptation::Shade,
            crossover: Crossover::Binomial,
            f: 0.5,
            cr: 0.9,
            p_best: 0.11,
            archive_scale: 1.0,
            history: memory::DEFAULT_HISTORY,
            reduce_population: true,
            boundary: Boundary::Midpoint,
            stall_generations: 0,
            seed: 1,
        }
    }
}

impl DeConfig {
    /// A configuration with an explicit seed.
    pub fn seeded(seed: u64) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    /// Population size after the automatic rule.
    pub fn resolved_population(&self, dim: usize) -> usize {
        if self.population > 0 {
            return self.population.max(4);
        }
        (18 * dim).clamp(40, 1000)
    }

    /// Evaluation budget after the automatic rule.
    pub fn resolved_budget(&self, dim: usize) -> usize {
        if self.budget > 0 {
            return self.budget.max(2);
        }
        DEFAULT_BUDGET_PER_DIM * dim
    }

    fn validate(&self, dim: usize) {
        assert!(self.f > 0.0 && self.f <= 1.0, "F must be in (0, 1]");
        assert!((0.0..=1.0).contains(&self.cr), "CR must be in [0, 1]");
        assert!(
            self.p_best > 0.0 && self.p_best <= 1.0,
            "p_best must be in (0, 1]"
        );
        assert!(self.history >= 1, "history must have at least one slot");
        assert!(dim > 0, "dimension must be positive");
        assert!(
            self.archive_scale >= 0.0,
            "archive scale must not be negative"
        );
    }
}

/// What one run produced.
#[derive(Clone, Debug)]
pub struct DeOutcome {
    /// Best decision vector found.
    pub best_x: Vec<f64>,
    /// Score of `best_x`.
    pub best: Score,
    /// Number of objective evaluations spent.
    pub evaluations: usize,
    /// Number of generations completed.
    pub generations: usize,
    /// Number of stagnation restarts performed.
    pub restarts: usize,
    /// Best objective after each generation, for convergence plots.
    pub history: Vec<f64>,
    /// Final population size.
    pub population: usize,
    /// Mean distance to the population centroid, relative to the box diagonal.
    pub diversity: f64,
}

impl DeOutcome {
    /// Best single objective value.
    pub fn best_objective(&self) -> f64 {
        self.best.objective()
    }
}

/// Run differential evolution on a single objective problem.
///
/// # Panics
///
/// Panics when the problem has more than one objective, because this engine
/// selects with a total order. Use [`crate::moead::minimize`] for a front.
pub fn minimize<P: Problem + ?Sized>(problem: &P, config: &DeConfig) -> DeOutcome {
    assert_eq!(
        problem.objectives(),
        1,
        "differential evolution needs a single objective; use moead::minimize instead"
    );
    let dim = problem.dim();
    config.validate(dim);

    let budget = config.resolved_budget(dim);
    let max_population = config.resolved_population(dim).min(budget.max(4));
    let min_population = 4.min(max_population);
    let mut rng = Rng::new(config.seed);
    let bounds = problem.bounds();

    let mut population: Vec<Vec<f64>> = (0..max_population)
        .map(|_| bounds.sample(&mut rng))
        .collect();
    let mut scores = evaluate_population(problem, &population);
    let mut evaluations = population.len();

    let mut memory = ParameterMemory::new(config.history);
    let mut archive: Vec<Vec<f64>> = Vec::new();
    let archive_cap = (config.archive_scale * max_population as f64).round() as usize;

    // jDE keeps a parameter pair per individual.
    let mut params: Vec<(f64, f64)> = vec![(config.f, config.cr); population.len()];

    let mut history = Vec::new();
    let mut generations = 0;
    let mut restarts = 0;
    let mut stall = 0usize;
    let mut best_seen_key = best_key(&scores);

    while evaluations < budget && !population.is_empty() {
        // The linear reduction is part of L-SHADE, so it only applies there.
        if config.reduce_population
            && config.adaptation == Adaptation::Shade
            && population.len() > min_population
        {
            let progress = evaluations as f64 / budget as f64;
            let target = (max_population as f64
                - (max_population - min_population) as f64 * progress)
                .round()
                .max(min_population as f64) as usize;
            if target < population.len() {
                shrink(&mut population, &mut scores, &mut params, target);
            }
        }

        let np = population.len();
        let batch = np.min(budget - evaluations);
        let ranked = ranked_indices(&scores);
        let pbest_pool = ((config.p_best * np as f64).round() as usize)
            .max(2)
            .min(np);

        // Build every trial first, then evaluate the batch in parallel. This
        // keeps all random draws on one thread and the order deterministic.
        let mut trials: Vec<Vec<f64>> = Vec::with_capacity(batch);
        let mut trial_params: Vec<(f64, f64)> = Vec::with_capacity(batch);
        let mut weights: Vec<f64> = Vec::with_capacity(batch);
        for i in 0..batch {
            let (f, cr) = match config.adaptation {
                Adaptation::Shade => {
                    let slot = rng.below(memory.len());
                    (
                        sample_f(memory.mean_f(slot), &mut rng),
                        sample_cr(memory.mean_cr(slot), &mut rng),
                    )
                }
                Adaptation::Jde => {
                    let (mut f, mut cr) = params[i];
                    if rng.chance(0.1) {
                        f = 0.1 + 0.9 * rng.next_f64();
                        cr = rng.next_f64();
                    }
                    (f, cr)
                }
                Adaptation::Static => (config.f, config.cr),
            };

            let parent = &population[i];
            let mutant = match config.adaptation {
                Adaptation::Shade => {
                    let pbest = ranked[rng.below(pbest_pool)];
                    let r1 = distinct(&mut rng, np, &[i]);
                    let r2 = donor_index(&mut rng, np, archive.len(), &[i, r1]);
                    let donor = donor(&population, &archive, r2);
                    current_to_pbest(parent, &population[pbest], &population[r1], donor, f)
                }
                _ => match config.strategy {
                    Strategy::Rand1Bin => {
                        let r1 = distinct(&mut rng, np, &[i]);
                        let r2 = distinct(&mut rng, np, &[i, r1]);
                        let r3 = distinct(&mut rng, np, &[i, r1, r2]);
                        rand_1(&population[r1], &population[r2], &population[r3], f)
                    }
                    Strategy::Best1Bin => {
                        let best = ranked[0];
                        let r1 = distinct(&mut rng, np, &[i, best]);
                        let r2 = distinct(&mut rng, np, &[i, best, r1]);
                        ops::best_1(&population[best], &population[r1], &population[r2], f)
                    }
                    Strategy::CurrentToPBest1 => {
                        let pbest = ranked[rng.below(pbest_pool)];
                        let r1 = distinct(&mut rng, np, &[i]);
                        let r2 = distinct(&mut rng, np, &[i, r1]);
                        current_to_pbest(
                            parent,
                            &population[pbest],
                            &population[r1],
                            &population[r2],
                            f,
                        )
                    }
                },
            };

            let mut trial = match config.crossover {
                Crossover::Binomial => binomial(parent, &mutant, cr, &mut rng),
                Crossover::Exponential => exponential(parent, &mutant, cr, &mut rng),
            };
            repair(&mut trial, parent, bounds, config.boundary);
            trials.push(trial);
            trial_params.push((f, cr));
            weights.push(0.0);
        }

        let trial_scores = evaluate_population(problem, &trials);
        evaluations += batch;
        generations += 1;

        // Greedy selection under Deb's feasibility rules.
        for i in 0..batch {
            if trial_scores[i].dominates(&scores[i]) {
                let gain = improvement(&scores[i], &trial_scores[i]);
                weights[i] = gain;
                if config.adaptation == Adaptation::Shade && archive_cap > 0 {
                    if archive.len() >= archive_cap {
                        let victim = rng.below(archive.len());
                        archive[victim] = std::mem::take(&mut population[i]);
                    } else {
                        archive.push(std::mem::take(&mut population[i]));
                    }
                }
                population[i] = std::mem::take(&mut trials[i]);
                scores[i] = trial_scores[i].clone();
                if config.adaptation == Adaptation::Jde {
                    params[i] = trial_params[i];
                }
            }
        }

        if config.adaptation == Adaptation::Shade {
            for i in 0..batch {
                if weights[i] > 0.0 {
                    memory.record(trial_params[i].0, trial_params[i].1, weights[i]);
                }
            }
            memory.commit();
        }

        let current_index = best_index(&scores);
        let current_key = best_key(&scores);
        history.push(scores[current_index].objective());

        if improves(current_key, best_seen_key) {
            best_seen_key = current_key;
            stall = 0;
        } else {
            stall += 1;
        }

        if config.stall_generations > 0 && stall >= config.stall_generations {
            restart_worst_half(problem, &mut population, &mut scores, &mut rng, &mut params);
            memory.reset();
            archive.clear();
            stall = 0;
            restarts += 1;
            best_seen_key = best_key(&scores);
        }
    }

    // Selection is greedy and the population reduction keeps the best ranks,
    // so the best point ever seen is still in the final population. Reading it
    // from the final scores also survives an index shift from shrinking.
    let idx = best_index(&scores);
    let diversity = diversity(&population, bounds);
    DeOutcome {
        best_x: population[idx].clone(),
        best: scores[idx].clone(),
        evaluations,
        generations,
        restarts,
        history,
        population: population.len(),
        diversity,
    }
}

fn sample_f(mean: f64, rng: &mut Rng) -> f64 {
    // SHADE draws F from a Cauchy centred on the memory slot. Values at or
    // below zero are redrawn, values above one are capped, which is exactly the
    // published rule.
    for _ in 0..32 {
        let f = rng.cauchy(mean, 0.1);
        if f > 0.0 {
            return f.min(1.0);
        }
    }
    mean
}

fn sample_cr(mean: f64, rng: &mut Rng) -> f64 {
    rng.normal(mean, 0.1).clamp(0.0, 1.0)
}

/// A random index below `n` that is not in `exclude`.
fn distinct(rng: &mut Rng, n: usize, exclude: &[usize]) -> usize {
    debug_assert!(n > exclude.len(), "not enough individuals to choose from");
    loop {
        let candidate = rng.below(n);
        if !exclude.contains(&candidate) {
            return candidate;
        }
    }
}

/// Pick a donor from the union of the population and the archive.
fn donor_index(rng: &mut Rng, population: usize, archive: usize, exclude: &[usize]) -> usize {
    let total = population + archive;
    loop {
        let candidate = rng.below(total);
        if !exclude.contains(&candidate) {
            return candidate;
        }
    }
}

fn donor<'a>(population: &'a [Vec<f64>], archive: &'a [Vec<f64>], index: usize) -> &'a [f64] {
    if index < population.len() {
        &population[index]
    } else {
        &archive[index - population.len()]
    }
}

/// Ranking key for a single objective score: feasible points first, then by
/// objective, and infeasible points by violation.
fn key(score: &Score) -> (u8, f64) {
    if score.is_feasible() {
        (0, score.objective())
    } else {
        (1, score.violation())
    }
}

fn best_key(scores: &[Score]) -> (u8, f64) {
    scores
        .iter()
        .map(key)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or((1, f64::INFINITY))
}

fn best_index(scores: &[Score]) -> usize {
    let mut best = 0;
    for i in 1..scores.len() {
        if key(&scores[i]) < key(&scores[best]) {
            best = i;
        }
    }
    best
}

fn ranked_indices(scores: &[Score]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| key(&scores[a]).partial_cmp(&key(&scores[b])).unwrap());
    order
}

fn improves(current: (u8, f64), previous: (u8, f64)) -> bool {
    if current.0 != previous.0 {
        return current.0 < previous.0;
    }
    // Any strict change counts as progress for the stall counter, even a tiny
    // one, so that a slow but real descent is not treated as stagnation.
    current.1 < previous.1 - 1e-14
}

/// Weight of a successful trial for the SHADE memory.
///
/// With constraints in play the violation reduction matters more than the
/// objective change, so it supplies the weight whenever either point is
/// infeasible.
fn improvement(old: &Score, new: &Score) -> f64 {
    if old.violation() > 0.0 || new.violation() > 0.0 {
        (old.violation() - new.violation()).max(0.0)
    } else {
        (old.objective() - new.objective()).abs()
    }
}

fn shrink(
    population: &mut Vec<Vec<f64>>,
    scores: &mut Vec<Score>,
    params: &mut Vec<(f64, f64)>,
    target: usize,
) {
    let mut order = ranked_indices(scores);
    order.truncate(target);
    order.sort_unstable();
    let mut kept_population = Vec::with_capacity(target);
    let mut kept_scores = Vec::with_capacity(target);
    let mut kept_params = Vec::with_capacity(target);
    for i in order {
        kept_population.push(population[i].clone());
        kept_scores.push(scores[i].clone());
        kept_params.push(params[i]);
    }
    *population = kept_population;
    *scores = kept_scores;
    *params = kept_params;
}

fn restart_worst_half<P: Problem + ?Sized>(
    problem: &P,
    population: &mut [Vec<f64>],
    scores: &mut [Score],
    rng: &mut Rng,
    params: &mut [(f64, f64)],
) {
    let mut survivors = ranked_indices(scores);
    let keep = (population.len() / 2).max(2).min(population.len());
    survivors.truncate(keep);
    survivors.sort_unstable();
    // The better half stays, the rest is drawn again, so a restart really does
    // explore new ground instead of reshuffling the same points.
    let mut slot = 0usize;
    for i in 0..population.len() {
        if slot < survivors.len() && survivors[slot] == i {
            slot += 1;
            continue;
        }
        population[i] = problem.bounds().sample(rng);
        scores[i] = problem.score(&population[i]);
        params[i] = (0.5, 0.9);
    }
}

fn diversity(population: &[Vec<f64>], bounds: &crate::problem::Bounds) -> f64 {
    if population.len() < 2 {
        return 0.0;
    }
    let dim = bounds.dim();
    let mut centroid = vec![0.0; dim];
    for x in population {
        for (c, v) in centroid.iter_mut().zip(x) {
            *c += v;
        }
    }
    for c in centroid.iter_mut() {
        *c /= population.len() as f64;
    }
    let mut total = 0.0;
    for x in population {
        let mut d2 = 0.0;
        for i in 0..dim {
            let span = bounds.span(i);
            let d = if span > 0.0 {
                (x[i] - centroid[i]) / span
            } else {
                0.0
            };
            d2 += d * d;
        }
        total += d2.sqrt();
    }
    total / population.len() as f64
}
