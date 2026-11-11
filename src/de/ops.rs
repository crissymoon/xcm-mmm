//! Mutation, crossover and repair operators shared by DE and MOEA/D.

use crate::problem::{Boundary, Bounds};
use crate::rng::Rng;

/// `current-to-pbest/1`: the SHADE and jSO mutation.
///
/// `v = x + F (x_pbest - x) + F (x_r1 - x_r2)`
pub fn current_to_pbest(x: &[f64], pbest: &[f64], r1: &[f64], r2: &[f64], f: f64) -> Vec<f64> {
    x.iter()
        .zip(pbest)
        .zip(r1)
        .zip(r2)
        .map(|(((xi, p), a), b)| xi + f * (p - xi) + f * (a - b))
        .collect()
}

/// `rand/1`: the classic mutation, good on separable and rotated alike.
///
/// `v = x_r1 + F (x_r2 - x_r3)`
pub fn rand_1(r1: &[f64], r2: &[f64], r3: &[f64], f: f64) -> Vec<f64> {
    r1.iter()
        .zip(r2)
        .zip(r3)
        .map(|((a, b), c)| a + f * (b - c))
        .collect()
}

/// `best/1`: greedy and fast, but it loses diversity early.
///
/// `v = x_best + F (x_r1 - x_r2)`
pub fn best_1(best: &[f64], r1: &[f64], r2: &[f64], f: f64) -> Vec<f64> {
    best.iter()
        .zip(r1)
        .zip(r2)
        .map(|((b, a), c)| b + f * (a - c))
        .collect()
}

/// Binomial crossover, with one coordinate taken from the mutant by force.
pub fn binomial(parent: &[f64], mutant: &[f64], cr: f64, rng: &mut Rng) -> Vec<f64> {
    let dim = parent.len();
    let forced = rng.below(dim);
    let mut trial = Vec::with_capacity(dim);
    for j in 0..dim {
        if j == forced || rng.next_f64() < cr {
            trial.push(mutant[j]);
        } else {
            trial.push(parent[j]);
        }
    }
    trial
}

/// Exponential crossover, the original Storn and Price scheme.
///
/// A run of coordinates starting at a random index comes from the mutant; the
/// run grows until a draw fails the rate. Kept for the classic strategy and for
/// reproducing older results.
pub fn exponential(parent: &[f64], mutant: &[f64], cr: f64, rng: &mut Rng) -> Vec<f64> {
    let dim = parent.len();
    let start = rng.below(dim);
    let mut trial = parent.to_vec();
    let mut j = start;
    loop {
        trial[j] = mutant[j];
        j = (j + 1) % dim;
        if rng.next_f64() >= cr || j == start {
            break;
        }
    }
    trial
}

/// Bring a trial vector back inside the box, coordinate by coordinate.
///
/// The parent is needed by the midpoint strategy, which halves an offending
/// step rather than clipping it.
pub fn repair(trial: &mut [f64], parent: &[f64], bounds: &Bounds, strategy: Boundary) {
    for i in 0..trial.len() {
        trial[i] = bounds.repair(i, trial[i], parent[i], strategy);
    }
}

/// Polynomial mutation, the bounded variant used by MOEA/D-DE.
///
/// `eta` controls how far a child can move: 20 is the usual value.
pub fn polynomial_mutation(x: &mut [f64], bounds: &Bounds, eta: f64, rng: &mut Rng) {
    let dim = x.len();
    let rate = 1.0 / dim as f64;
    for (i, xi) in x.iter_mut().enumerate() {
        if rng.next_f64() >= rate {
            continue;
        }
        let lo = bounds.lower(i);
        let hi = bounds.upper(i);
        let span = hi - lo;
        if span <= 0.0 {
            continue;
        }
        let delta1 = (*xi - lo) / span;
        let delta2 = (hi - *xi) / span;
        let u = rng.next_f64();
        let power = 1.0 / (eta + 1.0);
        let delta = if u < 0.5 {
            let value = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(eta + 1.0);
            value.powf(power) - 1.0
        } else {
            let value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(eta + 1.0);
            1.0 - value.powf(power)
        };
        *xi = (*xi + delta * span).clamp(lo, hi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutations_follow_their_formulas() {
        let x = vec![1.0, 1.0];
        let p = vec![0.5, 0.5];
        let a = vec![0.25, 0.75];
        let b = vec![0.0, 0.5];
        let v = current_to_pbest(&x, &p, &a, &b, 0.5);
        assert!((v[0] - (1.0 + 0.5 * (-0.5) + 0.5 * 0.25)).abs() < 1e-12);
        let r = rand_1(&a, &b, &x, 0.5);
        assert!((r[0] - (0.25 + -0.5)).abs() < 1e-12);
        let g = best_1(&p, &x, &b, 1.0);
        assert!((g[0] - (0.5 + 1.0 * 1.0)).abs() < 1e-12);
    }

    #[test]
    fn binomial_always_takes_one_mutant_gene() {
        let parent = vec![0.0; 20];
        let mutant = vec![1.0; 20];
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            let child = binomial(&parent, &mutant, 0.0, &mut rng);
            assert_eq!(child.iter().filter(|&&v| v == 1.0).count(), 1);
        }
    }

    #[test]
    fn exponential_runs_are_contiguous() {
        let parent = vec![0.0; 30];
        let mutant = vec![1.0; 30];
        let mut rng = Rng::new(4);
        let child = exponential(&parent, &mutant, 0.5, &mut rng);
        let hits: Vec<usize> = child
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == 1.0)
            .map(|(i, _)| i)
            .collect();
        assert!(!hits.is_empty());
    }

    #[test]
    fn repair_respects_every_strategy() {
        let bounds = Bounds::unit(3);
        let parent = vec![0.5, 0.5, 0.5];
        for strategy in [Boundary::Clip, Boundary::Midpoint, Boundary::Reflect] {
            let mut trial = vec![-2.0, 0.5, 3.0];
            repair(&mut trial, &parent, &bounds, strategy);
            assert!(bounds.contains(&trial), "{strategy:?} gave {trial:?}");
        }
    }

    #[test]
    fn mutation_stays_inside_the_box() {
        let bounds = Bounds::new(vec![-1.0, 0.0], vec![1.0, 2.0]);
        let mut rng = Rng::new(9);
        for _ in 0..200 {
            let mut x = vec![0.0, 1.0];
            polynomial_mutation(&mut x, &bounds, 20.0, &mut rng);
            assert!(bounds.contains(&x));
        }
    }
}
