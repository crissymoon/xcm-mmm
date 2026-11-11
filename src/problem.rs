//! The problem contract: bounds, objectives, constraints and the score type.

use crate::rng::Rng;

/// Box bounds for a continuous search space.
#[derive(Clone, Debug, PartialEq)]
pub struct Bounds {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Bounds {
    /// Build bounds from matching lower and upper vectors.
    ///
    /// Panics on a length mismatch or when any lower bound exceeds its upper
    /// bound, because both are programming errors rather than data problems.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(lower.len(), upper.len(), "bounds length mismatch");
        assert!(!lower.is_empty(), "bounds must not be empty");
        for (i, (lo, hi)) in lower.iter().zip(&upper).enumerate() {
            assert!(lo.is_finite() && hi.is_finite(), "bound {i} is not finite");
            assert!(lo < hi, "bound {i} is empty: {lo} >= {hi}");
        }
        Self { lower, upper }
    }

    /// The same interval on every coordinate.
    pub fn symmetric(dim: usize, lo: f64, hi: f64) -> Self {
        Self::new(vec![lo; dim], vec![hi; dim])
    }

    /// The unit hypercube.
    pub fn unit(dim: usize) -> Self {
        Self::symmetric(dim, 0.0, 1.0)
    }

    /// Number of coordinates.
    pub fn dim(&self) -> usize {
        self.lower.len()
    }

    /// Lower bound on coordinate `i`.
    pub fn lower(&self, i: usize) -> f64 {
        self.lower[i]
    }

    /// Upper bound on coordinate `i`.
    pub fn upper(&self, i: usize) -> f64 {
        self.upper[i]
    }

    /// Interval width on coordinate `i`.
    pub fn span(&self, i: usize) -> f64 {
        self.upper[i] - self.lower[i]
    }

    /// Midpoint on coordinate `i`.
    pub fn mid(&self, i: usize) -> f64 {
        0.5 * (self.lower[i] + self.upper[i])
    }

    /// True when the point lies inside every bound.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dim()
            && x.iter()
                .zip(&self.lower)
                .zip(&self.upper)
                .all(|((v, lo), hi)| v.is_finite() && *v >= *lo && *v <= *hi)
    }

    /// Clamp a point into the box.
    pub fn clamp(&self, x: &mut [f64]) {
        for ((v, lo), hi) in x.iter_mut().zip(&self.lower).zip(&self.upper) {
            *v = v.clamp(*lo, *hi);
        }
    }

    /// A uniformly random point in the box.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        (0..self.dim())
            .map(|i| rng.uniform(self.lower[i], self.upper[i]))
            .collect()
    }

    /// Repair one coordinate that left the box.
    ///
    /// Every strategy is deterministic given its inputs, so a run stays
    /// reproducible. The midpoint strategy needs the parent value, which is
    /// what makes it the gentlest of the three.
    pub fn repair(&self, i: usize, value: f64, parent: f64, strategy: Boundary) -> f64 {
        let lo = self.lower[i];
        let hi = self.upper[i];
        if value >= lo && value <= hi {
            return value;
        }
        match strategy {
            Boundary::Clip => value.clamp(lo, hi),
            Boundary::Midpoint => {
                let v = 0.5 * (parent + lo);
                if value < lo {
                    v
                } else {
                    0.5 * (parent + hi)
                }
            }
            Boundary::Reflect => {
                let span = hi - lo;
                if !value.is_finite() {
                    return self.mid(i);
                }
                // Fold the value back into the interval by reflection.
                let shifted = (value - lo).rem_euclid(2.0 * span);
                if shifted <= span {
                    lo + shifted
                } else {
                    hi - (shifted - span)
                }
            }
        }
    }
}

/// How a candidate that leaves the box is brought back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Move it onto the nearest face. Cheap and common, but it piles points on
    /// the edge.
    Clip,
    /// Halve the step towards the parent, the jSO and L-SHADE choice.
    Midpoint,
    /// Mirror it back inside, which keeps the step size and the diversity.
    Reflect,
}

/// A continuous, possibly multi-objective, possibly constrained problem.
///
/// Implementors must be `Sync` because evaluations run on a rayon pool. They
/// must also be pure: the same point returns the same objectives, or the
/// algorithms lose their reproducibility guarantee.
pub trait Problem: Sync {
    /// Short name used by the CLI, the reports and the tests.
    fn name(&self) -> &str;

    /// Number of decision variables.
    fn dim(&self) -> usize;

    /// Number of objectives, at least one.
    fn objectives(&self) -> usize {
        1
    }

    /// Number of inequality constraints `g(x) <= 0`, possibly zero.
    fn constraints(&self) -> usize {
        0
    }

    /// Box bounds of the search space.
    fn bounds(&self) -> &Bounds;

    /// Fill `objectives` and `constraints` for the point `x`.
    ///
    /// The slices arrive at the right length, so an implementation never
    /// allocates on the hot path.
    fn evaluate(&self, x: &[f64], objectives: &mut [f64], constraints: &mut [f64]);

    /// Convenience wrapper that allocates one score.
    fn score(&self, x: &[f64]) -> Score {
        let mut objectives = vec![0.0; self.objectives()];
        let mut constraints = vec![0.0; self.constraints()];
        self.evaluate(x, &mut objectives, &mut constraints);
        Score::new(objectives, constraints)
    }
}

/// One evaluated point: objectives, constraint values and the derived
/// feasibility flag.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    objectives: Vec<f64>,
    constraints: Vec<f64>,
    violation: f64,
}

impl Score {
    /// Build a score and derive the total constraint violation.
    pub fn new(objectives: Vec<f64>, constraints: Vec<f64>) -> Self {
        // The violation is the sum of the positive constraint values. A
        // constraint is satisfied when it is at or below zero.
        let violation = constraints
            .iter()
            .map(|&g| if g > 0.0 { g } else { 0.0 })
            .sum();
        Self {
            objectives,
            constraints,
            violation,
        }
    }

    /// A score with no constraints.
    pub fn feasible(objectives: Vec<f64>) -> Self {
        Self::new(objectives, Vec::new())
    }

    /// Objective values.
    pub fn objectives(&self) -> &[f64] {
        &self.objectives
    }

    /// Constraint values, each satisfied when at or below zero.
    pub fn constraints(&self) -> &[f64] {
        &self.constraints
    }

    /// Sum of the positive constraint values.
    pub fn violation(&self) -> f64 {
        self.violation
    }

    /// True when every constraint holds.
    pub fn is_feasible(&self) -> bool {
        self.violation <= 0.0
    }

    /// Single objective value. Panics for a multi-objective problem.
    pub fn objective(&self) -> f64 {
        assert_eq!(
            self.objectives.len(),
            1,
            "score has more than one objective"
        );
        self.objectives[0]
    }

    /// Pareto dominance on objectives alone, ignoring feasibility.
    pub fn pareto_dominates(&self, other: &Self) -> bool {
        crate::eval::dominates(&self.objectives, &other.objectives)
    }

    /// Deb's constrained dominance: feasible beats infeasible, two infeasible
    /// points compare by violation, two feasible points compare by Pareto
    /// dominance.
    pub fn dominates(&self, other: &Self) -> bool {
        match (self.is_feasible(), other.is_feasible()) {
            (true, false) => true,
            (false, true) => false,
            (false, false) => self.violation < other.violation,
            (true, true) => self.pareto_dominates(other),
        }
    }

    /// True when `self` is at least as good as `other` under Deb's rules.
    pub fn weakly_dominates(&self, other: &Self) -> bool {
        !other.dominates(self)
    }

    /// Total order used by selection: strictly better means `Less`.
    ///
    /// Infeasible points never beat feasible ones, so a comparison sort over a
    /// population puts every feasible individual first.
    pub fn better_than(&self, other: &Self) -> bool {
        self.dominates(other)
    }
}
