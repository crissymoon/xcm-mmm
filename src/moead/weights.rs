//! Weight vectors and neighbourhoods for MOEA/D.

use crate::metrics::das_dennis;

/// Weight vectors plus the neighbourhood of each one.
#[derive(Clone, Debug)]
pub struct WeightSet {
    /// One weight vector per subproblem, each summing to one.
    pub weights: Vec<Vec<f64>>,
    /// Indices of the closest weights, the neighbourhood, for each subproblem.
    pub neighbourhoods: Vec<Vec<usize>>,
}

impl WeightSet {
    /// Build the simplex lattice and the neighbourhoods.
    ///
    /// `divisions` is the number of steps along each axis, `objectives` the
    /// dimension and `neighbourhood` the neighbourhood size. Ties in distance
    /// are broken by index, so the layout is deterministic.
    pub fn build(divisions: usize, objectives: usize, neighbourhood: usize) -> Self {
        let weights = das_dennis(divisions, objectives);
        let neighbourhood = neighbourhood.clamp(1, weights.len());
        let mut neighbourhoods = Vec::with_capacity(weights.len());
        for i in 0..weights.len() {
            let mut order: Vec<usize> = (0..weights.len()).collect();
            order.sort_by(|&a, &b| {
                distance_sq(&weights[a], &weights[i])
                    .partial_cmp(&distance_sq(&weights[b], &weights[i]))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            order.truncate(neighbourhood);
            neighbourhoods.push(order);
        }
        Self {
            weights,
            neighbourhoods,
        }
    }

    /// Number of subproblems.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True when there are no subproblems, which `build` never produces.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Neighbourhood of subproblem `i`.
    pub fn neighbours(&self, i: usize) -> &[usize] {
        &self.neighbourhoods[i]
    }
}

fn distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Number of points on a simplex lattice with `divisions` steps and `m`
/// objectives, which is `C(divisions + m - 1, m - 1)`.
pub fn lattice_count(divisions: usize, objectives: usize) -> usize {
    // The product form avoids the overflow a factorial would hit.
    let k = (objectives - 1).min(divisions);
    let n = divisions + objectives - 1;
    let mut count = 1u128;
    for i in 1..=k {
        count = count * (n - k + i) as u128 / i as u128;
    }
    usize::try_from(count).unwrap_or(usize::MAX)
}

/// The largest number of divisions whose lattice fits in `target` points.
///
/// Used when the caller asks for a population size rather than a lattice
/// resolution. The result is at least one.
pub fn divisions_for(target: usize, objectives: usize) -> usize {
    let mut d = 1;
    while lattice_count(d + 1, objectives) <= target && d < 5000 {
        d += 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lattice_counts_are_the_binomial_coefficients() {
        assert_eq!(lattice_count(4, 2), 5);
        assert_eq!(lattice_count(4, 3), 15);
        assert_eq!(lattice_count(12, 3), 91);
        assert_eq!(lattice_count(99, 2), 100);
    }

    #[test]
    fn divisions_respect_the_target() {
        for target in [10, 56, 100, 300] {
            let d = divisions_for(target, 3);
            assert!(lattice_count(d, 3) <= target);
            assert!(lattice_count(d + 1, 3) > target || d >= 5000);
        }
    }

    #[test]
    fn weights_sum_to_one_and_neighbourhoods_contain_self() {
        let set = WeightSet::build(6, 3, 8);
        assert_eq!(set.len(), 28);
        for (i, w) in set.weights.iter().enumerate() {
            assert_relative_eq!(w.iter().sum::<f64>(), 1.0);
            assert!(set.neighbours(i).contains(&i));
            assert_eq!(set.neighbours(i).len(), 8);
        }
    }

    #[test]
    fn neighbourhoods_are_closer_than_far_weights() {
        let set = WeightSet::build(8, 2, 5);
        for i in 0..set.len() {
            let own = set.neighbours(i);
            let worst = own
                .iter()
                .map(|&j| distance_sq(&set.weights[j], &set.weights[i]))
                .fold(0.0, f64::max);
            for j in 0..set.len() {
                if !own.contains(&j) {
                    assert!(distance_sq(&set.weights[j], &set.weights[i]) >= worst - 1e-12);
                }
            }
        }
    }
}
