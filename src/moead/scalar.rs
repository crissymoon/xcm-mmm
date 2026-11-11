//! Scalarising functions that turn a vector objective into one number per
//! subproblem.

/// How MOEA/D compares candidates against one weight vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalarization {
    /// Tchebycheff: `max_i w_i |f_i - z_i|`. Finds points on non-convex fronts,
    /// which weighted sum cannot.
    Tchebycheff,
    /// Penalty based intersection: distance along the weight direction plus a
    /// penalty for drifting sideways. Balanced fronts, needs `theta`.
    Pbi {
        /// Penalty on the perpendicular distance, 5.0 in the original paper.
        theta: f64,
    },
    /// Weighted sum: `sum_i w_i f_i`. Simple, but blind to concave regions.
    WeightedSum,
}

impl Scalarization {
    /// Scalar value of `f` for weight `w` and ideal point `z`. Lower is better.
    pub fn value(self, f: &[f64], ideal: &[f64], weight: &[f64]) -> f64 {
        debug_assert_eq!(f.len(), ideal.len());
        debug_assert_eq!(f.len(), weight.len());
        match self {
            Scalarization::Tchebycheff => f
                .iter()
                .zip(ideal)
                .zip(weight)
                // A zero weight would remove that objective from the maximum
                // and make every comparison degenerate, so it is floored.
                .map(|((fi, zi), wi)| wi.max(1e-6) * (fi - zi).abs())
                .fold(0.0, f64::max),
            Scalarization::WeightedSum => f.iter().zip(weight).map(|(fi, wi)| wi * fi).sum::<f64>(),
            Scalarization::Pbi { theta } => {
                let norm = weight.iter().map(|w| w * w).sum::<f64>().sqrt();
                if norm <= 0.0 {
                    return f64::INFINITY;
                }
                let diff: Vec<f64> = f.iter().zip(ideal).map(|(fi, zi)| fi - zi).collect();
                let projection = diff.iter().zip(weight).map(|(d, w)| d * w).sum::<f64>() / norm;
                let perpendicular = diff
                    .iter()
                    .zip(weight)
                    .map(|(d, w)| {
                        let along = projection * w / norm;
                        (d - along) * (d - along)
                    })
                    .sum::<f64>()
                    .sqrt();
                projection + theta * perpendicular
            }
        }
    }

    /// Short name for reports.
    pub fn name(self) -> &'static str {
        match self {
            Scalarization::Tchebycheff => "tchebycheff",
            Scalarization::Pbi { .. } => "pbi",
            Scalarization::WeightedSum => "weighted-sum",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tchebycheff_takes_the_worst_weighted_gap() {
        let s = Scalarization::Tchebycheff;
        let z = [0.0, 0.0];
        let w = [0.5, 0.5];
        assert!((s.value(&[1.0, 2.0], &z, &w) - 1.0).abs() < 1e-12);
        assert!((s.value(&[2.0, 1.0], &z, &w) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_sum_is_the_dot_product() {
        let s = Scalarization::WeightedSum;
        assert!((s.value(&[2.0, 4.0], &[0.0, 0.0], &[0.25, 0.75]) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn pbi_penalises_perpendicular_distance() {
        let s = Scalarization::Pbi { theta: 5.0 };
        let z = [0.0, 0.0];
        let w = [0.5, 0.5];
        let on_axis = s.value(&[1.0, 1.0], &z, &w);
        let off_axis = s.value(&[1.5, 0.5], &z, &w);
        assert!(off_axis > on_axis);
        assert!(on_axis > 0.0);
    }

    #[test]
    fn ideal_point_shift_never_increases_the_value() {
        let s = Scalarization::Tchebycheff;
        let w = [0.3, 0.7];
        let f = [1.0, 1.0];
        assert_eq!(s.value(&f, &[0.0, 0.0], &w), 0.7);
        assert!(s.value(&f, &[0.5, 0.5], &w) < 0.7);
    }
}
