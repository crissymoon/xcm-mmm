//! The two losses the pair schema needs.
//!
//! A task is one of two shapes, and the loss follows the shape rather than the
//! task name:
//!
//! * A pair carries an order, so the score difference is pushed through a
//!   logistic and the loss is the negative log likelihood of that order. This is
//!   the Bradley-Terry form, and it is what ranking, preference, contrast,
//!   correction and retrieval rows all use.
//! * A labelled row carries a direction, so a single score is pushed through a
//!   logistic and the loss is binary cross entropy.
//!
//! Both losses return a scalar and the derivative to each score, in the same
//! call, because the backward pass wants the two together and splitting them
//! would let the two drift apart. The test module checks each derivative
//! against a central finite difference.

use crate::transformer::tensor::{sigmoid, softplus, Real};

/// A loss value and its derivative to the score of each side.
#[derive(Clone, Copy, Debug)]
pub struct Gradient {
    /// The scalar loss.
    pub loss: Real,
    /// Derivative to the first score.
    pub d_first: Real,
    /// Derivative to the second score. Zero for a pointwise row.
    pub d_second: Real,
}

/// The logistic loss on a pair whose first side should score higher.
///
/// `weight` scales both the loss and the gradients, which is how a task weight
/// or a margin weight enters without a second code path.
pub fn pairwise(first: Real, second: Real, weight: Real) -> Gradient {
    let difference = first - second;
    let probability = sigmoid(difference);
    Gradient {
        loss: softplus(-difference) * weight,
        d_first: (probability - 1.0) * weight,
        d_second: (1.0 - probability) * weight,
    }
}

/// Binary cross entropy on one score and a target in `[0, 1]`.
pub fn pointwise(score: Real, target: Real, weight: Real) -> Gradient {
    let probability = sigmoid(score);
    Gradient {
        loss: (softplus(score) - target * score) * weight,
        d_first: (probability - target) * weight,
        d_second: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pairwise_gradient_matches_finite_difference() {
        let h = 1e-6;
        for (a, b, w) in [(0.3, -0.2, 1.0), (2.0, 2.0, 0.5), (-1.4, 0.8, 2.0)] {
            let loss = |x: Real, y: Real| pairwise(x, y, w).loss;
            let numeric_a = (loss(a + h, b) - loss(a - h, b)) / (2.0 * h);
            let numeric_b = (loss(a, b + h) - loss(a, b - h)) / (2.0 * h);
            let gradient = pairwise(a, b, w);
            assert_relative_eq!(gradient.d_first, numeric_a, epsilon = 1e-7);
            assert_relative_eq!(gradient.d_second, numeric_b, epsilon = 1e-7);
        }
    }

    #[test]
    fn pointwise_gradient_matches_finite_difference() {
        let h = 1e-6;
        for (score, target, w) in [(0.1, 1.0, 1.0), (-0.7, 0.0, 0.5), (1.3, 1.0, 2.0)] {
            let loss = |s: Real| pointwise(s, target, w).loss;
            let numeric = (loss(score + h) - loss(score - h)) / (2.0 * h);
            assert_relative_eq!(pointwise(score, target, w).d_first, numeric, epsilon = 1e-7);
        }
    }

    #[test]
    fn a_confident_wrong_pair_costs_more_than_a_confident_right_one() {
        let right = pairwise(1.0, 0.0, 1.0).loss;
        let wrong = pairwise(0.0, 1.0, 1.0).loss;
        assert!(wrong > right);
        assert!(right > 0.0);
    }
}
