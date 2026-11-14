//! A small logistic regression, fitted by fixed step gradient descent.
//!
//! The banks are a few thousand sentences and the features are a handful of
//! standardised readings, so an iterative solver would be a dependency and a
//! tolerance argument for nothing. This fit is 800 fixed steps, which converges
//! on this data and, more usefully, produces the same weights on every machine
//! because it draws no random numbers.
//!
//! Two details come from the published detector and are kept so the readings
//! mean the same thing. Every feature is standardised on the training rows and
//! clipped at four standard deviations, because a count feature that is rare in
//! the banks has a small scale and one occurrence in a new ad can standardise to
//! a dozen. The penalty is L2 on the weights and never on the intercept, so a
//! lopsided bank does not get its base rate squeezed towards a half.

use serde::{Deserialize, Serialize};

/// Where the optimiser stops and how far it steps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitParams {
    pub steps: usize,
    pub rate: f64,
    pub penalty: f64,
    pub clip: f64,
}

impl Default for FitParams {
    fn default() -> Self {
        Self {
            steps: 800,
            rate: 0.5,
            penalty: 0.15,
            clip: 4.0,
        }
    }
}

/// A fitted model on standardised features.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Logistic {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub clip: f64,
}

impl Logistic {
    /// Fit on rows of raw feature values and their labels.
    pub fn fit(rows: &[Vec<f64>], labels: &[bool], params: &FitParams) -> Self {
        let dim = rows.first().map(Vec::len).unwrap_or(0);
        let n = rows.len().max(1) as f64;
        let mean: Vec<f64> = (0..dim)
            .map(|j| rows.iter().map(|row| row[j]).sum::<f64>() / n)
            .collect();
        let scale: Vec<f64> = (0..dim)
            .map(|j| {
                let variance = rows
                    .iter()
                    .map(|row| (row[j] - mean[j]).powi(2))
                    .sum::<f64>()
                    / n;
                let sd = variance.sqrt();
                if sd > 1e-12 {
                    sd
                } else {
                    1.0
                }
            })
            .collect();
        let design: Vec<Vec<f64>> = rows
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, value)| {
                        ((value - mean[j]) / scale[j]).clamp(-params.clip, params.clip)
                    })
                    .collect()
            })
            .collect();

        let mut coefficients = vec![0.0; dim];
        let mut intercept = 0.0;
        let positive = labels.iter().filter(|l| **l).count() as f64;
        if positive > 0.0 && positive < n {
            let rate = (positive / n).clamp(1e-6, 1.0 - 1e-6);
            intercept = (rate / (1.0 - rate)).ln();
        }
        for step in 0..params.steps {
            let mut grad = vec![0.0; dim];
            let mut grad_b = 0.0;
            for (row, label) in design.iter().zip(labels) {
                let error =
                    sigmoid(dot(&coefficients, row) + intercept) - f64::from(u8::from(*label));
                for (j, value) in row.iter().enumerate() {
                    grad[j] += error * value;
                }
                grad_b += error;
            }
            let learning = params.rate / (1.0 + step as f64 / 100.0);
            for j in 0..dim {
                coefficients[j] -= learning * (grad[j] / n + params.penalty * coefficients[j]);
            }
            intercept -= learning * grad_b / n;
        }
        Self {
            mean,
            scale,
            coefficients,
            intercept,
            clip: params.clip,
        }
    }

    /// Probability of the positive class for one text's raw feature values.
    pub fn predict(&self, values: &[f64]) -> f64 {
        let mut total = self.intercept;
        for (j, value) in values.iter().enumerate() {
            let scale = self.scale.get(j).copied().unwrap_or(1.0);
            let mean = self.mean.get(j).copied().unwrap_or(0.0);
            let z = ((value - mean) / scale).clamp(-self.clip, self.clip);
            total += self.coefficients.get(j).copied().unwrap_or(0.0) * z;
        }
        sigmoid(total)
    }
}

fn dot(weights: &[f64], row: &[f64]) -> f64 {
    weights.iter().zip(row).map(|(w, x)| w * x).sum()
}

/// The logistic function, guarded against overflow at both ends.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_separable_line_is_learned() {
        let rows = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.2],
            vec![0.2, 0.1],
            vec![3.0, 3.0],
            vec![3.2, 2.8],
            vec![2.8, 3.1],
        ];
        let labels = vec![false, false, false, true, true, true];
        let model = Logistic::fit(&rows, &labels, &FitParams::default());
        assert!(model.predict(&[0.0, 0.0]) < 0.5);
        assert!(model.predict(&[3.0, 3.0]) > 0.5);
        assert!(model.predict(&[0.5, 0.5]) < model.predict(&[2.5, 2.5]));
    }

    #[test]
    fn the_fit_is_reproducible() {
        let rows = vec![vec![1.0], vec![2.0], vec![8.0], vec![9.0]];
        let labels = vec![false, false, true, true];
        let a = Logistic::fit(&rows, &labels, &FitParams::default());
        let b = Logistic::fit(&rows, &labels, &FitParams::default());
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.intercept, b.intercept);
    }

    #[test]
    fn one_class_returns_the_base_rate() {
        let rows = vec![vec![1.0], vec![2.0]];
        let labels = vec![true, true];
        let model = Logistic::fit(&rows, &labels, &FitParams::default());
        assert!(model.predict(&[1.5]) > 0.9);
    }

    #[test]
    fn a_constant_feature_does_not_divide_by_zero() {
        let rows = vec![vec![2.0], vec![2.0], vec![2.0]];
        let labels = vec![false, true, true];
        let model = Logistic::fit(&rows, &labels, &FitParams::default());
        assert_eq!(model.scale, vec![1.0]);
        assert!(model.predict(&[2.0]).is_finite());
    }
}
