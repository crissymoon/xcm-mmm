//! Dense matrices and the operators the transformer needs.
//!
//! Everything is `f64` and row major. The shapes are small enough that a
//! blocked kernel would be noise, but every operator the backward pass needs
//! has its transpose form here, so the model never allocates a transposed copy
//! just to multiply. Layer normalisation carries its own cache because the
//! backward pass needs the input, the mean and the inverse standard deviation,
//! and recomputing them would hide a bug rather than save work.
//!
//! The unit tests check the operators against a naive implementation and check
//! the layer normalisation gradient against a finite difference, which is the
//! cheapest way to be sure the hand written backward pass is right.

#![allow(clippy::needless_range_loop)]

use crate::linalg::dot;

/// The scalar type. One alias so a move to `f32` is a one line change.
pub type Real = f64;

/// A dense matrix, row major.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Real>,
}

impl Matrix {
    /// A zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// A matrix from row major data.
    ///
    /// Panics when the length does not agree with the shape, because that is a
    /// programming error rather than a runtime condition.
    pub fn new(rows: usize, cols: usize, data: Vec<Real>) -> Self {
        assert_eq!(rows * cols, data.len(), "matrix shape does not match data");
        Self { rows, cols, data }
    }

    /// One element.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Real {
        self.data[row * self.cols + col]
    }

    /// Set one element.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Real) {
        self.data[row * self.cols + col] = value;
    }

    /// One row, borrowed.
    #[inline]
    pub fn row(&self, row: usize) -> &[Real] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// One row, mutably borrowed.
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [Real] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

/// `a * b`, shapes `(m, k) x (k, n)`.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "matmul shape mismatch");
    let mut out = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        let arow = a.row(i);
        let orow = &mut out.data[i * b.cols..(i + 1) * b.cols];
        for k in 0..a.cols {
            let scale = arow[k];
            if scale == 0.0 {
                continue;
            }
            let brow = b.row(k);
            for j in 0..b.cols {
                orow[j] += scale * brow[j];
            }
        }
    }
    out
}

/// `a^T * b`, shapes `(m, k)^T x (m, n)` giving `(k, n)`.
pub fn matmul_atb(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.rows, b.rows, "atb shape mismatch");
    let mut out = Matrix::zeros(a.cols, b.cols);
    for i in 0..a.rows {
        let arow = a.row(i);
        let brow = b.row(i);
        for k in 0..a.cols {
            let scale = arow[k];
            if scale == 0.0 {
                continue;
            }
            let orow = &mut out.data[k * b.cols..(k + 1) * b.cols];
            for j in 0..b.cols {
                orow[j] += scale * brow[j];
            }
        }
    }
    out
}

/// `a * b^T`, shapes `(m, k) x (n, k)^T` giving `(m, n)`.
pub fn matmul_abt(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.cols, "abt shape mismatch");
    let mut out = Matrix::zeros(a.rows, b.rows);
    for i in 0..a.rows {
        let arow = a.row(i);
        for j in 0..b.rows {
            out.data[i * b.rows + j] = dot(arow, b.row(j));
        }
    }
    out
}

/// Add a row vector to every row of a matrix.
pub fn add_bias(matrix: &mut Matrix, bias: &[Real]) {
    assert_eq!(matrix.cols, bias.len(), "bias shape mismatch");
    for row in 0..matrix.rows {
        let target = matrix.row_mut(row);
        for (value, b) in target.iter_mut().zip(bias) {
            *value += b;
        }
    }
}

/// Sum every column into one row.
pub fn col_sum(matrix: &Matrix) -> Vec<Real> {
    let mut out = vec![0.0; matrix.cols];
    for row in 0..matrix.rows {
        for (slot, value) in out.iter_mut().zip(matrix.row(row)) {
            *slot += value;
        }
    }
    out
}

/// Mean of every column.
pub fn col_mean(matrix: &Matrix) -> Vec<Real> {
    let mut out = col_sum(matrix);
    if matrix.rows > 0 {
        let scale = 1.0 / matrix.rows as Real;
        for value in &mut out {
            *value *= scale;
        }
    }
    out
}

/// The statistics a layer normalisation backward pass needs.
#[derive(Clone, Copy, Debug)]
pub struct NormCache {
    /// Mean of the input.
    pub mean: Real,
    /// Inverse standard deviation, with the epsilon already applied.
    pub inv: Real,
}

/// Layer normalise one vector.
pub fn layer_norm(
    x: &[Real],
    gain: &[Real],
    bias: &[Real],
    eps: Real,
    out: &mut [Real],
) -> NormCache {
    let n = x.len();
    assert_eq!(gain.len(), n);
    assert_eq!(bias.len(), n);
    assert_eq!(out.len(), n);
    let mean = x.iter().sum::<Real>() / n as Real;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<Real>() / n as Real;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] - mean) * inv * gain[i] + bias[i];
    }
    NormCache { mean, inv }
}

/// Gradient of [`layer_norm`].
///
/// `dx`, `dgain` and `dbias` are accumulated into, not overwritten, so a caller
/// can share one gradient buffer across rows.
pub fn layer_norm_backward(
    dy: &[Real],
    x: &[Real],
    cache: &NormCache,
    gain: &[Real],
    dx: &mut [Real],
    dgain: &mut [Real],
    dbias: &mut [Real],
) {
    let n = x.len();
    let mut sum_dxhat = 0.0;
    let mut sum_dxhat_centered = 0.0;
    for i in 0..n {
        let centered = x[i] - cache.mean;
        let xhat = centered * cache.inv;
        let dxhat = dy[i] * gain[i];
        dgain[i] += dy[i] * xhat;
        dbias[i] += dy[i];
        sum_dxhat += dxhat;
        sum_dxhat_centered += dxhat * centered;
    }
    let dvar = sum_dxhat_centered * -0.5 * cache.inv * cache.inv * cache.inv;
    let dmean = -sum_dxhat * cache.inv;
    let scale = 1.0 / n as Real;
    for i in 0..n {
        let centered = x[i] - cache.mean;
        dx[i] += dy[i] * gain[i] * cache.inv + dvar * 2.0 * centered * scale + dmean * scale;
    }
}

/// Softmax of one row, in place, with the maximum removed for stability.
pub fn softmax_in_place(row: &mut [Real]) {
    let mut max = Real::NEG_INFINITY;
    for value in row.iter() {
        if *value > max {
            max = *value;
        }
    }
    let mut total = 0.0;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    if total > 0.0 {
        let inv = 1.0 / total;
        for value in row.iter_mut() {
            *value *= inv;
        }
    }
}

const GELU_SCALE: Real = 0.797_884_560_802_865_4;
const GELU_CUBIC: Real = 0.044_715;

/// The tanh approximation of GELU.
pub fn gelu(x: Real) -> Real {
    let inner = GELU_SCALE * (x + GELU_CUBIC * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Derivative of [`gelu`].
pub fn gelu_grad(x: Real) -> Real {
    let inner = GELU_SCALE * (x + GELU_CUBIC * x * x * x);
    let t = inner.tanh();
    let base = 0.5 * (1.0 + t);
    let slope = 0.5 * (1.0 - t * t) * GELU_SCALE * (1.0 + 3.0 * GELU_CUBIC * x * x);
    base + x * slope
}

/// The logistic function, stable at both ends.
pub fn sigmoid(x: Real) -> Real {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + exp(x))`, stable for large and small `x`.
pub fn softplus(x: Real) -> Real {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut rng = crate::rng::Rng::new(seed);
        let data = (0..rows * cols).map(|_| rng.standard_normal()).collect();
        Matrix::new(rows, cols, data)
    }

    #[test]
    fn transpose_products_match_naive() {
        let a = sample(3, 4, 1);
        let b = sample(3, 5, 2);
        let c = sample(4, 5, 3);
        let atb = matmul_atb(&a, &b);
        for k in 0..4 {
            for j in 0..5 {
                let mut sum = 0.0;
                for i in 0..3 {
                    sum += a.get(i, k) * b.get(i, j);
                }
                assert_relative_eq!(atb.get(k, j), sum, epsilon = 1e-12);
            }
        }
        let mm = matmul(&a, &c);
        for i in 0..3 {
            for j in 0..5 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += a.get(i, k) * c.get(k, j);
                }
                assert_relative_eq!(mm.get(i, j), sum, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn softmax_is_a_distribution() {
        let mut row = vec![1.0, 2.0, -3.0, 7.5];
        softmax_in_place(&mut row);
        assert_relative_eq!(row.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        assert!(row.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn gelu_gradient_matches_finite_difference() {
        for x in [-2.5, -0.4, 0.0, 0.7, 3.1] {
            let h = 1e-6;
            let numeric = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            assert_relative_eq!(gelu_grad(x), numeric, epsilon = 1e-6);
        }
    }

    #[test]
    fn layer_norm_gradient_matches_finite_difference() {
        let x = vec![0.3, -1.2, 0.8, 2.0, -0.4];
        let gain = vec![1.1, 0.7, 1.4, 0.9, 1.0];
        let bias = vec![0.1, -0.2, 0.0, 0.3, 0.05];
        let dy = vec![0.5, -0.3, 0.8, 0.1, -0.6];
        let eps = 1e-5;
        let mut out = vec![0.0; x.len()];
        let cache = layer_norm(&x, &gain, &bias, eps, &mut out);
        let mut dx = vec![0.0; x.len()];
        let mut dg = vec![0.0; x.len()];
        let mut db = vec![0.0; x.len()];
        layer_norm_backward(&dy, &x, &cache, &gain, &mut dx, &mut dg, &mut db);

        let loss = |x: &[f64]| -> f64 {
            let mut y = vec![0.0; x.len()];
            layer_norm(x, &gain, &bias, eps, &mut y);
            y.iter().zip(&dy).map(|(a, b)| a * b).sum()
        };
        let h = 1e-6;
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += h;
            minus[i] -= h;
            let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
            assert_relative_eq!(dx[i], numeric, epsilon = 1e-6);
        }
    }
}
