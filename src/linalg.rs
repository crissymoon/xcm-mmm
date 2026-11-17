//! Small linear algebra helpers shared by the forecasting front end.
//!
//! Only what a ridge regression and a rank correlation need, with no external
//! dependency. The solvers are Cholesky based and return `None` on a matrix that
//! is not positive definite, which the callers treat as a failed fit rather than
//! as a panic.

/// Arithmetic mean. Returns zero for an empty slice.
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

/// Population variance.
pub fn variance(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64
}

/// Population standard deviation.
pub fn std_dev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

/// Dot product of two equal length slices.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Centre a slice in place and divide by its standard deviation.
///
/// A constant column is left as zeros instead of producing NaNs, and the caller
/// can then treat it as uninformative.
pub fn standardize_in_place(x: &mut [f64]) -> (f64, f64) {
    let m = mean(x);
    let s = std_dev(x);
    if s > 0.0 {
        for v in x.iter_mut() {
            *v = (*v - m) / s;
        }
    } else {
        for v in x.iter_mut() {
            *v = 0.0;
        }
    }
    (m, s)
}

/// Cholesky factorisation in place: on success `a` holds the lower triangle.
// The row and column indices are the factorisation itself, not a walk over one
// slice.
#[allow(clippy::needless_range_loop)]
pub fn cholesky(a: &mut [Vec<f64>]) -> Option<()> {
    let n = a.len();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= a[i][k] * a[j][k];
            }
            if i == j {
                if sum <= 1e-14 || !sum.is_finite() {
                    return None;
                }
                a[i][j] = sum.sqrt();
            } else {
                a[i][j] = sum / a[j][j];
            }
        }
        for j in (i + 1)..n {
            a[i][j] = 0.0;
        }
    }
    Some(())
}

/// Solve `L L^T x = b` from a Cholesky factor.
pub fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// Solve a symmetric positive definite system, adding a ridge when asked.
///
/// Returns `None` when the matrix is singular even after the ridge, which the
/// caller should read as "this feature set cannot be fitted".
pub fn solve_spd(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, ridge: f64) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(Vec::new());
    }
    debug_assert_eq!(b.len(), n);
    for (i, row) in a.iter_mut().enumerate() {
        if !row[i].is_finite() {
            return None;
        }
        row[i] += ridge;
    }
    for v in b.iter_mut() {
        if !v.is_finite() {
            return None;
        }
    }
    cholesky(&mut a)?;
    let x = cholesky_solve(&a, &b);
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Pearson correlation, zero when either side is constant.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let (ma, mb) = (mean(&a[..n]), mean(&b[..n]));
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let x = a[i] - ma;
        let y = b[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Average ranks, with ties receiving the mean of the positions they occupy.
pub fn ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (x[order[j + 1]] - x[order[i]]).abs() <= 1e-12 {
            j += 1;
        }
        let rank = 0.5 * ((i + 1) as f64 + (j + 1) as f64);
        for k in i..=j {
            out[order[k]] = rank;
        }
        i = j + 1;
    }
    out
}

/// Spearman rank correlation, the usual information coefficient for forecasts.
///
/// Returns zero when either side is constant, so a feature that never changes
/// scores no better than noise.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 0.0;
    }
    pearson(&ranks(&a[..n]), &ranks(&b[..n]))
}

/// Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations.
///
/// Returns the eigenvalues, with the matching eigenvectors as the columns of the
/// second value, in the order the rotations leave them. The rotations are chosen
/// so the off-diagonal mass falls on every sweep; thirty sweeps settle any
/// matrix this crate asks about, and a matrix still rotating after that is
/// reported as a refusal instead of as a reading.
///
/// The size is capped at 64. A sweep is cubic in the size and every caller here
/// works in eight dimensions, so a larger matrix means the caller is confused
/// about what it is decomposing.
pub fn jacobi_eigen(matrix: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    if n == 0 || n > 64 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    // The symmetry check and the rotations read a square matrix on both axes,
    // which is the arithmetic rather than a walk over one slice.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        for j in 0..n {
            if !a[i][j].is_finite() {
                return None;
            }
            // A rotation only maps a symmetric matrix to a symmetric matrix, so
            // anything else is refused rather than silently symmetrised.
            if (a[i][j] - a[j][i]).abs() > 1e-9 * (1.0 + a[i][j].abs() + a[j][i].abs()) {
                return None;
            }
            a[j][i] = a[i][j];
        }
    }
    let mut vectors = vec![vec![0.0; n]; n];
    for (d, row) in vectors.iter_mut().enumerate() {
        row[d] = 1.0;
    }
    for _ in 0..30 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off <= 1e-24 {
            let values = (0..n).map(|d| a[d][d]).collect();
            return Some((values, vectors));
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let pivot = a[p][q];
                if pivot == 0.0 {
                    continue;
                }
                // The stable root of t^2 + 2t*theta - 1 = 0, which avoids the
                // cancellation the textbook form suffers when theta is large.
                let theta = (a[q][q] - a[p][p]) / (2.0 * pivot);
                let root = (theta * theta + 1.0).sqrt();
                let t = if theta >= 0.0 {
                    1.0 / (theta + root)
                } else {
                    -1.0 / (-theta + root)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                a[p][p] -= t * pivot;
                a[q][q] += t * pivot;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                #[allow(clippy::needless_range_loop)]
                for r in 0..n {
                    if r != p && r != q {
                        let (rp, rq) = (a[r][p], a[r][q]);
                        a[r][p] = c * rp - s * rq;
                        a[p][r] = a[r][p];
                        a[r][q] = s * rp + c * rq;
                        a[q][r] = a[r][q];
                    }
                }
                for row in vectors.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        let x = solve_spd(a, b, 0.0).unwrap();
        assert!((4.0 * x[0] + x[1] - 1.0).abs() < 1e-10);
        assert!((x[0] + 3.0 * x[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn ridge_saves_a_singular_system() {
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(solve_spd(a.clone(), vec![1.0, 1.0], 0.0).is_none());
        let x = solve_spd(a, vec![1.0, 1.0], 1e-6).unwrap();
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spearman_handles_ties_and_monotone_maps() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-12);
        let tied = [1.0, 1.0, 2.0, 3.0];
        let r = spearman(&tied, &[1.0, 2.0, 3.0, 4.0]);
        assert!(r > 0.7 && r < 1.0);
        assert_eq!(spearman(&[2.0; 5], &a), 0.0);
    }

    #[test]
    fn eigen_of_a_known_symmetric_matrix() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (values, vectors) = jacobi_eigen(&a).unwrap();
        let mut sorted = values.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 3.0).abs() < 1e-12);
        // Rebuild from the decomposition: the pair has to reproduce the input.
        for i in 0..2 {
            for j in 0..2 {
                let value: f64 = (0..2)
                    .map(|m| vectors[i][m] * values[m] * vectors[j][m])
                    .sum();
                assert!((value - a[i][j]).abs() < 1e-12);
            }
        }
        // Columns are orthonormal.
        let dot = vectors[0][0] * vectors[0][1] + vectors[1][0] * vectors[1][1];
        assert!(dot.abs() < 1e-12);
    }

    #[test]
    fn eigen_recovers_a_near_singular_ratio() {
        // A rotation of a diagonal matrix whose axes differ by six orders of
        // magnitude: this is the reading the mixture bound turns on, so it has
        // to survive the rotation and be found.
        let (c, s) = (0.6f64, 0.8f64);
        let a = vec![
            vec![c * c * 1.0 + s * s * 1e-6, c * s * (1.0 - 1e-6)],
            vec![c * s * (1.0 - 1e-6), s * s * 1.0 + c * c * 1e-6],
        ];
        let (values, _) = jacobi_eigen(&a).unwrap();
        let longest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shortest = values.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((longest - 1.0).abs() < 1e-9);
        assert!((longest / shortest - 1e6).abs() < 1e-2 * 1e6);
    }

    #[test]
    fn eigen_refuses_what_it_cannot_trust() {
        assert!(jacobi_eigen(&[]).is_none());
        assert!(jacobi_eigen(&[vec![1.0, 2.0], vec![3.0, 1.0]]).is_none());
        assert!(jacobi_eigen(&[vec![1.0, 0.0], vec![0.0]]).is_none());
        let big = vec![vec![0.0; 65]; 65];
        assert!(jacobi_eigen(&big).is_none());
    }

    #[test]
    fn standardize_denominator_never_zero() {
        let mut x = vec![3.0, 3.0, 3.0];
        let (m, s) = standardize_in_place(&mut x);
        assert_eq!(m, 3.0);
        assert_eq!(s, 0.0);
        assert!(x.iter().all(|&v| v == 0.0));
    }
}
