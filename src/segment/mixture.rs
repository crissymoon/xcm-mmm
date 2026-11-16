//! The likelihood a partition is read through, and the three that compete.
//!
//! A k-means fit is not only a clustering: it is the maximum likelihood fit of
//! a Gaussian mixture whose components are spheres of one shared width. That is
//! a strong statement about the data, and it is the statement this folder used
//! to ship without saying so. Every column was standardised to unit variance, so
//! a sphere in that space says the eight readings are equally spread inside a
//! group and that no two of them co-vary. Neither is true of an advertising
//! panel: an advertiser's volume, its spend and the number of terms it buys
//! against move together.
//!
//! Three families are therefore fitted and compared, in increasing freedom:
//!
//! * [`Family::Spherical`] one width per group, shared across columns, which is
//!   the shape k-means implies;
//! * [`Family::Diagonal`] one width per group per column, so a group may be
//!   tight on spend and loose on cadence;
//! * [`Family::Full`] a covariance per group, so a group may be stretched along
//!   the direction two columns jointly run.
//!
//! Estimation is expectation maximisation. Half the seeds start from a k-means
//! partition, which is the classic initialisation; the other half start from a
//! soft random assignment, because a hard start can only refine the split
//! k-means already made. Two groups that overlap along the axis k-means cuts are
//! found by the soft start and not by the hard one, so the mix is not decoration.
//! The best likelihood wins and every draw comes from the fit's seed.
//!
//! The covariance floor is stated in z units: a component may not claim a width
//! narrower than a thousandth of a standard deviation on any axis, because the
//! panel holds repeated readings and a component sitting on three identical rows
//! would otherwise report an unbounded density and win every comparison by
//! memorising them. A component that no row claims is reset to the whole panel
//! rather than kept at an exploding mean.
//!
//! Nothing here chooses. [`super::select`] owns the choice and its reasoning;
//! this module only states what a given (family, k) believes and how well that
//! belief predicts rows it did not see.

use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::rng::Rng;

use super::kmeans;

/// A component's width may not fall below this, in z units.
pub const FLOOR: f64 = 1e-3;

const LOG_2PI: f64 = 1.837_877_066_409_345_3;

/// How free a group is allowed to be about its own shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    /// One width per group, shared across columns: the k-means shape.
    #[default]
    Spherical,
    /// One width per group per column.
    Diagonal,
    /// A full covariance per group.
    Full,
}

impl Family {
    /// All three, simplest first, which is also the order the choice walks.
    pub const ALL: [Family; 3] = [Family::Spherical, Family::Diagonal, Family::Full];

    pub fn name(self) -> &'static str {
        match self {
            Family::Spherical => "spherical",
            Family::Diagonal => "diagonal",
            Family::Full => "full",
        }
    }

    /// The words the card uses, so the difference is not a bare name.
    pub fn shape(self) -> &'static str {
        match self {
            Family::Spherical => "one width per group, shared across columns",
            Family::Diagonal => "one width per group per column",
            Family::Full => "a covariance per group",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "spherical" => Some(Family::Spherical),
            "diagonal" => Some(Family::Diagonal),
            "full" => Some(Family::Full),
            _ => None,
        }
    }

    /// Free parameters of a mixture of this family: weights, means, covariances.
    pub fn parameters(self, k: usize, dim: usize) -> usize {
        let weights = k.saturating_sub(1);
        let means = k * dim;
        let covariance = match self {
            Family::Spherical => 1,
            Family::Diagonal => k * dim,
            Family::Full => k * dim * (dim + 1) / 2,
        };
        weights + means + covariance
    }
}

/// One group's belief about itself: how likely it is, where it sits, how wide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub weight: f64,
    pub mean: Vec<f64>,
    /// The covariance actually used, after the floor, as a full square matrix.
    pub covariance: Vec<Vec<f64>>,
}

/// A Cholesky factor and the log determinant that falls out of it.
#[derive(Clone, Debug)]
struct Factor {
    chol: Vec<Vec<f64>>,
    logdet: f64,
}

impl Factor {
    /// `log p(x | mean, covariance)` without the weight.
    fn log_density(&self, row: &[f64], mean: &[f64]) -> f64 {
        let dim = row.len();
        let mut residual = Vec::with_capacity(dim);
        for d in 0..dim {
            residual.push(row[d] - mean[d]);
        }
        let solved = linalg::cholesky_solve(&self.chol, &residual);
        let maha: f64 = solved.iter().zip(&residual).map(|(a, b)| a * b).sum();
        -0.5 * (maha + self.logdet + dim as f64 * LOG_2PI)
    }
}

/// Factorise, raising the diagonal by the floor on each retry.
///
/// The raised matrix is returned, not just the factor: the covariance a reader
/// finds in the artifact has to be the one the density was computed from. The
/// maximisation step already applies the floor to the diagonal, so the retries
/// only fire when a full covariance is indefinite despite it.
fn factorise(covariance: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Factor)> {
    let dim = covariance.len();
    for attempt in 0..8 {
        let mut trial = covariance.to_vec();
        if attempt > 0 {
            let bump = FLOOR * attempt as f64;
            for (d, row) in trial.iter_mut().enumerate() {
                row[d] += bump;
            }
        }
        let mut factor = trial.clone();
        if linalg::cholesky(&mut factor).is_none() {
            continue;
        }
        let logdet: f64 = 2.0 * (0..dim).map(|d| factor[d][d].ln()).sum::<f64>();
        if !logdet.is_finite() {
            continue;
        }
        return Some((
            trial,
            Factor {
                chol: factor,
                logdet,
            },
        ));
    }
    None
}

/// A fitted mixture: what it believes and how well it explains its own rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mixture {
    pub family: Family,
    pub components: Vec<Component>,
    /// Total log likelihood on the rows it was fitted to.
    pub loglik: f64,
    pub iterations: usize,
}

impl Default for Mixture {
    fn default() -> Self {
        Self {
            family: Family::Spherical,
            components: Vec::new(),
            loglik: f64::NEG_INFINITY,
            iterations: 0,
        }
    }
}

impl Mixture {
    pub fn k(&self) -> usize {
        self.components.len()
    }

    pub fn means(&self) -> Vec<Vec<f64>> {
        self.components.iter().map(|c| c.mean.clone()).collect()
    }

    /// Fit by expectation maximisation, best of `restarts` seeds. Even runs
    /// start from a k-means partition, odd runs from a soft random assignment,
    /// so the search is not confined to refinements of one hard split. A larger
    /// log likelihood wins; a tie keeps the lower run so the fit is
    /// reproducible.
    pub fn fit(
        rows: &[Vec<f64>],
        k: usize,
        family: Family,
        restarts: usize,
        iterations: usize,
        seed: u64,
    ) -> Mixture {
        assert!(!rows.is_empty(), "fit on an empty matrix");
        assert!(k >= 1 && k <= rows.len(), "k out of range");
        let mut best: Option<Mixture> = None;
        for run in 0..restarts.max(1) {
            let start = if run % 2 == 0 {
                let fit = kmeans::fit(rows, k, 1, iterations, seed.wrapping_add(run as u64));
                one_hot(&fit.labels, k)
            } else {
                soft_start(rows.len(), k, seed.wrapping_add(run as u64))
            };
            let candidate = em(rows, k, family, iterations, &start);
            if candidate.components.is_empty() {
                continue;
            }
            let better = match &best {
                None => true,
                Some(current) => candidate.loglik > current.loglik + 1e-9,
            };
            if better {
                best = Some(candidate);
            }
        }
        best.unwrap_or_default()
    }

    /// The model a set of hard labels implies: one M step, then the likelihood.
    ///
    /// This is how the k-means incumbent is scored on the same scale as the
    /// fitted mixtures, so the comparison is like for like rather than a
    /// silhouette beside a likelihood.
    pub fn plugged_in(rows: &[Vec<f64>], labels: &[usize], k: usize) -> Mixture {
        let resp = one_hot(labels, k);
        let (components, factors) = m_step(rows, &resp, k, Family::Spherical);
        Mixture {
            family: Family::Spherical,
            loglik: total(&components, &factors, rows),
            components,
            iterations: 0,
        }
    }

    fn factored(&self) -> Option<Vec<Factor>> {
        self.components
            .iter()
            .map(|component| factorise(&component.covariance).map(|(_, f)| f))
            .collect()
    }

    /// Log density of one row under the whole mixture.
    pub fn log_density(&self, row: &[f64]) -> f64 {
        match self.factored() {
            Some(factors) => log_sum(&self.components, &factors, row),
            None => f64::NEG_INFINITY,
        }
    }

    /// The posterior weight of every component for one row, summing to one.
    pub fn posterior(&self, row: &[f64]) -> Vec<f64> {
        let k = self.k().max(1);
        let Some(factors) = self.factored() else {
            return vec![1.0 / k as f64; k];
        };
        let mut logs: Vec<f64> = self
            .components
            .iter()
            .zip(&factors)
            .map(|(component, factor)| {
                component.weight.max(1e-300).ln() + factor.log_density(row, &component.mean)
            })
            .collect();
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return vec![1.0 / k as f64; k];
        }
        let mut mass = 0.0;
        for value in logs.iter_mut() {
            *value = (*value - top).exp();
            mass += *value;
        }
        if mass > 0.0 {
            for value in logs.iter_mut() {
                *value /= mass;
            }
        }
        logs
    }

    /// The component each row belongs to, by posterior, ties to the lower index.
    pub fn assign(&self, rows: &[Vec<f64>]) -> Vec<usize> {
        let Some(factors) = self.factored() else {
            return vec![0; rows.len()];
        };
        rows.iter()
            .map(|row| {
                let mut best = (0usize, f64::NEG_INFINITY);
                for (id, (component, factor)) in self.components.iter().zip(&factors).enumerate() {
                    let value = component.weight.max(1e-300).ln()
                        + factor.log_density(row, &component.mean);
                    if value > best.1 {
                        best = (id, value);
                    }
                }
                best.0
            })
            .collect()
    }

    /// Total log likelihood of rows this model did not see.
    pub fn total_loglik(&self, rows: &[Vec<f64>]) -> f64 {
        match self.factored() {
            Some(factors) => total(&self.components, &factors, rows),
            None => f64::NEG_INFINITY,
        }
    }

    /// Bayesian information criterion, lower is better.
    pub fn bic(&self, rows: usize, dim: usize) -> f64 {
        let parameters = self.family.parameters(self.k(), dim);
        -2.0 * self.loglik + parameters as f64 * (rows.max(2) as f64).ln()
    }

    /// Mahalanobis distance from a row to one component's centre, in standard
    /// deviations, so `segment assign` still prints a readable gap.
    pub fn distance(&self, row: &[f64], component: usize) -> f64 {
        let Some(component) = self.components.get(component) else {
            return f64::INFINITY;
        };
        let Some((_, factor)) = factorise(&component.covariance) else {
            return f64::INFINITY;
        };
        let residual: Vec<f64> = row
            .iter()
            .zip(&component.mean)
            .map(|(x, m)| x - m)
            .collect();
        let solved = linalg::cholesky_solve(&factor.chol, &residual);
        let maha: f64 = solved.iter().zip(&residual).map(|(a, b)| a * b).sum();
        maha.max(0.0).sqrt()
    }
}

/// The component a row belongs to and its Mahalanobis gap, factored once.
///
/// `Artifact::classify` calls this so placing an entity reads the same belief the
/// fit was written with, rather than a Euclidean distance to a mean.
pub fn classify(components: &[Component], row: &[f64]) -> (usize, f64) {
    let mut factors = Vec::with_capacity(components.len());
    for component in components {
        match factorise(&component.covariance) {
            Some((_, factor)) => factors.push(factor),
            None => return (0, f64::INFINITY),
        }
    }
    let mut best = (0usize, f64::NEG_INFINITY);
    for (id, (component, factor)) in components.iter().zip(&factors).enumerate() {
        let value = component.weight.max(1e-300).ln() + factor.log_density(row, &component.mean);
        if value > best.1 {
            best = (id, value);
        }
    }
    let residual: Vec<f64> = row
        .iter()
        .zip(&components[best.0].mean)
        .map(|(x, m)| x - m)
        .collect();
    let solved = linalg::cholesky_solve(&factors[best.0].chol, &residual);
    let maha: f64 = solved.iter().zip(&residual).map(|(a, b)| a * b).sum();
    (best.0, maha.max(0.0).sqrt())
}

fn log_sum(components: &[Component], factors: &[Factor], row: &[f64]) -> f64 {
    let logs: Vec<f64> = components
        .iter()
        .zip(factors)
        .map(|(component, factor)| {
            component.weight.max(1e-300).ln() + factor.log_density(row, &component.mean)
        })
        .collect();
    let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return f64::NEG_INFINITY;
    }
    top + logs
        .iter()
        .map(|value| (value - top).exp())
        .sum::<f64>()
        .ln()
}

fn total(components: &[Component], factors: &[Factor], rows: &[Vec<f64>]) -> f64 {
    if components.is_empty() || factors.len() != components.len() {
        return f64::NEG_INFINITY;
    }
    rows.iter()
        .map(|row| log_sum(components, factors, row))
        .sum()
}

fn one_hot(labels: &[usize], k: usize) -> Vec<Vec<f64>> {
    let mut resp = vec![vec![0.0; k]; labels.len()];
    for (i, &label) in labels.iter().enumerate() {
        resp[i][label.min(k - 1)] = 1.0;
    }
    resp
}

/// A uniform draw on the simplex for every row: `k` exponentials, normalised.
///
/// This is the start that can leave the split k-means already made, which is the
/// whole reason it is here. The generator is the fit's own, so it reproduces.
fn soft_start(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::stream(seed, 0x51ed_270b);
    let mut resp = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row: Vec<f64> = (0..k).map(|_| -rng.open_f64().ln()).collect();
        let total: f64 = row.iter().sum();
        for value in row.iter_mut() {
            *value /= total;
        }
        resp.push(row);
    }
    resp
}

/// The maximisation step: weights, means and the covariance of the family.
///
/// The covariance loops index a square matrix on both axes, which reads closer to
/// the arithmetic than an iterator chain would.
#[allow(clippy::needless_range_loop)]
fn m_step(
    rows: &[Vec<f64>],
    resp: &[Vec<f64>],
    k: usize,
    family: Family,
) -> (Vec<Component>, Vec<Factor>) {
    let n = rows.len();
    let dim = rows[0].len();
    let mut counts = vec![0.0; k];
    for row in resp {
        for (j, value) in row.iter().enumerate() {
            counts[j] += *value;
        }
    }

    // The whole panel, used to reset a component no row claims. Without this an
    // abandoned component divides by a floor and reports a mean off the data.
    let mut centre = vec![0.0; dim];
    for row in rows {
        for d in 0..dim {
            centre[d] += row[d];
        }
    }
    for value in centre.iter_mut() {
        *value /= n as f64;
    }
    let mut spread = vec![FLOOR; dim];
    for row in rows {
        for d in 0..dim {
            let gap = row[d] - centre[d];
            spread[d] += gap * gap / n as f64;
        }
    }

    let mut means = vec![vec![0.0; dim]; k];
    for (i, row) in rows.iter().enumerate() {
        for j in 0..k {
            let weight = resp[i][j];
            if weight == 0.0 {
                continue;
            }
            for d in 0..dim {
                means[j][d] += weight * row[d];
            }
        }
    }
    for j in 0..k {
        if counts[j] < 1e-6 * n as f64 {
            means[j] = centre.clone();
            continue;
        }
        let count = counts[j];
        for d in 0..dim {
            means[j][d] /= count;
        }
    }

    // One width shared by every group, which is exactly what k-means assumes.
    let shared = if family == Family::Spherical {
        let mut sum = 0.0;
        for (i, row) in rows.iter().enumerate() {
            for j in 0..k {
                let weight = resp[i][j];
                if weight != 0.0 {
                    sum += weight * kmeans::dist2(row, &means[j]);
                }
            }
        }
        (sum / (n as f64 * dim as f64)).max(FLOOR)
    } else {
        FLOOR
    };

    let mut components = Vec::with_capacity(k);
    let mut factors = Vec::with_capacity(k);
    for j in 0..k {
        if counts[j] < 1e-6 * n as f64 {
            let mut covariance = vec![vec![0.0; dim]; dim];
            for d in 0..dim {
                covariance[d][d] = spread[d].max(FLOOR);
            }
            let Some((floored, factor)) = factorise(&covariance) else {
                return (Vec::new(), Vec::new());
            };
            components.push(Component {
                weight: 0.0,
                mean: centre.clone(),
                covariance: floored,
            });
            factors.push(factor);
            continue;
        }
        let count = counts[j];
        let mut covariance = vec![vec![0.0; dim]; dim];
        match family {
            Family::Spherical => {
                for (d, row) in covariance.iter_mut().enumerate() {
                    row[d] = shared;
                }
            }
            Family::Diagonal => {
                for (i, row) in rows.iter().enumerate() {
                    let weight = resp[i][j];
                    if weight == 0.0 {
                        continue;
                    }
                    for d in 0..dim {
                        let gap = row[d] - means[j][d];
                        covariance[d][d] += weight * gap * gap;
                    }
                }
                for d in 0..dim {
                    covariance[d][d] = (covariance[d][d] / count).max(FLOOR);
                }
            }
            Family::Full => {
                for (i, row) in rows.iter().enumerate() {
                    let weight = resp[i][j];
                    if weight == 0.0 {
                        continue;
                    }
                    let gaps: Vec<f64> = (0..dim).map(|d| row[d] - means[j][d]).collect();
                    for a in 0..dim {
                        for b in a..dim {
                            covariance[a][b] += weight * gaps[a] * gaps[b];
                        }
                    }
                }
                for a in 0..dim {
                    for b in a..dim {
                        let value = covariance[a][b] / count;
                        covariance[a][b] = value;
                        covariance[b][a] = value;
                    }
                    covariance[a][a] = covariance[a][a].max(FLOOR);
                }
            }
        }
        let Some((floored, factor)) = factorise(&covariance) else {
            return (Vec::new(), Vec::new());
        };
        components.push(Component {
            weight: counts[j] / n as f64,
            mean: means[j].clone(),
            covariance: floored,
        });
        factors.push(factor);
    }
    (components, factors)
}

/// Expectation maximisation from one starting assignment.
fn em(
    rows: &[Vec<f64>],
    k: usize,
    family: Family,
    iterations: usize,
    start: &[Vec<f64>],
) -> Mixture {
    let n = rows.len();
    let mut resp = start.to_vec();
    let mut state = m_step(rows, &resp, k, family);
    let mut previous = f64::NEG_INFINITY;
    let mut used = 0;
    for step in 0..iterations.max(1) {
        if state.0.is_empty() {
            break;
        }
        let mut total_ll = 0.0;
        for (i, row) in rows.iter().enumerate() {
            let logs: Vec<f64> = state
                .0
                .iter()
                .zip(&state.1)
                .map(|(component, factor)| {
                    component.weight.max(1e-300).ln() + factor.log_density(row, &component.mean)
                })
                .collect();
            let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if !top.is_finite() {
                continue;
            }
            let mass: f64 = logs.iter().map(|value| (value - top).exp()).sum();
            total_ll += top + mass.ln();
            for j in 0..k {
                resp[i][j] = (logs[j] - top).exp() / mass.max(1e-300);
            }
        }
        used = step + 1;
        if (total_ll - previous).abs() <= 1e-8 * n as f64 {
            break;
        }
        previous = total_ll;
        let next = m_step(rows, &resp, k, family);
        if next.0.is_empty() {
            break;
        }
        state = next;
    }
    let loglik = total(&state.0, &state.1, rows);
    Mixture {
        family,
        components: state.0,
        loglik,
        iterations: used,
    }
}

#[cfg(test)]
mod tests;
