//! The two-part reading of a share that piles up on a boundary.
//!
//! Two of the eight readings are shares, and both pile up on an edge: 151 of 251
//! advertisers repeat no creative at all, and 161 run every ad from one
//! fingerprint. A Gaussian has no way to describe a pile: it can only mimic one
//! with a component of almost no width, and then the reading is a density where
//! the data are a probability. The card used to report that imitation as a z.
//!
//! A gate states the pile as what it is. For each group the share is either at
//! the boundary, with probability `mass`, or inside, with the readings of that
//! advertiser beside it, and the inside part is a normal whose mean moves with
//! the other readings. That is the classical two-part model of a semicontinuous
//! variable, and it is fitted here on the responsibilities of the chosen
//! partition, after the partition is chosen.
//!
//! Two consequences are worth stating before the code.
//!
//! The gate is not inside the expectation maximisation loop, so it does not move
//! a group's boundary. The reason is that a likelihood with an atom and a
//! likelihood without one are densities against different measures, and this
//! repository chooses its partition on held-out likelihood. Comparing the two
//! directly is not a comparison: on an atom row the gate reports a probability
//! and the Gaussian reports a density, and at this panel's spacing the density
//! times the spacing exceeds one, which is how a continuous model "wins" a
//! lattice it cannot represent. What is comparable is the predictive probability
//! each model gives the observed value on the lattice the share actually lives
//! on, and [`audit`] measures exactly that, paired by fold.
//!
//! The clip is not applied to a gated column. A share with a boundary at 1 and a
//! small spread would otherwise have its atom clipped into the interior, which
//! would make the gate's boundary value meaningless. On this panel the clip never
//! fired on either share (it fires on ad volume, cadence, burstiness, spend and
//! terms), so this is a guard rather than a repair, and the card says which
//! readings it does fire on.

use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::num::round4;
use crate::stats::normal_cdf;

use super::feature::FeatureSpec;
use super::mixture::Component;
use super::SegmentConfig;

const LOG_2PI: f64 = 1.837_877_066_409_345_3;

/// Width below which an interior part is treated as a spike rather than a spread.
const FLOOR: f64 = 1e-3;

/// How close to the boundary a reading has to be to count as on it.
const TOUCH: f64 = 1e-9;

/// A reading declared to be a share that may pile up on a boundary.
///
/// This is data, not code: the feature table says which readings are shares and
/// where the edge is, and the fit decides from the panel whether the pile is
/// large enough to be worth a gate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateSpec {
    pub name: String,
    pub column: usize,
    /// The edge in plain units, such as 0 for a share of nothing.
    pub boundary: f64,
    /// Where that edge sits in z, which is where the point mass is.
    pub atom: f64,
    /// Share of the panel observed on the edge.
    pub share: f64,
}

impl GateSpec {
    /// Whether one row sits on the edge.
    pub fn holds(&self, row: &[f64]) -> bool {
        row.get(self.column)
            .map(|x| (x - self.atom).abs() <= TOUCH)
            .unwrap_or(false)
    }
}

/// A fitted gate: the pile, and what the inside looks like in each group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    pub name: String,
    pub column: usize,
    /// The edge in plain units and in z, carried so a reader can rebuild both.
    pub boundary: f64,
    pub atom: f64,
    /// Share of the panel on the edge, before any grouping.
    pub share: f64,
    /// Columns the interior part is regressed on: every reading the gates do
    /// not hold as a point mass.
    pub columns: Vec<usize>,
    /// Panel means of those columns, so the centre below is at a stated place.
    pub origin: Vec<f64>,
    /// Per group: probability of the edge.
    pub mass: Vec<f64>,
    /// Per group: mean of the inside at the panel centre.
    pub centre: Vec<f64>,
    /// Per group: how the inside moves with the other readings.
    pub slope: Vec<Vec<f64>>,
    /// Per group: variance of the inside, in z squared.
    pub variance: Vec<f64>,
}

impl Gate {
    /// Probability of the edge in one group.
    pub fn mass(&self, component: usize) -> f64 {
        self.mass.get(component).copied().unwrap_or(0.0)
    }

    /// Mean of the inside part in one group, at a row's other readings.
    pub fn inside_mean(&self, component: usize, row: &[f64]) -> f64 {
        let centre = self.centre.get(component).copied().unwrap_or(0.0);
        let slope = self.slope.get(component);
        let mut mean = centre;
        if let Some(slope) = slope {
            for (k, column) in self.columns.iter().enumerate() {
                if let (Some(beta), Some(x), Some(origin)) =
                    (slope.get(k), row.get(*column), self.origin.get(k))
                {
                    mean += beta * (x - origin);
                }
            }
        }
        mean
    }

    /// The log of this gate's contribution under one group: a probability on the
    /// edge, a density times the inside's normal weight otherwise.
    pub fn log_term(&self, component: usize, row: &[f64]) -> f64 {
        let mass = self.mass(component).clamp(1e-12, 1.0 - 1e-12);
        if self.holds(row) {
            return mass.ln();
        }
        let variance = self
            .variance
            .get(component)
            .copied()
            .unwrap_or(1.0)
            .max(FLOOR);
        let mean = self.inside_mean(component, row);
        let gap = row.get(self.column).copied().unwrap_or(mean) - mean;
        (1.0 - mass).ln() - 0.5 * (gap * gap / variance + variance.ln() + LOG_2PI)
    }

    fn holds(&self, row: &[f64]) -> bool {
        row.get(self.column)
            .map(|x| (x - self.atom).abs() <= TOUCH)
            .unwrap_or(false)
    }
}

/// Detect the shares whose pile is large enough to be modelled as a gate.
///
/// The table declares which readings are shares and where their edge is; the
/// panel decides whether the pile is real. A declared share with fewer than
/// `min_share` of its rows on the edge is left to the Gaussian, and the card
/// prints the share so the decision is visible either way.
pub fn detect(
    specs: &[FeatureSpec],
    rows: &[Vec<f64>],
    mean: &[f64],
    sd: &[f64],
    min_share: f64,
) -> Vec<GateSpec> {
    let n = rows.len().max(1);
    let mut out = Vec::new();
    for (column, spec) in specs.iter().enumerate() {
        let Some(boundary) = spec.boundary else {
            continue;
        };
        let edge = spec.apply(boundary);
        let width = sd.get(column).copied().unwrap_or(0.0);
        if width <= 1e-9 {
            continue;
        }
        let hits = rows
            .iter()
            .filter(|row| {
                row.get(column)
                    .map(|x| (x - edge).abs() <= TOUCH)
                    .unwrap_or(false)
            })
            .count();
        let share = hits as f64 / n as f64;
        if share < min_share {
            continue;
        }
        out.push(GateSpec {
            name: spec.name.clone(),
            column,
            boundary,
            atom: (edge - mean.get(column).copied().unwrap_or(0.0)) / width,
            share,
        });
    }
    out
}

/// Fit one gate per spec against a partition's responsibilities.
///
/// The edge probability and the inside are read from the rows a group claims,
/// weighted by how strongly it claims them. The inside is a weighted least
/// squares of the share on the readings the gates do not hold, with the
/// predictors centred on the panel so the intercept reads as a mean at a stated
/// place rather than at an arbitrary origin.
pub fn bars(
    rows: &[Vec<f64>],
    resp: &[Vec<f64>],
    specs: &[GateSpec],
    columns: &[usize],
    origin: &[f64],
) -> Vec<Gate> {
    let k = resp.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let width = columns.len();
        let mut mass = vec![0.0; k];
        let mut centre = vec![0.0; k];
        let mut slope = vec![vec![0.0; width]; k];
        let mut variance = vec![1.0; k];
        for g in 0..k {
            let weights: Vec<f64> = resp
                .iter()
                .map(|r| r.get(g).copied().unwrap_or(0.0))
                .collect();
            let total: f64 = weights.iter().sum();
            let edge: f64 = rows
                .iter()
                .zip(&weights)
                .filter(|(row, _)| spec.holds(row))
                .map(|(_, w)| *w)
                .sum();
            mass[g] = if total > 0.0 {
                (edge / total).clamp(1e-4, 1.0 - 1e-4)
            } else {
                spec.share.clamp(1e-4, 1.0 - 1e-4)
            };
            let inside: Vec<usize> = (0..rows.len())
                .filter(|&i| !spec.holds(&rows[i]) && weights[i] > 1e-8)
                .collect();
            if inside.len() < width + 3 {
                // Too few inside rows to separate a slope from noise: the mean
                // of the inside is reported and the slope stays at zero.
                let (sum, wsum) = inside.iter().fold((0.0, 0.0), |(s, w), &i| {
                    (s + weights[i] * rows[i][spec.column], w + weights[i])
                });
                centre[g] = if wsum > 0.0 { sum / wsum } else { spec.atom };
                continue;
            }
            let (fit, var) = weighted(rows, &weights, spec.column, columns, origin, &inside);
            centre[g] = fit[0];
            slope[g] = fit[1..].to_vec();
            variance[g] = var;
        }
        out.push(Gate {
            name: spec.name.clone(),
            column: spec.column,
            boundary: spec.boundary,
            atom: spec.atom,
            share: spec.share,
            columns: columns.to_vec(),
            origin: origin.to_vec(),
            mass,
            centre,
            slope,
            variance,
        });
    }
    out
}

/// One weighted least squares fit of a gated column, with the standard error of
/// every coefficient.
///
/// The standard errors are what a group's slope is judged by. A slope fitted on
/// thirty five rows carries a wider one than a slope fitted on ninety, and a model
/// that wants to pool the two has to know which is which. The covariance of the
/// estimate is `sigma^2 (X'WX + ridge I)^-1`, so the diagonal of that inverse,
/// solved for one unit vector per coefficient, is the variance of each beta.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Inside {
    pub beta: Vec<f64>,
    pub variance: f64,
    pub se: Vec<f64>,
}

/// Weighted least squares of one column on the others, plus its residual variance.
///
/// Shared with [`super::pattern`], which nests the same regression inside the
/// mixture rather than fitting it after the partition is chosen.
pub(super) fn weighted(
    rows: &[Vec<f64>],
    weights: &[f64],
    column: usize,
    columns: &[usize],
    origin: &[f64],
    inside: &[usize],
) -> (Vec<f64>, f64) {
    match fit_inside(rows, weights, column, columns, origin, inside) {
        Some(fit) => (fit.beta, fit.variance),
        None => (vec![0.0; columns.len() + 1], 1.0),
    }
}

/// The same fit as [`weighted`], with the standard errors a pooling decision needs.
///
/// Returns `None` when the system cannot be solved at all, which keeps the
/// caller's fallback identical to the one [`weighted`] has always used.
pub(super) fn fit_inside(
    rows: &[Vec<f64>],
    weights: &[f64],
    column: usize,
    columns: &[usize],
    origin: &[f64],
    inside: &[usize],
) -> Option<Inside> {
    let p = columns.len() + 1;
    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    let mut wsum = 0.0;
    for &i in inside {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        let mut design = Vec::with_capacity(p);
        design.push(1.0);
        for (k, c) in columns.iter().enumerate() {
            design.push(rows[i][*c] - origin.get(k).copied().unwrap_or(0.0));
        }
        let y = rows[i][column];
        for a in 0..p {
            rhs[a] += w * design[a] * y;
            for b in a..p {
                gram[a][b] += w * design[a] * design[b];
            }
        }
        wsum += w;
    }
    for a in 1..p {
        let (above, below) = gram.split_at_mut(a);
        for (b, row) in above.iter().enumerate() {
            below[0][b] = row[a];
        }
    }
    if wsum <= 0.0 {
        return None;
    }
    let ridge = 1e-6 * wsum.max(1.0);
    for (a, row) in gram.iter_mut().enumerate() {
        row[a] += ridge;
    }
    let mut factor = gram;
    linalg::cholesky(&mut factor)?;
    let beta = linalg::cholesky_solve(&factor, &rhs);
    if !beta.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut err = 0.0;
    for &i in inside {
        let w = weights[i];
        let mut predicted = beta[0];
        for (k, c) in columns.iter().enumerate() {
            predicted += beta[k + 1] * (rows[i][*c] - origin.get(k).copied().unwrap_or(0.0));
        }
        let gap = rows[i][column] - predicted;
        err += w * gap * gap;
    }
    let variance = (err / wsum).clamp(FLOOR, 1e6);
    let mut se = vec![0.0; p];
    for (d, slot) in se.iter_mut().enumerate() {
        let mut unit = vec![0.0; p];
        unit[d] = 1.0;
        let inverse = linalg::cholesky_solve(&factor, &unit);
        *slot = (variance * inverse[d]).max(0.0).sqrt();
    }
    Some(Inside { beta, variance, se })
}

/// What each group's gated readings look like, in plain units.
///
/// Every group carries two numbers per gated share: the fraction of its members
/// on the edge, and the mean of the members inside it. That pair is what the
/// card prints instead of a z that a Gaussian spent a component imitating. A
/// group with no members on the edge reports zero for the first, and a group
/// with no members inside it reports its own edge value for the second, because
/// an empty inside has no mean and zero would be a claim.
pub fn segment_shares(
    rows: &[Vec<f64>],
    labels: &[usize],
    k: usize,
    specs: &[GateSpec],
    features: &[FeatureSpec],
    scaling: &super::feature::Scaling,
) -> Vec<Vec<(String, f64, f64)>> {
    let mut out = vec![Vec::new(); k];
    for spec in specs {
        let feature = features.get(spec.column);
        let mu = scaling.mean.get(spec.column).copied().unwrap_or(0.0);
        let sd = scaling.sd.get(spec.column).copied().unwrap_or(1.0);
        // A mean held in z is a mean in the space the fit worked in, so it is
        // read back through the standardisation before the transform.
        let plain = |z: f64| match feature {
            Some(feature) => feature.invert(z * sd + mu),
            None => z * sd + mu,
        };
        for (cluster, list) in out.iter_mut().enumerate() {
            let members: Vec<usize> = (0..rows.len()).filter(|&i| labels[i] == cluster).collect();
            if members.is_empty() {
                continue;
            }
            let edge = members.iter().filter(|&&i| spec.holds(&rows[i])).count() as f64
                / members.len() as f64;
            let inside: Vec<f64> = members
                .iter()
                .filter(|&&i| !spec.holds(&rows[i]))
                .map(|&i| rows[i][spec.column])
                .collect();
            let inside_mean = if inside.is_empty() {
                spec.boundary
            } else {
                plain(inside.iter().sum::<f64>() / inside.len() as f64)
            };
            list.push((spec.name.clone(), edge, inside_mean));
        }
    }
    out
}

/// How well a gate predicts its edge, against the Gaussian it replaces.
///
/// Both sides are scored as the predictive probability of the observed value on
/// the lattice the share lives on, conditioning on the other readings only. That
/// is the same event under both models, which the joint densities are not: on an
/// edge row a gate reports a probability and a Gaussian reports a density, and
/// the two are not the same kind of number. The gate is refit on the training
/// rows of each fold; the partition is the shipped one, so the level is
/// optimistic and the paired difference is the reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateAudit {
    pub name: String,
    /// Mean hold-out log probability of the observed value, under the gate.
    pub log_gate: f64,
    /// The same under the Gaussian conditional the gate replaces.
    pub log_gauss: f64,
    /// Brier score of the edge probability, and of the panel rate beside it.
    pub brier_gate: f64,
    pub brier_gauss: f64,
    /// Share of hold-out rows whose edge state is called correctly.
    pub accuracy: f64,
    /// The panel rate, which is what a gate that predicts nothing scores.
    pub rate: f64,
    pub folds: usize,
}

impl GateAudit {
    /// The mean gain per row, in nats, of the gate over the Gaussian.
    pub fn gain(&self) -> f64 {
        self.log_gate - self.log_gauss
    }
}

/// What each of the two models puts on one row's observed value, and on the edge.
///
/// `gate` and `gauss` are probabilities of the same event, the observed value of
/// the gated reading given the readings the gates do not hold, which is what
/// makes them comparable. `edge_gate` and `edge_gauss` are the probability of
/// the edge state itself, which is what a Brier score and a hit rate are read
/// from.
pub(super) struct Cell {
    pub gate: f64,
    pub gauss: f64,
    pub edge_gate: f64,
    pub edge_gauss: f64,
}

/// Score one row under a fitted gate and the Gaussian conditional it replaces.
#[allow(clippy::too_many_arguments)]
pub(super) fn scored(
    row: &[f64],
    spacing: f64,
    spec: &GateSpec,
    fit: &Gate,
    components: &[Component],
    resp: &[f64],
    columns: &[usize],
    scaling: &super::feature::Scaling,
) -> Cell {
    let on = spec.holds(row);
    let mut out = Cell {
        gate: 0.0,
        gauss: 0.0,
        edge_gate: 0.0,
        edge_gauss: 0.0,
    };
    for (g_id, component) in components.iter().enumerate() {
        let post = resp.get(g_id).copied().unwrap_or(0.0);
        let Some((mean, sd)) = conditional(component, columns, spec.column, row) else {
            continue;
        };
        let cell = cell_probability(row[spec.column], mean, sd, spec, scaling, spacing);
        let mass = fit.mass(g_id);
        let inside_sd = fit
            .variance
            .get(g_id)
            .copied()
            .unwrap_or(1.0)
            .max(FLOOR)
            .sqrt();
        let inside_cell = cell_probability(
            row[spec.column],
            fit.inside_mean(g_id, row),
            inside_sd,
            spec,
            scaling,
            spacing,
        );
        out.gate += post * if on { mass } else { (1.0 - mass) * inside_cell };
        out.gauss += post * cell;
        out.edge_gate += post * mass;
        out.edge_gauss += post * cell_probability(spec.atom, mean, sd, spec, scaling, spacing);
    }
    out
}

/// Score every gate on held-out folds against the Gaussian it replaces.
pub fn audit(
    rows: &[Vec<f64>],
    ads: &[f64],
    components: &[Component],
    specs: &[GateSpec],
    scaling: &super::feature::Scaling,
    cfg: &SegmentConfig,
) -> Vec<GateAudit> {
    if components.is_empty() || specs.is_empty() || rows.is_empty() {
        return Vec::new();
    }
    let columns: Vec<usize> = (0..rows[0].len())
        .filter(|c| !specs.iter().any(|s| s.column == *c))
        .collect();
    let origin: Vec<f64> = columns
        .iter()
        .map(|c| rows.iter().map(|r| r[*c]).sum::<f64>() / rows.len() as f64)
        .collect();
    let folds = super::select::folds_for(rows.len(), cfg);
    // One reading of which group a row belongs to, given only the readings the
    // gates do not hold. The partition is the shipped one on both sides.
    let resp = responsibilities(rows, components, &columns);
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut log_gate = Vec::new();
        let mut log_gauss = Vec::new();
        let mut brier_gate = Vec::new();
        let mut brier_gauss = Vec::new();
        let mut accuracy = Vec::new();
        let mut rate = Vec::new();
        for fold in &folds {
            let train: Vec<usize> = (0..rows.len()).filter(|i| !fold.contains(i)).collect();
            let held: Vec<usize> = fold.clone();
            if train.is_empty() || held.is_empty() {
                continue;
            }
            let bars = bars(
                &train.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>(),
                &train.iter().map(|&i| resp[i].clone()).collect::<Vec<_>>(),
                std::slice::from_ref(spec),
                &columns,
                &origin,
            );
            let Some(gate) = bars.first() else {
                continue;
            };
            rate.push(
                held.iter().filter(|&&i| spec.holds(&rows[i])).count() as f64 / held.len() as f64,
            );
            let mut g = Vec::new();
            let mut c = Vec::new();
            let mut bg = Vec::new();
            let mut bc = Vec::new();
            let mut hit = 0.0;
            for &i in &held {
                let row = &rows[i];
                let spacing = lattice(ads.get(i).copied().unwrap_or(1.0), spec, scaling);
                let on = spec.holds(row);
                let scored = scored(
                    row, spacing, spec, gate, components, &resp[i], &columns, scaling,
                );
                let truth = if on { 1.0 } else { 0.0 };
                g.push(scored.gate.max(1e-12).ln());
                c.push(scored.gauss.max(1e-12).ln());
                bg.push((scored.edge_gate - truth) * (scored.edge_gate - truth));
                bc.push((scored.edge_gauss - truth) * (scored.edge_gauss - truth));
                if (scored.edge_gate > 0.5) == on {
                    hit += 1.0;
                }
            }
            log_gate.push(mean(&g));
            log_gauss.push(mean(&c));
            brier_gate.push(mean(&bg));
            brier_gauss.push(mean(&bc));
            accuracy.push(hit / held.len() as f64);
        }
        out.push(GateAudit {
            name: spec.name.clone(),
            log_gate: round4(mean(&log_gate)),
            log_gauss: round4(mean(&log_gauss)),
            brier_gate: round4(mean(&brier_gate)),
            brier_gauss: round4(mean(&brier_gauss)),
            accuracy: round4(mean(&accuracy)),
            rate: round4(mean(&rate)),
            folds: log_gate.len(),
        });
    }
    out
}

/// The probability the model puts on the lattice cell around an observed value.
///
/// The cell is integrated rather than approximated by density times spacing: at
/// this panel's spacing the approximation reports mass above one, and a model
/// cannot be credited with a probability it cannot represent.
pub(super) fn cell_probability(
    value: f64,
    mean: f64,
    sd: f64,
    spec: &GateSpec,
    scaling: &super::feature::Scaling,
    spacing: f64,
) -> f64 {
    let width = scaling
        .sd
        .get(spec.column)
        .copied()
        .unwrap_or(1.0)
        .max(1e-9);
    let low = (0.0 - scaling.mean.get(spec.column).copied().unwrap_or(0.0)) / width;
    let high = (1.0 - scaling.mean.get(spec.column).copied().unwrap_or(0.0)) / width;
    let sd = sd.max(FLOOR.sqrt());
    let left = (value - 0.5 * spacing).max(low);
    let right = (value + 0.5 * spacing).min(high);
    let mass = normal_cdf((right - mean) / sd) - normal_cdf((left - mean) / sd);
    let support = normal_cdf((high - mean) / sd) - normal_cdf((low - mean) / sd);
    (mass / support.max(1e-12)).clamp(1e-12, 1.0)
}

/// The spacing of the lattice a share can take, in z units.
pub(super) fn lattice(ads: f64, spec: &GateSpec, scaling: &super::feature::Scaling) -> f64 {
    let width = scaling
        .sd
        .get(spec.column)
        .copied()
        .unwrap_or(1.0)
        .max(1e-9);
    (1.0 / ads.max(1.0)) / width
}

/// Component posterior given the readings the gates do not hold.
pub(super) fn responsibilities(
    rows: &[Vec<f64>],
    components: &[Component],
    columns: &[usize],
) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let logs: Vec<f64> = components
                .iter()
                .map(|component| {
                    let weight = component.weight.max(1e-300).ln();
                    weight + marginal(component, columns, row).unwrap_or(f64::NEG_INFINITY)
                })
                .collect();
            soften(logs)
        })
        .collect()
}

pub(super) fn soften(logs: Vec<f64>) -> Vec<f64> {
    let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return vec![1.0 / logs.len().max(1) as f64; logs.len()];
    }
    let mass: f64 = logs.iter().map(|v| (v - top).exp()).sum();
    logs.iter().map(|v| (v - top).exp() / mass).collect()
}

/// One component's density over a subset of its readings, factorised once.
///
/// [`super::pattern`] reads the same marginal for every row that shows the same
/// readings, so the factorisation is worth keeping: a fold model factorises one
/// matrix per component per distinct pattern rather than one per row.
pub(super) struct Marginal {
    columns: Vec<usize>,
    chol: Vec<Vec<f64>>,
    logdet: f64,
    mean: Vec<f64>,
}

impl Marginal {
    pub(super) fn build(component: &Component, columns: &[usize]) -> Option<Self> {
        let dim = columns.len();
        let mut chol = vec![vec![0.0; dim]; dim];
        for (a, &ca) in columns.iter().enumerate() {
            for (b, &cb) in columns.iter().enumerate() {
                chol[a][b] = *component.covariance.get(ca)?.get(cb)?;
                if a == b {
                    chol[a][b] += FLOOR;
                }
            }
        }
        linalg::cholesky(&mut chol)?;
        let logdet = 2.0 * (0..dim).map(|d| chol[d][d].ln()).sum::<f64>();
        if !logdet.is_finite() {
            return None;
        }
        Some(Self {
            columns: columns.to_vec(),
            chol,
            logdet,
            mean: columns
                .iter()
                .map(|&c| component.mean.get(c).copied().unwrap_or(0.0))
                .collect(),
        })
    }

    /// Log density of the readings in this marginal, with the weight left out.
    pub(super) fn log_density(&self, row: &[f64]) -> f64 {
        let residual: Vec<f64> = self
            .columns
            .iter()
            .zip(&self.mean)
            .map(|(&c, mean)| row.get(c).copied().unwrap_or(0.0) - mean)
            .collect();
        let solved = linalg::cholesky_solve(&self.chol, &residual);
        let maha: f64 = solved.iter().zip(&residual).map(|(a, b)| a * b).sum();
        -0.5 * (maha + self.logdet + self.columns.len() as f64 * LOG_2PI)
    }
}

/// The Gaussian density over a subset of the readings, with the weight left out.
fn marginal(component: &Component, columns: &[usize], row: &[f64]) -> Option<f64> {
    Marginal::build(component, columns).map(|marginal| marginal.log_density(row))
}

/// The conditional mean and spread of one reading given the others, in one group.
pub(super) fn conditional(
    component: &Component,
    columns: &[usize],
    target: usize,
    row: &[f64],
) -> Option<(f64, f64)> {
    let dim = columns.len();
    let mut cov = vec![vec![0.0; dim]; dim];
    for (a, &ca) in columns.iter().enumerate() {
        for (b, &cb) in columns.iter().enumerate() {
            cov[a][b] = *component.covariance.get(ca)?.get(cb)?;
            if a == b {
                cov[a][b] += FLOOR;
            }
        }
    }
    let mut chol = cov;
    linalg::cholesky(&mut chol)?;
    let mut cross: Vec<f64> = Vec::with_capacity(dim);
    for column in columns {
        cross.push(*component.covariance.get(*column)?.get(target)?);
    }
    let solved = linalg::cholesky_solve(&chol, &cross);
    let residual: Vec<f64> = columns
        .iter()
        .map(|&c| {
            row.get(c).copied().unwrap_or(0.0) - component.mean.get(c).copied().unwrap_or(0.0)
        })
        .collect();
    let shift: f64 = solved.iter().zip(&residual).map(|(a, b)| a * b).sum();
    let mean = component.mean.get(target).copied().unwrap_or(0.0) + shift;
    let variance = component
        .covariance
        .get(target)
        .and_then(|r| r.get(target))
        .copied()
        .unwrap_or(1.0)
        - solved.iter().zip(&cross).map(|(a, b)| a * b).sum::<f64>();
    Some((mean, variance.max(FLOOR).sqrt()))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::feature::Transform;

    fn feature(name: &str, boundary: Option<f64>) -> FeatureSpec {
        FeatureSpec {
            name: name.into(),
            label: name.into(),
            family: "creative".into(),
            unit: "share".into(),
            transform: Transform::Identity,
            description: "a share".into(),
            boundary,
        }
    }

    #[test]
    fn a_share_with_a_large_pile_is_detected_and_a_small_one_is_not() {
        let table = vec![
            feature("repeat", Some(0.0)),
            feature("concentrate", Some(1.0)),
        ];
        let mut rows = Vec::new();
        for i in 0..20 {
            let edge = if i < 15 {
                0.0
            } else {
                0.25 * (i as f64 - 14.0)
            };
            // concentrate varies here rather than mirroring the edge, so only
            // one column in the table has a pile to find.
            rows.push(vec![edge, 0.2 + 0.01 * i as f64]);
        }
        let mean = vec![0.1, 0.3];
        let sd = vec![0.2, 0.1];
        let found = detect(&table, &rows, &mean, &sd, 0.5);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "repeat");
        assert!((found[0].share - 0.75).abs() < 1e-9);
        assert!((found[0].atom + 0.5).abs() < 1e-9);
        assert!(detect(&table, &rows, &mean, &sd, 0.9).is_empty());
    }

    #[test]
    fn a_gate_reads_a_pile_as_a_probability_not_a_density() {
        let spec = GateSpec {
            name: "repeat".into(),
            column: 0,
            boundary: 0.0,
            atom: -0.5,
            share: 0.75,
        };
        let rows = vec![
            vec![-0.5, 0.2],
            vec![-0.5, 0.4],
            vec![0.6, 0.1],
            vec![0.9, 0.3],
        ];
        let resp: Vec<Vec<f64>> = rows.iter().map(|_| vec![1.0]).collect();
        let gates = bars(&rows, &resp, std::slice::from_ref(&spec), &[1], &[0.25]);
        let gate = &gates[0];
        assert!((gate.mass[0] - 0.5).abs() < 1e-9);
        let on = gate.log_term(0, &rows[0]);
        let off = gate.log_term(0, &rows[2]);
        assert!((on - 0.5f64.ln()).abs() < 1e-9);
        assert!(off < 0.0);
        assert!(on > off - 2.0);
    }

    #[test]
    fn the_inside_mean_moves_with_the_readings_beside_it() {
        let spec = GateSpec {
            name: "repeat".into(),
            column: 0,
            boundary: 0.0,
            atom: -0.5,
            share: 0.5,
        };
        // inside values climb with column 1, so the slope should be positive
        let rows = vec![
            vec![-0.5, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![0.4, 0.3],
            vec![0.6, 0.4],
            vec![0.8, 0.5],
        ];
        let resp: Vec<Vec<f64>> = rows.iter().map(|_| vec![1.0]).collect();
        let gates = bars(&rows, &resp, std::slice::from_ref(&spec), &[1], &[0.25]);
        assert!(gates[0].slope[0][0] > 0.9);
        assert!(gates[0].inside_mean(0, &[0.5, 0.1]) < gates[0].inside_mean(0, &[0.5, 0.5]));
    }

    #[test]
    fn a_group_on_the_edge_is_described_by_its_share_not_by_a_mean() {
        let specs = vec![GateSpec {
            name: "repeat".into(),
            column: 0,
            boundary: 0.0,
            atom: -0.8,
            share: 0.5,
        }];
        let features = vec![feature("repeat", Some(0.0))];
        let scaling = super::super::feature::Scaling {
            mean: vec![0.4],
            sd: vec![0.5],
            clip: 3.0,
            clip_exempt: vec![0],
        };
        // members 0 and 1 are on the edge, member 2 sits inside at 0.4
        let rows = vec![vec![-0.8], vec![-0.8], vec![0.0]];
        let labels = vec![0, 0, 1];
        let shares = segment_shares(&rows, &labels, 2, &specs, &features, &scaling);
        assert!((shares[0][0].1 - 1.0).abs() < 1e-9);
        assert!((shares[0][0].2 - 0.0).abs() < 1e-9);
        assert!((shares[1][0].1 - 0.0).abs() < 1e-9);
        // z of 0 is a plain share of 0.4, not 0 and not a z
        assert!((shares[1][0].2 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn the_audit_reads_the_gate_ahead_of_the_gaussian_on_a_pile() {
        // Two groups told apart by the first column; the share is an atom on
        // half the panel and spread above it on the other half.
        let mut rows = Vec::new();
        let mut ads = Vec::new();
        for i in 0..120 {
            // the left group is on the edge nine times in ten, the right group
            // is inside it nine times in ten, so a gate can say which
            let side = if i % 2 == 0 { -1.2 } else { 1.2 };
            let jitter = 0.05 * (i % 5) as f64;
            let share = if i % 2 == 0 {
                if i % 20 == 0 {
                    0.30 + 0.05 * (i % 7) as f64
                } else {
                    0.0
                }
            } else if i % 20 == 1 {
                0.0
            } else {
                0.30 + 0.05 * (i % 7) as f64
            };
            rows.push(vec![share, side + jitter]);
            ads.push(8.0);
        }
        let table = vec![feature("repeat", Some(0.0))];
        let cfg = SegmentConfig {
            folds: 4,
            ..SegmentConfig::default()
        };
        // rows are already in z, so the scaling is the identity
        let mean = vec![0.0, 0.0];
        let sd = vec![1.0, 1.0];
        let found = detect(&table, &rows, &mean, &sd, 0.2);
        assert_eq!(found.len(), 1);
        let components = super::super::mixture::Mixture::fit(
            &rows,
            2,
            super::super::mixture::Family::Full,
            2,
            120,
            7,
        )
        .components;
        let scaling = super::super::feature::Scaling {
            mean: mean.clone(),
            sd: sd.clone(),
            clip: 3.0,
            clip_exempt: vec![0],
        };
        let audits = audit(&rows, &ads, &components, &found, &scaling, &cfg);
        assert_eq!(audits.len(), 1);
        assert!(audits[0].gain() > 0.0, "gate {:?}", audits[0]);
        assert!(
            audits[0].accuracy > audits[0].rate + 0.2,
            "gate {:?}",
            audits[0]
        );
    }

    #[test]
    fn a_gated_column_keeps_its_edge_out_of_the_clip() {
        // A share with five per cent on an edge at 1 has its edge at 4.36 z, and
        // the clip would fold that edge into the interior at 3, which is where
        // the gate's atom is supposed to sit.
        let mut rows = Vec::new();
        for i in 0..40 {
            let edge = if i < 2 { 1.0 } else { 0.02 };
            rows.push(vec![edge]);
        }
        let table = vec![feature("concentrate", Some(1.0))];
        let mean = rows.iter().map(|r| r[0]).sum::<f64>() / 40.0;
        let sd = (rows.iter().map(|r| (r[0] - mean).powi(2)).sum::<f64>() / 40.0).sqrt();
        let found = detect(&table, &rows, &[mean], &[sd], 0.04);
        assert_eq!(found.len(), 1);
        assert!(found[0].atom > 3.0, "the edge sits beyond the clip");
        let scaling = super::super::feature::Scaling {
            mean: vec![mean],
            sd: vec![sd],
            clip: 3.0,
            clip_exempt: vec![0],
        };
        assert!(scaling.z(0, 1.0) > 3.0, "a gated edge is not clipped");
        let plain = super::super::feature::Scaling {
            mean: vec![mean],
            sd: vec![sd],
            clip: 3.0,
            clip_exempt: Vec::new(),
        };
        assert!((plain.z(0, 1.0) - 3.0).abs() < 1e-9);
    }
}
