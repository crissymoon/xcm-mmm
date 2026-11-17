//! Descriptive columns in the mixing weights, never in the density.
//!
//! The feature table holds behaviour on purpose. Recency is the obvious
//! omission, and the reason it was omitted still holds: the panel is a snapshot
//! weighted toward recent ads, so the distance from an advertiser's last start
//! measures the archive's coverage as much as the advertiser's activity. Putting
//! that distance into the density would let a sampling artefact set the width of
//! a group and would make every profile a statement about the snapshot.
//!
//! There is a second place a mixture can use a column, and it is a weaker claim.
//! The mixing weights answer "which group does this row come from", not "what is
//! this group like". A gating model writes that answer as `logit(pi_k(x))`, a
//! function of descriptive columns, and leaves the densities on behaviour
//! (Dayton and Macready, *JASA* 83, 1988; Jacobs et al., *Neural Computation* 3,
//! 1991). A recency effect in the weights says the archive over-represents recent
//! advertisers *in some groups*; it does not say recency is what a group is.
//!
//! This module implements that model and, more importantly, the audit that
//! decides whether this panel supports it. The acceptance was fixed before the
//! fit was run: the gated weights must add held-out likelihood, and no component
//! mean may move more than a tenth of a standard deviation. The second condition
//! is what keeps the item honest. If the means move, the partition has changed
//! and the groups mean something else, which is a decision for a reader rather
//! than a gain to be banked.
//!
//! Three implementation choices carry the same reasoning:
//!
//! * The gate is a multinomial logistic regression on soft responsibilities,
//!   fitted by damped Fisher scoring with a ridge on every non-intercept
//!   coefficient. Soft targets are what expectation maximisation hands it, and
//!   the ridge is what keeps four coefficient vectors identifiable when a
//!   covariate separates two groups almost perfectly.
//! * The gated loop reuses [`Mixture::step`], which does not apply the shape
//!   bound, and bounds the model once at the end. The previous round measured
//!   what in-loop clipping costs, and there is no reason for the gate audit to
//!   pay it again.
//! * The comparison is paired by fold on the same folds the choice used, and the
//!   movement is read between two fits that start from the same model, so both
//!   numbers describe the same rows.
//!
//! Nothing here changes which partition ships. The audit either reports that the
//! gating model is supported, in which case the numbers say by how much, or it
//! reports that it is not, which is the answer a card can state without moving
//! the segments a reader has already seen.

use std::io;

use serde::{Deserialize, Serialize};

use super::ad::{Day, Entity};
use super::mixture::{Family, Mixture};
use super::SegmentConfig;
use crate::num::round4;

/// Ridge on every slope, never on an intercept.
///
/// The gate has four coefficient vectors over three standardised columns, and a
/// column can separate two groups with almost no overlap. The penalty is small
/// enough to leave a real effect alone and large enough that the fit is defined
/// when a covariate is close to constant inside a group.
const RIDGE: f64 = 1.0;

/// Fisher scoring rounds per maximisation step. The objective is smooth and
/// strictly concave after the ridge, so this converges long before it matters.
const GATE_ROUNDS: usize = 40;

/// A descriptive column, declared so a card can name it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Covariate {
    pub name: String,
    pub label: String,
    pub unit: String,
    /// Share of this column's own variance the partition explains. A column the
    /// groups already separate cannot add much to the weights.
    #[serde(default)]
    pub eta2: f64,
}

/// The standardised covariate matrix the gate reads, one row per entity.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub covariates: Vec<Covariate>,
    pub z: Vec<Vec<f64>>,
}

impl Block {
    /// Recency, campaign span and test share for a panel, in that order.
    ///
    /// Recency is days from the panel's last day back to the entity's last
    /// start, so the anchor is the panel, not the wall clock: a rebuild from the
    /// same file gives the same covariate. Span is the distance between the first
    /// and last start. Test share is the share of its ads the archive marked as a
    /// split, which is a fact about the export rather than about the advertiser
    /// and is therefore descriptive in exactly the way this module means.
    ///
    /// `wanted` names the columns this block carries, in the declared order. The
    /// pre-registered reading gates on `recency` alone, because that is the column
    /// the item is about; the wider set is a supplementary reading rather than a
    /// second chance to find a gain, and the card says so next to it.
    ///
    /// A covariate that does not vary is dropped rather than standardised to a
    /// column of zeros, and a panel with no covariate left is refused by
    /// [`measure`] instead of silently fitting intercepts.
    pub fn build(entities: &[Entity], wanted: &[&str]) -> io::Result<Block> {
        if entities.is_empty() {
            return Err(bad("no entities to gate"));
        }
        let mut first = Vec::with_capacity(entities.len());
        let mut last = Vec::with_capacity(entities.len());
        for entity in entities {
            let start = Day::parse(&entity.first)
                .ok_or_else(|| bad(&format!("entity {} has no readable first day", entity.id)))?;
            let end = Day::parse(&entity.last)
                .ok_or_else(|| bad(&format!("entity {} has no readable last day", entity.id)))?;
            if end.0 < start.0 {
                return Err(bad(&format!("entity {} ends before it starts", entity.id)));
            }
            first.push(start.0);
            last.push(end.0);
        }
        let anchor = last.iter().copied().max().unwrap_or(0);
        let recency: Vec<f64> = last
            .iter()
            .map(|end| ((anchor - end) as f64 + 1.0).ln())
            .collect();
        let span: Vec<f64> = first
            .iter()
            .zip(&last)
            .map(|(start, end)| ((end - start) as f64 + 1.0).ln())
            .collect();
        let test_share: Vec<f64> = entities
            .iter()
            .map(|entity| {
                let ads = entity.ads.max(1) as f64;
                entity.test_ads.min(entity.ads) as f64 / ads
            })
            .collect();
        let declared = [
            (
                Covariate {
                    name: "recency".into(),
                    label: "recency".into(),
                    unit: "days since last start".into(),
                    eta2: 0.0,
                },
                recency,
            ),
            (
                Covariate {
                    name: "span".into(),
                    label: "campaign span".into(),
                    unit: "days from first to last start".into(),
                    eta2: 0.0,
                },
                span,
            ),
            (
                Covariate {
                    name: "test_share".into(),
                    label: "test share".into(),
                    unit: "share of ads in a split".into(),
                    eta2: 0.0,
                },
                test_share,
            ),
        ];
        let mut covariates = Vec::new();
        let mut columns = Vec::new();
        for (covariate, values) in declared {
            if !wanted.contains(&covariate.name.as_str()) {
                continue;
            }
            let mean = mean(&values);
            let sd = spread(&values, mean);
            if sd <= 1e-9 {
                continue;
            }
            let z: Vec<f64> = values
                .iter()
                .map(|value| clip((value - mean) / sd))
                .collect();
            covariates.push(covariate);
            columns.push(z);
        }
        if covariates.is_empty() {
            return Err(bad("no descriptive column varies on this panel"));
        }
        let mut z = Vec::with_capacity(entities.len());
        for row in 0..entities.len() {
            z.push(columns.iter().map(|column| column[row]).collect());
        }
        Ok(Block { covariates, z })
    }

    /// The rows this block holds, so a caller can align folds.
    pub fn len(&self) -> usize {
        self.z.len()
    }

    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }
}

/// A multinomial logistic gate: one coefficient vector per component.
///
/// The first coefficient of a row is the intercept, the rest are slopes on the
/// standardised covariates in the order [`Block`] holds them. Weights are a
/// softmax, which is why the ridge never touches an intercept and why the
/// coefficient vectors are not separately identified without it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    pub coef: Vec<Vec<f64>>,
}

impl Gate {
    /// Uniform weights, the model a plain mixture implies.
    pub fn uniform(k: usize, dim: usize) -> Gate {
        Gate {
            coef: vec![vec![0.0; dim + 1]; k],
        }
    }

    pub fn k(&self) -> usize {
        self.coef.len()
    }

    /// Log weights for one row, before normalisation.
    pub fn log_weights(&self, x: &[f64]) -> Vec<f64> {
        self.coef
            .iter()
            .map(|row| {
                let mut value = row.first().copied().unwrap_or(0.0);
                for (slope, xj) in row.iter().skip(1).zip(x) {
                    value += slope * xj;
                }
                value
            })
            .collect()
    }

    /// Mixing weights for one row, summing to one.
    pub fn weights(&self, x: &[f64]) -> Vec<f64> {
        softmax(&self.log_weights(x))
    }

    /// Wide enough to shift a group's weight, quoted for a one sd move in each
    /// covariate, which is the size a reader can picture.
    pub fn slopes(&self) -> Vec<f64> {
        let dim = self
            .coef
            .first()
            .map(|row| row.len().saturating_sub(1))
            .unwrap_or(0);
        (0..dim)
            .map(|column| {
                self.coef
                    .iter()
                    .map(|row| row.get(column + 1).copied().unwrap_or(0.0).abs())
                    .fold(0.0f64, f64::max)
            })
            .collect()
    }
}

/// A mixture whose weights come from the gate rather than from the component.
#[derive(Clone, Debug)]
pub struct Gated {
    pub mixture: Mixture,
    pub gate: Gate,
    /// Gated log likelihood on the rows it was fitted to.
    pub loglik: f64,
    pub rounds: usize,
}

impl Gated {
    /// Fit the gated model from scratch, with the same restarts and iterations a
    /// plain fit of this (family, k) would use.
    ///
    /// Starting from a plain fit rather than from random parameters is the
    /// standard route: the gate then has to earn its improvement against the
    /// model it is compared with, instead of arriving at a different local
    /// optimum and calling that a gain.
    pub fn fit(
        rows: &[Vec<f64>],
        x: &[Vec<f64>],
        k: usize,
        family: Family,
        restarts: usize,
        iterations: usize,
        seed: u64,
    ) -> Gated {
        let base = Mixture::fit(rows, k, family, restarts, iterations, seed);
        Gated::refine(base, rows, x, iterations)
    }

    /// Refine an existing mixture by alternating the gate and the components.
    pub fn refine(base: Mixture, rows: &[Vec<f64>], x: &[Vec<f64>], iterations: usize) -> Gated {
        let k = base.k();
        if k == 0 || rows.is_empty() || x.len() != rows.len() {
            return Gated {
                mixture: base,
                gate: Gate::uniform(k.max(1), 1),
                loglik: f64::NEG_INFINITY,
                rounds: 0,
            };
        }
        let dim = x.first().map(|row| row.len()).unwrap_or(0);
        let mut mixture = base;
        let mut resp: Vec<Vec<f64>> = rows.iter().map(|row| mixture.posterior(row)).collect();
        let mut gate = fit_gate(x, &resp, k, dim);
        let mut previous = f64::NEG_INFINITY;
        let mut rounds = 0;
        for _ in 0..iterations.max(1) {
            let (next, loglik) = responsibilities(&mixture, &gate, rows, x);
            resp = next;
            mixture = Mixture::step(rows, &resp, k, mixture.family);
            gate = fit_gate(x, &resp, k, dim);
            rounds += 1;
            if (loglik - previous).abs() < 1e-8 * rows.len() as f64 {
                break;
            }
            previous = loglik;
        }
        let mixture = mixture.bounded(rows);
        let loglik = Gated::score(&mixture, &gate, rows, x) * rows.len() as f64;
        Gated {
            mixture,
            gate,
            loglik,
            rounds,
        }
    }

    /// Gated log likelihood of `rows`, in nats per row.
    pub fn score(mixture: &Mixture, gate: &Gate, rows: &[Vec<f64>], x: &[Vec<f64>]) -> f64 {
        let (_, total) = responsibilities(mixture, gate, rows, x);
        if rows.is_empty() {
            f64::NEG_INFINITY
        } else {
            total / rows.len() as f64
        }
    }
}

/// One audit of the gating model against the one that ships.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    pub covariates: Vec<Covariate>,
    pub folds: usize,
    /// Held-out nats per entity with constant weights and with gated weights.
    pub constant: f64,
    pub gated: f64,
    /// Paired difference per fold, gated minus constant. Positive is better.
    pub delta: f64,
    pub delta_sd: f64,
    pub delta_se: f64,
    pub per_fold: Vec<f64>,
    /// How far a component mean moved when the weights became a function of the
    /// covariates, in standard deviations of the column it moved on.
    pub shift_mean: f64,
    pub shift_max: f64,
    /// Where the largest move sits, as a column name.
    pub shift_at: String,
    /// A one sd move in the widest slope, as a change in that group's weight.
    pub widest_slope: f64,
    /// The two halves of the acceptance, kept apart so a card can print which
    /// half failed.
    pub added: bool,
    pub steady: bool,
    pub verdict: String,
}

impl Audit {
    /// Whether the panel supports gating on these columns.
    pub fn honest(&self) -> bool {
        self.added && self.steady
    }

    pub fn rounded(mut self) -> Audit {
        self.constant = round4(self.constant);
        self.gated = round4(self.gated);
        self.delta = round4(self.delta);
        self.delta_sd = round4(self.delta_sd);
        self.delta_se = round4(self.delta_se);
        self.per_fold = self.per_fold.iter().map(|v| round4(*v)).collect();
        self.shift_mean = round4(self.shift_mean);
        self.shift_max = round4(self.shift_max);
        self.widest_slope = round4(self.widest_slope);
        self
    }
}

/// Measure the gating model on the same folds the choice was scored on.
///
/// `folds` are indices into `rows`, in the order `select::folds_for` returned
/// them, so the paired difference is read on rows the fit never saw and on the
/// same rows for both models. `names` names the behavioural columns, so the
/// movement reading points at a reading rather than at a column number.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    rows: &[Vec<f64>],
    x: &[Vec<f64>],
    labels: &[usize],
    mixture: &Mixture,
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
    block: &Block,
    names: &[String],
) -> Audit {
    let k = mixture.k();
    let mut audit = Audit {
        covariates: block.covariates.clone(),
        folds: folds.len(),
        ..Audit::default()
    };
    if k == 0 || rows.len() != x.len() {
        audit.verdict =
            "the gating model was not measured: the panel and the covariates are misaligned".into();
        return audit;
    }
    // How much of each covariate's own spread the partition already explains.
    // A column the densities already separate cannot add much to the weights, and
    // this is the number that says so before any fit is run.
    for (column, covariate) in audit.covariates.iter_mut().enumerate() {
        covariate.eta2 = round4(eta2(x, column, labels, k));
    }
    let mut differences = Vec::with_capacity(folds.len());
    let mut constant_scores = Vec::with_capacity(folds.len());
    let mut gated_scores = Vec::with_capacity(folds.len());
    for (index, fold) in folds.iter().enumerate() {
        let held: Vec<usize> = fold.clone();
        let train: Vec<usize> = (0..rows.len()).filter(|i| !held.contains(i)).collect();
        if train.len() <= k || held.is_empty() {
            continue;
        }
        let train_rows: Vec<Vec<f64>> = train.iter().map(|&i| rows[i].clone()).collect();
        let train_x: Vec<Vec<f64>> = train.iter().map(|&i| x[i].clone()).collect();
        let held_rows: Vec<Vec<f64>> = held.iter().map(|&i| rows[i].clone()).collect();
        let held_x: Vec<Vec<f64>> = held.iter().map(|&i| x[i].clone()).collect();
        let restarts = cfg.em_restarts.clamp(1, 2);
        let seed = cfg.seed.wrapping_add(index as u64);
        let plain = Mixture::fit(
            &train_rows,
            k,
            mixture.family,
            restarts,
            cfg.iterations,
            seed,
        );
        if plain.components.is_empty() {
            continue;
        }
        let constant = plain.total_loglik(&held_rows) / held_rows.len() as f64;
        let gated = Gated::fit(
            &train_rows,
            &train_x,
            k,
            mixture.family,
            restarts,
            cfg.iterations,
            seed,
        );
        if gated.rounds == 0 {
            continue;
        }
        let gated_score = Gated::score(&gated.mixture, &gated.gate, &held_rows, &held_x);
        constant_scores.push(constant);
        gated_scores.push(gated_score);
        differences.push(gated_score - constant);
    }
    if differences.is_empty() {
        audit.verdict = "every fold was too small to fit the gating model".into();
        return audit;
    }
    let (delta, delta_sd) = mean_sd(&differences);
    audit.delta = delta;
    audit.delta_sd = delta_sd;
    audit.delta_se = delta_sd / (differences.len() as f64).sqrt();
    audit.per_fold = differences.clone();
    audit.constant = constant_scores.iter().sum::<f64>() / constant_scores.len() as f64;
    audit.gated = gated_scores.iter().sum::<f64>() / gated_scores.len() as f64;
    // A rough two standard error rule, stated as one rather than dressed as a
    // test: with five folds the per fold spread is the only estimate of the
    // noise available, and a gain that does not clear twice its own standard
    // error is not a gain this panel can see. The threshold is fixed here, not
    // chosen after the numbers were read.
    audit.added = delta > 0.0 && delta > 2.0 * audit.delta_se;

    // The movement reading. Both fits start from the shipped model, so a moved
    // mean is the gate's doing and not a different local optimum's.
    let whole = Gated::refine(mixture.clone(), rows, x, cfg.iterations);
    let (shift_mean, shift_max, shift_at) = movement(mixture, &whole.mixture, names);
    audit.shift_mean = shift_mean;
    audit.shift_max = shift_max;
    audit.shift_at = shift_at;
    audit.steady = shift_max < 0.1;
    audit.widest_slope = whole.gate.slopes().iter().copied().fold(0.0f64, f64::max);
    audit.verdict = verdict(&audit);
    audit
}

/// Mean and largest move between two component means, in column standard
/// deviations.
///
/// The gated loop updates each component from its own starting point, so the
/// order is preserved and the comparison is component by component rather than
/// by nearest neighbour. A component that lost all its rows is reported at the
/// panel centre by [`Mixture::step`], which shows up here as a large move,
/// which is the honest reading.
fn movement(before: &Mixture, after: &Mixture, names: &[String]) -> (f64, f64, String) {
    let mut sum = 0.0;
    let mut count = 0;
    let mut worst = 0.0;
    let mut at = 0;
    for (a, b) in before.components.iter().zip(&after.components) {
        for (column, (x, y)) in a.mean.iter().zip(&b.mean).enumerate() {
            let shift = (x - y).abs();
            sum += shift;
            count += 1;
            if shift > worst {
                worst = shift;
                at = column;
            }
        }
    }
    let mean = if count == 0 { 0.0 } else { sum / count as f64 };
    let label = names
        .get(at)
        .cloned()
        .unwrap_or_else(|| format!("column {at}"));
    (mean, worst, label)
}

fn verdict(audit: &Audit) -> String {
    let columns: Vec<&str> = audit
        .covariates
        .iter()
        .map(|covariate| covariate.label.as_str())
        .collect();
    let names = columns.join(", ");
    if audit.honest() {
        format!(
            "Gating on {names} adds {:.3} nats per entity on held-out rows (standard error {:.3}) and moves no mean more than {:.3} sd, so the panel supports it.",
            audit.delta, audit.delta_se, audit.shift_max
        )
    } else if !audit.added && !audit.steady {
        format!(
            "Gating on {names} adds {:.3} nats per entity against a standard error of {:.3} and moves a mean by {:.3} sd, so it fails both halves of the acceptance and stays out.",
            audit.delta, audit.delta_se, audit.shift_max
        )
    } else if !audit.added {
        format!(
            "Gating on {names} adds {:.3} nats per entity on held-out rows against a standard error of {:.3}, which is not a gain, so it stays out of the fit and the densities keep their constant weights.",
            audit.delta, audit.delta_se
        )
    } else {
        format!(
            "Gating on {names} adds {:.3} nats per entity but moves a mean by {:.3} sd, so the groups would mean something else and the change is left to a reader rather than banked.",
            audit.delta, audit.shift_max
        )
    }
}

/// Responsibilities and total log likelihood under a gate and a mixture.
fn responsibilities(
    mixture: &Mixture,
    gate: &Gate,
    rows: &[Vec<f64>],
    x: &[Vec<f64>],
) -> (Vec<Vec<f64>>, f64) {
    let k = mixture.k().max(1);
    let mut out = Vec::with_capacity(rows.len());
    let mut total = 0.0;
    for (row, covariate) in rows.iter().zip(x) {
        let densities = mixture.log_components(row);
        // The gate contributes a log probability, not a logit: without the
        // normalising constant the score would be `ln k` nats per row too high
        // and would not be comparable with the constant weight model it is
        // measured against.
        let logits = gate.log_weights(covariate);
        let normaliser = log_sum_exp(&logits);
        let mut logs: Vec<f64> = logits
            .iter()
            .take(k)
            .zip(&densities)
            .map(|(weight, density)| weight - normaliser + density)
            .collect();
        while logs.len() < k {
            logs.push(f64::NEG_INFINITY);
        }
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            out.push(vec![1.0 / k as f64; k]);
            continue;
        }
        let mut mass = 0.0;
        for value in logs.iter_mut() {
            *value = (*value - top).exp();
            mass += *value;
        }
        if mass <= 0.0 {
            out.push(vec![1.0 / k as f64; k]);
            continue;
        }
        total += top + mass.ln();
        for value in logs.iter_mut() {
            *value /= mass;
        }
        logs.resize(k, 0.0);
        out.push(logs);
    }
    (out, total)
}

/// Multinomial logistic regression on soft responsibilities.
///
/// The objective is `sum_i sum_k r_ik log softmax_k(B x_i)` minus the ridge, and
/// it is maximised by damped Fisher scoring. A step is taken from the diagonal
/// of the information matrix, which is exact per coefficient for a logistic
/// model and needs no matrix inverse at this width; a step that lowers the
/// objective is halved until it does not.
fn fit_gate(x: &[Vec<f64>], resp: &[Vec<f64>], k: usize, dim: usize) -> Gate {
    let mut gate = Gate::uniform(k, dim);
    if x.is_empty() || resp.is_empty() {
        return gate;
    }
    let mut current = objective(&gate, x, resp);
    for _ in 0..GATE_ROUNDS {
        let mut step = vec![vec![0.0; dim + 1]; k];
        for (row, target) in x.iter().zip(resp) {
            let p = gate.weights(row);
            for (j, row_step) in step.iter_mut().enumerate() {
                let probability = p.get(j).copied().unwrap_or(0.0);
                let residual = target.get(j).copied().unwrap_or(0.0) - probability;
                let curvature = probability * (1.0 - probability);
                let intercept = curvature * x.len() as f64;
                row_step[0] += residual / (intercept + 1e-9);
                for (column, value) in row.iter().enumerate() {
                    let information = curvature * value * value * x.len() as f64 + RIDGE;
                    row_step[column + 1] += residual * value / information;
                }
            }
        }
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..20 {
            let mut candidate = gate.clone();
            for (j, row) in candidate.coef.iter_mut().enumerate() {
                for (column, value) in row.iter_mut().enumerate() {
                    *value += scale * step[j][column];
                }
            }
            if objective(&candidate, x, resp) >= current {
                gate = candidate;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
        let after = objective(&gate, x, resp);
        if (after - current).abs() < 1e-10 * x.len() as f64 {
            break;
        }
        current = after;
    }
    gate
}

fn objective(gate: &Gate, x: &[Vec<f64>], resp: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for (row, target) in x.iter().zip(resp) {
        let logs = gate.log_weights(row);
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            return f64::NEG_INFINITY;
        }
        let mass: f64 = logs.iter().map(|value| (value - top).exp()).sum();
        let norm = top + mass.ln();
        for (weight, log) in target.iter().zip(&logs) {
            total += weight * (log - norm);
        }
    }
    for row in &gate.coef {
        for value in row.iter().skip(1) {
            total -= 0.5 * RIDGE * value * value;
        }
    }
    total
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        let k = logits.len().max(1);
        return vec![1.0 / k as f64; k];
    }
    let mut out: Vec<f64> = logits.iter().map(|value| (value - top).exp()).collect();
    let mass: f64 = out.iter().sum();
    if mass > 0.0 {
        for value in out.iter_mut() {
            *value /= mass;
        }
    }
    out
}

/// `ln sum exp(values)`, the normaliser a log probability needs.
fn log_sum_exp(values: &[f64]) -> f64 {
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return 0.0;
    }
    top + values
        .iter()
        .map(|value| (value - top).exp())
        .sum::<f64>()
        .ln()
}

fn eta2(z: &[Vec<f64>], column: usize, labels: &[usize], k: usize) -> f64 {
    if z.is_empty() || labels.len() != z.len() {
        return 0.0;
    }
    let values: Vec<f64> = z.iter().map(|row| row[column]).collect();
    let grand = mean(&values);
    let mut between = 0.0;
    let mut within = 0.0;
    for group in 0..k {
        let members: Vec<f64> = values
            .iter()
            .zip(labels)
            .filter(|(_, label)| **label == group)
            .map(|(value, _)| *value)
            .collect();
        if members.is_empty() {
            continue;
        }
        let centre = mean(&members);
        between += members.len() as f64 * (centre - grand) * (centre - grand);
        within += members
            .iter()
            .map(|value| (value - centre) * (value - centre))
            .sum::<f64>();
    }
    let total = between + within;
    if total <= 1e-12 {
        0.0
    } else {
        between / total
    }
}

fn clip(value: f64) -> f64 {
    value.clamp(-3.0, 3.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn spread(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mu) * (value - mu))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let mu = mean(values);
    (mu, spread(values, mu))
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests;
