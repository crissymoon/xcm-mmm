//! The atom gate inside the expectation maximisation loop: a pattern mixture.
//!
//! [`super::gate`] fits the two shares to a partition that was chosen without
//! them. That is one step short of the model the pile implies. A row that repeats
//! no creative is not a row whose repeat reading is one and a fifth standard
//! deviations: it is a row whose reading was never drawn from the continuous law
//! at all. Feeding the edge in as a value makes the Gaussian spend a component on
//! a spike, and the group boundaries are then drawn partly by that spike.
//!
//! A pattern mixture states the pattern as data (Ghahramani and Jordan,
//! *Supervised learning from incomplete data*, Technical Report CMU-CS-93-116,
//! 1993). A gated reading that sits on its edge is *absent*: the row is scored on
//! the readings it shows, the absent coordinate is marginalised out of the joint
//! covariance, and the probability of the edge becomes its own parameter per
//! group per gate. Inside the maximisation step the absent coordinate is filled
//! with its conditional mean given the readings the row shows, and the
//! conditional covariance is added to the second moment. That is the standard
//! missing-data step, and it is what keeps the eight column dependence: a
//! covariance entry is estimated from every row that shows both of its readings,
//! which is more rows than a two-part model fitted after the fact ever sees.
//!
//! Three statements about what this does and does not settle.
//!
//! The likelihood maximised here is a density against the mixed measure, the
//! product of Lebesgue on the readings and counting on the edges, so it is a
//! proper likelihood for these data and the maximisation is monotone. It is not
//! comparable with a Gaussian likelihood, because the two are densities against
//! different measures. Nothing in this module is used to choose a family, and
//! nothing in it replaces [`super::select`].
//!
//! What is comparable is the held-out probability of one event: the observed
//! value of a gated reading, given the readings the gates do not hold. [`compare`]
//! measures that for this model and for the gate fitted to the partition, on the
//! same rows and paired by fold, and the paired difference is the reading.
//!
//! The partition this model implies is not shipped by this module. A pattern
//! mixture draws different boundaries, and moving the artifact onto them would
//! move every segment a reader has seen, on the strength of a comparison that
//! covers the two shares only. The comparison says whether that move is
//! warranted; it does not make it.
//!
//! Measured on the shipped panel, it is not. The pattern mixture scores
//! -0.099 nats per entity against -0.988 for the gate fitted to the partition on
//! `repeat`, which is 1.6 fold standard errors and not a separation, and -0.142
//! against -0.915 on `concentrate`, which is 10.5 and is a shortfall. The
//! decomposition says why: 93 percent of it sits on rows that are inside the
//! edge, where the two-part gate fits a slope and a variance per group on
//! exactly those rows while this model derives the same conditional from a joint
//! covariance estimated partly across rows whose share was imputed. On the rows
//! that are on the edge the two models are 0.015 nats per row apart, so the atom
//! is predicted about as well by the model that does not move the partition.
//! That is the finding, and it agrees with the two-part literature rather than
//! with the hope: a two-part model's conditional mean is not the conditional
//! mean of a joint normal (Duan, Manning, Morris and Newhouse, 1983).
//!
//! The model also draws a different partition: 58.1 percent of held out rows
//! keep their group and 105 of 251 move, because the atom enters the grouping
//! and a group is allowed an edge probability near one. That is a different
//! segmentation rather than a better one, which is the second reason the
//! artifact does not move.
//!
//! Take the comparison as a standing test rather than a one-off. On a panel
//! drawn from this model the pattern mixture must not lose to the gate fitted to
//! the partition, and a test asserts that, because otherwise the numbers above
//! would be measuring a defect. That test has already earned its place: the
//! first maximisation step added both orders of the conditional covariance to
//! one upper-triangle cell, which doubled the off diagonal, made the average
//! indefinite and silently ended every fit after one step. The signs of that
//! bug were a comparison that looked decisive against the model and a fit that
//! converged in one iteration.

use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::num::round4;

use super::feature::Scaling;
use super::gate::{self, GateSpec};
use super::kmeans;
use super::mixture::{self, Component, Family, Mixture, FLOOR};
use super::SegmentConfig;

/// Probability of an edge is clamped away from zero and one, because a group
/// that puts a zero there would read a row on the edge as impossible and take
/// the whole likelihood with it.
const EPS: f64 = 1e-6;

/// How close to the edge a reading has to be to count as on it.
const TOUCH: f64 = 1e-9;

/// The readings a row shows, and the gated readings it holds on their edge.
#[derive(Clone, Debug, PartialEq, Eq)]
struct View {
    observed: Vec<usize>,
    missing: Vec<usize>,
}

impl View {
    fn of(row: &[f64], specs: &[GateSpec], dim: usize) -> Self {
        let mut observed = Vec::with_capacity(dim);
        let mut missing = Vec::new();
        for column in 0..dim {
            let held = specs
                .iter()
                .any(|spec| spec.column == column && spec.holds(row));
            if held {
                missing.push(column);
            } else {
                observed.push(column);
            }
        }
        Self { observed, missing }
    }

    fn holds(&self, column: usize) -> bool {
        self.missing.contains(&column)
    }
}

/// A pattern mixture: groups that read an edge as a state rather than a value.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub k: usize,
    pub components: Vec<Component>,
    /// Gated columns, in the order [`Pattern::atom`] is indexed by.
    pub gates: Vec<usize>,
    /// The edge of each gate in z, so a row's state can be read from it.
    pub edges: Vec<f64>,
    /// Per gate, per group: probability of the edge.
    pub atom: Vec<Vec<f64>>,
    /// Observed-data log likelihood of the panel the model was fitted to.
    pub loglik: f64,
    pub iterations: usize,
    /// Worst condition number across the groups, after the shipping bound.
    pub conditions: f64,
}

impl Pattern {
    /// Free parameters: weights, means, a covariance per group, an atom per
    /// gate, and the probability of every atom.
    pub fn parameters(&self, dim: usize) -> usize {
        let k = self.k;
        k.saturating_sub(1) + k * dim + k * dim * (dim + 1) / 2 + k * self.gates.len()
    }

    /// Fit by expectation maximisation on the observed-data likelihood.
    ///
    /// Even runs start from a k-means partition of the panel with the edges
    /// mean-filled, which is an initialisation and not a model; odd runs start
    /// from a soft random assignment, for the reason [`super::mixture`] records.
    /// A larger observed-data likelihood wins and a tie keeps the lower run, so
    /// the fit is reproducible.
    pub fn fit(
        rows: &[Vec<f64>],
        specs: &[GateSpec],
        k: usize,
        restarts: usize,
        iterations: usize,
        seed: u64,
    ) -> Self {
        if rows.is_empty() || k == 0 || k > rows.len() || specs.is_empty() {
            return Self::default();
        }
        let dim = rows[0].len();
        let views: Vec<View> = rows.iter().map(|row| View::of(row, specs, dim)).collect();
        let filled = filled(rows, &views);
        let mut best: Option<Self> = None;
        for run in 0..restarts.max(1) {
            let start = if run % 2 == 0 {
                let fit = kmeans::fit(&filled, k, 1, iterations, seed.wrapping_add(run as u64));
                one_hot(&fit.labels, k)
            } else {
                mixture::soft_start(rows.len(), k, seed.wrapping_add(run as u64))
            };
            let candidate = em(rows, &views, k, specs, iterations, &start);
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

    /// The group each row belongs to, reading its edge states as data.
    ///
    /// This is the partition the pattern mixture itself draws, which is what a
    /// reader would ship if the comparison had cleared. It is reported, not
    /// used: the artifact keeps the partition it chose on the Gaussian
    /// likelihood.
    pub fn assign(&self, row: &[f64]) -> usize {
        let posterior = self.posterior(row);
        let mut best = (0usize, f64::NEG_INFINITY);
        for (id, value) in posterior.iter().enumerate() {
            if *value > best.1 {
                best = (id, *value);
            }
        }
        best.0
    }

    /// Posterior weight of every group for one row, reading its edge states as
    /// data.
    pub fn posterior(&self, row: &[f64]) -> Vec<f64> {
        let k = self.k.min(self.components.len());
        if k == 0 {
            return Vec::new();
        }
        let dim = self.components[0].mean.len();
        let view = self.view(row, dim);
        let mut logs = Vec::with_capacity(k);
        for j in 0..k {
            let mut log = self.components[j].weight.max(1e-300).ln();
            for (g, column) in self.gates.iter().enumerate() {
                let mass = self
                    .atom
                    .get(g)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(1e-12, 1.0 - 1e-12);
                log += if view.holds(*column) {
                    mass.ln()
                } else {
                    (1.0 - mass).ln()
                };
            }
            match gate::Marginal::build(&self.components[j], &view.observed) {
                Some(marginal) => log += marginal.log_density(row),
                None => log = f64::NEG_INFINITY,
            }
            logs.push(log);
        }
        gate::soften(logs)
    }

    fn view(&self, row: &[f64], dim: usize) -> View {
        let mut observed = Vec::with_capacity(dim);
        let mut missing = Vec::new();
        for (column, value) in row.iter().enumerate().take(dim) {
            let held = self
                .gates
                .iter()
                .position(|gate| *gate == column)
                .and_then(|g| self.edges.get(g))
                .map(|edge| (value - edge).abs() <= TOUCH)
                .unwrap_or(false);
            if held {
                missing.push(column);
            } else {
                observed.push(column);
            }
        }
        View { observed, missing }
    }

    /// Log probability of the observed value of one gated reading, on its
    /// lattice, given the readings the gates do not hold.
    ///
    /// The other readings are the same conditioning set the two-part gate is
    /// scored on, so the two numbers are probabilities of the same event and the
    /// difference between them is a like-for-like comparison. The other gate's
    /// coordinate is not conditioned on: this model reads it as a state, and a
    /// state is not a value to condition on.
    pub fn score(
        &self,
        row: &[f64],
        spec: &GateSpec,
        columns: &[usize],
        scaling: &Scaling,
        spacing: f64,
    ) -> f64 {
        let Some(g) = self.gates.iter().position(|c| *c == spec.column) else {
            return f64::NEG_INFINITY;
        };
        let k = self.k.min(self.components.len()).min(self.atom[g].len());
        if k == 0 {
            return f64::NEG_INFINITY;
        }
        let on = (row[spec.column] - spec.atom).abs() <= TOUCH;
        let mut logs = Vec::with_capacity(k);
        for j in 0..k {
            let mass = self.atom[g][j].clamp(1e-12, 1.0 - 1e-12);
            let mut log = self.components[j].weight.max(1e-300).ln()
                + if on { mass.ln() } else { (1.0 - mass).ln() };
            match gate::Marginal::build(&self.components[j], columns) {
                Some(marginal) => log += marginal.log_density(row),
                None => log = f64::NEG_INFINITY,
            }
            logs.push(log);
        }
        let post = gate::soften(logs);
        let mass = if on {
            (0..k)
                .map(|j| post[j] * self.atom[g][j].clamp(1e-12, 1.0 - 1e-12))
                .sum::<f64>()
        } else {
            let mut total = 0.0;
            for (j, weight) in post.iter().enumerate().take(k) {
                let Some((mean, sd)) =
                    gate::conditional(&self.components[j], columns, spec.column, row)
                else {
                    continue;
                };
                let cell =
                    gate::cell_probability(row[spec.column], mean, sd, spec, scaling, spacing);
                total += weight * (1.0 - self.atom[g][j].clamp(1e-12, 1.0 - 1e-12)) * cell;
            }
            total
        };
        mass.max(1e-12).ln()
    }
}

/// One share, scored by both models on the same held-out rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub name: String,
    /// Held-out log probability of the observed value, nats per entity.
    pub two_stage: f64,
    pub pattern: f64,
    /// The paired difference per fold, and its standard error.
    pub gain: f64,
    pub se: f64,
    /// The gain in standard errors, which is what the gate reads.
    pub z: f64,
    pub folds: usize,
    /// The same scores split by the row's true state, which says where a
    /// difference comes from: the edge probability or the inside value.
    pub two_stage_edge: f64,
    pub pattern_edge: f64,
    pub two_stage_inside: f64,
    pub pattern_inside: f64,
    /// Rows on the edge and inside it, over all folds.
    pub edge_rows: usize,
    pub inside_rows: usize,
    /// Share of held-out rows the pattern mixture places in the shipped group,
    /// and how many it moves. Descriptive: the partition is not shipped.
    pub agreement: f64,
    pub moved: usize,
    pub verdict: String,
}

impl Verdict {
    /// Whether the pattern mixture cleared twice its own standard error.
    pub fn cleared(&self) -> bool {
        self.verdict.starts_with("cleared")
    }
}

/// Fit the pattern mixture per fold and score it against the two-part gate on
/// the same held-out rows.
///
/// Both sides are refit on the training rows of each fold, including the mixture
/// the two-part gate is fitted to, so the difference is the model and not the
/// data. The family is the shipped one, because the question is whether the atom
/// belongs inside that likelihood rather than beside it.
#[allow(clippy::too_many_arguments)]
pub fn compare(
    rows: &[Vec<f64>],
    ads: &[f64],
    components: &[Component],
    family: Family,
    labels: &[usize],
    specs: &[GateSpec],
    scaling: &Scaling,
    cfg: &SegmentConfig,
) -> Vec<Verdict> {
    if rows.is_empty() || components.is_empty() || specs.is_empty() {
        return Vec::new();
    }
    let dim = rows[0].len();
    let gates: Vec<usize> = specs.iter().map(|spec| spec.column).collect();
    let columns: Vec<usize> = (0..dim).filter(|c| !gates.contains(c)).collect();
    let origin: Vec<f64> = columns
        .iter()
        .map(|c| rows.iter().map(|row| row[*c]).sum::<f64>() / rows.len() as f64)
        .collect();
    let folds = super::select::folds_for(rows.len(), cfg);
    let mut ledger = vec![Ledger::default(); specs.len()];
    for (number, fold) in folds.iter().enumerate() {
        let held = fold.clone();
        let train: Vec<usize> = (0..rows.len()).filter(|i| !held.contains(i)).collect();
        if train.is_empty() || held.is_empty() {
            continue;
        }
        let train_rows: Vec<Vec<f64>> = train.iter().map(|&i| rows[i].clone()).collect();
        let held_rows: Vec<Vec<f64>> = held.iter().map(|&i| rows[i].clone()).collect();
        // The mixture the two-part gate is fitted to is this fold's own, so the
        // comparison does not credit it with a model that saw the held rows.
        let fold_mixture = Mixture::fit(
            &train_rows,
            components.len(),
            family,
            cfg.em_restarts.clamp(1, 2),
            cfg.iterations,
            cfg.seed.wrapping_add(number as u64),
        );
        if fold_mixture.components.is_empty() {
            continue;
        }
        let resp_train = gate::responsibilities(&train_rows, &fold_mixture.components, &columns);
        let resp_held = gate::responsibilities(&held_rows, &fold_mixture.components, &columns);
        let fitted = Pattern::fit(
            &train_rows,
            specs,
            components.len(),
            cfg.em_restarts.clamp(1, 2),
            cfg.iterations,
            cfg.seed.wrapping_add(number as u64),
        );
        if fitted.components.is_empty() {
            continue;
        }
        // The pattern mixture numbers its groups by the order its seeds found
        // them, so its placement is read through the same alignment the
        // sharpness audit uses.
        let mapping = super::sharp::align(components, &fitted.components);
        let mut kept = 0;
        let mut moved = 0;
        for &i in &held {
            let placed = mapping
                .get(fitted.assign(&rows[i]))
                .copied()
                .unwrap_or(usize::MAX);
            if placed == labels.get(i).copied().unwrap_or(usize::MAX) {
                kept += 1;
            } else {
                moved += 1;
            }
        }
        let agreement = kept as f64 / held.len() as f64;
        for ledger in ledger.iter_mut() {
            ledger.agreement.push(agreement);
            ledger.moved += moved;
        }
        for (index, spec) in specs.iter().enumerate() {
            let bars = gate::bars(
                &train_rows,
                &resp_train,
                std::slice::from_ref(spec),
                &columns,
                &origin,
            );
            let Some(fit) = bars.first() else {
                continue;
            };
            let mut two = 0.0;
            let mut pattern = 0.0;
            let mut two_edge = 0.0;
            let mut pattern_edge = 0.0;
            let mut two_inside = 0.0;
            let mut pattern_inside = 0.0;
            let mut on_edge = 0;
            let mut inside = 0;
            for (slot, &i) in held.iter().enumerate() {
                let spacing = gate::lattice(ads.get(i).copied().unwrap_or(1.0), spec, scaling);
                let scored = gate::scored(
                    &rows[i],
                    spacing,
                    spec,
                    fit,
                    &fold_mixture.components,
                    &resp_held[slot],
                    &columns,
                    scaling,
                );
                let two_score = scored.gate.max(1e-12).ln();
                let pattern_score = fitted.score(&rows[i], spec, &columns, scaling, spacing);
                two += two_score;
                pattern += pattern_score;
                if spec.holds(&rows[i]) {
                    two_edge += two_score;
                    pattern_edge += pattern_score;
                    on_edge += 1;
                } else {
                    two_inside += two_score;
                    pattern_inside += pattern_score;
                    inside += 1;
                }
            }
            let count = held.len() as f64;
            ledger[index].two.push(two / count);
            ledger[index].pattern.push(pattern / count);
            ledger[index].two_edge.push(share(two_edge, on_edge as f64));
            ledger[index]
                .pattern_edge
                .push(share(pattern_edge, on_edge as f64));
            ledger[index]
                .two_inside
                .push(share(two_inside, inside as f64));
            ledger[index]
                .pattern_inside
                .push(share(pattern_inside, inside as f64));
            ledger[index].edge_rows += on_edge;
            ledger[index].inside_rows += inside;
        }
    }
    specs
        .iter()
        .zip(&ledger)
        .map(|(spec, ledger)| verdict(spec, ledger))
        .collect()
}

/// What the comparison accumulates for one share.
#[derive(Clone, Debug, Default)]
struct Ledger {
    two: Vec<f64>,
    pattern: Vec<f64>,
    two_edge: Vec<f64>,
    pattern_edge: Vec<f64>,
    two_inside: Vec<f64>,
    pattern_inside: Vec<f64>,
    agreement: Vec<f64>,
    edge_rows: usize,
    inside_rows: usize,
    moved: usize,
}

fn share(sum: f64, count: f64) -> f64 {
    if count > 0.0 {
        sum / count
    } else {
        0.0
    }
}

fn verdict(spec: &GateSpec, ledger: &Ledger) -> Verdict {
    let folds = ledger.two.len().min(ledger.pattern.len());
    if folds == 0 {
        return Verdict {
            name: spec.name.clone(),
            two_stage: 0.0,
            pattern: 0.0,
            gain: 0.0,
            se: 0.0,
            z: 0.0,
            folds: 0,
            two_stage_edge: 0.0,
            pattern_edge: 0.0,
            two_stage_inside: 0.0,
            pattern_inside: 0.0,
            edge_rows: 0,
            inside_rows: 0,
            agreement: 0.0,
            moved: 0,
            verdict: "not measured: no fold produced a fit on both sides".into(),
        };
    }
    let paired: Vec<f64> = (0..folds)
        .map(|i| ledger.pattern[i] - ledger.two[i])
        .collect();
    let gain = mean(&paired);
    let se = spread(&paired);
    let z = gain / se.max(1e-12);
    let verdict = if folds < 3 {
        format!("not measured: {folds} fold is too few to separate a gain from fold noise")
    } else if gain > 2.0 * se {
        format!(
            "cleared: modelling the edge inside the loop gives the observed value {gain:+.3} nats per entity, {z:.1} standard errors, and the partition is still the shipped one"
        )
    } else if gain < -2.0 * se {
        format!(
            "refused: the loop model is {:.3} nats per entity worse, {z:.1} standard errors, so the gate fitted to the partition stands and the edge stays where it is",
            -gain
        )
    } else {
        format!(
            "not separated: {gain:+.3} nats per entity is {z:.1} standard errors, so nothing was gained; the partition-fitted gate stands because it is the simpler statement"
        )
    };
    Verdict {
        name: spec.name.clone(),
        two_stage: round4(mean(&ledger.two)),
        pattern: round4(mean(&ledger.pattern)),
        gain: round4(gain),
        se: round4(se),
        z: round4(z),
        folds,
        two_stage_edge: round4(mean(&ledger.two_edge)),
        pattern_edge: round4(mean(&ledger.pattern_edge)),
        two_stage_inside: round4(mean(&ledger.two_inside)),
        pattern_inside: round4(mean(&ledger.pattern_inside)),
        edge_rows: ledger.edge_rows,
        inside_rows: ledger.inside_rows,
        agreement: round4(mean(&ledger.agreement)),
        moved: ledger.moved,
        verdict,
    }
}

/// One maximisation step from soft responsibilities, with the edges absent.
///
/// The conditional law of an absent reading is taken from the previous step's
/// parameters, which is the classical missing-data maximisation step. It is
/// cached per distinct pattern, because every row that shows the same readings
/// needs the same matrix and only the residual differs.
fn m_step(
    rows: &[Vec<f64>],
    views: &[View],
    resp: &[Vec<f64>],
    k: usize,
    specs: &[GateSpec],
    previous: &[Component],
) -> (Vec<Component>, Vec<Vec<f64>>) {
    let n = rows.len();
    let dim = rows[0].len();
    let gates: Vec<usize> = specs.iter().map(|spec| spec.column).collect();
    let mut counts = vec![0.0; k];
    for row in resp.iter() {
        for (j, value) in row.iter().enumerate() {
            counts[j] += *value;
        }
    }

    // The whole panel, used to reset a group no row claims.
    let mut centre = vec![0.0; dim];
    for row in rows {
        for (d, value) in row.iter().enumerate() {
            centre[d] += value;
        }
    }
    for value in centre.iter_mut() {
        *value /= n as f64;
    }
    let mut block = vec![FLOOR; dim];
    for row in rows {
        for (d, value) in row.iter().enumerate() {
            let gap = value - centre[d];
            block[d] += gap * gap / n as f64;
        }
    }

    let (patterns, index) = patterns_of(views);
    let cache: Vec<Vec<Option<Impute>>> = patterns
        .iter()
        .map(|(observed, missing)| {
            previous
                .iter()
                .map(|component| Impute::build(component, observed, missing))
                .collect()
        })
        .collect();

    let mut first = vec![vec![0.0; dim]; k];
    let mut second = vec![vec![vec![0.0; dim]; dim]; k];
    let mut edge = vec![vec![0.0; k]; gates.len()];
    for i in 0..n {
        let (observed, missing) = &patterns[index[i]];
        for j in 0..k {
            let weight = resp[i][j];
            if weight <= 0.0 {
                continue;
            }
            for (g, column) in gates.iter().enumerate() {
                if missing.contains(column) {
                    edge[g][j] += weight;
                }
            }
            let (values, extra) = if missing.is_empty() {
                (rows[i].clone(), None)
            } else {
                match cache[index[i]][j].as_ref() {
                    Some(impute) => {
                        let (cond_mean, cond_cov) = impute.apply(&rows[i], observed);
                        let mut values = rows[i].clone();
                        for (t, &column) in missing.iter().enumerate() {
                            values[column] = cond_mean[t];
                        }
                        (values, Some(cond_cov))
                    }
                    // No factorisation for this pattern: the row is taken as it
                    // stands, which is the mean-filled initialisation.
                    None => (rows[i].clone(), None),
                }
            };
            for (d, value) in values.iter().enumerate() {
                first[j][d] += weight * value;
            }
            for a in 0..dim {
                for b in a..dim {
                    second[j][a][b] += weight * values[a] * values[b];
                }
            }
            if let Some(cov) = extra {
                // The conditional covariance is symmetric and only the upper
                // triangle of the second moment is accumulated, so each distinct
                // pair is added once. Adding both orders doubles the off
                // diagonal term and the average stops being a second moment,
                // which makes the maximisation step indefinite and silently
                // ends the fit after one step.
                for (ta, &ma) in missing.iter().enumerate() {
                    for (tb, &mb) in missing.iter().enumerate().skip(ta) {
                        let (low, high) = if ma <= mb { (ma, mb) } else { (mb, ma) };
                        second[j][low][high] += weight * cov[ta][tb];
                    }
                }
            }
        }
    }

    let mut components = Vec::with_capacity(k);
    for j in 0..k {
        if counts[j] < 1e-6 * n as f64 {
            let mut covariance = vec![vec![0.0; dim]; dim];
            for d in 0..dim {
                covariance[d][d] = block[d].max(FLOOR);
            }
            let Some(covariance) = repair(covariance) else {
                return (Vec::new(), Vec::new());
            };
            components.push(Component {
                weight: 0.0,
                mean: centre.clone(),
                covariance,
            });
            continue;
        }
        let count = counts[j];
        let mean: Vec<f64> = first[j].iter().map(|value| value / count).collect();
        let mut covariance = vec![vec![0.0; dim]; dim];
        // Only the upper triangle is accumulated, so the lower triangle is
        // mirrored once here. Index loops are the clear form for a triangular
        // half of a matrix, where there is no row being walked.
        #[allow(clippy::needless_range_loop)]
        for a in 0..dim {
            for b in a..dim {
                covariance[a][b] = second[j][a][b] / count - mean[a] * mean[b];
            }
        }
        #[allow(clippy::needless_range_loop)]
        for a in 0..dim {
            covariance[a][a] = covariance[a][a].max(FLOOR);
            for b in 0..a {
                covariance[a][b] = covariance[b][a];
            }
        }
        let Some(covariance) = repair(covariance) else {
            return (Vec::new(), Vec::new());
        };
        components.push(Component {
            weight: count / n as f64,
            mean,
            covariance,
        });
    }

    let mut atom = vec![vec![0.0; k]; gates.len()];
    for (g, spec) in specs.iter().enumerate() {
        for j in 0..k {
            atom[g][j] = if counts[j] > 0.0 {
                (edge[g][j] / counts[j]).clamp(EPS, 1.0 - EPS)
            } else {
                spec.share.clamp(EPS, 1.0 - EPS)
            };
        }
    }
    (components, atom)
}

/// The expectation step: responsibilities and the observed-data log likelihood.
///
/// Each component's density over a pattern is factorised once per pattern, so the
/// cost of a fold model is one factorisation per group per distinct pattern
/// rather than one per row.
fn e_step(
    components: &[Component],
    atom: &[Vec<f64>],
    gates: &[usize],
    views: &[View],
    rows: &[Vec<f64>],
) -> (Vec<Vec<f64>>, f64) {
    let k = components.len();
    let n = rows.len();
    let (patterns, index) = patterns_of(views);
    let cache: Vec<Vec<Option<gate::Marginal>>> = (0..k)
        .map(|j| {
            patterns
                .iter()
                .map(|(observed, _)| gate::Marginal::build(&components[j], observed))
                .collect()
        })
        .collect();
    let mut resp = vec![vec![0.0; k]; n];
    let mut loglik = 0.0;
    for i in 0..n {
        let mut logs = Vec::with_capacity(k);
        for j in 0..k {
            let mut log = components[j].weight.max(1e-300).ln();
            for (g, column) in gates.iter().enumerate() {
                let mass = atom
                    .get(g)
                    .and_then(|row| row.get(j))
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(1e-12, 1.0 - 1e-12);
                log += if views[i].holds(*column) {
                    mass.ln()
                } else {
                    (1.0 - mass).ln()
                };
            }
            match &cache[j][index[i]] {
                Some(marginal) => log += marginal.log_density(&rows[i]),
                None => log = f64::NEG_INFINITY,
            }
            logs.push(log);
        }
        let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !top.is_finite() {
            resp[i] = vec![1.0 / k as f64; k];
            continue;
        }
        let mut total = 0.0;
        for (j, value) in logs.iter().enumerate() {
            let mass = (value - top).exp();
            resp[i][j] = mass;
            total += mass;
        }
        for value in resp[i].iter_mut() {
            *value /= total.max(1e-300);
        }
        loglik += top + total.max(1e-300).ln();
    }
    (resp, loglik)
}

/// Expectation maximisation from one starting assignment, then the shipping
/// bound on shape.
fn em(
    rows: &[Vec<f64>],
    views: &[View],
    k: usize,
    specs: &[GateSpec],
    iterations: usize,
    start: &[Vec<f64>],
) -> Pattern {
    let gates: Vec<usize> = specs.iter().map(|spec| spec.column).collect();
    let filled = filled(rows, views);
    let mut previous = Mixture::step(&filled, start, k, Family::Full).components;
    if previous.is_empty() {
        return Pattern::default();
    }
    let mut atom = start_atom(rows, specs, start, k);
    let (mut resp, mut loglik) = e_step(&previous, &atom, &gates, views, rows);
    let mut used = 1;
    for step in 1..iterations.max(1) {
        let (components, next_atom) = m_step(rows, views, &resp, k, specs, &previous);
        if components.is_empty() {
            break;
        }
        previous = components;
        atom = next_atom;
        let (next, total) = e_step(&previous, &atom, &gates, views, rows);
        let settled = (total - loglik).abs() <= 1e-8 * rows.len() as f64;
        resp = next;
        loglik = total;
        used = step + 1;
        if settled {
            break;
        }
    }
    let _ = resp;
    let mut model = Pattern {
        k,
        components: previous,
        gates: gates.clone(),
        edges: specs.iter().map(|spec| spec.atom).collect(),
        atom,
        loglik,
        iterations: used,
        conditions: 0.0,
    };
    if let Some(bounded) = mixture::bounded_components(&model.components) {
        model.components = bounded;
        let (_, total) = e_step(&model.components, &model.atom, &gates, views, rows);
        model.loglik = total;
    }
    model.conditions = mixture::conditioning(&model.components);
    model
}

/// The conditional law of the absent readings, cached per pattern and group.
///
/// `gain` is `cov[missing][observed] * cov[observed][observed]^-1` and `cov` is
/// the conditional covariance of the absent block, which is what the second
/// moment needs: a row whose reading is absent contributes a spread, not a point.
struct Impute {
    gain: Vec<Vec<f64>>,
    cov: Vec<Vec<f64>>,
    absent: Vec<f64>,
    observed: Vec<f64>,
}

impl Impute {
    fn build(component: &Component, observed: &[usize], missing: &[usize]) -> Option<Self> {
        let o = observed.len();
        let m = missing.len();
        if o == 0 || m == 0 {
            return None;
        }
        let mut block = vec![vec![0.0; o]; o];
        for (a, &ca) in observed.iter().enumerate() {
            for (b, &cb) in observed.iter().enumerate() {
                block[a][b] = *component.covariance.get(ca)?.get(cb)?;
                if a == b {
                    block[a][b] += FLOOR;
                }
            }
        }
        let mut chol = block;
        linalg::cholesky(&mut chol)?;
        let mut cross = vec![vec![0.0; o]; m];
        for (t, &mc) in missing.iter().enumerate() {
            for (s, &oc) in observed.iter().enumerate() {
                cross[t][s] = *component.covariance.get(mc)?.get(oc)?;
            }
        }
        let mut gain = vec![vec![0.0; o]; m];
        for (t, row) in cross.iter().enumerate() {
            gain[t] = linalg::cholesky_solve(&chol, row);
        }
        let mut cov = vec![vec![0.0; m]; m];
        for a in 0..m {
            for b in 0..m {
                let removed: f64 = (0..o).map(|s| gain[a][s] * cross[b][s]).sum();
                cov[a][b] = *component.covariance.get(missing[a])?.get(missing[b])? - removed;
            }
        }
        Some(Self {
            gain,
            cov,
            absent: missing
                .iter()
                .map(|&c| component.mean.get(c).copied().unwrap_or(0.0))
                .collect(),
            observed: observed
                .iter()
                .map(|&c| component.mean.get(c).copied().unwrap_or(0.0))
                .collect(),
        })
    }

    /// Conditional mean of the absent readings, and their conditional covariance.
    fn apply(&self, row: &[f64], observed: &[usize]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let residual: Vec<f64> = observed
            .iter()
            .zip(&self.observed)
            .map(|(&c, mean)| row.get(c).copied().unwrap_or(0.0) - mean)
            .collect();
        let mean = self
            .gain
            .iter()
            .zip(&self.absent)
            .map(|(gain, absent)| {
                absent + gain.iter().zip(&residual).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect();
        (mean, self.cov.clone())
    }
}

/// Distinct patterns, and which one each row shows.
/// Distinct observed-and-absent column sets, and which set each row takes.
type Patterns = (Vec<(Vec<usize>, Vec<usize>)>, Vec<usize>);

fn patterns_of(views: &[View]) -> Patterns {
    let mut patterns: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let mut index = Vec::with_capacity(views.len());
    for view in views {
        let key = (view.observed.clone(), view.missing.clone());
        let id = match patterns.iter().position(|seen| *seen == key) {
            Some(id) => id,
            None => {
                patterns.push(key);
                patterns.len() - 1
            }
        };
        index.push(id);
    }
    (patterns, index)
}

/// The panel with each absent reading put at its column mean, for initialisation
/// only.
fn filled(rows: &[Vec<f64>], views: &[View]) -> Vec<Vec<f64>> {
    let dim = rows[0].len();
    let mut centre = vec![0.0; dim];
    for row in rows {
        for (d, value) in row.iter().enumerate() {
            centre[d] += value;
        }
    }
    for value in centre.iter_mut() {
        *value /= rows.len() as f64;
    }
    rows.iter()
        .zip(views)
        .map(|(row, view)| {
            let mut out = row.clone();
            for &column in &view.missing {
                out[column] = centre[column];
            }
            out
        })
        .collect()
}

fn start_atom(rows: &[Vec<f64>], specs: &[GateSpec], resp: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut atom = vec![vec![0.0; k]; specs.len()];
    for (g, spec) in specs.iter().enumerate() {
        for (j, cell) in atom[g].iter_mut().enumerate() {
            let mut total = 0.0;
            let mut edge = 0.0;
            for (i, row) in rows.iter().enumerate() {
                let weight = resp.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                total += weight;
                if spec.holds(row) {
                    edge += weight;
                }
            }
            *cell = if total > 0.0 {
                (edge / total).clamp(EPS, 1.0 - EPS)
            } else {
                spec.share.clamp(EPS, 1.0 - EPS)
            };
        }
    }
    atom
}
/// A covariance that a Cholesky factor accepts, raising the diagonal if needed.
fn repair(covariance: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    for attempt in 0..8 {
        let mut trial = covariance.clone();
        let bump = FLOOR * attempt as f64;
        for (d, row) in trial.iter_mut().enumerate() {
            row[d] += bump;
        }
        let mut chol = trial.clone();
        if linalg::cholesky(&mut chol).is_some() {
            return Some(trial);
        }
    }
    None
}

fn one_hot(labels: &[usize], k: usize) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&label| {
            let mut row = vec![0.0; k];
            if label < k {
                row[label] = 1.0;
            }
            row
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Standard error of a mean over independent folds.
fn spread(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::INFINITY;
    }
    let centre = mean(values);
    let variance: f64 = values
        .iter()
        .map(|value| (value - centre) * (value - centre))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (variance / values.len() as f64).sqrt()
}

#[cfg(test)]
mod tests;
