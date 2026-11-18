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
//! the readings it shows, and the probability of the edge becomes its own
//! parameter per group per gate. The conditional law of the inside reading is
//! estimated where the two-part gate estimates it, by weighted least squares on
//! the rows that show a value, and the stored covariance is completed from those
//! pieces so that conditioning it returns the regression exactly. That is the
//! change the measurement asked for: an earlier version filled the absent
//! coordinate at the conditional mean of a joint normal, which is the right
//! answer for a joint normal and a flatter conditional than a two-part model's
//! (Duan, Manning, Morris and Newhouse, 1983).
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
//! Measured on the shipped panel, it is not. The loop model scores -1.165 nats
//! per entity against -0.988 for the gate fitted to the partition on `repeat`,
//! which is 2.8 fold standard errors, and -1.165 against -0.915 on `concentrate`,
//! which is 5.8. The model with one group, so the conditional is pooled rather
//! than per group, reads -1.340 and -1.243, so grouping does buy something the
//! pooling cannot and the shortfall is not the regression being fitted badly.
//!
//! The explanation is a variance problem, and it is visible in the numbers the
//! comparison now carries. The loop model fits its inside residual on fewer rows
//! per group, 37.8 and 35.2 against 50.6 and 44.6, and gets a tighter residual in
//! sample, 0.187 and 0.422 against 0.285 and 0.460, while predicting worse out of
//! sample. It is also sharper in the block the score conditions on, a mean
//! winning posterior of 0.978 against 0.922. Six coefficients per group on
//! thirty-five rows fits the training inside spread and does not predict it. The
//! next move the numbers support is partial pooling: shrink each group's slope
//! toward the pooled slope by its own standard error.
//!
//! The model also draws a different partition, and more different than the
//! imputing version drew: 46.2 percent of held out rows keep their group and 135
//! of 251 move, against 58.1 and 105 before. The atom enters the grouping, so
//! groups are partly organised by which edges a row sits on. That is a different
//! segmentation rather than a better one, which is the second reason the artifact
//! does not move.
//!
//! Take the comparison as a standing test rather than a one-off. On a panel
//! drawn from this model the pattern mixture must not lose to the gate fitted to
//! the partition, and a test asserts that, because otherwise the numbers above
//! would be measuring a defect. Two more tests hold the completion to the
//! regression it is built from and hold the residual correlation to the one the
//! panel planted. They have already earned their place three times: a
//! maximisation step that added both orders of a conditional covariance to one
//! upper-triangle cell, which doubled an off diagonal and ended every fit after
//! one step; a mirror that copied raw sums into the lower triangle; and a floor
//! meant for a probability applied to a log score, which read -27.631 for every
//! row.

use serde::{Deserialize, Serialize};

use crate::num::round4;

use super::feature::Scaling;
use super::gate::{self, GateSpec};
use super::kmeans;
use super::mixture::{self, Component, Family, Mixture};
use super::SegmentConfig;

mod fit;

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
    /// The inside regression of each gate, per group, which is the conditional
    /// law the completed covariance encodes.
    #[serde(default)]
    pub regressions: Vec<GateFit>,
    /// Residual correlation between two gates, per group. Carried because
    /// conditional independence is a claim the panel can refuse, and here it does.
    #[serde(default)]
    pub coupling: Vec<Coupling>,
    /// Observed-data log likelihood of the panel the model was fitted to.
    pub loglik: f64,
    pub iterations: usize,
    /// Worst condition number across the groups, after the shipping bound.
    pub conditions: f64,
}

/// What one gate's inside regression decided, per group.
///
/// The numbers are here so the card can print the conditional law rather than
/// infer it from a mean: a slope a reader can compare against the two-part gate's,
/// the residual it leaves, and how many rows the estimate had. The row counts are
/// the honest part. A slope fitted on eight rows is not the same claim as one
/// fitted on ninety, and a card that prints only the slope hides which it is.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GateFit {
    pub name: String,
    /// Weighted rows that show this reading inside its edge, per group.
    pub rows: Vec<usize>,
    /// Residual standard deviation of the inside regression, per group.
    pub sd: Vec<f64>,
    /// Euclidean norm of the slope vector, so one number per group summarises it.
    pub slope: Vec<f64>,
    /// The intercept, which is the inside mean at the panel's centre.
    pub inside: Vec<f64>,
}

impl GateFit {
    pub(super) fn empty(spec: &GateSpec, k: usize) -> Self {
        Self {
            name: spec.name.clone(),
            rows: vec![0; k],
            sd: vec![0.0; k],
            slope: vec![0.0; k],
            inside: vec![0.0; k],
        }
    }
}

/// The residual correlation between two gated readings, given the free block.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Coupling {
    pub first: String,
    pub second: String,
    /// Per group, the weighted residual correlation.
    pub correlation: Vec<f64>,
    /// Weighted rows showing both readings inside, the sample the correlation is
    /// measured on.
    pub rows: Vec<usize>,
}

impl Pattern {
    /// Parameters the fit reports, which is not the number the likelihood
    /// identifies: the completed covariance is a function of the free block, the
    /// slopes and the residual covariance, so the gate block is counted through
    /// its regression rather than as a free covariance.
    pub fn parameters(&self, dim: usize) -> usize {
        let k = self.k;
        let gates = self.gates.len();
        let free = dim.saturating_sub(gates);
        let pairs = gates.saturating_sub(1) * gates / 2;
        k.saturating_sub(1)
            + k * free
            + k * free * (free + 1) / 2
            + k * gates
            + k * gates * (free + 1)
            + k * pairs
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
            let candidate = fit::em(rows, &views, k, specs, iterations, &start);
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
    #[serde(default)]
    pub two_stage_edge: f64,
    #[serde(default)]
    pub pattern_edge: f64,
    #[serde(default)]
    pub two_stage_inside: f64,
    #[serde(default)]
    pub pattern_inside: f64,
    /// The loop model with one group, so the conditional is pooled rather than
    /// per group. Zero when the reference did not fit.
    #[serde(default)]
    pub pooled: f64,
    /// Rows on the edge and inside it, over all folds.
    #[serde(default)]
    pub edge_rows: usize,
    #[serde(default)]
    pub inside_rows: usize,
    /// Mean weighted inside rows per group and mean inside residual, for the
    /// partition gate and for the loop model. These say whether a difference is
    /// the estimator or the sample the estimator had, which is the question a
    /// reader asks first when a conditional is refitted inside a loop.
    #[serde(default)]
    pub gate_rows: f64,
    #[serde(default)]
    pub gate_sd: f64,
    #[serde(default)]
    pub loop_rows: f64,
    #[serde(default)]
    pub loop_sd: f64,
    /// Mean winning posterior over the groups given the readings the score
    /// conditions on: how sharply each model separates its groups in that block.
    #[serde(default)]
    pub gate_sharp: f64,
    #[serde(default)]
    pub loop_sharp: f64,
    /// Share of held-out rows the pattern mixture places in the shipped group,
    /// and how many it moves. Descriptive: the partition is not shipped.
    #[serde(default)]
    pub agreement: f64,
    #[serde(default)]
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
        // One group, so the conditional is pooled over the whole panel. It is the
        // reference that says whether the groups buy anything on the conditional
        // question at all, which is the difference between a model that needs a
        // better estimator and one that needs a better partition.
        let pooled = Pattern::fit(
            &train_rows,
            specs,
            1,
            1,
            cfg.iterations,
            cfg.seed.wrapping_add(number as u64),
        );
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
        // How sharply each model separates its groups on the readings the score
        // conditions on. A mixture of four conditionals is flatter than one
        // conditional whenever its groups are indistinguishable in that block, so
        // this is the reading that decides whether a difference is the estimator
        // or the grouping.
        let loop_resp = gate::responsibilities(&held_rows, &fitted.components, &columns);
        let loop_sharp = winning(&loop_resp);
        let gate_sharp = winning(&resp_held);
        for ledger in ledger.iter_mut() {
            ledger.agreement.push(agreement);
            ledger.moved += moved;
            ledger.gate_sharp.push(gate_sharp);
            ledger.loop_sharp.push(loop_sharp);
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
            let inside_fit = fitted.regressions.get(index);
            let mut gate_rows = 0.0;
            let mut gate_sd = 0.0;
            let mut groups = 0.0;
            for (j, spec_fit) in fit.variance.iter().enumerate() {
                let rows: f64 = train_rows
                    .iter()
                    .enumerate()
                    .filter(|(i, row)| !spec.holds(row) && resp_train[*i][j] > 1e-8)
                    .count() as f64;
                if rows > 1.0 {
                    gate_rows += rows;
                    gate_sd += spec_fit.sqrt();
                    groups += 1.0;
                }
            }
            if groups > 0.0 {
                ledger[index].gate_rows.push(gate_rows / groups);
                ledger[index].gate_sd.push(gate_sd / groups);
            }
            if let Some(inside_fit) = inside_fit {
                let (mut rows, mut sd, mut seen) = (0.0, 0.0, 0.0);
                for (count, spread) in inside_fit.rows.iter().zip(&inside_fit.sd) {
                    if *count > 1 {
                        rows += *count as f64;
                        sd += spread;
                        seen += 1.0;
                    }
                }
                if seen > 0.0 {
                    ledger[index].loop_rows.push(rows / seen);
                    ledger[index].loop_sd.push(sd / seen);
                }
            }
            let mut two = 0.0;
            let mut pattern = 0.0;
            let mut pooled_score = 0.0;
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
                if !pooled.components.is_empty() {
                    // `score` already returns a log probability with its own
                    // floor. Clamping it with `.max(1e-12)` would clamp a
                    // negative number to a positive one and read -27.631 for
                    // every row, which is how this line was first written.
                    pooled_score += pooled.score(&rows[i], spec, &columns, scaling, spacing);
                }
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
            if !pooled.components.is_empty() {
                ledger[index].pooled.push(pooled_score / count);
            }
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
    pooled: Vec<f64>,
    agreement: Vec<f64>,
    gate_rows: Vec<f64>,
    gate_sd: Vec<f64>,
    loop_rows: Vec<f64>,
    loop_sd: Vec<f64>,
    gate_sharp: Vec<f64>,
    loop_sharp: Vec<f64>,
    edge_rows: usize,
    inside_rows: usize,
    moved: usize,
}

/// Mean winning posterior over a set of responsibilities.
fn winning(resp: &[Vec<f64>]) -> f64 {
    if resp.is_empty() {
        return 0.0;
    }
    let total: f64 = resp
        .iter()
        .map(|row| row.iter().copied().fold(0.0, f64::max))
        .sum();
    total / resp.len() as f64
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
            pooled: 0.0,
            edge_rows: 0,
            inside_rows: 0,
            gate_rows: 0.0,
            gate_sd: 0.0,
            loop_rows: 0.0,
            loop_sd: 0.0,
            gate_sharp: 0.0,
            loop_sharp: 0.0,
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
        pooled: if ledger.pooled.is_empty() {
            0.0
        } else {
            round4(mean(&ledger.pooled))
        },
        edge_rows: ledger.edge_rows,
        inside_rows: ledger.inside_rows,
        agreement: round4(mean(&ledger.agreement)),
        moved: ledger.moved,
        gate_rows: round4(mean(&ledger.gate_rows)),
        gate_sd: round4(mean(&ledger.gate_sd)),
        loop_rows: round4(mean(&ledger.loop_rows)),
        loop_sd: round4(mean(&ledger.loop_sd)),
        gate_sharp: round4(mean(&ledger.gate_sharp)),
        loop_sharp: round4(mean(&ledger.loop_sharp)),
        verdict,
    }
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

/// Distinct observed-and-absent column sets, and which set each row takes.
///
/// The expectation step factorises one matrix per group per distinct pattern, so a
/// fold model pays for the patterns the panel shows rather than for its rows.
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
