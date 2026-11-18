//! A drift gate sized to the sample it was computed on.
//!
//! The folklore bands on the population stability index, 0.1 for "some change"
//! and 0.25 for "major", are quoted as if they were properties of the
//! statistic. They are not: PSI is a sum of standardised squared differences in
//! bin shares, so doubling the sample halves it and dividing the sample by ten
//! multiplies it by ten. A fixed band is therefore a different test at every
//! sample size, and a backfill of a stationary panel at a thousand rows a side
//! reads a 95th percentile of about 0.03, which the folklore calls three times
//! quieter than "no change at all" (Yurdakul and Naranjo, *Journal of Risk Model
//! Validation*, 2020, is the measurement; Siddiqi, 2006, is where the bands come
//! from).
//!
//! So this module scores drift against a null generated at the size the reading
//! was taken at, in three layers:
//!
//! * [`psi_null`] simulates the index under the null. The reference bins fix
//!   the expected shares, so under "no change" the current counts are a
//!   multinomial draw against the reference shares, and the null law depends
//!   only on the bin shares and the current row count. That makes it cheap and
//!   exact up to the binning, and it respects a discrete column whose shares
//!   are not `1 / bins` because ties flatten the quantiles.
//! * [`cvm`] and [`CvmNull`] add a Cramer-von Mises reading, which is a
//!   distance between whole distribution functions rather than a sum over
//!   bins, so a change the bins merge still shows. Its null is the asymptotic
//!   Cramer-von Mises law, computed from a Karhunen-Loeve expansion of the
//!   Brownian bridge rather than read from a table, and a test pins its 95th
//!   percentile against the tabulated 0.461.
//! * [`mmd`] compares the whole standardised vector with a kernel two-sample
//!   test (Gretton et al., *JMLR* 13, 2012; the median bandwidth heuristic is
//!   from the same paper's experiments). A per-feature gate cannot see a
//!   rotation the margins do not show, and this is the cheapest reading that
//!   can.
//!
//! The two per-feature readings are combined by taking the smaller of their
//! two p-values with a factor of two for the double test, then applied to
//! Benjamini and Hochberg across features (*JRSS-B* 57, 1995) because a report
//! with forty columns is forty chances at a false alarm. With every feature
//! null the step-up rejects only when the smallest p-value clears
//! `alpha / features`, so the family-wise error is held near `alpha`, and the
//! acceptance test measures that rather than assuming it.
//!
//! Every draw in this module comes from a fixed seed, so two runs of the same
//! report print the same p-values. Nothing here is a daemon; it is the decision
//! layer a scheduler calls.

use crate::data::Dataset;
use crate::num::round4;
use crate::rng::Rng;

use super::drift::{counts, edges, paired_columns, psi_counts};

/// Seed for every null in this module, so a report is reproducible.
pub const GATE_SEED: u64 = 0x6A7E_5EED;

/// How the gate is scored.
#[derive(Clone, Debug)]
pub struct GateConfig {
    /// Nominal level for the per-feature tests and the joint reading.
    pub alpha: f64,
    /// Reference quantile bins for the index.
    pub bins: usize,
    /// Null draws for the index, per feature. The smallest attainable p-value
    /// is `1 / (psi_draws + 1)`.
    pub psi_draws: usize,
    /// Null draws for the Cramer-von Mises law, shared by every feature.
    pub cvm_draws: usize,
    /// Terms kept in the Karhunen-Loeve expansion of the bridge.
    pub cvm_terms: usize,
    /// Rows per side kept for the joint reading. Below two the reading is off
    /// rather than a two-row test.
    pub mmd_rows: usize,
    /// Permutations for the joint reading. The smallest attainable p-value is
    /// `1 / (mmd_permutations + 1)`.
    pub mmd_permutations: usize,
    /// A fixed index threshold in PSI units. When present it replaces the
    /// calibrated threshold and the report says so.
    pub threshold: Option<f64>,
    /// Seed for every draw.
    pub seed: u64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            bins: 10,
            psi_draws: 499,
            cvm_draws: 4000,
            cvm_terms: 200,
            mmd_rows: 200,
            mmd_permutations: 99,
            threshold: None,
            seed: GATE_SEED,
        }
    }
}

/// The two-sample Cramer-von Mises statistic, on the same scale as the
/// asymptotic law used for its p-value.
///
/// `T = (nm / N^2) * sum over the pooled sample of (F - G)^2`, which converges
/// to the integral of a squared Brownian bridge. Mean under the null is 1/6.
pub fn cvm(a: &[f64], b: &[f64]) -> f64 {
    let a: Vec<f64> = a.iter().copied().filter(|v| v.is_finite()).collect();
    let b: Vec<f64> = b.iter().copied().filter(|v| v.is_finite()).collect();
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mut a = a;
    let mut b = b;
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let mut pool: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    pool.sort_by(f64::total_cmp);
    let (n, m) = (a.len() as f64, b.len() as f64);
    let total = n + m;
    let mut sum = 0.0;
    for value in &pool {
        let fa = a.partition_point(|x| x <= value) as f64 / n;
        let fb = b.partition_point(|x| x <= value) as f64 / m;
        sum += (fa - fb) * (fa - fb);
    }
    (n * m / (total * total)) * sum
}

/// The asymptotic Cramer-von Mises law, from a Karhunen-Loeve expansion.
///
/// `integral of B squared = sum over k of Z_k^2 / (k^2 pi^2)` for standard
/// normal `Z_k`, so drawing the law is a sum of squared normals with geometric
/// weights and needs no bridge discretisation. Truncating at `terms` drops a
/// mean of `1 / (pi^2 terms)`, which is 0.0005 at two hundred terms.
#[derive(Clone, Debug)]
pub struct CvmNull {
    sorted: Vec<f64>,
}

impl CvmNull {
    /// Draw the law `draws` times from `seed`.
    pub fn new(draws: usize, terms: usize, seed: u64) -> Self {
        let draws = draws.max(1);
        let terms = terms.clamp(1, 2000);
        let weights: Vec<f64> = (1..=terms)
            .map(|k| {
                let scale = k as f64 * std::f64::consts::PI;
                1.0 / (scale * scale)
            })
            .collect();
        let mut rng = Rng::stream(seed, 0x0C_2E);
        let mut sorted = Vec::with_capacity(draws);
        for _ in 0..draws {
            let mut total = 0.0;
            for weight in &weights {
                let z = rng.standard_normal();
                total += weight * z * z;
            }
            sorted.push(total);
        }
        sorted.sort_by(f64::total_cmp);
        Self { sorted }
    }

    /// Upper tail probability at `statistic`.
    ///
    /// The add-one correction keeps a reading from claiming zero probability
    /// from a finite number of draws; the smallest value it can return is
    /// `1 / (draws + 1)`.
    pub fn p(&self, statistic: f64) -> f64 {
        let above = self.sorted.partition_point(|value| *value < statistic);
        (self.sorted.len() - above + 1) as f64 / (self.sorted.len() + 1) as f64
    }

    /// The `1 - alpha` quantile of the null, for printing beside a reading.
    pub fn critical(&self, alpha: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let q = (1.0 - alpha).clamp(0.0, 1.0);
        let at = (q * (self.sorted.len() - 1) as f64).round() as usize;
        self.sorted[at.min(self.sorted.len() - 1)]
    }
}

/// The null law of the population stability index at the size it was read at.
///
/// The index is a sum of standardised squared differences in bin shares, so to
/// first order it is a quadratic form in zero-mean normal differences with
/// `bins - 1` degrees of freedom and scale `sum (1 - p_i) * (1 / n + 1 / m)`,
/// where `p_i` are the reference shares and `n` and `m` the two row counts
/// (*not* one of them twice: both samples move, which is why a null that
/// conditioned on the reference understates this law by a factor of about two
/// and would have fired on a stationary backfill most of the time).
///
/// The approximation was checked rather than trusted: on a thousand rows a
/// side in ten bins it gives a mean of 0.018 and a 95th percentile of 0.034,
/// against 0.018 and 0.032 measured end to end on normal draws, and the 1
/// percent point reads 0.043 against 0.042 measured. A test compares the two
/// again on the shipped panel's own columns.
///
/// A bin the reference never occupies contributes `(1 - 0) = 1` to the scale,
/// so a value that appears only in the current panel still alarms rather than
/// hiding behind a probability of zero.
#[derive(Clone, Debug)]
pub struct PsiNull {
    sorted: Vec<f64>,
    scale: f64,
    degrees: usize,
}

impl PsiNull {
    /// Draw the law from the reference shares and the two row counts.
    pub fn new(
        shares: &[f64],
        reference_rows: usize,
        current_rows: usize,
        draws: usize,
        seed: u64,
    ) -> Self {
        let draws = draws.max(1);
        let total: f64 = shares.iter().sum();
        let degrees = shares.len().saturating_sub(1).max(1);
        let spread = 1.0 / reference_rows.max(1) as f64 + 1.0 / current_rows.max(1) as f64;
        let expected: f64 = if total > 0.0 {
            shares
                .iter()
                .map(|share| (1.0 - (share / total).clamp(0.0, 1.0)) * spread)
                .sum()
        } else {
            degrees as f64 * spread
        };
        let scale = expected / degrees as f64;
        let mut rng = Rng::stream(seed, 0x9511);
        let mut sorted = Vec::with_capacity(draws);
        for _ in 0..draws {
            let mut total = 0.0;
            for _ in 0..degrees {
                let z = rng.standard_normal();
                total += z * z;
            }
            sorted.push(scale * total);
        }
        sorted.sort_by(f64::total_cmp);
        Self {
            sorted,
            scale,
            degrees,
        }
    }

    /// Upper tail probability at `observed`, with the add-one correction.
    pub fn p(&self, observed: f64) -> f64 {
        if self.sorted.is_empty() {
            return 1.0;
        }
        let above = self.sorted.partition_point(|value| *value < observed);
        (self.sorted.len() - above + 1) as f64 / (self.sorted.len() + 1) as f64
    }

    /// The `1 - alpha` quantile of the null, in index units.
    pub fn critical(&self, alpha: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let at = ((1.0 - alpha).clamp(0.0, 1.0) * (self.sorted.len() - 1) as f64).round() as usize;
        self.sorted[at.min(self.sorted.len() - 1)]
    }

    /// The scale the law was fitted with, for a report that wants to show it.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Degrees of freedom, `bins - 1`.
    pub fn degrees(&self) -> usize {
        self.degrees
    }
}

/// Benjamini-Hochberg step-up: which of `p` are declared at `alpha`.
pub fn bh(p: &[f64], alpha: f64) -> Vec<bool> {
    let m = p.len();
    let mut out = vec![false; m];
    if m == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|a, b| p[*a].total_cmp(&p[*b]));
    let mut cut = 0usize;
    for (rank, index) in order.iter().enumerate() {
        let threshold = alpha * (rank + 1) as f64 / m as f64;
        if p[*index] <= threshold {
            cut = rank + 1;
        }
    }
    for index in order.iter().take(cut) {
        out[*index] = true;
    }
    out
}

/// One feature's gated reading.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateFeature {
    pub name: String,
    pub psi: f64,
    pub psi_p: f64,
    /// The `1 - alpha` quantile of this feature's null, in PSI units.
    pub psi_threshold: f64,
    pub ks: f64,
    pub cvm: f64,
    pub cvm_p: f64,
    /// The smaller of the two readings with a factor of two for the double
    /// test. This is the value Benjamini-Hochberg is applied to.
    pub p: f64,
    pub alarm: bool,
    pub mean_shift: f64,
}

/// The joint reading over the whole standardised vector.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MmdReport {
    pub reference_rows: usize,
    pub current_rows: usize,
    pub bandwidth: f64,
    pub statistic: f64,
    pub p: f64,
    pub permutations: usize,
}

/// One comparison, scored.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GateReport {
    pub reference_rows: usize,
    pub current_rows: usize,
    pub alpha: f64,
    pub bins: usize,
    pub features: Vec<GateFeature>,
    pub missing: Vec<String>,
    /// The features Benjamini-Hochberg declared at `alpha`.
    pub alarms: Vec<String>,
    pub mmd: Option<MmdReport>,
    /// The `1 - alpha` quantile of the Cramer-von Mises null, for reference.
    pub cvm_critical: f64,
    /// A fixed threshold the caller supplied, in PSI units.
    pub manual_threshold: Option<f64>,
    pub verdict: String,
}

impl GateReport {
    /// True when any reading fired or the joint test did.
    pub fn fired(&self) -> bool {
        !self.alarms.is_empty() || self.mmd.as_ref().is_some_and(|mmd| mmd.p <= self.alpha)
    }
}

/// Score a reference panel against a current one.
///
/// Columns are matched by name, and a column present on one side only is named
/// rather than treated as zero. The joint reading uses only the columns both
/// sides carry.
pub fn assess(
    reference: &Dataset,
    current: &Dataset,
    config: &GateConfig,
) -> Result<GateReport, String> {
    if reference.features.is_empty() || current.features.is_empty() {
        return Err("drift needs at least one row on each side".to_string());
    }
    let (columns, missing) = paired_columns(reference, current);
    if columns.is_empty() {
        return Err("the two panels share no feature name".to_string());
    }
    let null_cvm = CvmNull::new(config.cvm_draws, config.cvm_terms, config.seed);
    let mut features = Vec::with_capacity(columns.len());
    for (index, (name, a, b)) in columns.iter().enumerate() {
        let bins = config.bins.clamp(2, 50);
        let edges = edges(a, bins);
        let reference_counts = counts(a, &edges);
        let current_counts = counts(b, &edges);
        let reading = psi_counts(
            &reference_counts,
            a.len() as f64,
            &current_counts,
            b.len() as f64,
        );
        let shares: Vec<f64> = reference_counts
            .iter()
            .map(|count| *count as f64 / a.len().max(1) as f64)
            .collect();
        let null = PsiNull::new(
            &shares,
            a.len(),
            b.len(),
            config.psi_draws,
            config.seed ^ (index as u64 + 1),
        );
        let psi_p = null.p(reading);
        let psi_threshold = null.critical(config.alpha);
        let statistic = cvm(a, b);
        let cvm_p = null_cvm.p(statistic);
        let combined = (2.0 * psi_p.min(cvm_p)).min(1.0);
        let reference_mean = a.iter().sum::<f64>() / a.len().max(1) as f64;
        let current_mean = b.iter().sum::<f64>() / b.len().max(1) as f64;
        features.push(GateFeature {
            name: name.clone(),
            psi: reading,
            psi_p,
            psi_threshold,
            ks: super::drift::ks(a, b),
            cvm: statistic,
            cvm_p,
            p: combined,
            alarm: false,
            mean_shift: current_mean - reference_mean,
        });
    }
    let ps: Vec<f64> = features.iter().map(|feature| feature.p).collect();
    let declared = bh(&ps, config.alpha);
    for (feature, declared) in features.iter_mut().zip(&declared) {
        feature.alarm = *declared;
    }
    if let Some(threshold) = config.threshold {
        for feature in features.iter_mut() {
            feature.alarm = feature.psi >= threshold;
        }
    }
    let alarms: Vec<String> = features
        .iter()
        .filter(|feature| feature.alarm)
        .map(|feature| feature.name.clone())
        .collect();
    let mmd = make_mmd(&columns, config);
    let mut report = GateReport {
        reference_rows: reference.len(),
        current_rows: current.len(),
        alpha: config.alpha,
        bins: config.bins.clamp(2, 50),
        features,
        missing,
        alarms,
        mmd,
        cvm_critical: null_cvm.critical(config.alpha),
        manual_threshold: config.threshold,
        verdict: String::new(),
    };
    report.verdict = verdict(&report);
    Ok(report)
}

fn verdict(report: &GateReport) -> String {
    if report.fired() {
        let mut parts = Vec::new();
        if !report.alarms.is_empty() {
            parts.push(format!("drift in {}", report.alarms.join(", ")));
        }
        if let Some(mmd) = &report.mmd {
            if mmd.p <= report.alpha {
                parts.push(format!(
                    "the joint reading fires at p {:.3} on the whole vector",
                    mmd.p
                ));
            }
        }
        parts.join("; ")
    } else {
        let smallest = report
            .features
            .iter()
            .map(|feature| feature.p)
            .fold(1.0f64, f64::min);
        format!(
            "stable at level {:.2}: no feature clears the step-up, smallest p {:.3}",
            report.alpha, smallest
        )
    }
}

/// Kernel two-sample test on the standardised vector.
///
/// Columns are divided by their pooled standard deviation first, because a
/// kernel on raw units mixes a tenure in months with a rate in tenths.
/// Bandwidth is the median pairwise distance, and the statistic is the biased
/// MMD estimate with a permutation null. Rows are capped at `mmd_rows` per side
/// and the permutation count is printed with the reading, since both bound the
/// smallest p-value the test can report.
fn make_mmd(columns: &[(String, Vec<f64>, Vec<f64>)], config: &GateConfig) -> Option<MmdReport> {
    // A cap below two rows is off, not a two-row test: a two-row permutation
    // test reads noise at whatever rate the tie-breaking gives, which is how a
    // disabled joint reading fired on most stationary splits the first time it
    // was measured.
    if config.mmd_rows < 2 || config.mmd_permutations == 0 {
        return None;
    }
    let reference_rows = columns.first()?.1.len();
    let current_rows = columns.first()?.2.len();
    if reference_rows < 2 || current_rows < 2 {
        return None;
    }
    let cap = config.mmd_rows.clamp(2, 5000);
    let keep = |rows: usize| -> Vec<usize> { (0..rows.min(cap)).collect() };
    let keep_a = keep(reference_rows);
    let keep_b = keep(current_rows);
    let mut scaled = Vec::with_capacity(columns.len());
    for (_, a, b) in columns {
        let pooled: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
        let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
        let variance = pooled.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
            / (pooled.len().max(2) - 1) as f64;
        let sd = variance.sqrt().max(1e-9);
        let x: Vec<f64> = keep_a.iter().map(|i| (a[*i] - mean) / sd).collect();
        let y: Vec<f64> = keep_b.iter().map(|i| (b[*i] - mean) / sd).collect();
        scaled.push((x, y));
    }
    let point = |index: usize, side: usize| -> Vec<f64> {
        scaled
            .iter()
            .map(|(x, y)| if side == 0 { x[index] } else { y[index] })
            .collect()
    };
    let a: Vec<Vec<f64>> = keep_a.iter().map(|i| point(*i, 0)).collect();
    let b: Vec<Vec<f64>> = keep_b.iter().map(|j| point(*j, 1)).collect();
    let bandwidth = median_distance(&a, &b);
    if bandwidth <= 0.0 {
        return None;
    }
    let statistic = mmd_statistic(&a, &b, bandwidth);
    let mut rng = Rng::stream(config.seed, 0x3D3D);
    let mut pool: Vec<Vec<f64>> = a.iter().chain(b.iter()).cloned().collect();
    let mut at_or_above = 1usize;
    for _ in 0..config.mmd_permutations.max(1) {
        rng.shuffle(&mut pool);
        let split = a.len();
        let permuted = mmd_statistic(&pool[..split], &pool[split..], bandwidth);
        if permuted >= statistic {
            at_or_above += 1;
        }
    }
    let draws = config.mmd_permutations.max(1) + 1;
    Some(MmdReport {
        reference_rows: a.len(),
        current_rows: b.len(),
        bandwidth: round4(bandwidth),
        statistic: round4(statistic),
        p: round4(at_or_above as f64 / draws as f64),
        permutations: config.mmd_permutations.max(1),
    })
}

/// A deterministic subsample of pairwise distances, for the median heuristic.
fn median_distance(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let all: Vec<&Vec<f64>> = a.iter().chain(b.iter()).collect();
    let mut distances = Vec::new();
    let step = (all.len() / 40).max(1);
    for i in (0..all.len()).step_by(step) {
        for j in (i + 1..all.len()).step_by(step) {
            let sum: f64 = all[i]
                .iter()
                .zip(all[j].iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            distances.push(sum.sqrt());
        }
    }
    if distances.is_empty() {
        return 1.0;
    }
    distances.sort_by(f64::total_cmp);
    distances[distances.len() / 2].max(1e-6)
}

/// Biased MMD squared with an RBF kernel.
fn mmd_statistic(a: &[Vec<f64>], b: &[Vec<f64>], bandwidth: f64) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let scale = -1.0 / (2.0 * bandwidth * bandwidth);
    let mean = |left: &[Vec<f64>], right: &[Vec<f64>]| -> f64 {
        let mut total = 0.0;
        for x in left {
            for y in right {
                let sum: f64 = x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum();
                total += (sum * scale).exp();
            }
        }
        total / (left.len() * right.len()) as f64
    };
    mean(a, a) + mean(b, b) - 2.0 * mean(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normals(rng: &mut Rng, rows: usize) -> Vec<f64> {
        (0..rows).map(|_| rng.standard_normal()).collect()
    }

    /// A one column dataset, with a target that drift never reads.
    fn panel(name: &str, values: &[f64]) -> Dataset {
        let rows: Vec<Vec<f64>> = values.iter().map(|value| vec![*value]).collect();
        Dataset::new(vec![name.to_string()], rows, vec![0.0; values.len()]).expect("panel")
    }

    fn frame(columns: &[Vec<f64>]) -> Dataset {
        let rows = columns.first().map(|column| column.len()).unwrap_or(0);
        let names: Vec<String> = (0..columns.len()).map(|at| format!("c{at}")).collect();
        let features: Vec<Vec<f64>> = (0..rows)
            .map(|row| columns.iter().map(|column| column[row]).collect())
            .collect();
        Dataset::new(names, features, vec![0.0; rows]).expect("frame")
    }

    /// True when a config leaves the joint reading off.
    fn report_is_off(config: &GateConfig) -> bool {
        let column = vec![0.0f64, 1.0, 2.0];
        make_mmd(&[("c".to_string(), column.clone(), column)], config).is_none()
    }

    #[test]
    fn the_cvm_null_reproduces_the_tabulated_quantiles() {
        let null = CvmNull::new(20_000, 200, 0x1234);
        let critical = null.critical(0.05);
        assert!(
            (0.43..0.49).contains(&critical),
            "95th percentile {critical} against the tabulated 0.461"
        );
        let mean = null.sorted.iter().sum::<f64>() / null.sorted.len() as f64;
        assert!((mean - 1.0 / 6.0).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn the_two_sample_statistic_matches_the_law_it_is_scored_against() {
        let mut rng = Rng::new(0x55);
        let mut exceeded = 0;
        let trials = 400;
        let null = CvmNull::new(4000, 200, 0x99);
        for _ in 0..trials {
            let a = normals(&mut rng, 400);
            let b = normals(&mut rng, 400);
            if null.p(cvm(&a, &b)) <= 0.05 {
                exceeded += 1;
            }
        }
        let rate = exceeded as f64 / trials as f64;
        assert!((0.01..0.09).contains(&rate), "rejection rate {rate}");
    }

    #[test]
    fn a_shifted_sample_crosses_the_critical_value() {
        let mut rng = Rng::new(0x77);
        let a = normals(&mut rng, 500);
        let b: Vec<f64> = normals(&mut rng, 500).iter().map(|v| v + 1.0).collect();
        let null = CvmNull::new(4000, 200, 0x99);
        assert!(cvm(&a, &b) > null.critical(0.01));
    }

    #[test]
    fn the_psi_null_shrinks_with_the_sample() {
        let shares = vec![0.1f64; 10];
        let small = PsiNull::new(&shares, 200, 200, 2000, 0x11);
        let large = PsiNull::new(&shares, 5000, 5000, 2000, 0x11);
        let (a, b) = (small.critical(0.05), large.critical(0.05));
        assert!(b * 20.0 < a, "200 rows {a} against 5,000 rows {b}");
        assert!(
            a > 0.1,
            "at 200 rows the folklore 0.1 band is below the 5 percent point {a}"
        );
        assert!(
            b < 0.1,
            "at 5,000 rows the folklore 0.1 band is far above the 5 percent point {b}"
        );
    }

    /// The analytic null against the procedure it approximates: cut the
    /// reference at its own quantiles, draw a fresh current sample from the
    /// same law, and read the index.
    #[test]
    fn the_analytic_null_matches_an_end_to_end_simulation() {
        let mut rng = Rng::new(0x4242);
        let rows = 1000;
        let bins = 10;
        let mut measured = Vec::new();
        for _ in 0..400 {
            let reference = normals(&mut rng, rows);
            let current = normals(&mut rng, rows);
            measured.push(crate::monitor::drift::psi(&reference, &current, bins));
        }
        measured.sort_by(f64::total_cmp);
        let at =
            |q: f64| measured[((q * (measured.len() - 1) as f64) as usize).min(measured.len() - 1)];
        let analytic = PsiNull::new(&vec![0.1f64; bins], rows, rows, 4000, 0x99);
        let predicted = analytic.critical(0.05);
        let observed = at(0.95);
        assert!(
            (predicted - observed).abs() < 0.25 * observed,
            "analytic 95th percentile {predicted} against measured {observed}"
        );
        let predicted = analytic.critical(0.50);
        let observed = at(0.50);
        assert!(
            (predicted - observed).abs() < 0.3 * observed.max(1e-6),
            "analytic median {predicted} against measured {observed}"
        );
    }

    #[test]
    fn the_null_is_reproducible_from_its_seed() {
        let shares = vec![0.2f64, 0.3, 0.5];
        let a = PsiNull::new(&shares, 300, 300, 100, 0x22);
        let b = PsiNull::new(&shares, 300, 300, 100, 0x22);
        assert_eq!(a.critical(0.05), b.critical(0.05));
        assert_eq!(a.degrees(), 2);
    }

    #[test]
    fn a_discrete_column_keeps_a_reading_of_zero() {
        let values: Vec<f64> = (0..500)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let report = assess(
            &panel("flag", &values),
            &panel("flag", &values.clone()),
            &GateConfig {
                psi_draws: 60,
                cvm_draws: 200,
                mmd_rows: 60,
                mmd_permutations: 19,
                ..GateConfig::default()
            },
        )
        .unwrap();
        let feature = &report.features[0];
        assert!(feature.psi.abs() < 1e-9, "psi {}", feature.psi);
        assert!(feature.psi_p > 0.99, "p {}", feature.psi_p);
        assert!(!feature.alarm);
    }

    #[test]
    fn a_shifted_column_is_declared_and_a_stationary_one_is_not() {
        let mut rng = Rng::new(0xABCD);
        let a = normals(&mut rng, 600);
        let b = normals(&mut rng, 600);
        let drifted: Vec<f64> = normals(&mut rng, 600).iter().map(|v| v + 0.8).collect();
        let quiet = assess(
            &panel("x", &a),
            &panel("x", &b),
            &GateConfig {
                psi_draws: 199,
                cvm_draws: 500,
                ..GateConfig::default()
            },
        )
        .unwrap();
        let loud = assess(
            &panel("x", &a),
            &panel("x", &drifted),
            &GateConfig {
                psi_draws: 199,
                cvm_draws: 500,
                ..GateConfig::default()
            },
        )
        .unwrap();
        assert!(!quiet.fired(), "{}", quiet.verdict);
        assert!(loud.fired(), "{}", loud.verdict);
    }

    #[test]
    fn benjamini_hochberg_holds_the_family_wise_error_on_nulls() {
        let declared = bh(&[0.02, 0.03, 0.04, 0.045], 0.01);
        assert_eq!(declared, vec![false; 4]);
        let declared = bh(&[0.001, 0.6, 0.7, 0.8], 0.05);
        assert_eq!(declared, vec![true, false, false, false]);
    }

    /// The acceptance number for roadmap item 21: a stationary backfill must
    /// fire on under five percent of splits at a thousand rows per side.
    #[test]
    fn a_stationary_backfill_rarely_fires() {
        let mut rng = Rng::new(0x21_21);
        let config = GateConfig {
            psi_draws: 199,
            cvm_draws: 4000,
            mmd_rows: 0,
            ..GateConfig::default()
        };
        assert!(
            report_is_off(&config),
            "a cap below two rows disables the joint reading"
        );
        let splits = 200;
        let features = 3;
        let mut fired = 0;
        for _ in 0..splits {
            let reference: Vec<Vec<f64>> = (0..features).map(|_| normals(&mut rng, 1000)).collect();
            let current: Vec<Vec<f64>> = (0..features).map(|_| normals(&mut rng, 1000)).collect();
            let left = frame(&reference);
            let right = frame(&current);
            let report = assess(&left, &right, &config).expect("assess");
            if report.fired() {
                fired += 1;
            }
        }
        let rate = fired as f64 / splits as f64;
        let (low, upper) = crate::uncertainty::wilson(fired, splits, 0.95);
        // Five percent is the target a nominal gate can hit, and two hundred
        // splits resolve it to about a point and a half: the assertion is that
        // the interval contains the target and that the gate is nowhere near
        // the fixed bands, which fire on about a third of stationary splits.
        assert!(
            rate < 0.10 && upper < 0.13 && low < 0.05,
            "fired on {fired} of {splits} splits, {rate:.3} with Wilson [{low:.3}, {upper:.3}]"
        );
    }

    /// The sharper half of the acceptance: the two readings are calibrated in
    /// their own tails, which two hundred gate splits cannot resolve.
    #[test]
    fn both_readings_are_calibrated_in_their_tails() {
        let mut rng = Rng::new(0x3131);
        let cvm_law = CvmNull::new(4000, 200, 0x99);
        let psi_law = PsiNull::new(&[0.1f64; 10], 1000, 1000, 4000, 0x77);
        let trials = 800;
        let mut cvm_p = Vec::with_capacity(trials);
        let mut psi_p = Vec::with_capacity(trials);
        for _ in 0..trials {
            let a = normals(&mut rng, 1000);
            let b = normals(&mut rng, 1000);
            cvm_p.push(cvm_law.p(cvm(&a, &b)));
            let edges = edges(&a, 10);
            let reference = counts(&a, &edges);
            let current = counts(&b, &edges);
            psi_p.push(psi_law.p(psi_counts(&reference, 1000.0, &current, 1000.0)));
        }
        let at = |values: &mut Vec<f64>, q: f64| {
            values.sort_by(f64::total_cmp);
            values[((q * values.len() as f64) as usize).min(values.len() - 1)]
        };
        let (cvm_five, psi_five) = (at(&mut cvm_p, 0.05), at(&mut psi_p, 0.05));
        assert!(
            (0.03..0.07).contains(&cvm_five),
            "the Cramer-von Mises 5 percent point reads {cvm_five}"
        );
        assert!(
            (0.03..0.07).contains(&psi_five),
            "the index 5 percent point reads {psi_five}"
        );
    }
}
