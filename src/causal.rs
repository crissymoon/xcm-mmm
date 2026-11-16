//! Causal uplift for a binary outcome with an observed treatment.
//!
//! The churn coefficients are directions, not effects. This module answers the
//! different question: if we intervened, by how much would the outcome move,
//! and for whom? It needs what an observational table can honestly supply: a
//! treatment column, a control group, and enough covariate overlap that the two
//! groups are comparable after adjustment.
//!
//! Two estimators are fitted side by side.
//!
//! * A T-learner fits one outcome model on the treated rows and one on the
//!   control rows, with cross-fitting so no row is scored by a model that saw
//!   it. The difference of the two probabilities is the uplift.
//! * Inverse probability weighting fits a propensity model, clips the weights
//!   at a declared quantile, and reports the Horvitz-Thompson mean. It is a
//!   second reading of the same average effect under a different assumption.
//!
//! Heterogeneity is reported as the uplift difference across the median of each
//! covariate, with a standard error, and as a decile table of predicted uplift.
//! A placebo permutation of the treatment gives a p-value for the average
//! effect under the sharp null of no effect. None of this is identification
//! from observational data alone: unmeasured confounding is not repaired by a
//! propensity score, and the card says so.
//!
//! The headline average effect is a doubly robust (AIPW) estimate built from
//! the same cross-fitted nuisances. Each row contributes
//! `mu1 - mu0 + t (y - mu1) / e - (1 - t) (y - mu0) / (1 - e)`, where `mu0` and
//! `mu1` are the control and treated arm probabilities and `e` is the
//! propensity. Averaging that pseudo-outcome is Neyman orthogonal to first
//! order in the nuisances (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen,
//! Newey and Robins, 2018): a small nuisance error moves the estimate by second
//! order, which the plain difference of two arm models does not provide.
//!
//! What that buys is robustness to approximation error, not identification. The
//! estimate still adjusts for the observed covariates only, so a confounder
//! nobody measured is untouched. The E-value (VanderWeele and Ding, 2017) says
//! how strong such a confounder would have to be, on the risk ratio scale, to
//! explain the estimate away; a large E-value is a statement about how much
//! confounding the reading tolerates, never proof that confounding is absent.
//!
//! Two overlap estimands read the same pseudo-outcomes. The overlap weighted
//! effect weights rows by `e (1 - e)` (Li, Morgan and Zaslavsky, 2018), which
//! targets the population where a decision is actually possible. The Crump
//! trimmed effect keeps the rows inside a propensity bound chosen by the rule
//! of Crump, Hotz, Imbens and Mitnik (2009). The policy value curve gives the
//! decision reading: what a rule that treats the top fraction of a ranking is
//! worth, next to treating everyone and treating nobody. Qini stays as a
//! descriptive summary of the same ranking.

use serde::{Deserialize, Serialize};

use crate::detect::logistic::{FitParams, Logistic};
use crate::rng::Rng;
use crate::stats::z_for_level;
use crate::uncertainty::{bootstrap, Interval};

mod estimators;
mod table;

pub use estimators::PolicyPoint;
pub use table::{load_csv, CausalData, Row};

use estimators::{
    crump_rule, dr_mean, e_value_of_difference, overlap_weighted, policy_curve, pseudo_outcomes,
    CRUMP_FLOOR,
};

/// How a causal fit is configured.
#[derive(Clone, Debug)]
pub struct CausalConfig {
    /// Cross-fitting folds.
    pub folds: usize,
    /// Bootstrap resamples for the interval on the average effect.
    pub bootstrap: usize,
    /// Nominal interval level.
    pub level: f64,
    /// Placebo permutations.
    pub permutations: usize,
    /// Propensity weights above this quantile are clipped.
    pub trim: f64,
    /// Smallest row count the fit will touch.
    pub min_rows: usize,
    /// Seed for the folds, the bootstrap and the placebo.
    pub seed: u64,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            folds: 5,
            bootstrap: 400,
            level: 0.95,
            permutations: 200,
            trim: 0.95,
            min_rows: 80,
            seed: 20261115,
        }
    }
}

/// Uplift of one decile of the ranked score.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Decile {
    pub decile: usize,
    pub rows: usize,
    pub treated: usize,
    pub predicted: f64,
    pub observed_treated: f64,
    pub observed_control: f64,
    pub difference: f64,
    /// Stabilized IPW effect for this decile.
    pub effect: f64,
}

/// Uplift difference across the median of one covariate.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Heterogeneity {
    pub feature: String,
    pub low: f64,
    pub high: f64,
    pub difference: f64,
    pub standard_error: f64,
    pub z: f64,
}

/// What the causal fit found.
///
/// Every field added after the first release carries `serde(default)`, so a
/// report written by an older build still parses.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CausalReport {
    pub rows: usize,
    pub treated: usize,
    pub control: usize,
    pub features: usize,
    /// Cross-fitted T-learner average effect.
    pub ate: f64,
    pub ate_interval: Interval,
    /// Cross-fitted inverse probability weighted average effect.
    pub ipw_ate: f64,
    pub ipw_interval: Interval,
    /// Observation-weighted Qini coefficient of the uplift ranking.
    pub qini: f64,
    /// Share of rows inside the propensity overlap band `[0.1, 0.9]`.
    pub overlap: f64,
    /// Share of propensity weights clipped at the trim quantile.
    pub clipped: f64,
    /// Placebo p-value for the sharp null of no effect.
    pub placebo_p: f64,
    pub deciles: Vec<Decile>,
    pub heterogeneity: Vec<Heterogeneity>,
    pub note: String,
    /// Doubly robust (AIPW) average effect, the headline estimate.
    #[serde(default)]
    pub dr_ate: f64,
    /// Influence function standard error of `dr_ate`.
    #[serde(default)]
    pub dr_se: f64,
    /// Normal interval on `dr_ate`, built from its influence function.
    #[serde(default)]
    pub dr_interval: Interval,
    /// Rows whose bias correction term hit the declared cap.
    #[serde(default)]
    pub dr_clipped: usize,
    /// Overlap weighted average effect, weights `e (1 - e)` scaled to mean one.
    #[serde(default)]
    pub overlap_weighted_ate: f64,
    /// Influence function standard error of the overlap weighted effect.
    #[serde(default)]
    pub overlap_weighted_se: f64,
    /// Crump trimmed average effect on the retained rows.
    #[serde(default)]
    pub crump_ate: f64,
    /// Influence function standard error of the Crump trimmed effect.
    #[serde(default)]
    pub crump_se: f64,
    /// Rows the Crump rule retained.
    #[serde(default)]
    pub crump_rows: usize,
    /// Propensity bound the Crump rule selected, zero when it selected none.
    #[serde(default)]
    pub crump_bound: f64,
    /// True when the rule retained too few rows and the untrimmed value stands.
    #[serde(default)]
    pub crump_refused: bool,
    /// E-value of `dr_ate`, which is the better estimator here, so the reading
    /// is built on it rather than on the T-learner difference: an unmeasured
    /// confounder would need an association with both the treatment and the
    /// outcome of at least this risk ratio to explain the estimate away.
    #[serde(default)]
    pub e_value: f64,
    /// E-value of the `dr_interval` bound nearest the null, which is the
    /// confounding the interval itself tolerates.
    #[serde(default)]
    pub e_value_ci: f64,
    /// Policy value curve of the T-learner ranking, from treating nobody up.
    #[serde(default)]
    pub policy: Vec<PolicyPoint>,
    /// Value of treating every row, the mean pseudo-outcome.
    #[serde(default)]
    pub policy_value_all: f64,
    /// Value of treating no row, which is zero by construction.
    #[serde(default)]
    pub policy_value_none: f64,
}

/// A fitted model that scores a new row's uplift.
#[derive(Clone, Debug)]
pub struct CausalModel {
    pub names: Vec<String>,
    pub treated: Logistic,
    pub control: Logistic,
    pub propensity: Logistic,
    pub trim: f64,
}

impl CausalModel {
    /// Uplift for one row: treated probability minus control probability.
    pub fn uplift(&self, features: &[f64]) -> f64 {
        self.treated.predict(features) - self.control.predict(features)
    }

    /// Propensity for one row.
    pub fn propensity(&self, features: &[f64]) -> f64 {
        self.propensity.predict(features)
    }

    /// A row is inside the overlap band when both arms are plausible.
    pub fn in_overlap(&self, features: &[f64]) -> bool {
        let p = self.propensity(features);
        (0.1..=0.9).contains(&p)
    }
}

/// A report and the model that produced it.
#[derive(Clone, Debug)]
pub struct CausalFit {
    pub report: CausalReport,
    pub model: CausalModel,
}

/// Fit the T-learner, the propensity model, the intervals and the audits.
pub fn fit(data: &CausalData, config: &CausalConfig) -> Result<CausalFit, String> {
    if data.rows.len() < config.min_rows {
        return Err(format!(
            "{} rows is below the {} row floor for a causal fit",
            data.rows.len(),
            config.min_rows
        ));
    }
    if data.treated() < 10 || data.control() < 10 {
        return Err(format!(
            "the arms are too small: {} treated, {} control",
            data.treated(),
            data.control()
        ));
    }
    let folds = config.folds.max(2).min(data.rows.len() / 10);
    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    let mut rng = Rng::new(config.seed);
    for i in (1..order.len()).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let mut assignment = vec![0usize; data.rows.len()];
    for (position, &row) in order.iter().enumerate() {
        assignment[row] = position % folds;
    }

    let params = FitParams {
        steps: 400,
        ..FitParams::default()
    };
    // Cross-fitted nuisances. A fold that is skipped keeps these priors, which
    // are honest marginal rates rather than zeros that would distort a
    // pseudo-outcome.
    let control_prior = if data.control() == 0 {
        0.5
    } else {
        data.rows
            .iter()
            .filter(|row| !row.treatment)
            .map(|row| row.outcome)
            .sum::<f64>()
            / data.control() as f64
    };
    let treated_prior = if data.treated() == 0 {
        0.5
    } else {
        data.rows
            .iter()
            .filter(|row| row.treatment)
            .map(|row| row.outcome)
            .sum::<f64>()
            / data.treated() as f64
    };
    let propensity_prior = data.treated() as f64 / data.rows.len().max(1) as f64;
    let mut mu0 = vec![control_prior; data.rows.len()];
    let mut mu1 = vec![treated_prior; data.rows.len()];
    let mut uplift = vec![treated_prior - control_prior; data.rows.len()];
    let mut propensity = vec![propensity_prior; data.rows.len()];
    for fold in 0..folds {
        let train: Vec<usize> = (0..data.rows.len())
            .filter(|row| assignment[*row] != fold)
            .collect();
        let test: Vec<usize> = (0..data.rows.len())
            .filter(|row| assignment[*row] == fold)
            .collect();
        if train.is_empty() || test.is_empty() {
            continue;
        }
        let design = data.design(&train);
        let treated: Vec<Vec<f64>> = train
            .iter()
            .zip(&design)
            .filter(|(row, _)| data.rows[**row].treatment)
            .map(|(_, row)| row.clone())
            .collect();
        let treated_y: Vec<bool> = train
            .iter()
            .filter(|row| data.rows[**row].treatment)
            .map(|row| data.rows[*row].outcome > 0.5)
            .collect();
        let control: Vec<Vec<f64>> = train
            .iter()
            .zip(&design)
            .filter(|(row, _)| !data.rows[**row].treatment)
            .map(|(_, row)| row.clone())
            .collect();
        let control_y: Vec<bool> = train
            .iter()
            .filter(|row| !data.rows[**row].treatment)
            .map(|row| data.rows[*row].outcome > 0.5)
            .collect();
        if treated.len() < 5 || control.len() < 5 {
            continue;
        }
        let treated_model = Logistic::fit(&treated, &treated_y, &params);
        let control_model = Logistic::fit(&control, &control_y, &params);
        let treatment: Vec<bool> = train.iter().map(|row| data.rows[*row].treatment).collect();
        let propensity_model = Logistic::fit(&design, &treatment, &params);
        for row in test {
            let features = &data.rows[row].features;
            mu1[row] = treated_model.predict(features);
            mu0[row] = control_model.predict(features);
            uplift[row] = mu1[row] - mu0[row];
            propensity[row] = propensity_model.predict(features);
        }
    }

    let ate = mean(&uplift);
    let ate_interval = bootstrap(
        &uplift,
        &vec![true; uplift.len()],
        config.bootstrap,
        config.level,
        config.seed ^ 0xA17,
        |values, _| mean(values),
    );
    let (ipw_ate, clipped, weights) = ipw(data, &propensity, config.trim);
    let ipw_interval = bootstrap(
        &weights,
        &vec![true; weights.len()],
        config.bootstrap,
        config.level,
        config.seed ^ 0xB23,
        |values, _| mean(values),
    );
    let overlap = propensity
        .iter()
        .filter(|p| (0.1..=0.9).contains(*p))
        .count() as f64
        / propensity.len().max(1) as f64;

    let (qini, deciles) = qini_and_deciles(data, &uplift, &weights);
    let heterogeneity = heterogeneity(data, &uplift);
    let placebo_p = placebo(data, &propensity, ate, config);

    let (psi, dr_clipped) = pseudo_outcomes(data, &mu0, &mu1, &propensity);
    let (dr_ate, dr_se) = dr_mean(&psi, &(0..psi.len()).collect::<Vec<_>>());
    let z = z_for_level(config.level);
    let dr_interval = Interval {
        point: dr_ate,
        low: dr_ate - z * dr_se,
        high: dr_ate + z * dr_se,
        level: config.level,
        resamples: 0,
    };
    let (overlap_weighted_ate, overlap_weighted_se) = overlap_weighted(&psi, &propensity);
    let (crump_bound_value, kept) = crump_rule(&propensity);
    let crump_refused = kept.len() < CRUMP_FLOOR;
    let (crump_value, crump_error) = dr_mean(&psi, &kept);
    let (crump_ate, crump_se) = if crump_refused {
        (dr_ate, dr_se)
    } else {
        (crump_value, crump_error)
    };
    // The E-value reads the risk difference against the control arm risk, and
    // it is built on the doubly robust point rather than the T-learner because
    // the doubly robust reading is the less nuisance sensitive of the two.
    let e_value_point = e_value_of_difference(dr_ate, control_prior);
    let near_null = if dr_ate >= 0.0 {
        dr_interval.low
    } else {
        dr_interval.high
    };
    let e_value_ci = e_value_of_difference(near_null, control_prior);
    let policy = policy_curve(&uplift, &psi);
    let policy_value_all = dr_ate;
    let policy_value_none = 0.0;

    let design = data.design(&(0..data.rows.len()).collect::<Vec<_>>());
    let treatment: Vec<bool> = data.rows.iter().map(|row| row.treatment).collect();
    let treated_rows: Vec<Vec<f64>> = design
        .iter()
        .zip(&data.rows)
        .filter(|(_, row)| row.treatment)
        .map(|(row, _)| row.clone())
        .collect();
    let treated_y: Vec<bool> = data
        .rows
        .iter()
        .filter(|row| row.treatment)
        .map(|row| row.outcome > 0.5)
        .collect();
    let control_rows: Vec<Vec<f64>> = design
        .iter()
        .zip(&data.rows)
        .filter(|(_, row)| !row.treatment)
        .map(|(row, _)| row.clone())
        .collect();
    let control_y: Vec<bool> = data
        .rows
        .iter()
        .filter(|row| !row.treatment)
        .map(|row| row.outcome > 0.5)
        .collect();
    let model = CausalModel {
        names: data.names.clone(),
        treated: Logistic::fit(&treated_rows, &treated_y, &params),
        control: Logistic::fit(&control_rows, &control_y, &params),
        propensity: Logistic::fit(&design, &treatment, &params),
        trim: config.trim,
    };
    Ok(CausalFit {
        report: CausalReport {
            rows: data.rows.len(),
            treated: data.treated(),
            control: data.control(),
            features: data.names.len(),
            ate,
            ate_interval,
            ipw_ate,
            ipw_interval,
            qini,
            overlap,
            clipped,
            placebo_p,
            deciles,
            heterogeneity,
            note: "Observational estimate. The doubly robust mean is Neyman orthogonal to first order in the nuisances; it still adjusts for observed covariates only, and an E-value states what unmeasured confounding it would take, not that none exists.".to_string(),
            dr_ate,
            dr_se,
            dr_interval,
            dr_clipped,
            overlap_weighted_ate,
            overlap_weighted_se,
            crump_ate,
            crump_se,
            crump_rows: kept.len(),
            crump_bound: crump_bound_value,
            crump_refused,
            e_value: e_value_point,
            e_value_ci,
            policy,
            policy_value_all,
            policy_value_none,
        },
        model,
    })
}

fn ipw(data: &CausalData, propensity: &[f64], trim: f64) -> (f64, f64, Vec<f64>) {
    let cap = 1.0 / (1.0 - trim.clamp(0.5, 0.999));
    let mut treated_weight = 0.0;
    let mut control_weight = 0.0;
    let mut clipped = 0usize;
    let mut weights = Vec::with_capacity(data.rows.len());
    for (row, p) in data.rows.iter().zip(propensity) {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        let weight = if row.treatment {
            1.0 / p
        } else {
            1.0 / (1.0 - p)
        };
        if weight > cap {
            clipped += 1;
        }
        let weight = weight.min(cap);
        if row.treatment {
            treated_weight += weight;
        } else {
            control_weight += weight;
        }
        weights.push(weight);
    }
    // The stabilized (Hajek) form divides each arm by its own weight total.
    // It is the same estimand as the Horvitz-Thompson mean and much less
    // sensitive to one row with a small propensity.
    let mut scores = Vec::with_capacity(weights.len());
    for (row, weight) in data.rows.iter().zip(&weights) {
        let value = if row.treatment {
            row.outcome * weight / treated_weight.max(1e-9)
        } else {
            -row.outcome * weight / control_weight.max(1e-9)
        };
        scores.push(value);
    }
    let ate = scores.iter().sum::<f64>();
    // Put the contributions on a per-row scale so the bootstrap of their mean
    // estimates the average effect rather than the average effect divided by n.
    let scale = data.rows.len().max(1) as f64;
    for value in scores.iter_mut() {
        *value *= scale;
    }
    (ate, clipped as f64 / weights.len().max(1) as f64, scores)
}

/// Decile table of the uplift ranking and the Qini coefficient of its curve.
///
/// The curve is the cumulative within-decile treated-minus-control difference,
/// which is the usual uplift chart and therefore descriptive under confounding.
/// The random baseline is that same within-decile total spread evenly over the
/// ranking, not the marginal treated-minus-control difference across the whole
/// table. The two coincide only when the treated share is equal in every
/// decile, which it never is: in a confounded table the top decile can be
/// mostly treated while the bottom is mostly control, the marginal difference
/// is then much larger than the average within-decile difference, and a
/// correctly ranked model reports a negative Qini. That was the shipped
/// behaviour and it is fixed here.
///
/// The IPW `effect` column beside the observed difference is the adjusted
/// reading; neither is a substitute for a randomized experiment.
fn qini_and_deciles(data: &CausalData, uplift: &[f64], transform: &[f64]) -> (f64, Vec<Decile>) {
    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    order.sort_by(|a, b| uplift[*b].total_cmp(&uplift[*a]));
    let n = order.len();
    let bins = 10.min(n.max(1));
    let mut deciles = Vec::with_capacity(bins);
    // Within-decile difference and rows in the bin.
    let mut parts: Vec<(f64, usize)> = Vec::with_capacity(bins);

    for bin in 0..bins {
        let start = bin * n / bins;
        let end = (bin + 1) * n / bins;
        let slice = &order[start..end];
        let effect: f64 = slice.iter().map(|row| transform[*row]).sum();
        let treated: Vec<&Row> = slice
            .iter()
            .map(|row| &data.rows[*row])
            .filter(|row| row.treatment)
            .collect();
        let control: Vec<&Row> = slice
            .iter()
            .map(|row| &data.rows[*row])
            .filter(|row| !row.treatment)
            .collect();
        let observed_treated = if treated.is_empty() {
            f64::NAN
        } else {
            treated.iter().map(|row| row.outcome).sum::<f64>() / treated.len() as f64
        };
        let observed_control = if control.is_empty() {
            f64::NAN
        } else {
            control.iter().map(|row| row.outcome).sum::<f64>() / control.len() as f64
        };
        let difference = observed_treated - observed_control;
        parts.push((difference, slice.len()));
        deciles.push(Decile {
            decile: bin + 1,
            rows: slice.len(),
            treated: treated.len(),
            predicted: mean(&slice.iter().map(|row| uplift[*row]).collect::<Vec<_>>()),
            observed_treated,
            observed_control,
            difference,
            effect,
        });
    }

    let share = |rows: usize| rows as f64 / n.max(1) as f64;
    let total: f64 = parts
        .iter()
        .map(|(difference, rows)| {
            if difference.is_finite() {
                difference * share(*rows)
            } else {
                0.0
            }
        })
        .sum();
    let mut cumulative = 0.0;
    let mut cumulative_rows = 0usize;
    let mut model_area = 0.0;
    let mut random_area = 0.0;
    for (difference, rows) in &parts {
        if difference.is_finite() {
            cumulative += difference * share(*rows);
        }
        cumulative_rows += *rows;
        model_area += cumulative;
        random_area += total * share(cumulative_rows);
    }
    let qini = if total.abs() > 1e-12 {
        (model_area - random_area) / (bins as f64 * total.abs())
    } else {
        0.0
    };
    (qini, deciles)
}

fn heterogeneity(data: &CausalData, uplift: &[f64]) -> Vec<Heterogeneity> {
    let mut out = Vec::new();
    for (column, name) in data.names.iter().enumerate() {
        let mut values: Vec<f64> = data.rows.iter().map(|row| row.features[column]).collect();
        values.sort_by(f64::total_cmp);
        let median = crate::uncertainty::quantile(&values, 0.5);
        let low: Vec<f64> = data
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.features[column] <= median)
            .map(|(at, _)| uplift[at])
            .collect();
        let high: Vec<f64> = data
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.features[column] > median)
            .map(|(at, _)| uplift[at])
            .collect();
        if low.len() < 5 || high.len() < 5 {
            continue;
        }
        let (low_mean, high_mean) = (mean(&low), mean(&high));
        let low_var = variance(&low);
        let high_var = variance(&high);
        let standard_error = (low_var / low.len() as f64 + high_var / high.len() as f64).sqrt();
        let difference = high_mean - low_mean;
        out.push(Heterogeneity {
            feature: name.clone(),
            low: low_mean,
            high: high_mean,
            difference,
            standard_error,
            z: if standard_error > 0.0 {
                difference / standard_error
            } else {
                0.0
            },
        });
    }
    out.sort_by(|a, b| b.difference.abs().total_cmp(&a.difference.abs()));
    out
}

fn placebo(data: &CausalData, propensity: &[f64], observed: f64, config: &CausalConfig) -> f64 {
    let mut rng = Rng::new(config.seed ^ 0xC31);
    let mut treatment: Vec<bool> = data.rows.iter().map(|row| row.treatment).collect();
    let mut extreme = 0usize;
    for _ in 0..config.permutations.max(1) {
        for i in (1..treatment.len()).rev() {
            let j = rng.below(i + 1);
            treatment.swap(i, j);
        }
        let mut signed = 0.0;
        for ((row, p), flag) in data.rows.iter().zip(propensity).zip(&treatment) {
            let p = p.clamp(1e-6, 1.0 - 1e-6);
            if *flag {
                signed += row.outcome / p;
            } else {
                signed -= row.outcome / (1.0 - p);
            }
        }
        let value = signed / data.rows.len().max(1) as f64;
        if value.abs() >= observed.abs() {
            extreme += 1;
        }
    }
    (extreme + 1) as f64 / (config.permutations.max(1) + 1) as f64
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values
        .iter()
        .map(|value| (value - m) * (value - m))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panel with a known logit effect on half the rows, or a flat outcome
    /// when `constant` is set, which is how a null panel is expressed.
    fn panel(rows: usize, effect: f64, constant: Option<f64>) -> CausalData {
        let mut rng = Rng::new(77);
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let x0 = rng.normal(0.0, 1.0);
            let x1 = rng.normal(0.0, 1.0);
            let base = 1.0 / (1.0 + (-(-1.0 + 0.8 * x0)).exp());
            let treatment = rng.next_f64() < base;
            let shift = if treatment && x1 > 0.0 { effect } else { 0.0 };
            let probability = (1.0 / (1.0 + (-(0.2 + 0.9 * x0 + shift)).exp())).clamp(0.02, 0.98);
            let outcome = match constant {
                Some(value) => value,
                None => f64::from(u8::from(rng.next_f64() < probability)),
            };
            out.push(Row {
                features: vec![x0, x1],
                treatment,
                outcome,
            });
        }
        CausalData {
            names: vec!["x0".to_string(), "x1".to_string()],
            rows: out,
        }
    }

    /// A panel with a known positive effect on half the rows.
    fn synthetic() -> CausalData {
        panel(600, 1.5, None)
    }

    /// A panel with no treatment effect at all, for readings that must vanish.
    fn no_effect() -> CausalData {
        panel(1200, 0.0, None)
    }

    /// A panel whose outcome is the same number on every row.
    fn flat_outcome() -> CausalData {
        panel(1200, 0.0, Some(0.4))
    }

    fn quick(data: &CausalData) -> CausalReport {
        fit(
            data,
            &CausalConfig {
                bootstrap: 60,
                permutations: 20,
                ..CausalConfig::default()
            },
        )
        .expect("a causal fit")
        .report
    }

    #[test]
    fn the_doubly_robust_estimate_agrees_with_the_t_learner() {
        let report = quick(&synthetic());
        assert!(report.dr_ate > 0.0, "dr {}", report.dr_ate);
        assert!(
            (report.dr_ate - report.ate).abs() < 0.06,
            "dr {} t-learner {}",
            report.dr_ate,
            report.ate
        );
        assert!(report.dr_se > 0.0, "se {}", report.dr_se);
    }

    #[test]
    fn the_doubly_robust_interval_brackets_its_point() {
        let report = quick(&synthetic());
        assert!(report.dr_interval.low <= report.dr_ate);
        assert!(report.dr_interval.high >= report.dr_ate);
        assert_eq!(report.dr_interval.resamples, 0);
        assert!((report.dr_interval.level - 0.95).abs() < 1e-12);
    }

    #[test]
    fn a_planted_effect_survives_an_e_value_of_one_and_a_half() {
        let report = quick(&synthetic());
        assert!(
            report.e_value > 1.5,
            "e-value {} for dr {}",
            report.e_value,
            report.dr_ate
        );
        assert!(report.e_value_ci > 1.0, "e-value ci {}", report.e_value_ci);
    }

    #[test]
    fn a_flat_outcome_gives_an_e_value_near_one() {
        let report = quick(&flat_outcome());
        assert!(
            (report.dr_ate).abs() < 0.02,
            "dr {} on a constant outcome",
            report.dr_ate
        );
        assert!(
            report.e_value < 1.5,
            "e-value {} should be near one",
            report.e_value
        );
        assert!(report.e_value >= 1.0);
    }

    #[test]
    fn the_overlap_weighted_reading_is_finite_with_a_positive_error() {
        let report = quick(&synthetic());
        assert!(report.overlap_weighted_ate.is_finite());
        assert!(report.overlap_weighted_se > 0.0);
    }

    #[test]
    fn the_crump_rule_trims_some_rows_but_not_the_panel() {
        let report = quick(&synthetic());
        assert!(!report.crump_refused, "the rule should keep enough rows");
        assert!(
            report.crump_rows > 50,
            "kept {} of {}",
            report.crump_rows,
            report.rows
        );
        assert!(report.crump_rows < report.rows);
        assert!((0.0..0.5).contains(&report.crump_bound));
        assert!(report.crump_se > 0.0);
    }

    #[test]
    fn the_top_decile_is_worth_more_than_treating_everyone() {
        let report = quick(&synthetic());
        let top = report
            .policy
            .iter()
            .find(|point| (point.fraction - 0.1).abs() < 1e-12)
            .expect("a tenth of the rows");
        assert!(top.rows > 0);
        assert!(
            top.value > report.policy_value_all,
            "top decile {} against all {}",
            top.value,
            report.policy_value_all
        );
        assert_eq!(report.policy_value_none, 0.0);
        assert_eq!(report.policy.len(), 11);
        assert!((report.policy_value_all - report.dr_ate).abs() < 1e-12);
    }

    #[test]
    fn a_no_effect_panel_reports_no_effect() {
        let report = quick(&no_effect());
        assert!(
            report.dr_ate.abs() < 0.08,
            "dr {} on a null panel",
            report.dr_ate
        );
        assert!(report.e_value < 1.5, "e-value {}", report.e_value);
    }

    #[test]
    fn a_planted_effect_is_recovered_with_the_right_sign() {
        let data = synthetic();
        let fit = fit(
            &data,
            &CausalConfig {
                bootstrap: 100,
                permutations: 40,
                ..CausalConfig::default()
            },
        )
        .expect("a causal fit");
        assert!(
            fit.report.ate > 0.08,
            "ate {} treated {} control {}",
            fit.report.ate,
            fit.report.treated,
            fit.report.control
        );
        assert!(fit.report.ipw_ate > 0.05, "ipw {}", fit.report.ipw_ate);
        assert!(fit.report.ate_interval.low <= fit.report.ate);
        assert!(fit.report.ate_interval.high >= fit.report.ate);
        assert!(fit.report.placebo_p < 0.2);
    }

    #[test]
    fn heterogeneity_names_the_effect_modifier() {
        let data = synthetic();
        let fit = fit(
            &data,
            &CausalConfig {
                bootstrap: 60,
                permutations: 20,
                ..CausalConfig::default()
            },
        )
        .expect("a causal fit");
        assert!(!fit.report.heterogeneity.is_empty());
        let x1 = fit
            .report
            .heterogeneity
            .iter()
            .find(|row| row.feature == "x1")
            .expect("x1");
        assert!(x1.difference > 0.05, "x1 difference {}", x1.difference);
    }

    #[test]
    fn an_older_report_still_parses_without_the_new_fields() {
        let old = r#"{"rows":100,"treated":50,"control":50,"features":2,"ate":0.1,
            "ate_interval":{"point":0.1,"low":0.0,"high":0.2,"level":0.95,"resamples":20},
            "ipw_ate":0.1,"ipw_interval":{"point":0.1,"low":0.0,"high":0.2,"level":0.95,"resamples":20},
            "qini":0.0,"overlap":0.8,"clipped":0.0,"placebo_p":0.5,"deciles":[],
            "heterogeneity":[],"note":""}"#;
        let report: CausalReport = serde_json::from_str(old).expect("an older report parses");
        assert_eq!(report.rows, 100);
        assert_eq!(report.dr_ate, 0.0);
        assert!(report.policy.is_empty());
        assert_eq!(report.crump_rows, 0);
    }

    /// A panel whose outcome rises with one feature only under treatment, with
    /// the arms alternating so every decile holds both.
    fn ranked_panel() -> CausalData {
        CausalData {
            names: vec!["x".to_string()],
            rows: (0..200)
                .map(|i| {
                    let treated = i % 2 == 0;
                    Row {
                        features: vec![i as f64],
                        treatment: treated,
                        outcome: f64::from(u8::from(treated && i >= 100)),
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn the_qini_is_positive_when_the_ranking_is_right() {
        let data = ranked_panel();
        let weights = vec![1.0; data.rows.len()];
        let high_first: Vec<f64> = data.rows.iter().map(|row| row.features[0]).collect();
        let low_first: Vec<f64> = high_first.iter().map(|x| -x).collect();
        let (good, deciles) = qini_and_deciles(&data, &high_first, &weights);
        let (bad, _) = qini_and_deciles(&data, &low_first, &weights);
        assert!(good > 0.2, "qini {good}");
        assert!(bad < 0.0, "qini {bad}");
        // The deciles have to be front-loaded for the top bin to be the point.
        assert!(deciles[0].difference > deciles[9].difference);
    }

    #[test]
    fn a_constant_treatment_is_refused() {
        let data = CausalData {
            names: vec!["x".to_string()],
            rows: (0..100)
                .map(|i| Row {
                    features: vec![i as f64],
                    treatment: true,
                    outcome: f64::from(u8::from(i % 2 == 0)),
                })
                .collect(),
        };
        assert!(fit(&data, &CausalConfig::default()).is_err());
    }
}
