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

use std::collections::BTreeSet;
use std::io;

use serde::{Deserialize, Serialize};

use crate::churn::csvx;
use crate::detect::logistic::{FitParams, Logistic};
use crate::rng::Rng;
use crate::uncertainty::{bootstrap, Interval};

/// One row of the causal table.
#[derive(Clone, Debug)]
pub struct Row {
    pub features: Vec<f64>,
    pub treatment: bool,
    pub outcome: f64,
}

/// A causal table with named covariates.
#[derive(Clone, Debug)]
pub struct CausalData {
    pub names: Vec<String>,
    pub rows: Vec<Row>,
}

impl CausalData {
    /// Rows in the treated arm.
    pub fn treated(&self) -> usize {
        self.rows.iter().filter(|row| row.treatment).count()
    }

    /// Rows in the control arm.
    pub fn control(&self) -> usize {
        self.rows.len() - self.treated()
    }

    /// Feature matrix for the rows selected by `index`.
    pub fn design(&self, index: &[usize]) -> Vec<Vec<f64>> {
        index
            .iter()
            .map(|&at| self.rows[at].features.clone())
            .collect()
    }
}

/// Read a CSV with a binary treatment column and a numeric outcome.
///
/// Numeric covariates are used as they stand. A non-numeric column with at most
/// twelve levels becomes one indicator per level after the first, which is the
/// baseline. A column with one value per row is treated as an identifier and
/// skipped. A missing or unparseable numeric covariate is a refusal rather than
/// a zero, because a zero lands in the propensity model as data.
pub fn load_csv(path: &str, outcome: &str, treatment: &str) -> io::Result<CausalData> {
    let table = csvx::read(path)?;
    let outcome_at = table.require(outcome)?;
    let treatment_at = table.require(treatment)?;
    if outcome_at == treatment_at {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the outcome and the treatment are the same column",
        ));
    }
    if table.rows.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} holds no rows"),
        ));
    }
    let rows = table.rows.len();
    let mut numeric = Vec::new();
    let mut categorical = Vec::new();
    for column in 0..table.header.len() {
        if column == outcome_at || column == treatment_at {
            continue;
        }
        let values: Vec<&str> = table.rows.iter().map(|row| row[column].trim()).collect();
        if values.iter().all(|value| value.parse::<f64>().is_ok()) {
            numeric.push(column);
            continue;
        }
        let levels: BTreeSet<String> = values
            .iter()
            .filter(|value| !value.is_empty())
            .map(|value| value.to_lowercase())
            .collect();
        if levels.len() >= 2 && levels.len() <= 12 && levels.len() < rows {
            categorical.push((column, levels.into_iter().collect::<Vec<_>>()));
        }
    }
    let mut names = Vec::new();
    for &column in &numeric {
        names.push(table.header[column].clone());
    }
    for (column, levels) in &categorical {
        for level in levels.iter().skip(1) {
            names.push(format!("{}={level}", table.header[*column]));
        }
    }
    if names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the table has no usable covariates",
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for (number, cells) in table.rows.iter().enumerate() {
        let at = number + 2;
        let outcome = cells[outcome_at].trim().parse::<f64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row {at} has a non numeric outcome {:?}", cells[outcome_at]),
            )
        })?;
        let treatment = parse_treatment(&cells[treatment_at], at)?;
        let mut features = Vec::with_capacity(names.len());
        for &column in &numeric {
            let value = cells[column].trim().parse::<f64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "row {at} has a non numeric {} {:?}",
                        table.header[column], cells[column]
                    ),
                )
            })?;
            features.push(value);
        }
        for (column, levels) in &categorical {
            let value = cells[*column].trim().to_lowercase();
            for level in levels.iter().skip(1) {
                features.push(f64::from(u8::from(&value == level)));
            }
        }
        out.push(Row {
            features,
            treatment,
            outcome,
        });
    }
    Ok(CausalData { names, rows: out })
}

fn parse_treatment(cell: &str, at: usize) -> io::Result<bool> {
    match cell.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "treated" | "treatment" | "t" => Ok(true),
        "0" | "false" | "no" | "n" | "control" | "c" => Ok(false),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("row {at} has an unrecognised treatment value {other:?}"),
        )),
    }
}

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
    let mut uplift = vec![0.0; data.rows.len()];
    let mut propensity = vec![0.0; data.rows.len()];
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
            uplift[row] = treated_model.predict(features) - control_model.predict(features);
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
            note: "Observational estimate. Adjusts for observed covariates only; unmeasured confounding is not repaired by a propensity score.".to_string(),
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

fn qini_and_deciles(data: &CausalData, uplift: &[f64], transform: &[f64]) -> (f64, Vec<Decile>) {
    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    order.sort_by(|a, b| uplift[*b].total_cmp(&uplift[*a]));
    let n = order.len();
    let overall_treated: Vec<f64> = data
        .rows
        .iter()
        .filter(|row| row.treatment)
        .map(|row| row.outcome)
        .collect();
    let overall_control: Vec<f64> = data
        .rows
        .iter()
        .filter(|row| !row.treatment)
        .map(|row| row.outcome)
        .collect();
    let overall_diff = mean(&overall_treated) - mean(&overall_control);
    let bins = 10.min(n.max(1));
    let mut cumulative_effect = 0.0;
    let mut cumulative_rows = 0usize;
    let mut model_area = 0.0;
    let mut random_area = 0.0;
    let mut deciles = Vec::with_capacity(bins);

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
        let weighted = if difference.is_finite() {
            difference * slice.len() as f64 / n.max(1) as f64
        } else {
            0.0
        };
        cumulative_effect += weighted;
        cumulative_rows += slice.len();
        model_area += cumulative_effect;
        random_area += overall_diff * cumulative_rows as f64 / n.max(1) as f64;

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
    // The curve uses the observed decile difference, which is the usual uplift
    // chart, and is therefore descriptive under confounding. The IPW `effect`
    // column beside it is the adjusted reading; neither is a substitute for a
    // randomized experiment.
    let qini = (model_area - random_area) / (bins as f64 * overall_diff.abs().max(1e-9));
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

    /// A panel with a known positive effect on half the rows.
    fn synthetic() -> CausalData {
        let mut rng = Rng::new(77);
        let mut rows = Vec::new();
        for _ in 0..600 {
            let x0 = rng.normal(0.0, 1.0);
            let x1 = rng.normal(0.0, 1.0);
            let base = 1.0 / (1.0 + (-(-1.0 + 0.8 * x0)).exp());
            let treatment = rng.next_f64() < base;
            let effect = if treatment && x1 > 0.0 { 1.5 } else { 0.0 };
            let probability = (1.0 / (1.0 + (-(0.2 + 0.9 * x0 + effect)).exp())).clamp(0.02, 0.98);
            let outcome = f64::from(u8::from(rng.next_f64() < probability));
            rows.push(Row {
                features: vec![x0, x1],
                treatment,
                outcome,
            });
        }
        CausalData {
            names: vec!["x0".to_string(), "x1".to_string()],
            rows,
        }
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

    #[test]
    fn a_csv_round_trip_reads_numeric_and_categorical_covariates() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-causal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("causal.csv");
        let mut text = String::from("id,y,t,x,plan\n");
        for i in 0..60 {
            let treatment = u8::from(i % 2 == 0);
            let outcome = u8::from(i % 3 == 0);
            text.push_str(&format!(
                "c{i},{outcome},{treatment},{i},{}\n",
                if i % 4 == 0 { "basic" } else { "plus" }
            ));
        }
        std::fs::write(&path, text).unwrap();
        let data = load_csv(path.to_str().unwrap(), "y", "t").unwrap();
        assert_eq!(data.names, vec!["x".to_string(), "plan=plus".to_string()]);
        assert_eq!(data.treated(), 30);
        assert_eq!(data.control(), 30);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
