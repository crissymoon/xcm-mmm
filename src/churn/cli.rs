//! The `churn` command surface, beside the module it drives.
//!
//! `extract` writes the panel, the series and the reason readings; `fit` writes
//! the baseline and its card; `info` reads them back. The split matters because
//! the panel is derived from a customer table while the model is a choice made
//! on top of it: a reader can take the panel without accepting the regression.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use clap::{Args, Subcommand};

use super::fit;
use super::panel::{self, Report, Summary};
use crate::causal::{self, CausalConfig};
use crate::data::Dataset;
use crate::digest::sha256;
use crate::monitor::{self, Audit, Registry};

#[derive(Args)]
pub struct ChurnArgs {
    #[command(subcommand)]
    pub command: ChurnCommand,
}

#[derive(Subcommand)]
pub enum ChurnCommand {
    /// Build the panel, the retention series and the reason readings.
    Extract(ChurnExtractArgs),
    /// Fit the held-out retention baseline and write its card.
    Fit(ChurnFitArgs),
    /// Print the panel, the series, the drivers and the fit.
    Info(ChurnInfoArgs),
    /// Rebuild the folder from its source and report every disagreement.
    Check(ChurnCheckArgs),
    /// Estimate an uplift from a panel that carries a treatment column.
    Uplift(ChurnUpliftArgs),
    /// Compare a reference panel with a current one by PSI and KS.
    Monitor(ChurnMonitorArgs),
    /// Publish the current artifact as the next registry version.
    Publish(ChurnPublishArgs),
    /// Print the published versions, their audit trails and the log.
    Versions(ChurnVersionsArgs),
    /// Restore the previous registry version and move the pointer back.
    Rollback(ChurnRollbackArgs),
}

#[derive(Args)]
pub struct ChurnExtractArgs {
    #[arg(long, default_value = "data/churn")]
    pub dir: String,
}

#[derive(Args)]
pub struct ChurnFitArgs {
    #[arg(long, default_value = "data/churn")]
    pub dir: String,
    #[arg(long, default_value_t = fit::FOLDS)]
    pub folds: usize,
}

#[derive(Args)]
pub struct ChurnInfoArgs {
    #[arg(long, default_value = "data/churn")]
    pub dir: String,
}

#[derive(Args)]
pub struct ChurnCheckArgs {
    #[arg(long, default_value = "data/churn")]
    pub dir: String,
}

#[derive(Args)]
pub struct ChurnUpliftArgs {
    /// CSV with a treatment column and a binary outcome.
    #[arg(long, default_value = "data/churn/panel.csv")]
    pub data: String,
    /// Outcome column. A churn panel uses `churn`.
    #[arg(long, default_value = "churn")]
    pub outcome: String,
    /// Treatment column, coded 1/0, true/false, treated/control.
    #[arg(long)]
    pub treatment: String,
    /// Cross-fitting folds.
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    /// Bootstrap resamples for the average effect.
    #[arg(long, default_value_t = 400)]
    pub bootstrap: usize,
    /// Placebo permutations.
    #[arg(long, default_value_t = 200)]
    pub permutations: usize,
    /// Write the report as JSON to this path.
    #[arg(long)]
    pub out: Option<String>,
    /// Print JSON instead of a table.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct ChurnMonitorArgs {
    /// Reference panel, usually the rows the model was fitted on.
    #[arg(long, default_value = "data/churn/panel.csv")]
    pub reference: String,
    /// Current panel.
    #[arg(long)]
    pub current: String,
    /// Target column that [`Dataset`] keeps out of the feature matrix.
    #[arg(long, default_value = "churn")]
    pub target: String,
    /// A fixed index threshold in PSI units. Absent means the gate calibrates
    /// its own against a null at the row counts it is reading.
    #[arg(long)]
    pub threshold: Option<f64>,
    /// Reference quantile bins.
    #[arg(long, default_value_t = 10)]
    pub bins: usize,
    /// Level for the per-feature tests and the joint reading.
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,
    /// Null draws for the index, per feature.
    #[arg(long, default_value_t = 499)]
    pub psi_draws: usize,
    /// Null draws for the Cramer-von Mises law.
    #[arg(long, default_value_t = 4000)]
    pub cvm_draws: usize,
    /// Rows per side kept for the joint reading. Below two turns it off.
    #[arg(long, default_value_t = 200)]
    pub mmd_rows: usize,
    /// Permutations for the joint reading.
    #[arg(long, default_value_t = 99)]
    pub mmd_permutations: usize,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct ChurnPublishArgs {
    #[arg(long, default_value = "data/churn/registry")]
    pub registry: String,
    #[arg(long, default_value = "data/churn/model.json")]
    pub model: String,
}

#[derive(Args)]
pub struct ChurnVersionsArgs {
    #[arg(long, default_value = "data/churn/registry")]
    pub registry: String,
}

#[derive(Args)]
pub struct ChurnRollbackArgs {
    #[arg(long, default_value = "data/churn/registry")]
    pub registry: String,
    #[arg(long, default_value = "data/churn/model.json")]
    pub target: String,
}

/// Run one `churn` subcommand.
pub fn run(args: ChurnArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        ChurnCommand::Extract(a) => {
            let report = panel::extract(&a.dir)?;
            print_report(&report);
            println!("wrote     panel.csv, series.csv, reasons.csv, reasons.json");
            Ok(())
        }
        ChurnCommand::Fit(a) => {
            if !panel::exists(&format!("{}/panel.csv", a.dir)) {
                return Err(
                    format!("{}/panel.csv is missing: run churn extract first", a.dir).into(),
                );
            }
            let model = fit::build_with(&a.dir, a.folds)?;
            let score = &model.score;
            println!(
                "panel     {} rows  {} churned  {:.1}%",
                commas(score.rows),
                commas(score.positives),
                score.positives as f64 / score.rows.max(1) as f64 * 100.0
            );
            println!(
                "held out  AUC {:.3}  tenure only {:.3}  balanced {:.3}  majority {:.3}",
                score.auc, score.tenure_auc, score.balanced, model.metrics.majority
            );
            let folds: Vec<String> = score.per_fold.iter().map(|v| format!("{v:.3}")).collect();
            println!("folds     {}", folds.join(" "));
            if let Some(reading) =
                crate::support::against_ceiling(&score.learning, score.label_audit.ceiling)
            {
                println!(
                    "curve     {} sizes  accuracy {:.3} at 100 percent  ceiling {:.3}  gap {:+.3}",
                    score.learning.len(),
                    reading.curve,
                    reading.ceiling,
                    reading.gap
                );
                println!("reading   {}", reading.verdict);
            }
            println!("wrote     {}/model.json, card.md", a.dir);
            Ok(())
        }
        ChurnCommand::Info(a) => info(&a.dir),
        ChurnCommand::Check(a) => {
            let faults = panel::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     {} rebuilds from its own source", a.dir);
                Ok(())
            } else {
                for fault in &faults {
                    println!("fault     {fault}");
                }
                Err(format!("{} faults in {}", faults.len(), a.dir).into())
            }
        }
        ChurnCommand::Uplift(a) => uplift(a),
        ChurnCommand::Monitor(a) => monitor(a),
        ChurnCommand::Publish(a) => {
            let model = fit::read(&a.model)?;
            let audit = audit_for(&a.model, &model)?;
            let registry = Registry::open(&a.registry)?;
            let version = registry.publish_with(&a.model, audit.clone())?;
            println!(
                "published v{:04} {} from {}",
                version.id, version.file, a.model
            );
            println!("digest    sha256 {}", version.digest);
            println!("bytes     {}", commas(version.bytes as usize));
            println!(
                "parent    {}",
                version
                    .parent
                    .map(|id| format!("v{id:04}"))
                    .unwrap_or_else(|| "none, this is the first version".to_string())
            );
            println!(
                "data      {} rows  sha256 {}",
                commas(audit.data_rows),
                if audit.data_digest.is_empty() {
                    "not recorded".to_string()
                } else {
                    audit.data_digest.clone()
                }
            );
            println!(
                "range     {}",
                if audit.range.is_empty() {
                    "not recorded".to_string()
                } else {
                    audit.range.clone()
                }
            );
            for (name, value) in &audit.facts {
                println!("{name:<22}{value}");
            }
            Ok(())
        }
        ChurnCommand::Versions(a) => versions(&a.registry),
        ChurnCommand::Rollback(a) => {
            let registry = Registry::open(&a.registry)?;
            let version = registry.rollback(&a.target)?;
            println!(
                "rolled back to v{:04} {} -> {}",
                version.id, version.file, a.target
            );
            println!(
                "digest    sha256 {} (checked before restoring)",
                version.digest
            );
            Ok(())
        }
    }
}

/// The trail a published version carries: what the artifact was built from, on
/// what range, and the readings the card claims.
///
/// Everything here is read off the artifact and the panel beside it, so a
/// version cannot claim a reading its own fit did not produce. The seed is
/// absent because the fit draws nothing: folds are an id hash and the logistic
/// is deterministic from its inputs, which is a claim the rebuild check tests.
fn audit_for(model_path: &str, model: &fit::ChurnModel) -> Result<Audit, Box<dyn Error>> {
    let panel = Path::new(model_path)
        .parent()
        .map(|dir| dir.join("panel.csv"))
        .filter(|path| path.is_file());
    let data_digest = match &panel {
        Some(path) => sha256(&fs::read(path)?),
        None => String::new(),
    };
    let score = &model.score;
    let mut facts = BTreeMap::new();
    facts.insert(
        "held-out AUC".to_string(),
        format!(
            "{:.3} [{:.3}, {:.3}]",
            score.auc, score.auc_low, score.auc_high
        ),
    );
    facts.insert(
        "balanced accuracy".to_string(),
        format!("{:.3}", score.balanced),
    );
    facts.insert("ECE".to_string(), format!("{:.3}", score.ece));
    facts.insert(
        "conformal coverage".to_string(),
        format!(
            "{:.3} at {:.0} percent nominal",
            score.conformal_coverage,
            (1.0 - score.conformal_alpha) * 100.0
        ),
    );
    facts.insert("positives".to_string(), commas(score.positives));
    facts.insert("folds".to_string(), score.folds.to_string());
    facts.insert(
        "mature rows".to_string(),
        format!(
            "{} of {}",
            commas(score.cohorts.mature_rows),
            commas(score.rows)
        ),
    );
    facts.insert("built".to_string(), model.built_at.clone());
    facts.insert(
        "seed".to_string(),
        "none: the folds are an id hash and the fit draws nothing".to_string(),
    );
    let range = match &model.window {
        Some(window) if window.ok() => format!(
            "month {} to {} from time zero at {}",
            window.first_month, window.last_month, window.declared
        ),
        Some(window) => format!(
            "month {} to {}, which breaks the window contract",
            window.first_month, window.last_month
        ),
        None => String::new(),
    };
    Ok(Audit {
        data_digest,
        data_rows: score.rows,
        range,
        seed: None,
        facts,
    })
}

fn versions(root: &str) -> Result<(), Box<dyn Error>> {
    let registry = Registry::open(root)?;
    let versions = registry.list()?;
    if versions.is_empty() {
        println!("empty     nothing is published in {root}");
        return Ok(());
    }
    let current = registry.current()?.map(|version| version.id);
    println!("id    digest        rows    parent  state");
    for version in &versions {
        let state = match registry.verify(version.id)? {
            Some(true) => "verified",
            Some(false) => "DIGEST MISMATCH",
            None => "no digest recorded",
        };
        println!(
            "v{:04} {:<14}{:<8}{:<8}{}{}",
            version.id,
            short(&version.digest),
            commas(version.audit.data_rows),
            version
                .parent
                .map(|id| format!("v{id:04}"))
                .unwrap_or_else(|| "-".to_string()),
            if current == Some(version.id) {
                format!("current, {state}")
            } else {
                state.to_string()
            },
            if version.published.is_empty() {
                String::new()
            } else {
                format!(", published {}", version.published)
            }
        );
        if !version.audit.range.is_empty() {
            println!("      {:<32}{}", "range", version.audit.range);
        }
        for (name, value) in &version.audit.facts {
            println!("      {:<32}{}", name, value);
        }
    }
    let history = registry.history()?;
    if !history.is_empty() {
        println!("log");
        for event in &history {
            println!("      {:<22}{:<10}{}", event.at, event.action, event.note);
        }
    }
    Ok(())
}

/// The head of a digest or timestamp, for a table that has to stay readable.
fn short(text: &str) -> String {
    text.chars().take(12).collect()
}

fn uplift(args: ChurnUpliftArgs) -> Result<(), Box<dyn Error>> {
    let data = causal::load_csv(&args.data, &args.outcome, &args.treatment)?;
    let fit = causal::fit(
        &data,
        &CausalConfig {
            folds: args.folds,
            bootstrap: args.bootstrap,
            permutations: args.permutations,
            ..CausalConfig::default()
        },
    )?;
    let report = &fit.report;
    if let Some(path) = &args.out {
        fs::write(path, format!("{}\n", serde_json::to_string_pretty(report)?))?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    println!(
        "rows      {}  treated {}  control {}  features {}",
        commas(report.rows),
        commas(report.treated),
        commas(report.control),
        report.features
    );
    println!(
        "T-learner ATE {:+.4}  [{:+.4}, {:+.4}]  {} resamples",
        report.ate,
        report.ate_interval.low,
        report.ate_interval.high,
        report.ate_interval.resamples
    );
    println!(
        "IPW ATE      {:+.4}  [{:+.4}, {:+.4}]  clipped {:.1}%",
        report.ipw_ate,
        report.ipw_interval.low,
        report.ipw_interval.high,
        report.clipped * 100.0
    );
    println!(
        "DR ATE       {:+.4}  [{:+.4}, {:+.4}]  se {:.4}  clipped {} rows",
        report.dr_ate,
        report.dr_interval.low,
        report.dr_interval.high,
        report.dr_se,
        report.dr_clipped
    );
    println!(
        "overlap weighted {:+.4}  se {:.4}",
        report.overlap_weighted_ate, report.overlap_weighted_se
    );
    println!(
        "crump trimmed    {:+.4}  se {:.4}  rows {} of {}  bound {:.3}{}",
        report.crump_ate,
        report.crump_se,
        report.crump_rows,
        report.rows,
        report.crump_bound,
        if report.crump_refused {
            "  too few rows, untrimmed value kept"
        } else {
            ""
        }
    );
    println!(
        "E-value      {:.2}  near null bound {:.2}",
        report.e_value, report.e_value_ci
    );
    println!(
        "qini {:+.3}  overlap {:.1}%  placebo p {:.3}",
        report.qini,
        report.overlap * 100.0,
        report.placebo_p
    );
    println!("\nfraction  rows  policy value");
    for point in &report.policy {
        println!(
            "{:>8.1}  {:>4}  {:>+12.4}",
            point.fraction, point.rows, point.value
        );
    }
    println!(
        "policy value: treat all {:+.4}  treat none {:+.4}",
        report.policy_value_all, report.policy_value_none
    );
    println!("qini is descriptive; the policy value column is the decision reading");
    println!("\ndecile  rows  treated  predicted  observed treated  control  difference");
    for decile in &report.deciles {
        println!(
            "{:>6}  {:>4}  {:>7}  {:>+9.4}  {:>16.4}  {:>7.4}  {:>+10.4}",
            decile.decile,
            decile.rows,
            decile.treated,
            decile.predicted,
            decile.observed_treated,
            decile.observed_control,
            decile.difference
        );
    }
    println!("\nheterogeneity across the median");
    println!("feature                 low      high   difference       z");
    for row in report.heterogeneity.iter().take(8) {
        println!(
            "{:<22} {:>+8.4} {:>+8.4} {:>+12.4} {:>+7.2}",
            row.feature, row.low, row.high, row.difference, row.z
        );
    }
    println!("\n{}", report.note);
    Ok(())
}

fn monitor(args: ChurnMonitorArgs) -> Result<(), Box<dyn Error>> {
    let reference = Dataset::load(&args.reference, &args.target)?;
    let current = Dataset::load(&args.current, &args.target)?;
    let config = monitor::GateConfig {
        alpha: args.alpha,
        bins: args.bins,
        psi_draws: args.psi_draws,
        cvm_draws: args.cvm_draws,
        mmd_rows: args.mmd_rows,
        mmd_permutations: args.mmd_permutations,
        threshold: args.threshold,
        ..monitor::GateConfig::default()
    };
    let report = monitor::assess(&reference, &current, &config)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "reference {} rows  current {} rows  level {:.2}  bins {}",
        commas(report.reference_rows),
        commas(report.current_rows),
        report.alpha,
        report.bins
    );
    let smallest = report
        .features
        .iter()
        .map(|feature| feature.psi_threshold)
        .fold(f64::INFINITY, f64::min);
    let largest = report
        .features
        .iter()
        .map(|feature| feature.psi_threshold)
        .fold(0.0f64, f64::max);
    if smallest.is_finite() {
        println!(
            "psi       calibrated threshold {:.4} to {:.4} at these row counts",
            smallest, largest
        );
    }
    match report.manual_threshold {
        Some(threshold) => {
            println!("psi       fixed threshold {threshold:.3} supplied, calibration overridden")
        }
        None => println!(
            "psi       the folklore 0.1 band is not used: it is a different test at every size"
        ),
    }
    println!(
        "cvm       critical {:.3} at level {:.2}, {} draws",
        report.cvm_critical, report.alpha, config.cvm_draws
    );
    println!("feature                    psi    psi p       ks      cvm   cvm p   combined  alarm");
    let mut features = report.features.clone();
    features.sort_by(|a, b| a.p.total_cmp(&b.p));
    for feature in &features {
        println!(
            "{:<24} {:>8.4} {:>8.3} {:>8.3} {:>8.3} {:>7.3} {:>10.3}  {}",
            feature.name,
            feature.psi,
            feature.psi_p,
            feature.ks,
            feature.cvm,
            feature.cvm_p,
            feature.p,
            if feature.alarm { "yes" } else { "no" }
        );
    }
    if let Some(mmd) = &report.mmd {
        println!(
            "joint     mmd {:.4} p {:.3}  {} rows against {}  bandwidth {:.3}  {} permutations",
            mmd.statistic,
            mmd.p,
            commas(mmd.reference_rows),
            commas(mmd.current_rows),
            mmd.bandwidth,
            mmd.permutations
        );
    }
    if !report.missing.is_empty() {
        println!("missing   {}", report.missing.join(", "));
    }
    println!("verdict   {}", report.verdict);
    Ok(())
}

fn info(dir: &str) -> Result<(), Box<dyn Error>> {
    let samples = fit::load(&format!("{dir}/panel.csv"))?;
    let positives = samples.iter().filter(|s| s.churn).count();
    println!(
        "panel     {} rows  {} churned  {:.1}%",
        commas(samples.len()),
        commas(positives),
        positives as f64 / samples.len().max(1) as f64 * 100.0
    );
    let series = super::csvx::read(&format!("{dir}/series.csv"))?;
    let month = series.require("month")?;
    let at_risk = series.require("at_risk")?;
    let churned = series.require("churned")?;
    let hazard = series.require("hazard")?;
    let survival = series.require("survival")?;
    if let Ok(window) = super::contract::check_files(dir) {
        println!("window    {}", window.summary());
        for violation in &window.violations {
            println!("  broken  {violation}");
        }
    }
    println!("series    {} months", commas(series.rows.len()));
    println!("month  at risk  churned  hazard  survival");
    for row in &series.rows {
        let register: u32 = row[month].parse().unwrap_or(0);
        if ![1, 6, 12, 24, 36, 48, 60, 72].contains(&register) {
            continue;
        }
        println!(
            "  {:>3}  {:>7}  {:>7}  {:>6.3}  {:>8.3}",
            register,
            row[at_risk],
            row[churned],
            row[hazard].parse::<f64>().unwrap_or(0.0),
            row[survival].parse::<f64>().unwrap_or(0.0)
        );
    }
    match fs::read_to_string(format!("{dir}/reasons.json")) {
        Ok(text) => {
            let summary: Summary = serde_json::from_str(&text)?;
            println!(
                "reasons   {} rows across {} categories",
                commas(summary.rows),
                summary.by_category.len()
            );
            println!("driver                     rows  valence  arousal");
            for (name, count, valence, arousal) in panel::driver_rows(&summary) {
                println!(
                    "  {:<24} {:>5}  {:>+7.3}  {:>7.3}",
                    name, count, valence, arousal
                );
            }
            let dominant = summary
                .overall
                .dominant
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)));
            if let Some((emotion, count)) = dominant {
                println!(
                    "overall   valence {:+.3}  arousal {:.3}  commonest {emotion} ({count})",
                    summary.overall.valence, summary.overall.arousal
                );
            }
        }
        Err(_) => println!("reasons   reasons.json is missing: run churn extract"),
    }
    if panel::exists(&format!("{dir}/model.json")) {
        let model = fit::read(&format!("{dir}/model.json"))?;
        let score = &model.score;
        println!(
            "fit       AUC {:.3} [{:.3}, {:.3}]  tenure only {:.3}  balanced {:.3}  cut {:.3}",
            score.auc, score.auc_low, score.auc_high, score.tenure_auc, score.balanced, score.cut
        );
        println!(
            "quality   brier {:.3}  log loss {:.3}  ece {:.3}  conformal {:.3}/{:.2}  trees {:.3}",
            score.brier,
            score.log_loss,
            score.ece,
            score.conformal_coverage,
            score.conformal_set,
            score.tree_auc
        );
        if score.label_audit.rows > 0 {
            println!(
                "labels    noise {:.3}  ceiling {:.3}  contradicted {}  unconfident {}",
                score.label_audit.noise,
                score.label_audit.ceiling,
                commas(score.label_audit.suspected),
                commas(score.label_audit.unconfident)
            );
            println!(
                "gain      {:.3} is the smallest resolvable at {:.0}% power, floor {:.3}, {}",
                score.power.detectable,
                score.power.power * 100.0,
                score.power.floor,
                if score.power.resolvable {
                    "resolved"
                } else {
                    "not resolved"
                }
            );
        }
        let mut weights = model.coefficients.clone();
        weights.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        let top: Vec<String> = weights
            .iter()
            .take(5)
            .map(|c| format!("{} {:+.2}", c.name, c.weight))
            .collect();
        println!("weights   {}", top.join(", "));
        let cohorts = &score.cohorts;
        if !cohorts.rows.is_empty() {
            println!(
                "\ncohorts   {} month horizon  {} of {} rows mature",
                cohorts.horizon,
                commas(cohorts.mature_rows),
                commas(score.rows)
            );
            println!(
                "band               rows  churned   rate  tenure     auc     ece  coverage  paired"
            );
            for row in &cohorts.rows {
                let paired = match &row.against_tenure {
                    Some(pair) => {
                        format!("{:+.3} [{:+.3}, {:+.3}]", pair.delta, pair.low, pair.high)
                    }
                    None => row
                        .refused
                        .clone()
                        .map(|reason| format!("refused: {reason}"))
                        .unwrap_or_else(|| "refused".to_string()),
                };
                println!(
                    "{:<16} {:>5}  {:>7}  {:>5.3}  {:>6.1}  {:>5}  {:>6}  {:>8}  {}",
                    row.label,
                    commas(row.rows),
                    commas(row.churned),
                    row.rate,
                    row.mean_tenure,
                    row.auc
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "-".to_string()),
                    row.ece
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "-".to_string()),
                    row.coverage
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "-".to_string()),
                    paired
                );
            }
            println!(
                "mature    AUC {:.3} over {} rows",
                cohorts.mature_auc,
                commas(cohorts.mature_rows)
            );
        }
    }
    Ok(())
}

fn print_report(report: &Report) {
    println!(
        "panel     {} rows  {} churned  {:.1}%",
        commas(report.rows),
        commas(report.churned),
        report.churn_rate * 100.0
    );
    println!(
        "status    {} stayed  {} joined (censored)",
        commas(report.stayed),
        commas(report.joined)
    );
    println!(
        "series    {} months  {} reasons across {} categories",
        commas(report.months),
        commas(report.reasons),
        report.categories
    );
}

pub use crate::num::commas;
