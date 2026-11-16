//! The `churn` command surface, beside the module it drives.
//!
//! `extract` writes the panel, the series and the reason readings; `fit` writes
//! the baseline and its card; `info` reads them back. The split matters because
//! the panel is derived from a customer table while the model is a choice made
//! on top of it: a reader can take the panel without accepting the regression.

use std::error::Error;
use std::fs;

use clap::{Args, Subcommand};

use super::fit;
use super::panel::{self, Report, Summary};
use crate::causal::{self, CausalConfig};
use crate::data::Dataset;
use crate::monitor::{self, Registry};

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
    /// PSI above which a feature is called drifted.
    #[arg(long, default_value_t = 0.2)]
    pub threshold: f64,
    /// Reference quantile bins.
    #[arg(long, default_value_t = 10)]
    pub bins: usize,
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
            let registry = Registry::open(&a.registry)?;
            let version = registry.publish(&a.model)?;
            println!(
                "published v{:04} {} from {}",
                version.id, version.file, a.model
            );
            Ok(())
        }
        ChurnCommand::Rollback(a) => {
            let registry = Registry::open(&a.registry)?;
            let version = registry.rollback(&a.target)?;
            println!(
                "rolled back to v{:04} {} -> {}",
                version.id, version.file, a.target
            );
            Ok(())
        }
    }
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
    let report = monitor::compare(&reference, &current, args.threshold, args.bins)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "reference {} rows  current {} rows  threshold {:.2}  drifted {}",
        commas(report.reference_rows),
        commas(report.current_rows),
        report.threshold,
        report.drifted
    );
    println!("feature                       psi      ks  mean shift  alarm");
    let mut features = report.features.clone();
    features.sort_by(|a, b| b.psi.total_cmp(&a.psi));
    for feature in &features {
        println!(
            "{:<26} {:>8.4} {:>7.3}  {:>+10.4}  {}",
            feature.name,
            feature.psi,
            feature.ks,
            feature.mean_shift,
            if feature.drifted { "yes" } else { "no" }
        );
    }
    if !report.missing.is_empty() {
        println!("missing   {}", report.missing.join(", "));
    }
    let alarms = report.alarms();
    if alarms.is_empty() {
        println!("verdict   stable at this threshold");
    } else {
        println!("verdict   drift in {}", alarms.join(", "));
    }
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
        let mut weights = model.coefficients.clone();
        weights.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        let top: Vec<String> = weights
            .iter()
            .take(5)
            .map(|c| format!("{} {:+.2}", c.name, c.weight))
            .collect();
        println!("weights   {}", top.join(", "));
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

fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
