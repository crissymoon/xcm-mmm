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
    }
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
            "fit       AUC {:.3}  tenure only {:.3}  balanced {:.3}  cut {:.3}",
            score.auc, score.tenure_auc, score.balanced, score.cut
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
