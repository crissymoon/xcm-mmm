//! Turning one customer table into the three files an outcome model needs.
//!
//! The panel is one row per customer with the churn label and the risk
//! features, and no text: the reason text is only present for customers who
//! already left, so a model that read it would be predicting the past. The
//! series is the discrete hazard by tenure month, which is the shape an MMM
//! outcome takes: how many customers reached month m, how many left at m, and
//! the share who survived to that month. The reasons are the emotional read,
//! joined back to the category a customer service team already uses.
//!
//! Every file is rendered from the same rows in one pass, so the three cannot
//! disagree about which customers were in the table.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::csvx;
use super::load::{self, Row};
use crate::dataset::built_at;
use crate::digest::fnv64;
use crate::emotion::EmotionModel;

pub const SOURCE: &str = "source/telecom_customer_churn.csv";
pub const PANEL: &str = "panel.csv";
pub const SERIES: &str = "series.csv";
pub const REASONS: &str = "reasons.csv";
pub const SUMMARY: &str = "reasons.json";
pub const MANIFEST: &str = "manifest.json";
pub const EMOTION_MODEL: &str = "../emotion/model.json";

/// What the extract saw and wrote.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub rows: usize,
    pub churned: usize,
    pub stayed: usize,
    pub joined: usize,
    pub churn_rate: f64,
    pub months: usize,
    pub reasons: usize,
    pub categories: usize,
}

/// One month of the retention curve.
#[derive(Clone, Debug)]
pub struct Month {
    pub month: u32,
    pub at_risk: usize,
    pub churned: usize,
    pub hazard: f64,
    pub survival: f64,
}

/// The churn hazard by tenure month, with right censoring.
///
/// Takes the two columns the hazard needs rather than a whole source row, so
/// the window contract in [`super::contract`] can rebuild the same series from
/// a rendered panel and compare like with like.
pub fn series(rows: &[super::contract::Tenure]) -> Vec<Month> {
    let longest = rows.iter().map(|row| row.tenure).max().unwrap_or(0);
    let mut out = Vec::with_capacity(longest as usize);
    let mut survival = 1.0;
    for month in 1..=longest {
        let at_risk = rows.iter().filter(|row| row.tenure >= month).count();
        let churned = rows
            .iter()
            .filter(|row| row.churned && row.tenure == month)
            .count();
        let hazard = if at_risk == 0 {
            0.0
        } else {
            churned as f64 / at_risk as f64
        };
        survival *= 1.0 - hazard;
        out.push(Month {
            month,
            at_risk,
            churned,
            hazard,
            survival,
        });
    }
    out
}

/// Render the panel.
pub fn render_panel(rows: &[Row]) -> String {
    let mut out = String::from(
        "id,churn,tenure,monthly,total,age,dependents,referrals,contract,internet,payment,support,security\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{},{},{},{:.2},{:.2},{},{},{},{},{},{},{},{}\n",
            escape(&row.id),
            u8::from(row.churned),
            row.tenure,
            row.monthly,
            row.total,
            row.age,
            row.dependents,
            row.referrals,
            escape(&row.contract),
            escape(&row.internet),
            escape(&row.payment),
            u8::from(row.support),
            u8::from(row.security)
        ));
    }
    out
}

/// Render the series.
pub fn render_series(months: &[Month]) -> String {
    let mut out = String::from("month,at_risk,churned,hazard,survival\n");
    for month in months {
        out.push_str(&format!(
            "{},{},{},{:.4},{:.4}\n",
            month.month, month.at_risk, month.churned, month.hazard, month.survival
        ));
    }
    out
}

/// One reason text and its reading.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasonRow {
    pub id: String,
    pub category: String,
    pub dominant: String,
    pub valence: f64,
    pub arousal: f64,
    pub intensity: f64,
    pub text: String,
}

/// Read every churn reason with the emotion model.
pub fn reasons(rows: &[Row], model: &EmotionModel) -> Vec<ReasonRow> {
    let scanner = model.scanner();
    rows.iter()
        .filter_map(|row| {
            let text = row.reason.clone()?;
            let reading = scanner.read(&text);
            Some(ReasonRow {
                id: row.id.clone(),
                category: row.category.clone().unwrap_or_else(|| "Other".to_string()),
                dominant: reading.dominant,
                valence: reading.valence,
                arousal: reading.arousal,
                intensity: reading.intensity,
                text,
            })
        })
        .collect()
}

/// Render the reasons.
pub fn render_reasons(reasons: &[ReasonRow]) -> String {
    let mut out = String::from("id,category,dominant,valence,arousal,intensity,text\n");
    for reason in reasons {
        out.push_str(&format!(
            "{},{},{},{:.4},{:.4},{:.4},{}\n",
            escape(&reason.id),
            escape(&reason.category),
            escape(&reason.dominant),
            reason.valence,
            reason.arousal,
            reason.intensity,
            quote(&reason.text)
        ));
    }
    out
}

/// A category's emotional profile.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Profile {
    pub rows: usize,
    pub valence: f64,
    pub arousal: f64,
    pub dominant: BTreeMap<String, usize>,
}

/// The reason summary: what each churn driver feels like.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub rows: usize,
    pub overall: Profile,
    pub by_category: BTreeMap<String, Profile>,
    /// Categories ordered by mean valence, most negative first.
    pub drivers: Vec<String>,
}

pub fn summarise(reasons: &[ReasonRow]) -> Summary {
    let mut groups: BTreeMap<String, Vec<&ReasonRow>> = BTreeMap::new();
    for reason in reasons {
        groups
            .entry(reason.category.clone())
            .or_default()
            .push(reason);
    }
    let mut by_category = BTreeMap::new();
    for (name, group) in &groups {
        by_category.insert(name.clone(), profile(group.clone()));
    }
    let overall = profile(reasons.iter().collect());
    let mut drivers: Vec<(String, f64)> = by_category
        .iter()
        .map(|(name, profile)| (name.clone(), profile.valence))
        .collect();
    drivers.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    Summary {
        rows: reasons.len(),
        overall,
        by_category,
        drivers: drivers.into_iter().map(|(name, _)| name).collect(),
    }
}

fn profile(rows: Vec<&ReasonRow>) -> Profile {
    let n = rows.len().max(1) as f64;
    let mut dominant: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        *dominant.entry(row.dominant.clone()).or_insert(0) += 1;
    }
    Profile {
        rows: rows.len(),
        valence: round4(rows.iter().map(|r| r.valence).sum::<f64>() / n),
        arousal: round4(rows.iter().map(|r| r.arousal).sum::<f64>() / n),
        dominant,
    }
}

/// Build the panel, the series, the reason files and the manifest.
pub fn extract(dir: &str) -> io::Result<Report> {
    let source_path = format!("{dir}/{SOURCE}");
    let table = csvx::read(&source_path)?;
    let rows = load::read(&table)?;
    let model = EmotionModel::load(&format!("{dir}/{EMOTION_MODEL}"))?;
    let months = series(
        &rows
            .iter()
            .map(super::contract::Tenure::from_row)
            .collect::<Vec<_>>(),
    );
    let reasons = reasons(&rows, &model);
    let summary = summarise(&reasons);

    fs::create_dir_all(dir)?;
    fs::write(format!("{dir}/{PANEL}"), render_panel(&rows))?;
    fs::write(format!("{dir}/{SERIES}"), render_series(&months))?;
    fs::write(format!("{dir}/{REASONS}"), render_reasons(&reasons))?;
    let text = serde_json::to_string_pretty(&summary).map_err(io::Error::other)?;
    fs::write(format!("{dir}/{SUMMARY}"), format!("{text}\n"))?;

    let churned = rows.iter().filter(|row| row.churned).count();
    let joined = rows.iter().filter(|row| row.joined).count();
    let report = Report {
        rows: rows.len(),
        churned,
        stayed: rows.len() - churned - joined,
        joined,
        churn_rate: round4(churned as f64 / rows.len().max(1) as f64),
        months: months.len(),
        reasons: reasons.len(),
        categories: summary.by_category.len(),
    };
    stamp(dir, &report, &source_path)?;
    Ok(report)
}

fn stamp(dir: &str, report: &Report, source: &str) -> io::Result<()> {
    let mut files = Vec::new();
    for (name, rows) in [
        (SOURCE, report.rows),
        (PANEL, report.rows),
        (SERIES, report.months),
        (REASONS, report.reasons),
        (SUMMARY, report.categories),
    ] {
        let path = if name == SOURCE {
            source.to_string()
        } else {
            format!("{dir}/{name}")
        };
        let bytes = fs::read(&path)?;
        files.push(FileStamp {
            name: name.to_string(),
            rows,
            bytes: bytes.len() as u64,
            fnv64: fnv64(&bytes),
        });
    }
    let manifest = Manifest {
        name: "xcm-mmm-churn".to_string(),
        version: 1,
        built_at: built_at(),
        report: report.clone(),
        files,
    };
    let text = serde_json::to_string_pretty(&manifest).map_err(io::Error::other)?;
    fs::write(format!("{dir}/{MANIFEST}"), format!("{text}\n"))
}

/// One file and the digest that proves it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileStamp {
    pub name: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
}

/// The folder's account of itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub report: Report,
    pub files: Vec<FileStamp>,
}

/// Rebuild the three files in memory and compare them with the shipped ones.
pub fn check(dir: &str) -> io::Result<Vec<String>> {
    let mut faults = Vec::new();
    let table = csvx::read(&format!("{dir}/{SOURCE}"))?;
    let rows = load::read(&table)?;
    let model = EmotionModel::load(&format!("{dir}/{EMOTION_MODEL}"))?;
    let months = series(
        &rows
            .iter()
            .map(super::contract::Tenure::from_row)
            .collect::<Vec<_>>(),
    );
    let reasons = reasons(&rows, &model);
    // The window is a data contract, so a broken one is a fault beside the
    // rebuild comparison rather than a footnote: a panel that counts a row
    // before time zero scores well for the wrong reason.
    let window = super::contract::check(
        &rows
            .iter()
            .map(super::contract::Tenure::from_row)
            .collect::<Vec<_>>(),
        &months,
    );
    for violation in &window.violations {
        faults.push(format!("window: {violation}"));
    }
    for (name, rebuilt) in [
        (PANEL, render_panel(&rows)),
        (SERIES, render_series(&months)),
        (REASONS, render_reasons(&reasons)),
    ] {
        match fs::read_to_string(format!("{dir}/{name}")) {
            Ok(shipped) if shipped != rebuilt => {
                faults.push(format!("{name} does not rebuild from its source"));
            }
            Ok(_) => {}
            Err(_) => faults.push(format!("{name} is missing")),
        }
    }
    // The window is a data contract, so a broken one is a fault beside the
    // rebuild comparison rather than a footnote: a panel that counts a row
    // before time zero scores well for the wrong reason.
    let window = super::contract::check(
        &rows
            .iter()
            .map(super::contract::Tenure::from_row)
            .collect::<Vec<_>>(),
        &months,
    );
    for violation in &window.violations {
        faults.push(format!("window: {violation}"));
    }
    let summary = summarise(&reasons);
    let rebuilt = serde_json::to_string_pretty(&summary).map_err(io::Error::other)?;
    match fs::read_to_string(format!("{dir}/{SUMMARY}")) {
        Ok(shipped) if shipped.trim() != rebuilt.trim() => {
            faults.push(format!("{SUMMARY} does not rebuild from its reasons"));
        }
        Ok(_) => {}
        Err(_) => faults.push(format!("{SUMMARY} is missing")),
    }
    if let Ok(text) = fs::read_to_string(format!("{dir}/{MANIFEST}")) {
        match serde_json::from_str::<Manifest>(&text) {
            Ok(manifest) => {
                for file in &manifest.files {
                    let path = if file.name == SOURCE {
                        format!("{dir}/{SOURCE}")
                    } else {
                        format!("{dir}/{}", file.name)
                    };
                    match fs::read(&path) {
                        Ok(bytes) if fnv64(&bytes) != file.fnv64 => {
                            faults
                                .push(format!("{} does not match its manifest digest", file.name));
                        }
                        Ok(_) => {}
                        Err(_) => faults.push(format!("{} is missing", file.name)),
                    }
                }
            }
            Err(e) => faults.push(format!("{MANIFEST}: {e}")),
        }
    } else {
        faults.push(format!("{MANIFEST} is missing"));
    }
    Ok(faults)
}

/// A category ordered by how negative its reasons read.
pub fn driver_rows(summary: &Summary) -> Vec<(String, usize, f64, f64)> {
    summary
        .drivers
        .iter()
        .filter_map(|name| {
            summary
                .by_category
                .get(name)
                .map(|profile| (name.clone(), profile.rows, profile.valence, profile.arousal))
        })
        .collect()
}

fn escape(cell: &str) -> String {
    if cell.contains(',') || cell.contains('"') || cell.contains('\n') {
        quote(cell)
    } else {
        cell.to_string()
    }
}

fn quote(cell: &str) -> String {
    format!("\"{}\"", cell.replace('"', "\"\""))
}

use crate::num::round4;

/// True when a path exists.
pub fn exists(path: &str) -> bool {
    Path::new(path).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::churn::csvx;
    use crate::churn::load;

    fn rows() -> Vec<Row> {
        let table = csvx::parse(
            "Customer ID,Age,Number of Dependents,Number of Referrals,Tenure in Months,Contract,Monthly Charge,Total Charges,Customer Status,Churn Category,Churn Reason,Premium Tech Support,Online Security,Internet Type,Payment Method\n\
             1,30,0,0,1,Month-to-Month,50,50,Churned,Price,Too expensive,No,No,DSL,Credit Card\n\
             2,40,1,0,2,Two Year,40,80,Stayed,,,Yes,Yes,Fiber,Bank Withdrawal\n\
             3,50,0,1,3,One Year,60,180,Churned,Competitor,Better offer,No,No,Cable,Mailed Check\n",
        )
        .expect("a table");
        load::read(&table).expect("rows")
    }

    #[test]
    fn the_hazard_uses_everyone_who_reached_the_month() {
        let months = series(
            &rows()
                .iter()
                .map(crate::churn::contract::Tenure::from_row)
                .collect::<Vec<_>>(),
        );
        assert_eq!(months.len(), 3);
        assert_eq!(months[0].at_risk, 3);
        assert_eq!(months[0].churned, 1);
        assert!((months[0].hazard - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(months[1].at_risk, 2);
        assert_eq!(months[2].churned, 1);
    }

    #[test]
    fn survival_falls_by_the_hazard() {
        let months = series(
            &rows()
                .iter()
                .map(crate::churn::contract::Tenure::from_row)
                .collect::<Vec<_>>(),
        );
        let product: f64 = months.iter().map(|m| 1.0 - m.hazard).product();
        assert!((months.last().unwrap().survival - product).abs() < 1e-9);
    }

    #[test]
    fn the_panel_has_one_row_per_customer_and_no_text() {
        let text = render_panel(&rows());
        assert_eq!(text.lines().count(), 4);
        assert!(text.starts_with("id,churn,tenure"));
        assert!(!text.contains("expensive"));
    }

    #[test]
    fn a_reason_with_a_comma_stays_quoted() {
        let text = render_reasons(&[ReasonRow {
            id: "1".into(),
            category: "Price".into(),
            dominant: "anger".into(),
            valence: -0.4,
            arousal: 0.5,
            intensity: 0.3,
            text: "too expensive, honestly".into(),
        }]);
        assert!(text.contains("\"too expensive, honestly\""));
    }

    #[test]
    fn the_summary_orders_drivers_by_valence() {
        let reasons = vec![
            ReasonRow {
                id: "1".into(),
                category: "Price".into(),
                dominant: "anger".into(),
                valence: -0.5,
                arousal: 0.6,
                intensity: 0.4,
                text: "expensive".into(),
            },
            ReasonRow {
                id: "2".into(),
                category: "Other".into(),
                dominant: "neutral".into(),
                valence: 0.0,
                arousal: 0.1,
                intensity: 0.05,
                text: "moved house".into(),
            },
        ];
        let summary = summarise(&reasons);
        assert_eq!(summary.rows, 2);
        assert_eq!(summary.drivers.first().map(String::as_str), Some("Price"));
        assert_eq!(summary.overall.rows, 2);
    }
}
