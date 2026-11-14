//! The churn folder end to end: extract it, check it, fit it, break it.
//!
//! The fixture is the shipped telecom extract plus the shipped emotion model,
//! because the reason readings borrow the emotion scanner. The test asserts the
//! folder rebuilds from its own source, that a changed series is a fault, and
//! that the fourteen-feature fit ranks customers better than tenure alone.

use std::path::{Path, PathBuf};

use xcm_mmm::churn::{self, fit, panel};

fn root(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-churn-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("churn/source")).unwrap();
    std::fs::create_dir_all(dir.join("emotion")).unwrap();
    std::fs::copy(
        "data/churn/source/telecom_customer_churn.csv",
        dir.join("churn/source/telecom_customer_churn.csv"),
    )
    .unwrap();
    std::fs::copy("data/emotion/model.json", dir.join("emotion/model.json")).unwrap();
    dir
}

fn churn_dir(root: &Path) -> String {
    root.join("churn").to_str().unwrap().to_string()
}

#[test]
fn the_folder_extracts_and_checks_from_its_own_source() {
    let root = root("extract");
    let dir = churn_dir(&root);
    let report = panel::extract(&dir).expect("an extract");
    assert_eq!(report.rows, 7043);
    assert!(report.churned > 1000, "{} churned", report.churned);
    assert!(report.churn_rate > 0.2 && report.churn_rate < 0.35);
    assert_eq!(report.reasons, report.churned);
    assert!(report.months > 1);
    let faults = panel::check(&dir).expect("a check");
    assert!(faults.is_empty(), "{faults:?}");
}

#[test]
fn a_series_row_changed_after_the_extract_is_a_fault() {
    let root = root("tamper");
    let dir = churn_dir(&root);
    panel::extract(&dir).expect("an extract");
    let path = root.join("churn/series.csv");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    assert!(lines.len() > 2);
    lines[1] = "1900-01,0,0,0,0";
    std::fs::write(&path, format!("{}\n", lines.join("\n"))).unwrap();
    let faults = panel::check(&dir).expect("a check");
    assert!(
        faults.iter().any(|f| f.contains("series.csv")),
        "the series fault is named: {faults:?}"
    );
}

#[test]
fn the_fourteen_features_rank_better_than_tenure_alone() {
    let root = root("fit");
    let dir = churn_dir(&root);
    panel::extract(&dir).expect("an extract");
    let model = fit::build(&dir).expect("a fit");
    assert_eq!(model.features.len(), 14);
    assert_eq!(model.coefficients.len(), 14);
    assert!(
        model.score.auc > model.score.tenure_auc,
        "auc {:.3} vs tenure {:.3}",
        model.score.auc,
        model.score.tenure_auc
    );
    assert!(model.score.auc > 0.8, "{}", model.score.auc);
    assert!(root.join("churn/model.json").is_file());
    assert!(root.join("churn/card.md").is_file());
    let shipped = fit::read(root.join("churn/model.json").to_str().unwrap()).unwrap();
    assert_eq!(shipped.score.rows, model.score.rows);
}

#[test]
fn every_reason_carries_a_category_and_a_reading() {
    let root = root("reasons");
    let dir = churn_dir(&root);
    let report = panel::extract(&dir).expect("an extract");
    assert!(report.categories > 1);
    let text = std::fs::read_to_string(root.join("churn/reasons.csv")).unwrap();
    let mut lines = text.lines();
    let header = lines.next().expect("a header");
    assert!(header.contains("category"), "{header}");
    assert!(header.contains("valence"), "{header}");
    assert_eq!(lines.count(), report.reasons);
    let summary = std::fs::read_to_string(root.join("churn/reasons.json")).unwrap();
    assert!(summary.contains("\"drivers\""));
    assert!(churn::panel::exists(&format!("{dir}/series.csv")));
}
