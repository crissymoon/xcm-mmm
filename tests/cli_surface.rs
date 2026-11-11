//! The command line surface, exercised as a user would.

use std::path::PathBuf;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_xcm-mmm")
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn list_prints_every_problem() {
    let out = Command::new(binary()).arg("list").output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for name in ["sphere", "zdt1", "dtlz2", "rosenbrock"] {
        assert!(text.contains(name), "missing {name} in {text}");
    }
}

#[test]
fn de_json_report_is_well_formed() {
    let out = Command::new(binary())
        .args([
            "de",
            "--problem",
            "sphere",
            "--dim",
            "5",
            "--budget",
            "3000",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let best = value["best_objective"].as_f64().unwrap();
    assert!(best < 1e-3, "best {best}");
    assert_eq!(value["dim"].as_u64().unwrap(), 5);
    assert!(value["generations"].as_u64().unwrap() > 0);
}

#[test]
fn moead_json_report_has_a_front() {
    let out = Command::new(binary())
        .args([
            "moead",
            "--problem",
            "zdt1",
            "--dim",
            "6",
            "--population",
            "30",
            "--budget",
            "1500",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let front = value["front"].as_array().unwrap();
    assert!(!front.is_empty());
    assert!(value["points"].as_u64().unwrap() as usize == front.len());
}

#[test]
fn the_wrong_engine_for_a_problem_is_refused() {
    let out = Command::new(binary())
        .args(["de", "--problem", "zdt1"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("moead"));
}

#[test]
fn synth_then_forecast_runs_from_the_shell() {
    let dir = temp_dir();
    let csv = dir.join("panel.csv");
    let made = Command::new(binary())
        .args([
            "synth",
            "--rows",
            "200",
            "--features",
            "8",
            "--informative",
            "3",
            "--seed",
            "5",
            "--out",
        ])
        .arg(&csv)
        .output()
        .unwrap();
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stderr)
    );
    assert!(csv.exists());

    let out = Command::new(binary())
        .args(["forecast"])
        .arg(&csv)
        .args([
            "--method", "de", "--folds", "3", "--budget", "600", "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["rows"].as_u64().unwrap(), 200);
    assert_eq!(value["folds"].as_u64().unwrap(), 3);
    assert!(value["models"]["all"]["mse"].as_f64().unwrap().is_finite());
    assert!(value["models"]["de"]["mse"].as_f64().unwrap().is_finite());

    let _ = std::fs::remove_file(&csv);
    let _ = std::fs::remove_dir(&dir);
}
