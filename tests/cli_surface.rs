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
fn the_allocation_schedule_can_be_switched_off_from_the_shell() {
    let run = |window: &str| {
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
                "--ri-window",
                window,
                "--seed",
                "4",
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
        assert!(!value["front"].as_array().unwrap().is_empty());
        value["front"].clone()
    };
    let improved = run("50");
    let random = run("0");
    assert_ne!(improved, random, "the schedule must change the run");
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

#[test]
fn dataset_ships_a_folder_that_checks_clean_and_catches_an_edit() {
    let dir = temp_dir().join("dataset");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let bank = dir.join("bank.jsonl");
    let splits = ["train", "train", "valid", "test"];
    let mut lines = String::new();
    for i in 0..12 {
        let row = serde_json::json!({
            "id": format!("id{i:02}"),
            "task": "rank",
            "split": splits[i % splits.len()],
            "day": format!("2024-0{}-1{}", 1 + i % 9, i % 10),
            "a": "issuer",
            "b": "other",
            "text_a": "the anchor document for this pair",
            "text_b": "the other side of this pair",
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    std::fs::write(&bank, lines).unwrap();

    let out = dir.join("shipped");
    let built = Command::new(binary())
        .args(["dataset", "build", "--input"])
        .arg(&bank)
        .args(["--out"])
        .arg(&out)
        .args(["--budget", "9"])
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    for name in [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "manifest.json",
        "card.md",
    ] {
        assert!(out.join(name).exists(), "missing {name}");
    }

    let checked = Command::new(binary())
        .args(["dataset", "check", "--dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stdout)
    );
    let text = String::from_utf8_lossy(&checked.stdout);
    assert!(text.contains("agrees with its manifest"), "{text}");

    let train = out.join("train.jsonl");
    let mut rows = std::fs::read_to_string(&train).unwrap();
    rows.push('\n');
    std::fs::write(&train, rows).unwrap();
    let tampered = Command::new(binary())
        .args(["dataset", "check", "--dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stdout).contains("bytes on disk"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn detect_trains_scores_checks_and_catches_an_edit() {
    let dir = temp_dir().join("detect");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let bank = dir.join("bank.jsonl");
    let mut lines = String::new();
    for i in 0..6 {
        let positive = serde_json::json!({
            "axis": "legal", "label": 1, "source": "federal-register",
            "text": format!("The department shall promulgate section {i} pursuant to the act"),
        });
        let negative = serde_json::json!({
            "axis": "plain", "label": 0, "source": "ad",
            "text": format!("Vote with us on Tuesday at the park near the corner {i}"),
        });
        for row in [positive, negative] {
            lines.push_str(&serde_json::to_string(&row).unwrap());
            lines.push('\n');
        }
    }
    std::fs::write(&bank, lines).unwrap();

    let ads = dir.join("ads.jsonl");
    let mut lines = String::new();
    for i in 0..6 {
        let row = serde_json::json!({
            "id": format!("ad{i}"),
            "split": if i < 4 { "train" } else { "test" },
            "text": format!("We repaired the sewer line on Sixth Street this spring {i}"),
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    std::fs::write(&ads, lines).unwrap();

    let cues = dir.join("cues.json");
    let table = serde_json::json!({
        "version": 1,
        "axes": {},
        "features": {"legal": {"names": ["authority"], "patterns": {"authority": "(?i)\\bpursuant to\\b"}}}
    });
    std::fs::write(&cues, serde_json::to_string(&table).unwrap()).unwrap();

    let out = dir.join("detectors");
    let trained = Command::new(binary())
        .args(["detect", "train", "--bank"])
        .arg(&bank)
        .args(["--ads"])
        .arg(&ads)
        .args(["--cues"])
        .arg(&cues)
        .args(["--out"])
        .arg(&out)
        .args(["--folds", "2", "--min-rows", "3"])
        .output()
        .unwrap();
    assert!(
        trained.status.success(),
        "{}",
        String::from_utf8_lossy(&trained.stderr)
    );

    let scored = Command::new(binary())
        .args(["detect", "score", "--model"])
        .arg(out.join("model.json"))
        .args([
            "--text",
            "The department shall promulgate regulations pursuant to the act",
        ])
        .output()
        .unwrap();
    assert!(scored.status.success());
    let text = String::from_utf8_lossy(&scored.stdout);
    assert!(text.contains("legal"), "{text}");
    assert!(
        text.contains("weak") || text.contains("clear") || text.contains("strong"),
        "{text}"
    );

    let checked = Command::new(binary())
        .args(["detect", "check", "--dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stdout)
    );

    // A model edited after the manifest was stamped must not check clean.
    let model = out.join("model.json");
    let mut body = std::fs::read_to_string(&model).unwrap();
    body = body.replace("\"terms\"", "\"termz\"");
    std::fs::write(&model, body).unwrap();
    let broken = Command::new(binary())
        .args(["detect", "check", "--dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!broken.status.success());
}
