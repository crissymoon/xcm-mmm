//! The detector folder end to end: build it, score with it, check it, break it.
//!
//! The fixture is small on purpose: forty sentences a side would make the test
//! slow for no extra coverage, so the config asks for three rows a side and two
//! folds. What is being tested is the wiring and the refusals, not the accuracy
//! of a model fitted on six sentences.

use std::path::{Path, PathBuf};

use xcm_mmm::detect::{self, Artifact, Axis, Detector, TrainConfig};

fn folder(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-detect-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config() -> TrainConfig {
    let mut cfg = TrainConfig {
        name: "fixture-detectors".into(),
        folds: 2,
        min_rows: 3,
        ..Default::default()
    };
    // The defaults are tuned for a bank of thousands of sentences. Six rows a
    // side would lose every term to the frequency ceiling, so the ceiling is
    // lifted here and the fixture exercises the lexicon as well as the surface.
    cfg.lexicon.min_df = 1;
    cfg.lexicon.max_rate = 0.9;
    cfg.lexicon.max_plain_df = 0.9;
    cfg
}

/// A bank of legal-sounding positives, plain negatives, a cue table and ads.
fn fixture(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let bank = dir.join("bank.jsonl");
    let mut lines = String::new();
    for i in 0..6 {
        let row = serde_json::json!({
            "axis": "legal", "label": 1, "source": "federal-register",
            "text": format!("The department shall promulgate section {i} pursuant to the act and its regulations"),
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    for i in 0..6 {
        let row = serde_json::json!({
            "axis": "plain", "label": 0, "source": "ad",
            "text": format!("Vote with us on Tuesday and bring a friend to the rally at the park {i}"),
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    std::fs::write(&bank, lines).unwrap();

    let cues = dir.join("cues.json");
    let table = serde_json::json!({
        "version": 1,
        "axes": {
            "publicity": {
                "tag": "#PublicityPlay",
                "question": "Does the ad point at coverage or a spectacle?",
                "components": {"teaser": 1.0},
                "cues": {"teaser": "(?i)\\bwatch this\\b"},
                "gate": ["teaser"],
                "min_groups": 1,
                "halves": [["teaser"], ["teaser"]],
                "checks": [
                    {"text": "Watch this clip of the stunt.", "expected": true},
                    {"text": "We repaired the sewer line.", "expected": false}
                ]
            }
        },
        "features": {
            "legal": {
                "names": ["register", "authority"],
                "patterns": {"authority": "(?i)\\b(shall|pursuant|hereby)\\b"}
            }
        }
    });
    std::fs::write(&cues, serde_json::to_string_pretty(&table).unwrap()).unwrap();

    let ads = dir.join("ads.jsonl");
    let mut lines = String::new();
    for i in 0..6 {
        let split = if i < 4 { "train" } else { "test" };
        let row = serde_json::json!({
            "id": format!("ad{i}"),
            "split": split,
            "text": if i == 0 {
                "Watch this clip of the stunt the city filmed downtown".to_string()
            } else {
                format!("We repaired the sewer line on Sixth Street this spring number {i}")
            },
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    std::fs::write(&ads, lines).unwrap();
    (bank, ads, cues)
}

#[test]
fn a_built_folder_scores_checks_and_detects_an_edit() {
    let dir = folder("build");
    let (bank, ads, cues) = fixture(&dir);
    let out = dir.join("detectors");

    let artifact = detect::build(
        bank.to_str().unwrap(),
        ads.to_str().unwrap(),
        cues.to_str().unwrap(),
        out.to_str().unwrap(),
        &config(),
    )
    .unwrap();

    assert_eq!(artifact.axes.len(), 2, "one learned axis and one declared");
    for name in ["card.md", "manifest.json", "model.json"] {
        assert!(out.join(name).is_file(), "{name} was not written");
    }
    assert!(detect::check(out.to_str().unwrap()).unwrap().is_empty());

    let detector = Detector::load(out.join("model.json").to_str().unwrap()).unwrap();
    let readings =
        detector.score("The department shall promulgate regulations pursuant to the act.");
    let legal = readings.iter().find(|r| r.axis == "legal").unwrap();
    assert!(legal.label, "{legal:?}");
    assert!(legal.hits.iter().any(|h| h == "pursuant"), "{legal:?}");

    let plain = detector.score("Vote with us on Tuesday at the park.");
    let legal = plain.iter().find(|r| r.axis == "legal").unwrap();
    assert!(!legal.label, "{legal:?}");

    let stunt = detector.score("Watch this clip of the stunt.");
    let publicity = stunt.iter().find(|r| r.axis == "publicity").unwrap();
    assert!(publicity.label, "{publicity:?}");

    // A card that no longer matches the fit it describes is a fault, not a note.
    let model = out.join("model.json");
    let mut text = std::fs::read_to_string(&model).unwrap();
    text = text.replace("\"terms\"", "\"termz\"");
    std::fs::write(&model, text).unwrap();
    let faults = detect::check(out.to_str().unwrap()).unwrap();
    assert!(!faults.is_empty(), "an edited model must not check clean");
}

#[test]
fn a_bank_with_no_shared_negative_side_is_refused() {
    let dir = folder("refuse");
    let (bank, ads, cues) = fixture(&dir);
    let mut lines = String::new();
    for i in 0..6 {
        let row = serde_json::json!({
            "axis": "legal", "label": 1, "source": "federal-register",
            "text": format!("The department shall promulgate section {i}"),
        });
        lines.push_str(&serde_json::to_string(&row).unwrap());
        lines.push('\n');
    }
    std::fs::write(&bank, lines).unwrap();
    let error = detect::build(
        bank.to_str().unwrap(),
        ads.to_str().unwrap(),
        cues.to_str().unwrap(),
        dir.join("out").to_str().unwrap(),
        &config(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("shared"), "{error}");
}

#[test]
fn a_cue_table_whose_declared_example_fails_is_refused() {
    let dir = folder("cuecheck");
    let (bank, ads, _) = fixture(&dir);
    let cues = dir.join("cues-broken.json");
    let table = serde_json::json!({
        "version": 1,
        "axes": {
            "publicity": {
                "tag": "#PublicityPlay",
                "components": {"teaser": 1.0},
                "cues": {"teaser": "(?i)\\bthis phrase never occurs\\b"},
                "gate": ["teaser"],
                "min_groups": 1,
                "checks": [{"text": "Watch this clip of the stunt.", "expected": true}]
            }
        }
    });
    std::fs::write(&cues, serde_json::to_string(&table).unwrap()).unwrap();
    let error = detect::build(
        bank.to_str().unwrap(),
        ads.to_str().unwrap(),
        cues.to_str().unwrap(),
        dir.join("out").to_str().unwrap(),
        &config(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("declared example"), "{error}");
}

#[test]
fn a_shipped_folder_rebuilds_to_the_same_fit() {
    let dir = folder("rebuild");
    let (bank, ads, cues) = fixture(&dir);
    let out = dir.join("detectors");
    let first = detect::build(
        bank.to_str().unwrap(),
        ads.to_str().unwrap(),
        cues.to_str().unwrap(),
        out.to_str().unwrap(),
        &config(),
    )
    .unwrap();
    let second = detect::fit(
        bank.to_str().unwrap(),
        ads.to_str().unwrap(),
        cues.to_str().unwrap(),
        &config(),
    )
    .unwrap();
    assert!(detect::model::same_fit(&first, &second));
    let stored = Artifact::load(out.join("model.json").to_str().unwrap()).unwrap();
    assert!(detect::model::same_fit(&stored, &second));
    assert!(matches!(stored.axes["legal"], Axis::Lexicon(_)));
    assert!(matches!(stored.axes["publicity"], Axis::Index(_)));
}
