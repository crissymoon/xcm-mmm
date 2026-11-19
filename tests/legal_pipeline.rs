//! The legal front end, exercised as a user would.
//!
//! Four things are fixed here, because a silent change to any of them would put
//! a claim on a document that its own folder does not support.
//!
//! 1. The shipped folder matches its manifest, and a hand edit to one row is a
//!    refusal rather than a quiet change.
//! 2. A statute sentence is read for its drafting words and rewritten, with the
//!    count reported before and after.
//! 3. A citation and a quotation are left exactly as they were.
//! 4. The JSON reading is well formed and carries the band, the offsets and the
//!    rules that produced the edit.
//! 5. The register audit names the registers the bank cannot speak to, and the
//!    pair benchmark reports a rate with its interval and a floor that fails.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_xcm-mmm")
}

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("legal")
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-legal-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> Output {
    Command::new(binary()).args(args).output().unwrap()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn the_shipped_folder_matches_its_manifest() {
    let dir = data();
    let out = run(&["legal", "check", "--dir", dir.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    assert!(text.contains("ok:"), "{text}");
}

#[test]
fn a_statute_is_read_and_rewritten() {
    let dir = data();
    let out = run(&[
        "legal",
        "translate",
        "--dir",
        dir.to_str().unwrap(),
        "The landlord shall provide notification pursuant to Section 827 before the increase.",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    let first = text.lines().next().unwrap();
    assert_eq!(
        first,
        "The landlord must notify under Section 827 before the increase."
    );
    assert!(text.contains("3 edits"), "{text}");
}

#[test]
fn a_construction_is_rewritten_not_just_its_words() {
    // The clause rule has to win over `shall not`, or the modal ends up in front
    // of a noun: "must not be entitled to" was the old output. `whatsoever` and
    // `incidental to` have no glossary row, so this is the only test that can
    // show the clause table doing the work a word table cannot.
    let dir = data();
    let out = run(&[
        "legal",
        "translate",
        "--dir",
        dir.to_str().unwrap(),
        "The Contractor shall not be entitled to payment or reimbursement for any \
         services performed, nor for any incidental or administrative expenses \
         whatsoever incurred in or incidental to performance hereunder.",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    let first = text.lines().next().unwrap();
    assert!(
        first.starts_with("The Contractor shall not be paid for any services performed"),
        "{first}"
    );
    assert!(!first.contains("entitled"), "{first}");
    assert!(!first.contains("whatsoever"), "{first}");
    assert!(
        first.contains("in or related to performance under this document"),
        "{first}"
    );
    assert!(text.contains("not-entitled-to-payment"), "{text}");
}

#[test]
fn a_citation_and_a_quotation_are_left_alone() {
    let dir = data();
    let out = run(&[
        "legal",
        "translate",
        "--dir",
        dir.to_str().unwrap(),
        "The notice read “pursuant to the lease” under Civil Code § 1940.2.",
    ]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("“pursuant to the lease”"), "{text}");
    assert!(text.contains("§ 1940.2"), "{text}");
}

#[test]
fn a_json_reading_carries_the_band_and_the_rules() {
    let dir = data();
    let out = run(&[
        "legal",
        "scan",
        "--json",
        "--dir",
        dir.to_str().unwrap(),
        "The tenant herein shall object thereto.",
    ]);
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(value["stats"]["avoidable"].as_u64().unwrap() >= 2);
    assert!(value["stats"]["band"].is_string());
    let terms = value["terms"].as_array().unwrap();
    assert!(terms.iter().any(|t| t["id"] == "herein"));
    let first = &terms[0];
    assert!(first["span"]["start"].as_u64().is_some());
    assert!(first["plain"].is_string() || first["id"].is_string());
}

#[test]
fn an_edited_row_breaks_the_check() {
    let dir = temp_dir("edited");
    for name in [
        "terms.jsonl",
        "rewrites.jsonl",
        "settings.json",
        "manifest.json",
    ] {
        fs::copy(data().join(name), dir.join(name)).unwrap();
    }
    let path = dir.join("terms.jsonl");
    let original = fs::read_to_string(&path).unwrap();
    let edited = original.replacen("\"must\"", "\"must \"", 1);
    assert_ne!(original, edited, "the fixture must change one row");
    fs::write(&path, edited).unwrap();

    let out = run(&["legal", "check", "--dir", dir.to_str().unwrap()]);
    assert!(!out.status.success(), "a changed row must fail the check");
    let text = stdout(&out);
    assert!(text.contains("terms.jsonl"), "{text}");
    assert!(text.contains("fault") || text.contains("digest"), "{text}");
}

#[test]
fn terms_can_be_listed_by_kind() {
    let dir = data();
    let out = run(&[
        "legal",
        "terms",
        "--kind",
        "latin",
        "--dir",
        dir.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("inter-alia"), "{text}");
    assert!(text.contains("mutatis-mutandis"), "{text}");
    assert!(!text.contains("hereunder"), "{text}");
}

#[test]
fn the_register_audit_names_the_registers_the_bank_cannot_speak_to() {
    let dir = data();
    let out = run(&["legal", "coverage", "--dir", dir.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    // The four registers the bank does hold are reported with a verdict, and
    // the ones it does not are named rather than left out of the table.
    for register in [
        "court-rule",
        "statute",
        "self-help",
        "ad-copy",
        "contract",
        "opinion",
    ] {
        assert!(text.contains(register), "missing {register} in\n{text}");
    }
    assert!(text.contains("absent"), "{text}");
    assert!(text.contains("widen: add contract text"), "{text}");
    assert!(
        text.lines().any(|line| line.starts_with("widen: ")),
        "{text}"
    );
}

#[test]
fn the_register_audit_keeps_plain_copy_out_of_the_reach_number() {
    let dir = data();
    let out = run(&[
        "legal",
        "coverage",
        "--json",
        "--dir",
        dir.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let reach = value["reach"].as_f64().unwrap();
    let leak = value["leak"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&reach), "{reach}");
    assert!((0.0..=1.0).contains(&leak), "{leak}");
    let rows = value["rows"].as_array().unwrap();
    let ad = rows
        .iter()
        .find(|row| row["register"] == "ad-copy")
        .unwrap();
    assert_eq!(ad["role"], "contrast");
    assert_eq!(ad["verdict"], "clean");
}

#[test]
fn the_bench_scores_the_shipped_pairs_with_an_interval() {
    let dir = data();
    let out = run(&["legal", "bench", "--dir", dir.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    assert!(text.contains("71 pairs scored"), "{text}");
    assert!(text.contains("edited"), "{text}");
    assert!(text.contains("95% interval"), "{text}");
}

#[test]
fn the_bench_json_carries_the_rate_and_its_interval() {
    let dir = data();
    let out = run(&[
        "legal",
        "bench",
        "--json",
        "--resamples",
        "200",
        "--dir",
        dir.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let all = &value["all"];
    let rate = all["rate"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&rate), "{rate}");
    assert!(all["low"].as_f64().unwrap() <= rate);
    assert!(all["high"].as_f64().unwrap() >= rate);
    assert_eq!(
        value["pairs"].as_u64().unwrap(),
        value["cases"].as_array().unwrap().len() as u64
    );
    // The edited slice is never larger than the whole, and a pair that no rule
    // touched cannot be counted as an edit.
    let edited = value["when_edited"]["cases"].as_u64().unwrap();
    assert!(edited <= value["pairs"].as_u64().unwrap());
    assert_eq!(edited, value["edited"].as_u64().unwrap());
}

#[test]
fn a_bench_floor_above_the_measured_rate_is_a_failure() {
    let dir = data();
    let out = run(&[
        "legal",
        "bench",
        "--fail-under",
        "1.0",
        "--resamples",
        "0",
        "--dir",
        dir.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "no rewrite reaches every pair");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("below the 1 floor"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(&[
        "legal",
        "bench",
        "--fail-under",
        "0",
        "--resamples",
        "0",
        "--dir",
        dir.to_str().unwrap(),
    ]);
    assert!(out.status.success());
}
