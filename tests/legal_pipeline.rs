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
