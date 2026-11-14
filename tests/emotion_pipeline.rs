//! The emotion folder end to end: rebuild it, check it, break it, read with it.
//!
//! The fixture is a copy of the shipped seed, list and bank, so the test proves
//! the bytes in the repository are the bytes the model came from. A tampered
//! bank must be a fault rather than a silent refit, and a model that has not
//! seen a word must still answer it through the lower tiers instead of dropping
//! the token.

use std::io::Write;
use std::path::PathBuf;

use xcm_mmm::emotion::{self, EmotionModel};

fn folder(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-emotion-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for file in ["seed.json", "listed.tsv", "bank.jsonl"] {
        std::fs::copy(format!("data/emotion/{file}"), dir.join(file)).unwrap();
    }
    dir
}

fn shipped() -> EmotionModel {
    EmotionModel::load("data/emotion/model.json").expect("the shipped model")
}

#[test]
fn the_folder_rebuilds_from_its_own_seed_list_and_bank() {
    let dir = folder("rebuild");
    let model = emotion::build(dir.to_str().unwrap()).expect("a fit");
    assert!(model.metrics.leave_one_out > model.metrics.majority);
    assert!(dir.join("model.json").is_file());
    assert!(dir.join("manifest.json").is_file());
    assert!(dir.join("card.md").is_file());
    let faults = emotion::check(dir.to_str().unwrap()).expect("a check");
    assert!(faults.is_empty(), "{faults:?}");
}

#[test]
fn a_bank_row_added_after_the_fit_is_a_fault() {
    let dir = folder("tamper");
    emotion::build(dir.to_str().unwrap()).expect("a fit");
    std::fs::OpenOptions::new()
        .append(true)
        .open(dir.join("bank.jsonl"))
        .unwrap()
        .write_all(b"{\"id\":\"late_01\",\"text\":\"this line arrived after the fit\",\"label\":\"Happy\",\"source\":\"late\"}\n")
        .unwrap();
    let faults = emotion::check(dir.to_str().unwrap()).expect("a check");
    assert!(!faults.is_empty(), "a changed bank must not pass");
    assert!(
        faults.iter().any(|f| f.contains("bank")),
        "the bank fault is named: {faults:?}"
    );
}

#[test]
fn a_word_the_bank_never_saw_is_still_read_by_a_lower_tier() {
    let model = shipped();
    let reading = model.scanner().read("overjoyed");
    let hit = reading.hits.first().expect("overjoyed is answered");
    assert_ne!(hit.tier, "prior", "the word fell to the prior");
    assert!(reading.share("joy") > 0.0, "{:?}", reading.emotions);
}

#[test]
fn a_text_of_unknown_words_is_neutral_and_contributes_no_mass() {
    let model = shipped();
    let reading = model.scanner().read("qwerty zxcvbn plugh xyzzy");
    assert!(reading.is_neutral(), "{}", reading.dominant);
    assert!(
        reading.hits.is_empty(),
        "unknown words must not invent hits"
    );
    assert!((reading.valence).abs() < 1e-9, "{}", reading.valence);
}

#[test]
fn the_scan_is_case_and_punctuation_blind() {
    let model = shipped();
    let plain = model.scanner().read("happy");
    let loud = model.scanner().read("HAPPY, \"happy\"! (Happy)");
    assert_eq!(plain.dominant, loud.dominant);
    assert_eq!(plain.hits.len(), 1);
    assert_eq!(loud.hits.len(), 3);
    assert!(loud.hits.iter().all(|hit| hit.word == "happy"));
    assert!(loud.hits.iter().all(|hit| hit.tier == "exact"));
}

#[test]
fn every_word_in_a_mixed_list_is_answered_at_some_tier() {
    let dir = folder("coverage");
    let words = dir.join("words.txt");
    std::fs::write(
        &words,
        "joyful\njoyfulness\nunhappiness\nrejoicing\nterrified\nqwerty\nzzz\nhappy\n",
    )
    .unwrap();
    let coverage = emotion::measure_coverage(&shipped().scanner(), words.to_str().unwrap(), 2)
        .expect("coverage");
    assert_eq!(coverage.words, 8);
    let tiered: usize = coverage.tiers.values().sum();
    assert_eq!(tiered, coverage.words, "some word went unanswered");
    assert!(coverage.handled() >= 5, "{} handled", coverage.handled());
}
