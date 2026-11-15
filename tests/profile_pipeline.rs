//! The reader, exercised on the shipped models.
//!
//! These tests treat the three artifacts as the product they are: a report is a
//! function of the shipped files, so the same text has to read the same way
//! twice, the segment view has to name what it could not observe, and a text
//! that is a textbook manipulation has to leave the manipulation axis above
//! zero rather than at the floor.

use std::process::Command;

use xcm_mmm::profile::{build, Loaded, Options, Report, DETECTORS, EMOTION, SEGMENTS};

fn loaded() -> Loaded {
    Loaded::open(DETECTORS, EMOTION, SEGMENTS).unwrap()
}

fn report(text: &str) -> Report {
    build(&loaded(), text, &Options::default())
}

fn axis<'a>(report: &'a Report, name: &str) -> &'a xcm_mmm::profile::report::Axis {
    report
        .detectors
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("no axis {name}"))
}

#[test]
fn a_manipulation_text_fires_the_manipulation_and_fear_axes() {
    let text = "Guaranteed returns or your money back. Only 3 seats left, and the \
                insiders already know. Act now before the crash destroys your savings.";
    let report = report(text);
    assert!(
        axis(&report, "manipulation").score > 0.0,
        "manipulation read {}",
        axis(&report, "manipulation").score
    );
    assert!(
        axis(&report, "fear").score > 0.0,
        "fear read {}",
        axis(&report, "fear").score
    );
    assert_eq!(axis(&report, "manipulation").level, "weak");
    assert!(!axis(&report, "manipulation").hits.is_empty());
}

#[test]
fn a_statutory_text_reads_more_legal_than_a_slogan() {
    let statute = report(
        "Pursuant to Section 4 of the agreement, the vendor shall provide \
         documentation within 30 days, and non-compliance with subsection (b) \
         constitutes a material breach of the authority granted herein.",
    );
    let slogan = report("Fresh coffee, every morning, made the way you like it.");
    assert!(
        axis(&statute, "legal").score > axis(&slogan, "legal").score,
        "statute {} vs slogan {}",
        axis(&statute, "legal").score,
        axis(&slogan, "legal").score
    );
    assert_ne!(axis(&statute, "legal").level, "none");
    assert!(axis(&statute, "legal").score > 0.4);
    assert!(statute.semantics.surface["authority"] > 0.0);
    assert!(statute.semantics.surface["citation"] > 0.0);
}

#[test]
fn the_segment_view_says_what_it_could_not_observe() {
    let report = report("Short copy. One promise. One price.");
    let view = report
        .segment
        .expect("the shipped segment model is present");
    assert_eq!(
        view.basis,
        vec!["copy length".to_string(), "creative reuse".to_string()]
    );
    assert_eq!(view.unobserved.len(), 6);
    assert_eq!(view.ranked.len(), 4);
    assert!(view
        .ranked
        .windows(2)
        .all(|w| w[0].distance <= w[1].distance));
    assert!(view.margin >= 0.0);
    let total: f64 = view.ranked.iter().map(|at| at.fit).sum();
    assert!((total - 100.0).abs() < 0.01, "{total}");
    assert!(view.reach > 0.0 && view.reach < 1.0, "{}", view.reach);
    assert!(view.note.contains("does not assign"));
    let reading = &view.ranked[0].readings[0];
    assert_eq!(reading.value, 6.0);
    // The segment mean is a count of words per ad, so it can never read as the
    // negative number an inverted z would give.
    assert!(reading.segment > 1.0, "{reading:?}");
}

#[test]
fn a_missing_segment_model_drops_the_view_and_says_so() {
    let loaded = Loaded::open(DETECTORS, EMOTION, "tmp/no-such-segments.json").unwrap();
    let report = build(&loaded, "Some copy to read.", &Options::default());
    assert!(report.segment.is_none());
    assert!(report.models.segments.is_none());
    assert!(report.notes.iter().any(|note| note.contains("segment fit")));
}

#[test]
fn every_detector_answers_and_the_count_is_seven() {
    let report = report("A sentence with no particular claim in it at all.");
    assert_eq!(report.detectors.len(), 7);
    for reading in &report.detectors {
        assert!(!reading.level.is_empty(), "{}", reading.name);
        if reading.method == "index" {
            assert!(!reading.gate.is_empty(), "{}", reading.name);
            assert!(!reading.parts.is_empty(), "{}", reading.name);
        }
    }
}

#[test]
fn second_person_and_sentence_shape_are_counted_not_guessed() {
    let report = report("You bring the plan. You own the outcome. Your call.");
    assert_eq!(report.shape.sentences, 3);
    assert_eq!(report.shape.words, 10);
    assert!((report.semantics.second_person_share - 0.3).abs() < 1e-9);
}

#[test]
fn the_same_text_reads_the_same_twice() {
    let text = "Limited time. Join 2.4 million readers. Ends tonight.";
    let first = build(&loaded(), text, &Options::default());
    let second = build(&loaded(), text, &Options::default());
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
}

#[test]
fn a_report_survives_a_json_round_trip() {
    let report = report("Act now, before it is too late. They do not want you to know.");
    let text = serde_json::to_string_pretty(&report).unwrap();
    let back: Report = serde_json::from_str(&text).unwrap();
    assert_eq!(back.shape.words, report.shape.words);
    assert_eq!(back.detectors.len(), report.detectors.len());
    assert_eq!(back.models.detectors, report.models.detectors);
}

#[test]
fn the_command_line_prints_the_same_report_as_json() {
    let binary = env!("CARGO_BIN_EXE_xcm-mmm");
    let out = Command::new(binary)
        .args([
            "read",
            "--json",
            "--triggers",
            "3",
            "Guaranteed returns. Only 3 seats left.",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["detectors"].as_array().unwrap().len(), 7);
    assert!(value["shape"]["words"].as_u64().unwrap() > 0);
    assert!(value["triggers"].as_array().unwrap().len() <= 3);
    assert_eq!(value["segment"]["basis"].as_array().unwrap().len(), 2);
    assert!(value["language"]["reuse"].is_number());
    let fits: Vec<f64> = value["segment"]["ranked"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["fit"].as_f64().unwrap())
        .collect();
    assert!((fits.iter().sum::<f64>() - 100.0).abs() < 0.01);
}

#[test]
fn the_command_line_reads_a_file_and_stdin() {
    let binary = env!("CARGO_BIN_EXE_xcm-mmm");
    let dir = std::env::temp_dir().join(format!("xcm-mmm-read-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("copy.txt");
    std::fs::write(&path, "Pursuant to Section 9, the party shall comply.").unwrap();
    let out = Command::new(binary)
        .args(["read", "--json", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["models"]["emotion"].as_str(), Some(EMOTION));

    let out = Command::new(binary)
        .args(["read", "--json"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(b"You will lose everything unless you act tonight.")?;
            child.wait_with_output()
        })
        .unwrap();
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(value["detectors"].as_array().unwrap().len() == 7);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_repeated_hook_scores_as_reuse_and_fits_a_reuse_segment() {
    let repeating =
        report("Act now, act now, act now. Only three seats left, and the insiders already know.");
    let flat =
        report("We opened a small office this spring and hired two people from the neighborhood.");
    assert!(repeating.language.reuse > 0.0, "{:?}", repeating.language);
    assert!(
        repeating.language.reuse > flat.language.reuse,
        "reuse {} vs {}",
        repeating.language.reuse,
        flat.language.reuse
    );
    let view = repeating.segment.as_ref().unwrap();
    assert_eq!(view.hook, "act now act now");
    let reuse = view
        .ranked
        .iter()
        .flat_map(|at| at.readings.iter())
        .find(|reading| reading.name == "repeat")
        .unwrap();
    assert_eq!(reuse.source, "phrase reuse");
    assert!(reuse.value > 0.0);
    let fits: f64 = view.ranked.iter().map(|at| at.fit).sum();
    assert!((fits - 100.0).abs() < 0.01, "{fits}");
}

#[test]
fn a_named_buy_adds_the_terms_column_and_nothing_else() {
    let loaded = loaded();
    let options = Options {
        triggers: 6,
        terms: Some(2.0),
        set: None,
    };
    let with = build(
        &loaded,
        "Short copy with one promise and a price.",
        &options,
    );
    let without = build(
        &loaded,
        "Short copy with one promise and a price.",
        &Options::default(),
    );
    let with_view = with.segment.as_ref().unwrap();
    let without_view = without.segment.as_ref().unwrap();
    assert_eq!(with_view.basis.len(), without_view.basis.len() + 1);
    let reading = with_view
        .ranked
        .iter()
        .flat_map(|at| at.readings.iter())
        .find(|reading| reading.name == "terms")
        .expect("the terms column is observed when the caller names the buy");
    assert_eq!(reading.value, 2.0);
    assert_eq!(reading.source, "terms");
}

#[test]
fn the_learned_ad_map_is_withheld_and_says_why() {
    let report = report("Act now, act now, act now. Only three seats left.");
    let learned = report
        .learned
        .expect("the shipped map is loaded with the segments");
    assert_eq!(learned.unit, "ad");
    assert_eq!(learned.texts, 1);
    assert!(!learned.usable);
    assert!(learned.ranked.is_empty());
    assert!(learned.balanced < learned.majority);
    assert!(learned.note.contains("withheld"), "{}", learned.note);
    assert!(report.models.textmap.is_some());
}

#[test]
fn a_copy_set_is_scored_by_the_brand_map_and_sums_to_one_hundred() {
    let set = vec![
        "Fight for the neighbourhood you love. Vote on Tuesday.".to_string(),
        "Fight for the neighbourhood you love. Vote on Tuesday.".to_string(),
        "Our plan protects renters and funds the buses.".to_string(),
        "Our plan protects renters and funds the buses.".to_string(),
    ];
    let first = set[0].clone();
    let report = build(
        &loaded(),
        &first,
        &Options {
            set: Some(set),
            ..Options::default()
        },
    );
    let learned = report.learned.expect("a brand block");
    assert_eq!(learned.unit, "brand");
    assert_eq!(learned.texts, 4);
    assert!(learned.usable, "{}", learned.note);
    assert_eq!(learned.ranked.len(), 4);
    let total: f64 = learned.ranked.iter().map(|scored| scored.probability).sum();
    assert!((total - 100.0).abs() < 0.01, "{total}");
    assert!(learned.ranked[0].probability >= learned.ranked[3].probability);
}
