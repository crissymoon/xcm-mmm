//! The retention loop end to end: two weeks of files, one bank, one index, one
//! ledger, and a guard against rewriting a corpus the bank did not build.

use std::fs;
use std::path::{Path, PathBuf};

use xcm_mmm::pairs::retain::{self, Policy};
use xcm_mmm::pairs::{self, Pair, Task};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-retain-it-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch");
    dir
}

fn write_pairs(path: &Path, rows: &[(String, f64, f64, &str, &str)]) {
    let mut out = String::new();
    for (id, margin, difficulty, text, group) in rows {
        let mut pair = Pair::new(Task::Rank, "train");
        pair.a = format!("doc-{id}");
        pair.b = Some(format!("neg-{id}"));
        pair.text_a = text.to_string();
        pair.text_b = Some(format!("{text} against consensus"));
        pair.y_a = Some(*margin);
        pair.y_b = Some(0.0);
        pair.margin = Some(*margin);
        pair.difficulty = Some(*difficulty);
        pair.day = "2026-11-09".into();
        pair.group_a = group.to_string();
        pair.quality_a = Some(0.8);
        pair.quality_b = Some(0.7);
        pair.stamp();
        pair.id = id.clone();
        out.push_str(&pairs::to_line(&pair));
        out.push('\n');
    }
    fs::write(path, out).expect("write pairs");
}

fn sentence(tag: &str) -> String {
    format!(
        "{tag} coverage of the quarter, with the desk reading the release against consensus and \
         prior commentary while the sector traded on the print and the guidance that followed it."
    )
}

fn policy() -> Policy {
    Policy {
        budget: 1_000,
        per_day: 200,
        per_group: 100,
        per_anchor: 2,
        per_task: 500,
        min_score: 0.0,
        // Distinct anchors differ by more than a bit, so the near duplicate
        // screen stays out of these tests; it has its own unit test.
        novelty_bits: 0,
        ..Policy::default()
    }
}

#[test]
fn two_days_grow_one_bank() {
    let dir = scratch("grow");
    let bank = dir.join("week.jsonl");
    let day1 = dir.join("day1.jsonl");
    let day2 = dir.join("day2.jsonl");
    write_pairs(
        &day1,
        &[
            ("a1".into(), 0.40, 0.50, &sentence("Alpha"), "AAA"),
            ("a2".into(), 0.20, 0.55, &sentence("Beta"), "BBB"),
        ],
    );
    write_pairs(
        &day2,
        &[
            ("b1".into(), 0.50, 0.50, &sentence("Gamma"), "CCC"),
            ("a1".into(), 0.40, 0.50, &sentence("Alpha"), "AAA"),
        ],
    );

    let bank_path = bank.to_str().unwrap();
    let first = retain::update(&[day1.to_str().unwrap().to_string()], bank_path, policy())
        .expect("first day");
    assert_eq!(first.admitted, 2);
    assert_eq!(first.bank_rows, 2);

    let second = retain::update(&[day2.to_str().unwrap().to_string()], bank_path, policy())
        .expect("second day");
    assert_eq!(second.admitted, 1);
    assert_eq!(second.known, 1);
    assert_eq!(second.bank_rows, 3);

    let corpus = fs::read_to_string(&bank).expect("corpus");
    assert_eq!(corpus.lines().count(), 3);
    // Best first, and the best row is the one with the widest gap.
    let top: Pair = pairs::from_line(corpus.lines().next().unwrap()).unwrap();
    assert_eq!(top.id, "b1");

    let index = fs::read_to_string(retain::index_path(bank_path)).expect("index");
    assert!(index.contains("\"version\":1"));
    let ledger = fs::read_to_string(retain::ledger_path(bank_path)).expect("ledger");
    assert_eq!(ledger.lines().count(), 2);
    assert!(ledger.lines().all(|line| line.contains("\"admitted\"")));

    let replayed =
        retain::update(&[day1.to_str().unwrap().to_string()], bank_path, policy()).expect("replay");
    assert_eq!(replayed.admitted, 0);
    assert_eq!(replayed.bank_rows, 3);
}

#[test]
fn budget_pushes_out_the_weakest() {
    let dir = scratch("budget");
    let bank = dir.join("week.jsonl");
    let day = dir.join("day1.jsonl");
    write_pairs(
        &day,
        &[
            ("weak".into(), 0.02, 0.95, &sentence("Weak"), "AAA"),
            ("strong".into(), 0.70, 0.50, &sentence("Strong"), "BBB"),
            ("middle".into(), 0.30, 0.50, &sentence("Middle"), "CCC"),
        ],
    );
    let tight = Policy {
        budget: 2,
        ..policy()
    };
    let report = retain::update(
        &[day.to_str().unwrap().to_string()],
        bank.to_str().unwrap(),
        tight,
    )
    .expect("run");
    assert_eq!(report.admitted, 3);
    assert_eq!(report.evicted, 1);
    assert_eq!(report.bank_rows, 2);
    let corpus = fs::read_to_string(&bank).expect("corpus");
    assert!(!corpus.contains("\"id\":\"weak\""));
}

#[test]
fn refuses_to_rewrite_a_foreign_corpus() {
    let dir = scratch("guard");
    let bank = dir.join("week.jsonl");
    let day = dir.join("day1.jsonl");
    write_pairs(
        &day,
        &[("a1".into(), 0.40, 0.50, &sentence("Alpha"), "AAA")],
    );
    fs::write(&bank, "{\"id\":\"handmade\"}\n").expect("corpus");
    let error = retain::update(
        &[day.to_str().unwrap().to_string()],
        bank.to_str().unwrap(),
        policy(),
    )
    .expect_err("must refuse");
    assert!(error.to_string().contains("refusing"));
    assert_eq!(
        fs::read_to_string(&bank).unwrap(),
        "{\"id\":\"handmade\"}\n",
        "the corpus must be untouched"
    );
}

#[test]
fn missing_text_drops_the_row() {
    let dir = scratch("missing");
    let bank = dir.join("week.jsonl");
    let day1 = dir.join("day1.jsonl");
    let day2 = dir.join("day2.jsonl");
    write_pairs(
        &day1,
        &[
            ("keep".into(), 0.40, 0.50, &sentence("Alpha"), "AAA"),
            ("lose".into(), 0.35, 0.50, &sentence("Beta"), "BBB"),
        ],
    );
    write_pairs(
        &day2,
        &[("later".into(), 0.60, 0.50, &sentence("Gamma"), "CCC")],
    );
    let path = bank.to_str().unwrap();
    retain::update(&[day1.to_str().unwrap().to_string()], path, policy()).expect("day one");
    // The next step does not resupply the first day's text, and the corpus lost
    // one line between runs, so that row cannot be written again.
    let kept: Vec<String> = fs::read_to_string(&bank)
        .unwrap()
        .lines()
        .filter(|line| !line.contains("\"id\":\"lose\""))
        .map(str::to_string)
        .collect();
    fs::write(&bank, format!("{}\n", kept.join("\n"))).unwrap();

    let report =
        retain::update(&[day2.to_str().unwrap().to_string()], path, policy()).expect("day two");
    assert_eq!(report.missing, 1);
    assert_eq!(report.bank_rows, 2);
    let corpus = fs::read_to_string(&bank).unwrap();
    assert!(!corpus.contains("\"id\":\"lose\""));
    assert!(corpus.contains("\"id\":\"keep\""));
    assert!(corpus.contains("\"id\":\"later\""));
}
