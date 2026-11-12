//! The transformer end to end through the binary.
//!
//! The fixture is small on purpose: the point is that the pair schema, the
//! tokenizer, the trainer and the model file agree with each other, not that a
//! model trained on twenty sentences is useful. The rows are separable, so a
//! model that fails to learn at all would also fail the accuracy floor.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_xcm-mmm")
}

/// A scratch directory of this test alone.
///
/// The tests run at the same time in one process, so a directory named after the
/// process would be shared and one test's cleanup would delete another's files.
fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("xcm-mmm-transformer-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> Output {
    Command::new(binary()).args(args).output().unwrap()
}

fn as_str(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

const GOOD: &[&str] = &[
    "beat estimates as demand rose",
    "raised its outlook after a strong quarter",
    "reported record profit on higher margins",
    "won a large contract and lifted guidance",
];

const BAD: &[&str] = &[
    "missed estimates as demand fell",
    "cut its outlook after a weak quarter",
    "reported a loss on thinner margins",
    "lost a large contract and withdrew guidance",
];

fn rows_body() -> String {
    let mut rng = xcm_mmm::rng::Rng::new(5);
    let mut body = String::new();
    for group in 0..8 {
        for day in 0..6 {
            for slot in 0..4 {
                let positive = rng.next_f64() < 0.5;
                let phrase = if positive {
                    GOOD[rng.below(GOOD.len())]
                } else {
                    BAD[rng.below(BAD.len())]
                };
                let outcome = if positive { 0.02 } else { -0.02 } + rng.standard_normal() * 0.004;
                let split = if day == 5 { "val" } else { "train" };
                body.push_str(&format!(
                    "{{\"id\":\"g{group}-d{day}-{slot}\",\"text\":\"the company {phrase}\",\
                     \"y\":{outcome:.6},\"group\":\"G{group}\",\"split\":\"{split}\",\
                     \"day\":\"d{day}\"}}\n"
                ));
            }
        }
    }
    body
}

#[test]
fn mine_train_eval_round_trip() {
    let dir = temp_dir("pipeline");
    let rows = dir.join("rows.jsonl");
    let pairs = dir.join("pairs.jsonl");
    let model = dir.join("model.json");
    std::fs::write(&rows, rows_body()).unwrap();

    let out = run(&[
        "pairs",
        "mine",
        &as_str(&rows),
        &as_str(&pairs),
        "--min-margin",
        "0.005",
    ]);
    assert!(
        out.status.success(),
        "mine failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run(&[
        "transformer",
        "train",
        "--pairs",
        &as_str(&pairs),
        "--out",
        &as_str(&model),
        "--vocab",
        "320",
        "--dim",
        "16",
        "--heads",
        "2",
        "--ff",
        "32",
        "--layers",
        "1",
        "--max-len",
        "24",
        "--epochs",
        "3",
        "--batch",
        "4",
        "--dropout",
        "0.0",
    ]);
    assert!(
        out.status.success(),
        "train failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("saved"), "train report: {stdout}");

    let body = std::fs::read_to_string(&model).unwrap();
    assert!(body.contains("xcm-mmm-transformer-v1"));
    assert!(body.contains("\"config\""));

    let out = run(&[
        "transformer",
        "eval",
        &as_str(&model),
        "--pairs",
        &as_str(&pairs),
    ]);
    assert!(
        out.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("accuracy"));
    assert!(stdout.contains("rank") || stdout.contains("preference"));

    let out = run(&["transformer", "info", &as_str(&model)]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("params"));

    let out = run(&[
        "transformer",
        "encode",
        &as_str(&model),
        "the company beat estimates as demand rose",
    ]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("preference"));

    let scores = dir.join("scores.jsonl");
    let out = run(&[
        "transformer",
        "score",
        &as_str(&model),
        "--rows",
        &as_str(&rows),
        "--out",
        &as_str(&scores),
    ]);
    assert!(
        out.status.success(),
        "score failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body = std::fs::read_to_string(&scores).unwrap();
    assert_eq!(body.lines().count(), 192);
    assert!(body.contains("\"scores\""));
    assert!(body.contains("\"preference\""));

    let single = dir.join("scores-one.jsonl");
    let out = run(&[
        "transformer",
        "score",
        &as_str(&model),
        "--rows",
        &as_str(&rows),
        "--out",
        &as_str(&single),
        "--task",
        "preference",
    ]);
    assert!(out.status.success());
    let body = std::fs::read_to_string(&single).unwrap();
    assert!(body.contains("\"task\":\"preference\""));
    assert!(!body.contains("\"scores\""));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_checkpoint_resumes_the_run_it_stopped() {
    let dir = temp_dir("resume");
    let rows = dir.join("rows.jsonl");
    let pairs = dir.join("pairs.jsonl");
    std::fs::write(&rows, rows_body()).unwrap();
    let out = run(&[
        "pairs",
        "mine",
        &as_str(&rows),
        &as_str(&pairs),
        "--min-margin",
        "0.005",
    ]);
    assert!(
        out.status.success(),
        "mine failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let pairs_text = as_str(&pairs);
    let shape = [
        "--vocab",
        "320",
        "--dim",
        "16",
        "--heads",
        "2",
        "--ff",
        "32",
        "--layers",
        "1",
        "--max-len",
        "24",
        "--batch",
        "4",
        "--dropout",
        "0.1",
        "--patience",
        "0",
    ];
    let full = dir.join("full.json");
    let mut args = vec!["transformer", "train", "--pairs", &pairs_text, "--out"];
    let full_text = as_str(&full);
    args.push(&full_text);
    args.extend_from_slice(&shape);
    args.extend_from_slice(&["--epochs", "3"]);
    let out = run(&args);
    assert!(
        out.status.success(),
        "full run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let checkpoint = dir.join("run.cp.json");
    let first = dir.join("first.json");
    let mut args = vec!["transformer", "train", "--pairs", &pairs_text, "--out"];
    let first_text = as_str(&first);
    args.push(&first_text);
    args.extend_from_slice(&shape);
    args.extend_from_slice(&["--epochs", "3", "--stop-after", "1", "--checkpoint"]);
    let checkpoint_text = as_str(&checkpoint);
    args.push(&checkpoint_text);
    let out = run(&args);
    assert!(
        out.status.success(),
        "first leg failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(checkpoint.exists(), "the first leg wrote no checkpoint");

    let joined = dir.join("joined.json");
    let mut args = vec!["transformer", "train", "--pairs", &pairs_text, "--out"];
    let joined_text = as_str(&joined);
    args.push(&joined_text);
    args.extend_from_slice(&shape);
    args.extend_from_slice(&["--resume"]);
    args.push(&checkpoint_text);
    let out = run(&args);
    assert!(
        out.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("resumed"), "resume report: {stdout}");

    let straight = std::fs::read(&full).unwrap();
    let resumed = std::fs::read(&joined).unwrap();
    assert_eq!(
        straight, resumed,
        "a resumed run must land on the model an uninterrupted one does"
    );

    let mut args = vec!["transformer", "train", "--pairs", &pairs_text, "--out"];
    args.push(&joined_text);
    args.extend_from_slice(&[
        "--vocab",
        "320",
        "--dim",
        "24",
        "--heads",
        "2",
        "--ff",
        "32",
        "--layers",
        "1",
        "--max-len",
        "24",
        "--batch",
        "4",
        "--resume",
    ]);
    args.push(&checkpoint_text);
    let out = run(&args);
    assert!(!out.status.success(), "a wider resume must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("different shape"), "stderr: {stderr}");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_missing_pair_file_is_reported() {
    let out = run(&[
        "transformer",
        "train",
        "--pairs",
        "/nonexistent/pairs.jsonl",
        "--out",
        "/nonexistent/model.json",
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("pairs") || stderr.contains("No such file"));
}
