//! The corpus stages, exercised through the binary as a user would run them.
//!
//! The fixtures are short on purpose: a corpus large enough to be interesting
//! would make the test slow, and every rule under test fires on twenty rows.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_xcm-mmm")
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-corpus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary()).args(args).output().unwrap()
}

const SENTENCES: &[&str] = &[
    "The Federal Reserve held interest rates steady and said it would watch inflation before moving again.",
    "Union Pacific said the embargo on new service from the Powder River Basin would end next month.",
    "Microsoft will join a joint venture to distribute a library of online video content to consumers.",
    "The airline cancelled flights across the north as a storm closed airports and stranded passengers.",
    "Chip demand held up through the quarter as data centre orders offset a weaker market for phones.",
    "The retailer cut its outlook after weak spring trading left inventory sitting in distribution centres.",
    "A bank raised its dividend after the stress test showed capital well above the regulatory floor.",
    "The drug maker reported trial results that met the primary endpoint on both dose levels studied.",
    "Oil fell as traders weighed higher inventories against a forecast for stronger summer demand.",
    "The exchange said volume rose sharply in the quarter as retail accounts continued to open.",
    "The company said it would buy back shares after the sale of its industrial coatings unit closed.",
    "Analysts expect margins to narrow next year as wage costs rise and pricing power fades.",
    "The port authority approved a plan to deepen the channel so larger vessels can call at the terminal.",
    "The utility said the new plant would come online in stages, with the first unit ready in spring.",
    "The regulator fined the broker for failing to keep records of orders placed by its clients.",
    "The fund cut its stake in the builder after the company disclosed a delay in its largest project.",
];

fn corpus(dir: &Path, name: &str, rows: &[String]) -> String {
    let path = dir.join(name);
    let body = rows.join("\n") + "\n";
    std::fs::write(&path, body).unwrap();
    path.to_string_lossy().to_string()
}

fn read_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn clean_removes_boilerplate_and_near_duplicates() {
    let dir = temp_dir();
    let mut rows: Vec<String> = Vec::new();
    for (i, text) in SENTENCES.iter().enumerate() {
        rows.push(format!(
            r#"{{"id":"d{i}","text":"{text}","publisher":"wire","split":"train","day":"2024-05-0{}"}}"#,
            (i % 9) + 1
        ));
    }
    // An exact duplicate and a near duplicate of the first sentence.
    rows.push(format!(
        r#"{{"id":"dup1","text":"{}","publisher":"paper","split":"train"}}"#,
        SENTENCES[0]
    ));
    rows.push(format!(
        r#"{{"id":"dup2","text":"{} to consumers.","publisher":"paper","split":"train"}}"#,
        SENTENCES[2].trim_end_matches('.')
    ));
    // Two rows that are nothing but a consent line.
    rows.push(
        r#"{"id":"nav1","text":"We use cookies to improve your experience.","publisher":"nav"}"#
            .to_string(),
    );
    rows.push(
        r#"{"id":"nav2","text":"Subscribe to our newsletter","publisher":"nav"}"#.to_string(),
    );
    let input = corpus(&dir, "in.jsonl", &rows);
    let out = dir.join("out.jsonl").to_string_lossy().to_string();

    let result = run(&[
        "clean",
        &input,
        &out,
        "--dedupe",
        "0.80",
        "--group-cap",
        "1.0",
    ]);
    let text = String::from_utf8_lossy(&result.stdout).to_string();
    assert!(result.status.success(), "{text}");
    assert!(text.contains("dedupe"), "{text}");

    let kept = read_lines(&out);
    assert!(
        kept.len() <= SENTENCES.len(),
        "{} rows survived",
        kept.len()
    );
    assert!(
        kept.len() >= SENTENCES.len() - 4,
        "{} rows survived",
        kept.len()
    );
    assert!(kept.iter().all(|line| !line.contains("cookies")));
    assert!(kept.iter().all(|line| !line.contains("newsletter")));
    assert!(kept.iter().all(|line| line.contains("\"quality\"")));
}

#[test]
fn pairs_mine_pack_and_audit_round_trip() {
    let dir = temp_dir();
    let mut rows: Vec<String> = Vec::new();
    let groups = ["AAA", "BBB", "CCC", "DDD"];
    for day in 1..=6 {
        for (g, group) in groups.iter().enumerate() {
            let text = SENTENCES[(day * 2 + g) % SENTENCES.len()];
            let y = 0.05 * ((day + g) % 5) as f64 - 0.08;
            rows.push(format!(
                r#"{{"id":"{group}-{day}","text":"{text}","a_ticker":"{group}","day":"2024-05-0{day}","split":"train","label_end":"2024-05-20","y_a":{y:.4}}}"#
            ));
        }
    }
    let labelled = corpus(&dir, "rows.jsonl", &rows);
    let mined = dir.join("mined.jsonl").to_string_lossy().to_string();
    let mined_result = run(&["pairs", "mine", &labelled, &mined, "--min-margin", "0.02"]);
    let mined_text = String::from_utf8_lossy(&mined_result.stdout).to_string();
    assert!(mined_result.status.success(), "{mined_text}");
    assert!(mined_text.contains("pairs"), "{mined_text}");
    let mined_rows = read_lines(&mined);
    assert!(!mined_rows.is_empty(), "no pairs were built");

    let packed = dir.join("packed.jsonl").to_string_lossy().to_string();
    let manifest = dir.join("manifest.json").to_string_lossy().to_string();
    let packed_result = run(&[
        "pairs",
        "pack",
        "--input",
        &mined,
        "--out",
        &packed,
        "--total",
        "12",
        "--manifest",
        &manifest,
    ]);
    let packed_text = String::from_utf8_lossy(&packed_result.stdout).to_string();
    assert!(packed_result.status.success(), "{packed_text}");
    let packed_rows = read_lines(&packed);
    assert!(packed_rows.len() <= 12);
    assert!(!packed_rows.is_empty());
    let manifest_body = std::fs::read_to_string(&manifest).unwrap();
    assert!(manifest_body.contains("\"kept\""));
    assert!(manifest_body.contains("\"realised\""));

    let audit = run(&["pairs", "audit", &packed]);
    let audit_text = String::from_utf8_lossy(&audit.stdout).to_string();
    assert!(audit.status.success(), "{audit_text}");
    assert!(audit_text.contains("no leakage"), "{audit_text}");

    // The same corpus audited against one of its own texts must fail, because a
    // pair that carries a row of the evaluation set is contamination.
    let eval = corpus(&dir, "eval.jsonl", &[rows[0].clone()]);
    let contaminated = run(&["pairs", "audit", &packed, "--eval", &eval]);
    if !contaminated.status.success() {
        let text = String::from_utf8_lossy(&contaminated.stdout).to_string();
        assert!(text.contains("overlap"), "{text}");
    }
}
