//! Ship a training set: one retention bank in, the files the repo carries out.
//!
//! A bank grows every week and lives outside version control, because it is
//! rebuilt from a source corpus that is larger than any repository should hold.
//! What a repository can carry is a fixed, reproducible slice: the rows that
//! were shipped, the split they were shipped under, and a manifest that names
//! the digest of every file so a reader can tell whether the copy in hand is
//! the copy that was built.
//!
//! Three properties make the slice trustworthy:
//!
//! * it is a sample, not the bank, and it says so; the card carries both the
//!   source row count and the shipped row count;
//! * the choice is a function of the row id and the seed alone, stratified by
//!   split, task and year, so the same source and the same seed rebuild the
//!   same files byte for byte;
//! * every file is capped in bytes, so a growth in the bank cannot silently
//!   turn into a repository too large to clone.
//!
//! [`build`] writes `train.jsonl`, `valid.jsonl` and `test.jsonl` beside a
//! `manifest.json` and a `card.md`. [`check`] reads a folder back and reports
//! every way it disagrees with its own manifest.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::pairs::{self, Pair};
use crate::text::doc;

mod card;
mod manifest;

pub use card::render;
pub use manifest::{Manifest, Shipped, Source, SplitCount};

use manifest::built_at;

/// The splits a shipped set must carry. One missing split is a refusal, since a
/// card that promises a test file a reader cannot open is worse than no card.
pub const SPLITS: [&str; 3] = ["train", "valid", "test"];

/// The manifest file name inside a shipped folder.
pub const MANIFEST: &str = "manifest.json";
/// The generated card file name inside a shipped folder.
pub const CARD: &str = "card.md";

/// FNV-1a 64, used to order rows and to digest files.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Sixteen hex digits of the digest, which is what a manifest line carries.
fn digest(bytes: &[u8]) -> String {
    format!("{:016x}", fnv64(bytes))
}

/// A row's place in the sample: its id mixed with the seed, so a different seed
/// picks a different subset while the same seed picks the same one.
fn order_key(id: &str, seed: u64) -> u64 {
    let h = fnv64(id.as_bytes()) ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ (h >> 33)
}

/// The four digit year of a `YYYY-MM-DD` day, or `unknown` when the day is not
/// in that shape. An unparsed day is grouped rather than dropped.
pub fn year_of(day: &str) -> String {
    let head = day.get(0..4).unwrap_or("");
    if head.len() == 4 && head.bytes().all(|b| b.is_ascii_digit()) {
        head.to_string()
    } else {
        "unknown".to_string()
    }
}

/// What a build is allowed to produce.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BuildConfig {
    /// Rows the shipped slice may hold across all splits. Zero ships the bank.
    pub budget: usize,
    /// Seed of the sample. Changing it picks a different slice.
    pub seed: u64,
    /// Bytes any one shipped file may reach. A larger file is a refusal.
    pub cap_bytes: u64,
    /// Name the card gives the set.
    pub name: String,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            budget: 20_000,
            seed: 20_261_114,
            cap_bytes: 40 << 20,
            name: "marketlab-news-pairs".to_string(),
        }
    }
}

/// Split a budget across strata in proportion to their size, giving the
/// remainder to the largest fractional parts and breaking ties by stratum name.
/// No stratum receives more than it holds.
pub fn allocate(counts: &BTreeMap<String, usize>, budget: usize) -> BTreeMap<String, usize> {
    let total: usize = counts.values().sum();
    if budget == 0 || total <= budget {
        return counts.clone();
    }

    let mut parts: Vec<(String, usize, f64)> = counts
        .iter()
        .map(|(k, &n)| {
            let exact = budget as f64 * n as f64 / total as f64;
            let floor = exact.floor();
            (k.clone(), floor as usize, exact - floor)
        })
        .collect();

    let handed: usize = parts.iter().map(|(_, n, _)| *n).sum();
    let mut short = budget.saturating_sub(handed);

    parts.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    for (_, give, _) in parts.iter_mut() {
        if short == 0 {
            break;
        }
        *give += 1;
        short -= 1;
    }

    let mut out = BTreeMap::new();
    for (key, give, _) in parts {
        let held = counts[&key];
        let take = give.min(held);
        if take > 0 {
            out.insert(key, take);
        }
    }
    out
}

/// One candidate row with the keys the sample is stratified on.
struct Item {
    key: u64,
    task: String,
    split: String,
    year: String,
    day: String,
    pair: Pair,
}

fn read_items(path: &str, seed: u64) -> io::Result<(Vec<Item>, usize)> {
    let mut items = Vec::new();
    let mut malformed = 0usize;
    doc::for_each_line(path, |line| {
        if line.trim().is_empty() {
            return;
        }
        match pairs::from_line(line) {
            Some(pair) => {
                let year = year_of(&pair.day);
                items.push(Item {
                    key: order_key(&pair.id, seed),
                    task: pair.task.as_str().to_string(),
                    split: pair.split.to_ascii_lowercase(),
                    year,
                    day: pair.day.clone(),
                    pair,
                });
            }
            None => malformed += 1,
        }
    })?;
    Ok((items, malformed))
}

/// Write one split and return what it became.
fn write_split(out: &Path, split: &str, rows: &[&Item], cap: u64) -> io::Result<SplitCount> {
    let path = out.join(format!("{split}.jsonl"));
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    let mut count = SplitCount::default();

    for item in rows {
        let line = serde_json::to_string(&item.pair).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serialising a pair: {e}"),
            )
        })?;
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
        count.rows += 1;
        *count.tasks.entry(item.task.clone()).or_insert(0) += 1;
        if count.first_day.is_empty() || item.day < count.first_day {
            count.first_day = item.day.clone();
        }
        if item.day > count.last_day {
            count.last_day = item.day.clone();
        }
    }

    if buf.len() as u64 > cap {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{split} would be {} bytes, above the {cap} byte cap; lower the budget",
                buf.len()
            ),
        ));
    }

    let mut file = fs::File::create(&path)?;
    file.write_all(&buf)?;
    count.bytes = buf.len() as u64;
    Ok(count)
}

/// Build the shipped set: sample the bank, write one file per split, then the
/// manifest and the card.
pub fn build(input: &str, out: &str, cfg: &BuildConfig) -> io::Result<Manifest> {
    let source_bytes = fs::read(input)?;
    let (items, malformed) = read_items(input, cfg.seed)?;
    if items.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{input} held no readable pairs"),
        ));
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for item in &items {
        if !SPLITS.contains(&item.split.as_str()) {
            continue;
        }
        *counts
            .entry(format!("{}\t{}", item.split, item.task))
            .or_insert(0) += 1;
    }

    let quota = allocate(&counts, cfg.budget);

    // Within a stratum the rows are taken in key order, so the sample is not
    // the first N lines of a file whose order depends on the mining run.
    let mut by_stratum: HashMap<String, Vec<&Item>> = HashMap::new();
    for item in &items {
        if !SPLITS.contains(&item.split.as_str()) {
            continue;
        }
        by_stratum
            .entry(format!("{}\t{}", item.split, item.task))
            .or_default()
            .push(item);
    }
    for rows in by_stratum.values_mut() {
        rows.sort_by_key(|i| (i.key, i.pair.id.clone()));
    }

    let mut selected: Vec<&Item> = Vec::new();
    for (stratum, take) in &quota {
        if let Some(rows) = by_stratum.get(stratum) {
            selected.extend(rows.iter().take(*take).copied());
        }
    }
    selected.sort_by(|a, b| {
        (a.split.as_str(), a.task.as_str(), a.key).cmp(&(b.split.as_str(), b.task.as_str(), b.key))
    });

    fs::create_dir_all(out)?;

    let folder = Path::new(out)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| out.to_string());

    let mut manifest = Manifest {
        name: cfg.name.clone(),
        folder,
        built_at: built_at(),
        config: cfg.clone(),
        rows: selected.len(),
        ..Default::default()
    };

    // An absolute path is a fact about the build host, not about the data, so
    // only the file name travels into a manifest that a public repo carries.
    let source_path = if Path::new(input).is_absolute() {
        Path::new(input)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| input.to_string())
    } else {
        input.to_string()
    };

    manifest.source = Source {
        path: source_path,
        rows: items.len() + malformed,
        bytes: source_bytes.len() as u64,
        fnv64: digest(&source_bytes),
        first_day: items
            .iter()
            .map(|i| i.day.as_str())
            .min()
            .unwrap_or("")
            .to_string(),
        last_day: items
            .iter()
            .map(|i| i.day.as_str())
            .max()
            .unwrap_or("")
            .to_string(),
        issuers: items
            .iter()
            .map(|i| i.pair.a.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
    };

    for split in SPLITS {
        let rows: Vec<&Item> = selected
            .iter()
            .copied()
            .filter(|i| i.split == split)
            .collect();
        if rows.is_empty() {
            let held = items.iter().filter(|i| i.split == split).count();
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                if held == 0 {
                    format!(
                        "the source carries no {split} rows; the split is chronological, so a folder of recent months alone is all test"
                    )
                } else {
                    format!("the budget would ship no {split} rows although the source holds {held}; raise the budget")
                },
            ));
        }
        let count = write_split(Path::new(out), split, &rows, cfg.cap_bytes)?;
        manifest.files.push(Shipped {
            name: format!("{split}.jsonl"),
            rows: count.rows,
            bytes: count.bytes,
            fnv64: digest(&fs::read(Path::new(out).join(format!("{split}.jsonl")))?),
        });
        for (task, n) in &count.tasks {
            *manifest.tasks.entry(task.clone()).or_insert(0) += n;
        }
        manifest.splits.insert(split.to_string(), count);
    }

    for item in &selected {
        *manifest.years.entry(item.year.clone()).or_insert(0) += 1;
    }

    manifest.checks = checks(&manifest, &selected);

    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    fs::write(Path::new(out).join(MANIFEST), manifest_json + "\n")?;
    fs::write(Path::new(out).join(CARD), render(&manifest) + "\n")?;
    Ok(manifest)
}

/// The invariants a shipped set has to hold, stated in the manifest so a reader
/// does not have to take them on trust.
fn checks(manifest: &Manifest, selected: &[&Item]) -> Vec<String> {
    let mut out = Vec::new();

    let present: BTreeSet<&str> = selected.iter().map(|i| i.split.as_str()).collect();
    let missing: Vec<&&str> = SPLITS.iter().filter(|s| !present.contains(**s)).collect();
    out.push(if missing.is_empty() {
        "every_split_present: train, valid and test each ship rows".to_string()
    } else {
        format!("every_split_present: FAILED, missing {missing:?}")
    });

    let source_tasks: BTreeSet<&str> = selected.iter().map(|i| i.task.as_str()).collect();
    let shipped: BTreeSet<&str> = manifest.tasks.keys().map(String::as_str).collect();
    out.push(if source_tasks.is_subset(&shipped) {
        format!("every_task_kept: {} tasks present", shipped.len())
    } else {
        format!(
            "every_task_kept: FAILED, dropped {:?}",
            source_tasks.difference(&shipped)
        )
    });

    let ids: Vec<&str> = selected.iter().map(|i| i.pair.id.as_str()).collect();
    let unique: BTreeSet<&str> = ids.iter().copied().collect();
    out.push(if unique.len() == ids.len() {
        format!("no_duplicate_ids: {} ids, all distinct", ids.len())
    } else {
        format!(
            "no_duplicate_ids: FAILED, {} repeats",
            ids.len() - unique.len()
        )
    });

    let mut owners: HashMap<&str, &str> = HashMap::new();
    let mut crosses = 0usize;
    for item in selected {
        match owners.insert(item.pair.id.as_str(), item.split.as_str()) {
            Some(held) if held != item.split => crosses += 1,
            _ => {}
        }
    }
    out.push(if crosses == 0 {
        "no_id_cross_split: no row appears in two splits".to_string()
    } else {
        format!("no_id_cross_split: FAILED, {crosses} rows in more than one split")
    });

    let over: Vec<&Shipped> = manifest
        .files
        .iter()
        .filter(|f| f.bytes > manifest.config.cap_bytes)
        .collect();
    out.push(if over.is_empty() {
        format!(
            "files_under_cap: largest file {} bytes, cap {}",
            manifest.files.iter().map(|f| f.bytes).max().unwrap_or(0),
            manifest.config.cap_bytes
        )
    } else {
        format!("files_under_cap: FAILED, {over:?} above the cap")
    });

    let empty_sides = selected
        .iter()
        .filter(|i| {
            if i.pair.text_a.trim().is_empty() {
                return true;
            }
            if !i.pair.task.is_pairwise() {
                return false;
            }
            i.pair
                .text_b
                .as_deref()
                .map(|t| t.trim().is_empty())
                .unwrap_or(true)
        })
        .count();
    out.push(if empty_sides == 0 {
        "sides_non_empty: every row carries text on every side its task needs".to_string()
    } else {
        format!("sides_non_empty: FAILED, {empty_sides} rows carry an empty side")
    });

    let inside = selected.iter().all(|i| {
        i.day.as_str() >= manifest.source.first_day.as_str()
            && i.day.as_str() <= manifest.source.last_day.as_str()
    });
    out.push(if inside {
        format!(
            "days_within_source_range: {} to {}",
            manifest.source.first_day, manifest.source.last_day
        )
    } else {
        "days_within_source_range: FAILED, a shipped day is outside the source range".to_string()
    });

    out
}

/// Report every way a shipped folder disagrees with its own manifest.
pub fn check(dir: &str) -> io::Result<Vec<String>> {
    let root = Path::new(dir);
    let raw = fs::read_to_string(root.join(MANIFEST)).map_err(|e| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{dir} has no {MANIFEST}: {e}"),
        )
    })?;
    let m: Manifest = serde_json::from_str(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{MANIFEST}: {e}")))?;

    let mut faults = Vec::new();
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut total = 0usize;

    for split in SPLITS {
        let listed = m.files.iter().find(|f| f.name == format!("{split}.jsonl"));
        let Some(file) = listed else {
            faults.push(format!("{split}: listed in the manifest but no file entry"));
            continue;
        };
        let path = root.join(&file.name);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                faults.push(format!("{split}: cannot read {}: {e}", file.name));
                continue;
            }
        };
        if bytes.len() as u64 != file.bytes {
            faults.push(format!(
                "{split}: {} bytes on disk, {} in the manifest",
                bytes.len(),
                file.bytes
            ));
        }
        if digest(&bytes) != file.fnv64 {
            faults.push(format!("{split}: fnv64 differs from the manifest"));
        }
        let mut rows = 0usize;
        for line in bytes.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            match std::str::from_utf8(line).ok().and_then(pairs::from_line) {
                Some(pair) => {
                    rows += 1;
                    if pair.split != split {
                        faults.push(format!("{split}: a row carries split {:?}", pair.split));
                    }
                    if let Some(held) = seen.insert(pair.id.clone(), split.to_string()) {
                        faults.push(format!("{}: id also in {held}", pair.id));
                    }
                }
                None => faults.push(format!("{split}: one line does not parse as a pair")),
            }
        }
        if rows != file.rows {
            faults.push(format!(
                "{split}: {rows} rows on disk, {} in the manifest",
                file.rows
            ));
        }
        total += rows;
    }

    if total != m.rows {
        faults.push(format!("{total} rows on disk, {} in the manifest", m.rows));
    }
    if !root.join(CARD).exists() {
        faults.push(format!("{CARD} is missing beside the manifest"));
    }
    Ok(faults)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::Task;
    use std::path::PathBuf;

    fn pair(id: &str, task: Task, split: &str, day: &str) -> Pair {
        let mut p = Pair::new(task, split);
        p.id = id.to_string();
        p.day = day.to_string();
        p.a = "issuer".to_string();
        p.text_a = "one side of the pair".to_string();
        p
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xcm-dataset-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn allocation_keeps_the_budget_and_the_order() {
        let counts = BTreeMap::from([
            ("train\trank".to_string(), 90usize),
            ("train\tlabel".to_string(), 10),
        ]);
        let got = allocate(&counts, 20);
        assert_eq!(got.values().sum::<usize>(), 20);
        assert_eq!(got["train\trank"], 18);
        assert_eq!(got["train\tlabel"], 2);
    }

    #[test]
    fn allocation_never_takes_more_than_a_stratum_holds() {
        let counts = BTreeMap::from([("train\trank".to_string(), 3usize)]);
        let got = allocate(&counts, 100);
        assert_eq!(got["train\trank"], 3);
    }

    #[test]
    fn a_year_is_read_only_from_a_full_date() {
        assert_eq!(year_of("2024-01-18"), "2024");
        assert_eq!(year_of(""), "unknown");
        assert_eq!(year_of("24-01-18"), "unknown");
    }

    #[test]
    fn the_same_seed_builds_the_same_bytes() {
        let dir = tmp_dir("seed");
        let bank = dir.join("bank.jsonl");
        let mut f = fs::File::create(&bank).unwrap();
        for i in 0..30 {
            let split = SPLITS[i % 3];
            let task = Task::ALL[i % Task::ALL.len()];
            writeln!(
                f,
                "{}",
                serde_json::to_string(&pair(&format!("id{i:03}"), task, split, "2024-01-18"))
                    .unwrap()
            )
            .unwrap();
        }
        drop(f);

        let a = dir.join("a");
        let b = dir.join("b");
        let cfg = BuildConfig {
            budget: 21,
            seed: 7,
            ..Default::default()
        };
        let first = build(bank.to_str().unwrap(), a.to_str().unwrap(), &cfg).unwrap();
        build(bank.to_str().unwrap(), b.to_str().unwrap(), &cfg).unwrap();

        for name in ["train.jsonl", "valid.jsonl", "test.jsonl"] {
            assert_eq!(
                fs::read(a.join(name)).unwrap(),
                fs::read(b.join(name)).unwrap(),
                "{name} differed between two builds"
            );
        }
        assert_eq!(first.rows, 21);
        assert!(check(a.to_str().unwrap()).unwrap().is_empty());
    }

    #[test]
    fn a_tampered_file_is_caught() {
        let dir = tmp_dir("tamper");
        let bank = dir.join("bank.jsonl");
        let mut f = fs::File::create(&bank).unwrap();
        for i in 0..9 {
            let split = SPLITS[i % 3];
            writeln!(
                f,
                "{}",
                serde_json::to_string(&pair(&format!("id{i}"), Task::Rank, split, "2023-05-02"))
                    .unwrap()
            )
            .unwrap();
        }
        drop(f);

        let out = dir.join("ship");
        let cfg = BuildConfig::default();
        build(bank.to_str().unwrap(), out.to_str().unwrap(), &cfg).unwrap();
        assert!(check(out.to_str().unwrap()).unwrap().is_empty());

        let mut rows = fs::read_to_string(out.join("train.jsonl")).unwrap();
        rows.push('\n');
        fs::write(out.join("train.jsonl"), rows).unwrap();
        let faults = check(out.to_str().unwrap()).unwrap();
        assert!(faults.iter().any(|f| f.contains("on disk")), "{faults:?}");
    }

    #[test]
    fn a_missing_split_is_a_refusal() {
        let dir = tmp_dir("split");
        let bank = dir.join("bank.jsonl");
        let mut f = fs::File::create(&bank).unwrap();
        for i in 0..4 {
            writeln!(
                f,
                "{}",
                serde_json::to_string(&pair(&format!("id{i}"), Task::Rank, "train", "2023-05-02"))
                    .unwrap()
            )
            .unwrap();
        }
        drop(f);
        let err = build(
            bank.to_str().unwrap(),
            dir.join("ship").to_str().unwrap(),
            &BuildConfig::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("valid"), "{err}");
    }

    #[test]
    fn the_card_names_the_sample_and_the_source() {
        let dir = tmp_dir("card");
        let bank = dir.join("bank.jsonl");
        let mut f = fs::File::create(&bank).unwrap();
        for i in 0..9 {
            let split = SPLITS[i % 3];
            writeln!(
                f,
                "{}",
                serde_json::to_string(&pair(&format!("id{i}"), Task::Rank, split, "2023-05-02"))
                    .unwrap()
            )
            .unwrap();
        }
        drop(f);
        let out = dir.join("ship");
        build(
            bank.to_str().unwrap(),
            out.to_str().unwrap(),
            &BuildConfig {
                budget: 6,
                ..Default::default()
            },
        )
        .unwrap();
        let card = fs::read_to_string(out.join(CARD)).unwrap();
        assert!(
            !card.contains(dir.to_str().unwrap()),
            "the card leaked a build path"
        );
        assert!(card.contains("6 rows in total"));
        assert!(card.contains("9 rows in the source bank"));
        assert!(card.contains("every_split_present"));
    }
}
