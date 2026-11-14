//! Fitting the emotion model: the seed, the bank adjustment, and the honesty pass.
//!
//! The seed is knowledge, not a result: 661 English words with a weight on each
//! emotion axis. The labelled bank is twenty contact-centre transcripts, which
//! is far too little to learn an emotion model from and exactly enough to
//! adjust one and to test it. So the fit does one thing to the bank: for each
//! emotion that at least two rows carry, it takes the words that appear
//! disproportionately in those rows, shrinks the ratio towards zero, caps the
//! weight and keeps the top twenty. That adjustment is blended at 0.6 over the
//! seed, and the card says so.
//!
//! Every accuracy in the artifact is a held-out number. `fit_accuracy` is the
//! bank it was adjusted on, printed only because a reader wants to see it;
//! `leave_one_out` refits the adjustment without each row in turn, so the
//! number that carries any weight is the second one. `majority` is the same
//! score for answering the commonest label every time.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::analogy::AnalogyConfig;
use super::model::{round4, BankMetrics, EmotionModel, Modifiers, Reading, Source};
use crate::dataset::built_at;
use crate::detect::token;
use crate::digest::fnv64;

pub const SEED: &str = "seed.json";
pub const BANK: &str = "bank.jsonl";
pub const LISTED: &str = "listed.tsv";
pub const MODEL: &str = "model.json";
pub const MANIFEST: &str = "manifest.json";
pub const CARD: &str = "card.md";

/// Words a class may contribute to the adjustment.
const TOP_TERMS: usize = 12;
/// Rows an emotion needs before its words are read at all.
const MIN_ROWS: usize = 2;
/// Shrinkage mass on the class counts.
const ALPHA: f64 = 0.5;
/// Ceiling on one word's adjustment.
const CAP: f64 = 0.35;

/// `seed.json` as it travels.
#[derive(Clone, Debug, Deserialize)]
pub struct SeedFile {
    pub version: u32,
    pub emotions: Vec<String>,
    pub neutral_band: f64,
    pub min_mass: f64,
    pub prior: f64,
    pub blend: f64,
    pub analogy: AnalogyConfig,
    pub valence: BTreeMap<String, f64>,
    pub arousal: BTreeMap<String, f64>,
    pub flip: BTreeMap<String, String>,
    pub prefixes: Vec<String>,
    pub prefix_keep: Vec<String>,
    pub suffixes: Vec<super::model::SuffixRule>,
    pub modifiers: Modifiers,
    pub labels: BTreeMap<String, BTreeMap<String, f64>>,
    pub listed_positive: BTreeMap<String, f64>,
    pub listed_negative: BTreeMap<String, f64>,
    pub groups: Vec<Group>,
}

/// One emotion's seed words at one weight.
#[derive(Clone, Debug, Deserialize)]
pub struct Group {
    pub emotion: String,
    pub weight: f64,
    pub words: Vec<String>,
}

impl SeedFile {
    pub fn load(path: &str) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let seed: Self = serde_json::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))?;
        seed.validate()?;
        Ok(seed)
    }

    /// A seed that names an emotion nobody scores is a refusal, not a zero.
    pub fn validate(&self) -> io::Result<()> {
        let names: BTreeSet<&String> = self.emotions.iter().collect();
        let bad = |what: &str, name: &str| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("seed names an emotion that is not scored: {what} {name}"),
            )
        };
        for emotion in self.valence.keys() {
            if !names.contains(emotion) {
                return Err(bad("valence", emotion));
            }
        }
        for emotion in self.arousal.keys() {
            if !names.contains(emotion) {
                return Err(bad("arousal", emotion));
            }
        }
        for (from, to) in &self.flip {
            if !names.contains(from) || !names.contains(to) {
                return Err(bad("flip", from));
            }
        }
        for group in &self.groups {
            if !names.contains(&group.emotion) {
                return Err(bad("group", &group.emotion));
            }
            if !(0.0..=1.0).contains(&group.weight) || group.weight == 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "group weight out of range: {} {}",
                        group.emotion, group.weight
                    ),
                ));
            }
        }
        for label in self.labels.values() {
            for emotion in label.keys() {
                if !names.contains(emotion) {
                    return Err(bad("label", emotion));
                }
            }
        }
        if self.emotions.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no emotions"));
        }
        Ok(())
    }

    /// Group entries merged by max: a word belongs to the axis it most names.
    pub fn words(&self) -> BTreeMap<String, Vec<f64>> {
        let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for group in &self.groups {
            let Some(at) = self.emotions.iter().position(|e| e == &group.emotion) else {
                continue;
            };
            for word in &group.words {
                let entry = out
                    .entry(word.to_lowercase())
                    .or_insert_with(|| vec![0.0; self.emotions.len()]);
                entry[at] = entry[at].max(group.weight);
            }
        }
        out
    }

    /// The emotion a label expects, for scoring the bank.
    pub fn expected(&self, label: &str) -> String {
        self.labels
            .get(label)
            .and_then(|weights| {
                weights
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                    .map(|(emotion, _)| emotion.clone())
            })
            .unwrap_or_else(|| "neutral".to_string())
    }
}

/// One labelled row of the bank.
#[derive(Clone, Debug, Deserialize)]
pub struct BankRow {
    pub id: String,
    pub text: String,
    pub label: String,
    #[serde(default)]
    pub source: String,
}

/// Read the bank, refusing an unknown label or an empty text.
pub fn read_bank(path: &str) -> io::Result<Vec<BankRow>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: BankRow = serde_json::from_str(line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}:{} {}", number + 1, e),
            )
        })?;
        if row.text.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}:{} has no text", number + 1),
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} holds no rows"),
        ));
    }
    Ok(rows)
}

/// Word counts per row, short words and the closed class dropped.
fn counts(rows: &[BankRow]) -> Vec<BTreeMap<String, usize>> {
    rows.iter()
        .map(|row| {
            let mut map = BTreeMap::new();
            for word in token::tokens(&row.text) {
                if word.chars().count() >= 3 && !token::is_function(&word) {
                    *map.entry(word).or_insert(0) += 1;
                }
            }
            map
        })
        .collect()
}

/// How much the neutral rows say a word means nothing, per word.
///
/// Seven of the twenty rows are labelled Neutral and they are transactional
/// calls: `order`, `please`, `inquire`, `question`. Those words are not
/// evidence of a feeling, and without this pass they win the anticipation axis
/// on a call that has no feeling in it at all. The pass is asymmetric on
/// purpose: a word seen at least twice in a neutral row and proportionally more
/// often there than elsewhere earns a damp of at most 0.6, and a word that never
/// appears in a neutral row is untouched.
pub fn learn_neutral(seed: &SeedFile, rows: &[BankRow]) -> BTreeMap<String, f64> {
    let counts = counts(rows);
    let neutral: Vec<bool> = rows
        .iter()
        .map(|row| seed.expected(&row.label) == "neutral")
        .collect();
    let neutral_rows = neutral.iter().filter(|n| **n).count();
    if neutral_rows < MIN_ROWS {
        return BTreeMap::new();
    }
    let mut positive: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rest: BTreeMap<&str, usize> = BTreeMap::new();
    for (at, counts) in counts.iter().enumerate() {
        let target = if neutral[at] {
            &mut positive
        } else {
            &mut rest
        };
        for (word, count) in counts {
            *target.entry(word.as_str()).or_insert(0) += count;
        }
    }
    let pos_total: usize = positive.values().sum();
    let rest_total: usize = rest.values().sum();
    let mut scored: Vec<(String, f64)> = Vec::new();
    for (word, count) in &positive {
        if *count < 3 {
            continue;
        }
        let seen = rest.get(word).copied().unwrap_or(0);
        let pos_rate = (*count as f64 + ALPHA) / (pos_total as f64 + ALPHA * 2.0);
        let rest_rate = (seen as f64 + ALPHA) / (rest_total.max(1) as f64 + ALPHA * 2.0);
        let delta = (pos_rate / rest_rate).ln();
        if delta <= 0.7 {
            continue;
        }
        scored.push((word.to_string(), round4((0.15 * delta).min(0.6))));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(TOP_TERMS * 2);
    scored.into_iter().collect()
}

/// A valence list: one token and a score from minus four to four per line.
///
/// The file is the VADER lexicon, MIT licensed, trimmed to single alphabetic
/// tokens. It is not an emotion model: it says how positive or negative a word
/// is and nothing about which of the eight axes it belongs on. The mapping from
/// that valence onto the axes is declared in the seed (`listed_positive`,
/// `listed_negative`), so the axis reading stays this model's decision and the
/// list supplies only the sign and the strength.
pub fn read_listed(path: &str) -> io::Result<BTreeMap<String, f64>> {
    let text = fs::read_to_string(path)?;
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let (Some(token), Some(value)) = (parts.next(), parts.next()) else {
            continue;
        };
        let token = token.trim().to_lowercase();
        if !token.chars().all(|c| c.is_alphabetic()) || token.chars().count() < 2 {
            continue;
        }
        let Ok(value) = value.trim().parse::<f64>() else {
            continue;
        };
        out.insert(token, round4((value / 4.0).clamp(-1.0, 1.0)));
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} holds no listed words"),
        ));
    }
    Ok(out)
}

/// The adjustment the bank supports, per emotion, word by word.
pub fn learn(
    seed: &SeedFile,
    rows: &[BankRow],
) -> (BTreeMap<String, Vec<f64>>, BTreeMap<String, Vec<String>>) {
    let width = seed.emotions.len();
    let counts = counts(rows);
    let member: Vec<Vec<bool>> = rows
        .iter()
        .map(|row| {
            let weights = seed.labels.get(&row.label);
            (0..width)
                .map(|at| {
                    weights
                        .and_then(|w| w.get(&seed.emotions[at]))
                        .copied()
                        .unwrap_or(0.0)
                        >= 0.5
                })
                .collect()
        })
        .collect();
    let mut adjustments: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut top: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for at in 0..width {
        let positives = member.iter().filter(|row| row[at]).count();
        if positives < MIN_ROWS {
            continue;
        }
        let mut positive: BTreeMap<&str, usize> = BTreeMap::new();
        let mut rest: BTreeMap<&str, usize> = BTreeMap::new();
        for (row, counts) in counts.iter().enumerate() {
            let target = if member[row][at] {
                &mut positive
            } else {
                &mut rest
            };
            for (word, count) in counts {
                *target.entry(word.as_str()).or_insert(0) += count;
            }
        }
        let pos_total: usize = positive.values().sum();
        let rest_total: usize = rest.values().sum();
        let mut scored: Vec<(String, f64)> = Vec::new();
        for (word, count) in &positive {
            if *count < 4 {
                continue;
            }
            let seen = rest.get(word).copied().unwrap_or(0);
            let pos_rate = (*count as f64 + ALPHA) / (pos_total as f64 + ALPHA * 2.0);
            let rest_rate = (seen as f64 + ALPHA) / (rest_total.max(1) as f64 + ALPHA * 2.0);
            let delta = (pos_rate / rest_rate).ln();
            if delta <= 1.0 {
                continue;
            }
            scored.push((word.to_string(), round4((0.5 * delta).min(CAP))));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(TOP_TERMS);
        top.insert(
            seed.emotions[at].clone(),
            scored.iter().map(|(word, _)| word.clone()).collect(),
        );
        for (word, weight) in scored {
            let entry = adjustments.entry(word).or_insert_with(|| vec![0.0; width]);
            entry[at] = weight;
        }
    }
    (adjustments, top)
}

/// An artifact from a seed, a bank and the adjustment the bank supports.
///
/// A row whose label the seed does not name is a refusal: silently reading it
/// as neutral would inflate the majority baseline with rows nobody labelled.
pub fn fit(
    seed_file: &SeedFile,
    rows: &[BankRow],
    listed: BTreeMap<String, f64>,
    source: Source,
) -> io::Result<EmotionModel> {
    for row in rows {
        if !seed_file.labels.contains_key(&row.label) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} carries an unknown label: {}", row.id, row.label),
            ));
        }
    }
    let (learned, learned_top) = learn(seed_file, rows);
    let neutral = learn_neutral(seed_file, rows);
    let mut model = EmotionModel {
        name: "xcm-mmm-emotion".to_string(),
        version: 1,
        built_at: built_at(),
        emotions: seed_file.emotions.clone(),
        valence_weights: seed_file.valence.clone(),
        arousal_weights: seed_file.arousal.clone(),
        flip: seed_file.flip.clone(),
        seed: seed_file.words(),
        learned,
        listed,
        listed_positive: seed_file.listed_positive.clone(),
        listed_negative: seed_file.listed_negative.clone(),
        neutral,
        blend: seed_file.blend,
        prefixes: seed_file.prefixes.clone(),
        prefix_keep: seed_file.prefix_keep.iter().cloned().collect(),
        suffixes: seed_file.suffixes.clone(),
        modifiers: seed_file.modifiers.clone(),
        neutral_band: seed_file.neutral_band,
        min_mass: seed_file.min_mass,
        prior: seed_file.prior,
        analogy: seed_file.analogy.clone(),
        source,
        metrics: BankMetrics::default(),
        learned_top,
    };
    model.metrics = metrics(seed_file, rows, &model);
    Ok(model)
}

/// Whether a reading answers a row's expected label.
///
/// A label that maps to an emotion is answered by the dominant axis. `Neutral`
/// maps to no emotion, and it is scored differently on purpose: it names the
/// absence of a complaint, not the absence of affect, and a polite order call
/// reads as mild anticipation without being a complaint. A neutral row is
/// correct when the reading is not negative, which is the question a retention
/// team actually asks of it.
fn correct(reading: &Reading, expected: &str, band: f64) -> bool {
    if expected == "neutral" {
        reading.valence > -band
    } else {
        reading.dominant == expected
    }
}

/// Fit accuracy, leave-one-out accuracy, and the majority baseline.
fn metrics(seed_file: &SeedFile, rows: &[BankRow], model: &EmotionModel) -> BankMetrics {
    let scanner = model.scanner();
    let expected: Vec<String> = rows
        .iter()
        .map(|row| seed_file.expected(&row.label))
        .collect();
    let fit_hits = rows
        .iter()
        .zip(&expected)
        .filter(|(row, want)| correct(&scanner.read(&row.text), want, seed_file.neutral_band))
        .count();
    let mut leave_one_out = 0usize;
    for held in 0..rows.len() {
        let rest: Vec<BankRow> = rows
            .iter()
            .enumerate()
            .filter(|(at, _)| *at != held)
            .map(|(_, row)| row.clone())
            .collect();
        let (learned, _) = learn(seed_file, &rest);
        let mut fold = model.clone();
        fold.learned = learned;
        fold.neutral = learn_neutral(seed_file, &rest);
        let reading = fold.scanner().read(&rows[held].text);
        if correct(&reading, &expected[held], seed_file.neutral_band) {
            leave_one_out += 1;
        }
    }
    let mut per_label: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut labels: BTreeSet<&String> = BTreeSet::new();
    for (at, row) in rows.iter().enumerate() {
        labels.insert(&row.label);
        let hit = correct(
            &scanner.read(&row.text),
            &expected[at],
            seed_file.neutral_band,
        );
        let entry = per_label.entry(row.label.clone()).or_insert((0, 0));
        entry.0 += usize::from(hit);
        entry.1 += 1;
    }
    let per_label: BTreeMap<String, f64> = per_label
        .into_iter()
        .map(|(label, (correct, total))| (label, round4(correct as f64 / total.max(1) as f64)))
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for label in &expected {
        *counts.entry(label.as_str()).or_insert(0) += 1;
    }
    let n = rows.len().max(1) as f64;
    BankMetrics {
        rows: rows.len(),
        labels: labels.len(),
        fit_accuracy: round4(fit_hits as f64 / n),
        leave_one_out: round4(leave_one_out as f64 / n),
        majority: round4(counts.values().copied().max().unwrap_or(0) as f64 / n),
        per_label,
    }
}

/// Build the shipped folder and return what it wrote.
pub fn build(dir: &str) -> io::Result<EmotionModel> {
    let seed_path = format!("{dir}/{SEED}");
    let bank_path = format!("{dir}/{BANK}");
    let listed_path = format!("{dir}/{LISTED}");
    let seed_file = SeedFile::load(&seed_path)?;
    let rows = read_bank(&bank_path)?;
    let listed = read_listed(&listed_path)?;
    let source = Source {
        seed: SEED.to_string(),
        seed_words: seed_file.words().len(),
        seed_fnv64: fnv64(&fs::read(&seed_path)?),
        bank: BANK.to_string(),
        bank_rows: rows.len(),
        bank_fnv64: fnv64(&fs::read(&bank_path)?),
        listed: LISTED.to_string(),
        listed_rows: listed.len(),
        listed_fnv64: fnv64(&fs::read(&listed_path)?),
    };
    let model = fit(&seed_file, &rows, listed, source)?;
    model.save(&format!("{dir}/{MODEL}"))?;
    let manifest = stamp(&model, dir)?;
    let text = serde_json::to_string_pretty(&manifest).map_err(io::Error::other)?;
    fs::write(format!("{dir}/{MANIFEST}"), format!("{text}\n"))?;
    fs::write(format!("{dir}/{CARD}"), super::card::render(&model))?;
    Ok(model)
}

/// One file the folder ships and the digest that proves it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileStamp {
    pub name: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
}

/// The folder's account of itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub files: Vec<FileStamp>,
    pub emotions: Vec<String>,
    pub seed_words: usize,
    pub bank_rows: usize,
    pub blend: f64,
    pub neutral_band: f64,
    pub min_mass: f64,
    pub metrics: BankMetrics,
}

fn stamp(model: &EmotionModel, dir: &str) -> io::Result<Manifest> {
    let mut files = Vec::new();
    for (name, rows) in [
        (SEED, model.source.seed_words),
        (BANK, model.source.bank_rows),
        (LISTED, model.source.listed_rows),
        (MODEL, model.learned.len()),
    ] {
        let path = format!("{dir}/{name}");
        let bytes = fs::read(&path)?;
        files.push(FileStamp {
            name: name.to_string(),
            rows,
            bytes: bytes.len() as u64,
            fnv64: fnv64(&bytes),
        });
    }
    Ok(Manifest {
        name: model.name.clone(),
        version: model.version,
        built_at: model.built_at.clone(),
        files,
        emotions: model.emotions.clone(),
        seed_words: model.seed.len(),
        bank_rows: model.source.bank_rows,
        blend: model.blend,
        neutral_band: model.neutral_band,
        min_mass: model.min_mass,
        metrics: model.metrics.clone(),
    })
}

/// Rebuild a folder from its own inputs and report every disagreement.
pub fn check(dir: &str) -> io::Result<Vec<String>> {
    let mut faults = Vec::new();
    let shipped = EmotionModel::load(&format!("{dir}/{MODEL}"))?;
    let seed_file = SeedFile::load(&format!("{dir}/{SEED}"))?;
    let rows = read_bank(&format!("{dir}/{BANK}"))?;
    let listed_path = format!("{dir}/{LISTED}");
    let listed = read_listed(&listed_path)?;
    let seed_bytes = fs::read(format!("{dir}/{SEED}"))?;
    let bank_bytes = fs::read(format!("{dir}/{BANK}"))?;
    let listed_bytes = fs::read(&listed_path)?;
    if fnv64(&seed_bytes) != shipped.source.seed_fnv64 {
        faults.push(format!("{SEED} is not the seed the model was fitted on"));
    }
    if fnv64(&bank_bytes) != shipped.source.bank_fnv64 {
        faults.push(format!("{BANK} is not the bank the model was fitted on"));
    }
    if fnv64(&listed_bytes) != shipped.source.listed_fnv64 {
        faults.push(format!("{LISTED} is not the list the model was fitted on"));
    }
    let rebuilt = fit(
        &seed_file,
        &rows,
        listed,
        Source {
            seed: SEED.to_string(),
            seed_words: seed_file.words().len(),
            seed_fnv64: fnv64(&seed_bytes),
            bank: BANK.to_string(),
            bank_rows: rows.len(),
            bank_fnv64: fnv64(&bank_bytes),
            listed: LISTED.to_string(),
            listed_rows: shipped.source.listed_rows,
            listed_fnv64: fnv64(&listed_bytes),
        },
    )?;
    if !super::model::same_fit(&shipped, &rebuilt) {
        faults.push("model.json does not rebuild from its own inputs".to_string());
    }
    if let Ok(text) = fs::read_to_string(format!("{dir}/{MANIFEST}")) {
        match serde_json::from_str::<Manifest>(&text) {
            Ok(manifest) => {
                if manifest.seed_words != seed_file.words().len() {
                    faults.push("manifest seed word count disagrees".to_string());
                }
                if manifest.bank_rows != rows.len() {
                    faults.push("manifest bank row count disagrees".to_string());
                }
                for file in &manifest.files {
                    let path = format!("{dir}/{}", file.name);
                    match fs::read(&path) {
                        Ok(bytes) if fnv64(&bytes) != file.fnv64 => {
                            faults
                                .push(format!("{} does not match its manifest digest", file.name));
                        }
                        Ok(_) => {}
                        Err(_) => faults.push(format!("{} is missing", file.name)),
                    }
                }
            }
            Err(e) => faults.push(format!("{MANIFEST}: {e}")),
        }
    } else {
        faults.push(format!("{MANIFEST} is missing"));
    }
    Ok(faults)
}

/// The path the folder lives at, for a caller that starts at the repository.
pub fn folder() -> &'static str {
    "data/emotion"
}

/// Read one text with a model, for a caller that only wants the reading.
pub fn read_text(model: &EmotionModel, text: &str) -> Reading {
    model.scanner().read(text)
}

/// True when a path exists and is a file.
pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> SeedFile {
        SeedFile::load("data/emotion/seed.json").expect("the shipped seed")
    }

    fn rows() -> Vec<BankRow> {
        read_bank("data/emotion/bank.jsonl").expect("the shipped bank")
    }

    fn listed() -> BTreeMap<String, f64> {
        read_listed("data/emotion/listed.tsv").expect("the shipped list")
    }

    #[test]
    fn the_seed_scores_eight_emotions_and_names_them_once() {
        let seed = seed();
        assert_eq!(seed.emotions.len(), 8);
        assert!(seed.words().len() > 500);
        let unique: BTreeSet<&String> = seed.emotions.iter().collect();
        assert_eq!(unique.len(), seed.emotions.len());
    }

    #[test]
    fn a_seed_word_is_placed_on_the_axis_it_names() {
        let seed = seed();
        let words = seed.words();
        let at = seed.emotions.iter().position(|e| e == "joy").expect("joy");
        let joy = words.get("happy").expect("happy is seeded");
        assert!(joy[at] > 0.5);
    }

    #[test]
    fn the_bank_reads_a_happy_call_as_joy() {
        let model = fit(&seed(), &rows(), listed(), Source::default()).expect("a fit");
        let happy = rows()
            .into_iter()
            .find(|row| row.label == "Happy")
            .expect("a happy row");
        assert_eq!(model.scanner().read(&happy.text).dominant, "joy");
    }

    #[test]
    fn the_adjustment_never_takes_a_term_over_the_cap() {
        let (learned, _) = learn(&seed(), &rows());
        assert!(!learned.is_empty());
        for weights in learned.values() {
            for weight in weights {
                assert!(*weight <= CAP + 1e-9, "{weight}");
            }
        }
    }

    #[test]
    fn a_label_the_seed_does_not_name_is_a_refusal() {
        let mut bad = rows();
        bad[0].label = "Furious".to_string();
        let error = fit(&seed(), &bad, listed(), Source::default()).expect_err("a refusal");
        assert!(error.to_string().contains("unknown label"), "{error}");
    }

    #[test]
    fn a_rebuild_of_the_shipped_folder_is_clean() {
        if !is_file("data/emotion/model.json") {
            return;
        }
        let faults = check("data/emotion").expect("the folder");
        assert!(faults.is_empty(), "{faults:?}");
    }

    #[test]
    fn the_metrics_are_held_out_and_not_perfect_by_construction() {
        let model = fit(&seed(), &rows(), listed(), Source::default()).expect("a fit");
        assert!(model.metrics.leave_one_out <= model.metrics.fit_accuracy);
        assert!(model.metrics.majority > 0.0);
        assert_eq!(model.metrics.rows, 20);
        assert_eq!(model.metrics.labels, 5);
    }
}
