//! Fit the seven axes, and check a shipped folder against its own account.
//!
//! The two kinds of axis are fitted differently on purpose.
//!
//! A lexicon axis has a labelled bank, so it is fitted and then scored out of
//! fold: folds are assigned by the digest of the text, which puts a repeated
//! sentence in the same fold every time, and the cut is chosen inside the
//! training part of each fold so the held out part informs nothing. The same
//! loop runs again on shuffled labels as a control, because a bank of legal
//! abstracts against ad copy is easy enough to separate that a broken fit would
//! still look impressive next to a majority class.
//!
//! An index axis has no labels, so nothing is scored for accuracy. Its
//! component quantiles are fitted on the training ads, its cuts on the scores
//! those ads actually reach, and its audit reports whether two disjoint halves
//! of its own cues pick out the same ads. The declared examples in the cue table
//! are run as well: a cue set that cannot fire on text where the construct is
//! plainly present is a regression, and the build refuses rather than ship it.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use super::bank::{self, AdRow, BankRow, SHARED_NEGATIVE};
use super::card;
use super::eval::{self, Metrics};
use super::feature::{FeatureSpec, Features};
use super::index::{self, Index, IndexSpec};
use super::lexicon::{self, round4, Lexicon};
use super::logistic::Logistic;
use super::model::{
    Artifact, Audit, Axis, AxisStamp, CheckRun, FileStamp, IndexAxis, LexiconAxis, Manifest, Source,
};
use super::token;
use super::{TrainConfig, ADS, BANK, CARD, CUES, MANIFEST, MODEL};
use crate::dataset::built_at;
use crate::text::hash::{fnv1a, hex64};

/// One labelled sentence with the fold it belongs to, its text and its tokens.
struct Sample {
    fold: usize,
    label: bool,
    text: String,
    words: Vec<String>,
}

/// Fit every axis from three files. Pure: nothing is written.
pub fn fit(
    bank_path: &str,
    ads_path: &str,
    cues_path: &str,
    cfg: &TrainConfig,
) -> io::Result<Artifact> {
    let bank = bank::read_bank(bank_path)?;
    let ads = bank::read_ads(ads_path)?;
    let cues = bank::read_cues(cues_path)?;
    refuse(&bank.bad, BANK, "row")?;
    refuse(&ads.bad, ADS, "row")?;
    refuse(&cues.bad, CUES, "cue table")?;
    let cues_digest = cues.digest();
    let cues_file = cues
        .rows
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "the cue table is empty"))?;

    let faults = bank::validate(&bank.rows, cfg.min_rows);
    if !faults.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            faults.join("; "),
        ));
    }
    let faults = bank::validate_ads(&ads.rows, cfg.min_rows);
    if !faults.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            faults.join("; "),
        ));
    }

    let mut axes: BTreeMap<String, Axis> = BTreeMap::new();
    for (name, (positive, negative)) in banks(&bank.rows) {
        // An axis with no declared features is read on its lexicon density
        // alone: the density is the one reading that always exists, because it
        // is fitted from the bank the axis came from.
        let spec = cues_file
            .features
            .get(&name)
            .cloned()
            .unwrap_or_else(|| FeatureSpec {
                names: vec!["register".to_string()],
                patterns: BTreeMap::new(),
            });
        axes.insert(
            name.clone(),
            Axis::Lexicon(fit_lexicon(&positive, &negative, &spec, cfg)?),
        );
    }
    for (name, spec) in &cues_file.axes {
        axes.insert(name.clone(), Axis::Index(fit_index(spec, &ads.rows, cfg)?));
    }
    for (name, axis) in &axes {
        if let Axis::Lexicon(lex) = axis {
            if lex.metrics.accuracy <= lex.metrics.majority || lex.metrics.balanced <= 0.5 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "axis {name} scores {:.3} accuracy against a {:.3} majority and {:.3} balanced, so it is not a detector",
                        lex.metrics.accuracy, lex.metrics.majority, lex.metrics.balanced
                    ),
                ));
            }
            if let Some(shuffled) = lex.metrics.shuffled {
                if shuffled >= lex.metrics.balanced {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "axis {name} scores {:.3} on shuffled labels against {:.3} on real ones",
                            shuffled, lex.metrics.balanced
                        ),
                    ));
                }
            }
        }
    }

    Ok(Artifact {
        name: cfg.name.clone(),
        version: 1,
        built_at: built_at(),
        source: Source {
            bank: file_name(bank_path),
            bank_rows: bank.rows.len(),
            bank_fnv64: bank.digest(),
            ads: file_name(ads_path),
            ads_rows: ads.rows.len(),
            ads_fnv64: ads.digest(),
            cues: file_name(cues_path),
            cues_axes: cues_file.axes.len(),
            cues_fnv64: cues_digest,
        },
        config: cfg.clone(),
        axes,
    })
}

/// Fit every axis and write `model.json`, `card.md` and `manifest.json`.
pub fn build(
    bank_path: &str,
    ads_path: &str,
    cues_path: &str,
    out: &str,
    cfg: &TrainConfig,
) -> io::Result<Artifact> {
    let artifact = fit(bank_path, ads_path, cues_path, cfg)?;
    fs::create_dir_all(out)?;
    // A shipped folder carries the inputs it was fitted from, so `check` can
    // rebuild it without knowing where the data lived on the machine that built
    // it. A build in place, which is how this repository ships, copies nothing.
    for path in [bank_path, ads_path, cues_path] {
        place(path, out)?;
    }
    artifact.save(&format!("{out}/{MODEL}"))?;
    // The manifest stamps the model, never itself. The card is a description of
    // the folder rather than data it carries, so it has no digest: a reader who
    // edits a paragraph of prose has edited prose, not the fit.
    let manifest = manifest_of(&artifact, out)?;
    fs::write(format!("{out}/{CARD}"), card::render(&artifact, &manifest))?;
    write_json(&format!("{out}/{MANIFEST}"), &manifest)?;
    Ok(artifact)
}

/// Fit one lexicon axis: the lexicon, the logistic over its features, the out
/// of fold numbers and the cuts.
fn fit_lexicon(
    positive: &[&BankRow],
    negative: &[&BankRow],
    spec: &FeatureSpec,
    cfg: &TrainConfig,
) -> io::Result<LexiconAxis> {
    let features = Features::compile(spec.clone())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let folds = cfg.folds.max(2);
    let samples: Vec<Sample> = positive
        .iter()
        .map(|row| (row, true))
        .chain(negative.iter().map(|row| (row, false)))
        .map(|(row, label)| Sample {
            fold: (fnv1a(row.text.as_bytes()) % folds as u64) as usize,
            label,
            text: row.text.clone(),
            words: token::tokens(&row.text),
        })
        .collect();

    let (positive_rows, negative_rows) = split(&samples);
    let lexicon = lexicon::fit(&positive_rows, &negative_rows, &cfg.lexicon);

    let labels: Vec<bool> = samples.iter().map(|s| s.label).collect();
    let values: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| features.values(&s.text, &s.words, Some(&lexicon)))
        .collect();
    let model = Logistic::fit(&values, &labels, &cfg.logistic);

    let (metrics, cut) = cross_validate(&samples, &features, cfg);

    // The clear and strong bands are read off the positive half of the fitting
    // bank, floored by the firing cut so the three bands never cross.
    let mut live: Vec<f64> = samples
        .iter()
        .filter(|s| s.label)
        .map(|s| model.predict(&features.values(&s.text, &s.words, Some(&lexicon))))
        .collect();
    live.sort_by(f64::total_cmp);
    let reference = index::quantiles(&live, cfg.bins);
    let mut cuts = vec![cut];
    for point in [0.90_f64, 0.97] {
        let at = ((point * (reference.len() - 1) as f64) as usize).min(reference.len() - 1);
        let next = round4(reference[at].max(*cuts.last().unwrap_or(&cut)));
        cuts.push(next);
    }

    Ok(LexiconAxis {
        tag: String::new(),
        grain: "sentence".to_string(),
        features: spec.clone(),
        lexicon,
        model,
        cuts,
        metrics,
    })
}

fn split(samples: &[Sample]) -> (Vec<Vec<String>>, Vec<Vec<String>>) {
    let positive = samples
        .iter()
        .filter(|s| s.label)
        .map(|s| s.words.clone())
        .collect();
    let negative = samples
        .iter()
        .filter(|s| !s.label)
        .map(|s| s.words.clone())
        .collect();
    (positive, negative)
}

/// Out of fold predictions, the median cut the folds chose, and the control.
fn cross_validate(samples: &[Sample], features: &Features, cfg: &TrainConfig) -> (Metrics, f64) {
    let folds = cfg.folds.max(2);
    let labels: Vec<bool> = samples.iter().map(|s| s.label).collect();
    let shuffled = eval::shuffled(&labels, 17);

    let mut gold = Vec::new();
    let mut predicted = Vec::new();
    let mut control_gold = Vec::new();
    let mut control = Vec::new();
    let mut cuts: Vec<f64> = Vec::new();

    for fold in 0..folds {
        let train: Vec<usize> = (0..samples.len())
            .filter(|i| samples[*i].fold != fold)
            .collect();
        let held: Vec<usize> = (0..samples.len())
            .filter(|i| samples[*i].fold == fold)
            .collect();
        if train.is_empty() || held.is_empty() {
            continue;
        }
        if let Some((model, lexicon, cut)) = fold_fit(samples, &train, &labels, features, cfg) {
            cuts.push(cut);
            for index in &held {
                let values = features.values(
                    &samples[*index].text,
                    &samples[*index].words,
                    Some(&lexicon),
                );
                gold.push(samples[*index].label);
                predicted.push(model.predict(&values) >= cut);
            }
        }
        if let Some((model, lexicon, cut)) = fold_fit(samples, &train, &shuffled, features, cfg) {
            for index in &held {
                let values = features.values(
                    &samples[*index].text,
                    &samples[*index].words,
                    Some(&lexicon),
                );
                control_gold.push(shuffled[*index]);
                control.push(model.predict(&values) >= cut);
            }
        }
    }

    let mut metrics = eval::evaluate(&gold, &predicted);
    metrics.shuffled = Some(eval::evaluate(&control_gold, &control).balanced);

    cuts.sort_by(f64::total_cmp);
    let cut = cuts.get(cuts.len() / 2).copied().unwrap_or(0.5);
    (metrics, round4(cut))
}

/// Fit one fold's lexicon and logistic, and choose its cut on the training rows.
fn fold_fit(
    samples: &[Sample],
    train: &[usize],
    labels: &[bool],
    features: &Features,
    cfg: &TrainConfig,
) -> Option<(Logistic, Lexicon, f64)> {
    let positive: Vec<Vec<String>> = train
        .iter()
        .filter(|i| labels[**i])
        .map(|i| samples[*i].words.clone())
        .collect();
    let negative: Vec<Vec<String>> = train
        .iter()
        .filter(|i| !labels[**i])
        .map(|i| samples[*i].words.clone())
        .collect();
    if positive.is_empty() || negative.is_empty() {
        return None;
    }
    let lexicon = lexicon::fit(&positive, &negative, &cfg.lexicon);
    let values: Vec<Vec<f64>> = train
        .iter()
        .map(|i| features.values(&samples[*i].text, &samples[*i].words, Some(&lexicon)))
        .collect();
    let gold: Vec<bool> = train.iter().map(|i| labels[*i]).collect();
    let model = Logistic::fit(&values, &gold, &cfg.logistic);
    let scores: Vec<f64> = values.iter().map(|row| model.predict(row)).collect();
    Some((model, lexicon, eval::best_cut(&scores, &gold)))
}

/// Fit one declared index axis: reference quantiles, cuts, audit and checks.
fn fit_index(spec: &IndexSpec, ads: &[AdRow], cfg: &TrainConfig) -> io::Result<IndexAxis> {
    let index = Index::compile(spec.clone())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", spec.tag)))?;
    let train: Vec<&AdRow> = ads.iter().filter(|a| a.split == "train").collect();

    let readings: Vec<BTreeMap<String, f64>> = train
        .iter()
        .map(|ad| index.parts(&ad.text, &token::tokens(&ad.text), None))
        .collect();
    let mut reference: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut scale: BTreeMap<String, f64> = BTreeMap::new();
    for name in spec.components.keys() {
        let values: Vec<f64> = readings
            .iter()
            .map(|parts| parts.get(name).copied().unwrap_or(0.0))
            .collect();
        let quantiles = index::quantiles(&values, cfg.bins);
        scale.insert(name.clone(), round4(index::scale_of(&quantiles)));
        reference.insert(name.clone(), quantiles);
    }

    let scores: Vec<f64> = readings
        .iter()
        .map(|parts| index.score(parts, &reference))
        .collect();
    let (cuts, cuts_note) = index::cuts(&scores);

    let mut audit = Audit {
        rows: ads.len(),
        ..Default::default()
    };
    let mut agree = 0usize;
    let mut left_scores = Vec::new();
    let mut right_scores = Vec::new();
    for ad in ads {
        let words = token::tokens(&ad.text);
        let parts = index.parts(&ad.text, &words, None);
        if index.score(&parts, &reference) > 0.0 {
            audit.fired += 1;
        }
        for name in spec.components.keys() {
            if parts.get(name).copied().unwrap_or(0.0) > 0.0 {
                *audit.groups.entry(name.clone()).or_insert(0) += 1;
            }
        }
        if spec.halves.len() == 2 {
            let left = index.half_score(&parts, &spec.halves[0], &reference);
            let right = index.half_score(&parts, &spec.halves[1], &reference);
            if left > 0.0 || right > 0.0 {
                audit.halves_live += 1;
            }
            if left > 0.0 && right > 0.0 {
                audit.halves_both_fired += 1;
            }
            if (left > 0.0) == (right > 0.0) {
                agree += 1;
            }
            left_scores.push(left);
            right_scores.push(right);
        }
    }
    audit.prevalence = round4(audit.fired as f64 / audit.rows.max(1) as f64);
    audit.halves_agree = round4(agree as f64 / audit.rows.max(1) as f64);
    if audit.halves_both_fired >= 5 {
        audit.halves_correlation = pearson(&left_scores, &right_scores);
    }

    let mut checks = Vec::new();
    for check in &spec.checks {
        let words = token::tokens(&check.text);
        let score = index.score(&index.parts(&check.text, &words, None), &reference);
        let passed = (score > 0.0) == check.expected;
        if !passed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} did not behave on a declared example: {:?} scored {score}",
                    spec.tag, check.text
                ),
            ));
        }
        checks.push(CheckRun {
            text: check.text.clone(),
            expected: check.expected,
            score,
            passed,
        });
    }

    Ok(IndexAxis {
        tag: spec.tag.clone(),
        question: spec.question.clone(),
        components: spec.components.clone(),
        cues: spec.cues.clone(),
        gate: spec.gate.clone(),
        min_groups: spec.min_groups.max(1),
        reference,
        scale,
        cuts,
        cuts_note,
        audit,
        checks,
    })
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() < 5 || left.len() != right.len() {
        return None;
    }
    let n = left.len() as f64;
    let mean_left = left.iter().sum::<f64>() / n;
    let mean_right = right.iter().sum::<f64>() / n;
    let covariance: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - mean_left) * (b - mean_right))
        .sum();
    let var_left = left
        .iter()
        .map(|a| (a - mean_left).powi(2))
        .sum::<f64>()
        .sqrt();
    let var_right = right
        .iter()
        .map(|b| (b - mean_right).powi(2))
        .sum::<f64>()
        .sqrt();
    if var_left == 0.0 || var_right == 0.0 {
        return None;
    }
    Some(round4(covariance / (var_left * var_right)))
}

/// Positive rows per fitted axis, with the negative side each one is read
/// against: its own if the bank supplies one, otherwise the shared plain bank.
fn banks(rows: &[BankRow]) -> BTreeMap<String, (Vec<&BankRow>, Vec<&BankRow>)> {
    let mut positive: BTreeMap<String, Vec<&BankRow>> = BTreeMap::new();
    let mut own: BTreeMap<String, Vec<&BankRow>> = BTreeMap::new();
    let mut shared: Vec<&BankRow> = Vec::new();
    for row in rows {
        if row.label == 1 {
            positive.entry(row.axis.clone()).or_default().push(row);
        } else if row.axis == SHARED_NEGATIVE {
            shared.push(row);
        } else {
            own.entry(row.axis.clone()).or_default().push(row);
        }
    }
    positive
        .into_iter()
        .map(|(axis, rows)| {
            let negative = own.remove(&axis).unwrap_or_else(|| shared.clone());
            (axis, (rows, negative))
        })
        .collect()
}

fn refuse(bad: &[String], name: &str, kind: &str) -> io::Result<()> {
    match bad.first() {
        None => Ok(()),
        Some(first) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} holds {} unreadable {kind}s: {first}", bad.len()),
        )),
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

/// Copy an input into the shipped folder unless it is already there.
fn place(path: &str, out: &str) -> io::Result<()> {
    let source = Path::new(path);
    let name = file_name(path);
    let target = Path::new(out).join(&name);
    if source
        .canonicalize()
        .ok()
        .zip(target.canonicalize().ok())
        .is_some_and(|(a, b)| a == b)
    {
        return Ok(());
    }
    fs::copy(source, &target).map(|_| ())
}

/// Stamp the model the build wrote, with its row count and digest.
fn manifest_of(artifact: &Artifact, out: &str) -> io::Result<Manifest> {
    let mut files = Vec::new();
    if let Ok(bytes) = fs::read(format!("{out}/{MODEL}")) {
        files.push(FileStamp {
            name: MODEL.to_string(),
            rows: artifact.axes.len(),
            bytes: bytes.len() as u64,
            fnv64: hex64(fnv1a(&bytes)),
        });
    }
    let mut axes = BTreeMap::new();
    for (name, axis) in &artifact.axes {
        let stamp = match axis {
            Axis::Lexicon(lex) => AxisStamp {
                method: "lexicon".into(),
                tag: lex.tag.clone(),
                rows: lex.metrics.rows,
                balanced: lex.metrics.balanced,
                prevalence: round4(lex.metrics.positives as f64 / lex.metrics.rows.max(1) as f64),
            },
            Axis::Index(ind) => AxisStamp {
                method: "index".into(),
                tag: ind.tag.clone(),
                rows: ind.audit.rows,
                balanced: 0.0,
                prevalence: ind.audit.prevalence,
            },
        };
        axes.insert(name.clone(), stamp);
    }
    Ok(Manifest {
        name: artifact.name.clone(),
        version: artifact.version,
        built_at: artifact.built_at.clone(),
        files,
        config: artifact.config.clone(),
        axes,
    })
}

fn write_json<T: serde::Serialize>(path: &str, value: &T) -> io::Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    fs::write(path, format!("{text}\n"))
}

/// Read a folder back and report every way it disagrees with itself.
///
/// Three questions, in order of how much they are worth: is each file the file
/// the manifest describes, does the fit rebuild from the shipped inputs, and do
/// the declared examples still behave. Answering the second means running the
/// same fit, so a check cannot pass by checking nothing.
pub fn check(dir: &str) -> io::Result<Vec<String>> {
    let mut faults = Vec::new();
    let manifest: Manifest =
        serde_json::from_str(&fs::read_to_string(format!("{dir}/{MANIFEST}"))?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{MANIFEST}: {e}")))?;
    for stamp in &manifest.files {
        match fs::read(format!("{dir}/{}", stamp.name)) {
            Ok(bytes) => {
                if bytes.len() as u64 != stamp.bytes {
                    faults.push(format!(
                        "{} holds {} bytes, manifest {}",
                        stamp.name,
                        bytes.len(),
                        stamp.bytes
                    ));
                }
                if hex64(fnv1a(&bytes)) != stamp.fnv64 {
                    faults.push(format!("{} does not match its digest", stamp.name));
                }
            }
            Err(_) => faults.push(format!(
                "{} is named by the manifest and missing",
                stamp.name
            )),
        }
    }
    let artifact = match Artifact::load(&format!("{dir}/{MODEL}")) {
        Ok(artifact) => artifact,
        Err(e) => {
            faults.push(format!("{MODEL}: {e}"));
            return Ok(faults);
        }
    };
    match fit(
        &format!("{dir}/{BANK}"),
        &format!("{dir}/{ADS}"),
        &format!("{dir}/{CUES}"),
        &manifest.config,
    ) {
        Ok(fresh) => {
            if !super::model::same_fit(&artifact, &fresh) {
                faults.push("a rebuild disagrees with the shipped fit".to_string());
            }
        }
        Err(e) => faults.push(format!("a rebuild failed: {e}")),
    }
    for (name, axis) in &artifact.axes {
        if let Axis::Index(ind) = axis {
            if let Some(failed) = ind.checks.iter().find(|c| !c.passed) {
                faults.push(format!(
                    "{name} failed a declared example: {:?}",
                    failed.text
                ));
            }
        }
    }
    if !Path::new(&format!("{dir}/{CARD}")).is_file() {
        faults.push(format!("{CARD} is missing"));
    }
    Ok(faults)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(label: bool, text: &str, fold: usize) -> Sample {
        Sample {
            fold,
            label,
            text: text.to_string(),
            words: token::tokens(text),
        }
    }

    #[test]
    fn a_fold_fit_needs_both_classes() {
        let samples = vec![sample(true, "shall", 0), sample(true, "shall", 1)];
        let labels: Vec<bool> = samples.iter().map(|s| s.label).collect();
        let features = Features::compile(FeatureSpec {
            names: vec!["register".into()],
            patterns: BTreeMap::new(),
        })
        .unwrap();
        assert!(fold_fit(&samples, &[0], &labels, &features, &TrainConfig::default()).is_none());
    }

    #[test]
    fn pearson_needs_a_distribution_not_two_points() {
        assert!(pearson(&[1.0, 2.0], &[1.0, 2.0]).is_none());
        let left = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(pearson(&left, &left), Some(1.0));
    }

    #[test]
    fn the_shared_negative_bank_serves_every_positive_axis() {
        let rows = vec![
            BankRow {
                axis: "legal".into(),
                label: 1,
                source: "fr".into(),
                text: "shall".into(),
            },
            BankRow {
                axis: SHARED_NEGATIVE.into(),
                label: 0,
                source: "ad".into(),
                text: "join".into(),
            },
        ];
        let grouped = banks(&rows);
        assert_eq!(grouped.len(), 1);
        let (positive, negative) = &grouped["legal"];
        assert_eq!(positive.len(), 1);
        assert_eq!(negative.len(), 1);
    }
}
