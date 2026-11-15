//! Fitting the learned copy map and auditing it.
//!
//! The map is a multinomial logistic fit, trained here rather than imported,
//! because the labels already live in this repository: an ad's copy against the
//! segment its advertiser was assigned. Two units are fitted from the same
//! panel, one row per ad and one row per advertiser.
//!
//! The audit is grouped. An advertiser's ads share a hook and a vocabulary, so
//! an ad-level split would let the map memorise half a brand and be credited
//! with knowing the other half. Every fold holds out whole page ids instead,
//! and the folded numbers are the ones quoted everywhere.
//!
//! Nothing here ships ad copy. `text_features.jsonl` carries the readings and
//! the vocabulary buckets, which is what a rebuild needs and all that leaves
//! the machine.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::detect::token;
use crate::digest;
use crate::emotion::EmotionModel;
use crate::rng::Rng;
use crate::segment::{ad, Artifact};
use crate::text::hash;

use super::language;
use super::textmap::{
    brand, columns, features, ClassRow, Source, SplitReading, TextConfig, TextMap, Unit, TEXTMAP,
    TEXT_FEATURES, UNITS,
};

/// One training row: what a text or a set reads as, and the segment it should
/// be placed in.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureRow {
    pub page_id: String,
    pub segment: usize,
    #[serde(default)]
    pub split: String,
    pub x: Vec<f64>,
}

/// What a fold achieved.
#[derive(Clone, Debug, PartialEq)]
pub struct Eval {
    pub accuracy: f64,
    pub balanced: f64,
    pub confusion: Vec<Vec<usize>>,
}

impl Eval {
    fn recall(&self, class: usize) -> f64 {
        let row = &self.confusion[class];
        let total: usize = row.iter().sum();
        if total == 0 {
            0.0
        } else {
            row[class] as f64 / total as f64
        }
    }
}

/// The weights and the standardisation they were fitted against.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
}

/// Read the panel and turn every labelled ad into a training row.
///
/// The emotional reading comes from the shipped emotion model, the same scan a
/// report makes, so a training row and a live reading measure the same thing.
pub fn rows(panel: &str, model: &str, emotion: &str) -> io::Result<Vec<FeatureRow>> {
    let artifact = Artifact::load(model)?;
    let scanner = EmotionModel::load(emotion)?.scanner();
    let ads = ad::read_rows(panel)?;
    let mut out = Vec::new();
    let mut unlabelled = 0usize;
    let mut blank = 0usize;
    for row in &ads {
        let segment = match artifact.assignments.get(&row.page_id) {
            Some(segment) => *segment,
            None => {
                unlabelled += 1;
                continue;
            }
        };
        if row.text.trim().is_empty() {
            blank += 1;
            continue;
        }
        let words = token::tokens(&row.text);
        let sentences = row.sentences.max(1);
        let emotional = scanner.read(&row.text).hits.len();
        let view = language::read(&row.text, &words, sentences, emotional);
        out.push(FeatureRow {
            page_id: row.page_id.clone(),
            segment,
            split: row.split.clone(),
            // Rounded to the precision the map is quoted at, and rounded
            // before the fit, so a rebuild from the shipped rows reproduces
            // the shipped weights rather than approximating them.
            x: features(&view, &words).into_iter().map(round4).collect(),
        });
    }
    if out.is_empty() {
        return Err(bad(&format!(
            "{panel} holds no labelled ad copy: {unlabelled} unlabelled rows, {blank} blank"
        )));
    }
    Ok(out)
}

/// One advertiser's copy set: the mean of its ads and the size of the set.
pub fn brand_rows(rows: &[FeatureRow]) -> Vec<FeatureRow> {
    let mut grouped: BTreeMap<&str, Vec<&FeatureRow>> = BTreeMap::new();
    for row in rows {
        grouped.entry(row.page_id.as_str()).or_default().push(row);
    }
    grouped
        .into_iter()
        .map(|(page_id, ads)| {
            let stacked: Vec<Vec<f64>> = ads.iter().map(|row| row.x.clone()).collect();
            FeatureRow {
                page_id: page_id.to_string(),
                segment: ads[0].segment,
                split: ads[0].split.clone(),
                x: brand(&stacked, ads.len()),
            }
        })
        .collect()
}

/// Fit one unit and measure it on walks through advertisers it never saw.
fn fit_unit(
    label: &str,
    rows: &[FeatureRow],
    specs: &[ClassRow],
    artifact: &Artifact,
    cfg: &TextConfig,
    control: f64,
) -> io::Result<Unit> {
    let k = artifact.segments.len();
    if k < 2 {
        return Err(bad("the segment model holds fewer than two groups"));
    }
    if rows.len() < cfg.min_rows {
        return Err(bad(&format!(
            "{} rows for the {label} unit: the floor is {}",
            rows.len(),
            cfg.min_rows
        )));
    }
    for row in rows {
        if row.segment >= k {
            return Err(bad(&format!(
                "{} is assigned to segment {}, but the model holds {k}",
                row.page_id, row.segment
            )));
        }
        if row.x.len() != columns(label).len() {
            return Err(bad(&format!(
                "{} carries {} readings for the {label} unit, which needs {}",
                row.page_id,
                row.x.len(),
                columns(label).len()
            )));
        }
    }

    let ids: Vec<&str> = rows.iter().map(|row| row.page_id.as_str()).collect();
    let labels: Vec<usize> = rows.iter().map(|row| row.segment).collect();
    let matrix: Vec<Vec<f64>> = rows.iter().map(|row| row.x.clone()).collect();
    let eval = cross_validate(&matrix, &labels, &ids, k, cfg);

    let counts = count_by_class(&labels, k);
    let total = labels.len().max(1) as f64;
    let majority = counts.iter().copied().max().unwrap_or(0) as f64 / total;
    let chance = 1.0 / k as f64;
    let lift = if chance >= 1.0 {
        0.0
    } else {
        (eval.balanced - chance) / (1.0 - chance)
    };
    let usable = rows.len() >= cfg.min_rows
        && eval.balanced >= chance + cfg.margin
        && lift >= cfg.min_lift
        && eval.accuracy > majority
        && eval.balanced > control + cfg.margin;
    let split = if label == "ad" {
        split_reading(&matrix, &labels, rows, k, cfg)
    } else {
        None
    };
    let fitted = train(&matrix, &labels, k, cfg);

    let classes = specs
        .iter()
        .map(|spec| ClassRow {
            id: spec.id,
            label: spec.label.clone(),
            share: spec.share,
            train: counts[spec.id],
            recall: round4(eval.recall(spec.id)),
        })
        .collect();

    Ok(Unit {
        unit: label.to_string(),
        rows: rows.len(),
        classes,
        accuracy: round4(eval.accuracy),
        balanced: round4(eval.balanced),
        chance: round4(chance),
        majority: round4(majority),
        lift: round4(lift),
        control: round4(control),
        usable,
        note: note_of(
            label,
            rows.len(),
            &eval,
            majority,
            chance,
            control,
            cfg,
            usable,
        ),
        confusion: eval.confusion.clone(),
        weights: fitted.weights,
        bias: fitted.bias,
        mean: fitted.mean,
        sd: fitted.sd,
        split,
    })
}

/// What one unit's measurement means, in a sentence built from the numbers.
#[allow(clippy::too_many_arguments)]
fn note_of(
    label: &str,
    rows: usize,
    eval: &Eval,
    majority: f64,
    chance: f64,
    control: f64,
    cfg: &TextConfig,
    usable: bool,
) -> String {
    let article = if label == "ad" { "an" } else { "a" };
    let measured = format!(
        "grouped folds recover {:.0}% balanced and {:.0}% overall, against {:.0}% chance, {:.0}% for the largest class and {:.0}% on shuffled labels",
        eval.balanced * 100.0,
        eval.accuracy * 100.0,
        chance * 100.0,
        majority * 100.0,
        control * 100.0
    );
    if usable {
        let lift = (eval.balanced - chance) / (1.0 - chance);
        return format!(
            "{article} {label} map over {rows} rows clears its gate: {measured}; lift over chance {:.0}% of the distance to perfect",
            lift * 100.0
        );
    }
    let why = if eval.balanced <= chance + cfg.margin {
        "balanced accuracy sits within two points of chance".to_string()
    } else if eval.accuracy <= majority {
        "the largest class alone scores higher".to_string()
    } else if eval.balanced <= control + cfg.margin {
        "shuffled labels score as well".to_string()
    } else {
        format!(
            "lift over chance is under the {:.0}% floor",
            cfg.min_lift * 100.0
        )
    };
    format!(
        "{article} {label} map over {rows} rows is withheld: {measured}, so {why}; it is not used to place this text"
    )
}

fn count_by_class(labels: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &label in labels {
        if label < k {
            counts[label] += 1;
        }
    }
    counts
}

/// A fold holds out every advertiser whose id hashes to it, so no brand is
/// half in training and half in scoring.
fn fold_of(page_id: &str, folds: usize) -> usize {
    (hash::word_hash(page_id) % folds as u64) as usize
}

fn cross_validate(
    matrix: &[Vec<f64>],
    labels: &[usize],
    ids: &[&str],
    k: usize,
    cfg: &TextConfig,
) -> Eval {
    let folds = cfg.folds.max(2);
    let mut pred = vec![usize::MAX; matrix.len()];
    for fold in 0..folds {
        let held: Vec<usize> = (0..matrix.len())
            .filter(|&i| fold_of(ids[i], folds) == fold)
            .collect();
        if held.is_empty() {
            continue;
        }
        let kept: Vec<usize> = (0..matrix.len())
            .filter(|&i| fold_of(ids[i], folds) != fold)
            .collect();
        if kept.len() < 2 {
            continue;
        }
        let train_x: Vec<Vec<f64>> = kept.iter().map(|&i| matrix[i].clone()).collect();
        let train_y: Vec<usize> = kept.iter().map(|&i| labels[i]).collect();
        let fitted = train(&train_x, &train_y, k, cfg);
        for &i in &held {
            pred[i] = argmax(&logits(&fitted, &matrix[i]));
        }
    }
    score(&pred, labels, k)
}

fn split_reading(
    matrix: &[Vec<f64>],
    labels: &[usize],
    rows: &[FeatureRow],
    k: usize,
    cfg: &TextConfig,
) -> Option<SplitReading> {
    let known: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].split != "test")
        .collect();
    let test: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].split == "test")
        .collect();
    if known.len() < cfg.min_rows || test.is_empty() {
        return None;
    }
    let train_x: Vec<Vec<f64>> = known.iter().map(|&i| matrix[i].clone()).collect();
    let train_y: Vec<usize> = known.iter().map(|&i| labels[i]).collect();
    let fitted = train(&train_x, &train_y, k, cfg);
    let right = test
        .iter()
        .filter(|&&i| argmax(&logits(&fitted, &matrix[i])) == labels[i])
        .count();
    Some(SplitReading {
        train: known.len(),
        test: test.len(),
        accuracy: round4(right as f64 / test.len() as f64),
    })
}

fn score(pred: &[usize], labels: &[usize], k: usize) -> Eval {
    let mut confusion = vec![vec![0usize; k]; k];
    let mut right = 0usize;
    let mut total = 0usize;
    for (guessed, &gold) in pred.iter().zip(labels) {
        if *guessed == usize::MAX {
            continue;
        }
        confusion[gold][*guessed] += 1;
        total += 1;
        if *guessed == gold {
            right += 1;
        }
    }
    let eval = Eval {
        accuracy: if total == 0 {
            0.0
        } else {
            right as f64 / total as f64
        },
        balanced: 0.0,
        confusion,
    };
    let recalls: Vec<f64> = (0..k)
        .filter(|&class| eval.confusion[class].iter().sum::<usize>() > 0)
        .map(|class| eval.recall(class))
        .collect();
    let balanced = if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    };
    Eval { balanced, ..eval }
}

/// Full-batch softmax regression with square-root class weights.
///
/// The weights matter: the largest segment holds a third of the panel, and an
/// unweighted fit would buy overall accuracy by never predicting the other
/// three. Square roots balance the classes without handing a segment of
/// thirty-seven advertisers the same vote as one of a hundred and fifteen.
fn train(matrix: &[Vec<f64>], labels: &[usize], k: usize, cfg: &TextConfig) -> Fitted {
    let (mean, sd) = standardise(matrix);
    let z: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, value)| (value - mean[column]) / sd[column])
                .collect()
        })
        .collect();
    let dim = mean.len();
    let counts = count_by_class(labels, k);
    let total = labels.len().max(1) as f64;
    let weight_of: Vec<f64> = labels
        .iter()
        .map(|&label| {
            let count = counts.get(label).copied().unwrap_or(0).max(1) as f64;
            (total / (k as f64 * count)).sqrt()
        })
        .collect();
    let weight_sum: f64 = weight_of.iter().sum::<f64>().max(1e-9);

    let mut weights = vec![vec![0.0; dim]; k];
    let mut bias = vec![0.0; k];
    for _ in 0..cfg.iterations {
        let mut grad_w = vec![vec![0.0; dim]; k];
        let mut grad_b = vec![0.0; k];
        for (row, &label) in z.iter().zip(labels) {
            let mut scores = vec![0.0; k];
            for class in 0..k {
                scores[class] = bias[class]
                    + weights[class]
                        .iter()
                        .zip(row)
                        .map(|(w, v)| w * v)
                        .sum::<f64>();
            }
            let peak = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut mass: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let sum: f64 = mass.iter().sum();
            for value in &mut mass {
                *value /= sum;
            }
            let error = weight_of[label] / weight_sum;
            for class in 0..k {
                let target = if class == label { 1.0 } else { 0.0 };
                let delta = (mass[class] - target) * error;
                grad_b[class] += delta;
                for column in 0..dim {
                    grad_w[class][column] += delta * row[column];
                }
            }
        }
        for class in 0..k {
            bias[class] -= cfg.learning_rate * grad_b[class];
            for column in 0..dim {
                let penalty = cfg.l2 * weights[class][column];
                weights[class][column] -= cfg.learning_rate * (grad_w[class][column] + penalty);
            }
        }
    }
    Fitted {
        weights,
        bias,
        mean,
        sd,
    }
}

fn standardise(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let dim = matrix.first().map(|row| row.len()).unwrap_or(0);
    let n = matrix.len().max(1) as f64;
    let mut mean = vec![0.0; dim];
    for row in matrix {
        for (column, value) in row.iter().enumerate() {
            mean[column] += value;
        }
    }
    for value in &mut mean {
        *value /= n;
    }
    let mut sd = vec![0.0; dim];
    for row in matrix {
        for (column, value) in row.iter().enumerate() {
            sd[column] += (value - mean[column]).powi(2);
        }
    }
    // A column that never varies would divide by zero; it carries no signal,
    // so the floor leaves it at a constant zero rather than a NaN.
    for value in &mut sd {
        *value = (*value / n).sqrt().max(1e-9);
    }
    (mean, sd)
}

fn logits(fitted: &Fitted, row: &[f64]) -> Vec<f64> {
    fitted
        .weights
        .iter()
        .zip(&fitted.bias)
        .map(|(weights, bias)| {
            bias + weights
                .iter()
                .enumerate()
                .map(|(column, weight)| {
                    weight * (row[column] - fitted.mean[column]) / fitted.sd[column]
                })
                .sum::<f64>()
        })
        .collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Shuffled-label controls. If a map scores as well with the labels scrambled
/// as with them intact, it has learned the panel's shape rather than the copy.
fn controls(rows: &[FeatureRow], k: usize, cfg: &TextConfig) -> f64 {
    if cfg.controls == 0 {
        return 0.0;
    }
    let ids: Vec<&str> = rows.iter().map(|row| row.page_id.as_str()).collect();
    let matrix: Vec<Vec<f64>> = rows.iter().map(|row| row.x.clone()).collect();
    let labels: Vec<usize> = rows.iter().map(|row| row.segment).collect();
    let mut rng = Rng::stream(cfg.seed, 0x7e77);
    let mut scores = Vec::with_capacity(cfg.controls);
    for _ in 0..cfg.controls {
        let mut shuffled = labels.clone();
        rng.shuffle(&mut shuffled);
        scores.push(cross_validate(&matrix, &shuffled, &ids, k, cfg).balanced);
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Assemble the two units from rows already in hand.
fn assemble(
    ads: &[FeatureRow],
    artifact: &Artifact,
    cfg: &TextConfig,
    source: Source,
    built_at: String,
) -> io::Result<TextMap> {
    let specs: Vec<ClassRow> = artifact
        .segments
        .iter()
        .map(|segment| ClassRow {
            id: segment.id,
            label: segment.label.clone(),
            share: round4(segment.share),
            train: 0,
            recall: 0.0,
        })
        .collect();
    let k = artifact.segments.len();

    let ad_control = controls(ads, k, cfg);
    let ad = fit_unit("ad", ads, &specs, artifact, cfg, ad_control)?;
    let brands = brand_rows(ads);
    let brand_control = controls(&brands, k, cfg);
    let brand = fit_unit("brand", &brands, &specs, artifact, cfg, brand_control)?;

    let mut units = BTreeMap::new();
    units.insert("ad".to_string(), ad);
    units.insert("brand".to_string(), brand);
    let columns: BTreeMap<String, Vec<String>> = UNITS
        .iter()
        .map(|unit| (unit.to_string(), columns(unit)))
        .collect();
    let in_use: Vec<String> = UNITS
        .iter()
        .filter_map(|unit| units.get(*unit))
        .filter(|unit| unit.usable)
        .map(|unit| unit.unit.clone())
        .collect();
    Ok(TextMap {
        name: cfg.name.clone(),
        version: 1,
        built_at,
        source,
        config: cfg.clone(),
        columns,
        units,
        note: format!(
            "one map, two units: the copy of a single ad and the copy set of an advertiser, both learned from this panel. {} in use; every other unit is fitted and withheld. Folds hold out whole advertisers, and the label is behavioural, so copy recovers only the part of a segment that copy shows",
            if in_use.is_empty() {
                "no unit".to_string()
            } else {
                in_use.join(" and ")
            }
        ),
    })
}

/// Fit both units from the panel and write the folder.
///
/// The derived rows are written beside the map so `segment text-check` can
/// rebuild without the panel and without any ad copy.
pub fn build(
    panel: &str,
    model: &str,
    emotion: &str,
    out: &str,
    cfg: &TextConfig,
) -> io::Result<TextMap> {
    let artifact = Artifact::load(model)?;
    let ads = rows(panel, model, emotion)?;
    let text = jsonl(&ads)?;
    fs::create_dir_all(out)?;
    fs::write(format!("{out}/{TEXT_FEATURES}"), &text)?;
    let source = Source {
        // The file name, as the segment model records its own source: a
        // published artifact says what it was fitted from, not where on this
        // machine that file happened to sit.
        panel: Path::new(panel)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| panel.to_string()),
        model: model.to_string(),
        ads: ads.len(),
        entities: ads
            .iter()
            .map(|row| row.page_id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        fnv64: digest::fnv64(text.as_bytes()),
    };
    let map = assemble(
        &ads,
        &artifact,
        cfg,
        source,
        crate::segment::model::now_iso8601(),
    )?;
    map.save(&format!("{out}/{TEXTMAP}"))?;
    Ok(map)
}

/// Rebuild the shipped map from its derived rows and report every disagreement.
pub fn check(dir: &str) -> io::Result<Vec<String>> {
    let mut faults = Vec::new();
    let map_path = format!("{dir}/{TEXTMAP}");
    let map = match TextMap::load(&map_path) {
        Ok(map) => map,
        Err(e) => {
            faults.push(format!("no usable learned map at {map_path}: {e}"));
            return Ok(faults);
        }
    };
    let features_path = format!("{dir}/{TEXT_FEATURES}");
    let text = match fs::read_to_string(&features_path) {
        Ok(text) => text,
        Err(e) => {
            faults.push(format!("{features_path}: {e}"));
            return Ok(faults);
        }
    };
    if digest::fnv64(text.as_bytes()) != map.source.fnv64 {
        // A file that no longer matches its digest cannot be the one the map
        // was fitted from, so nothing downstream of it means anything.
        faults.push(format!("{TEXT_FEATURES} does not match the map digest"));
        return Ok(faults);
    }
    let ads = match read_rows(&features_path) {
        Ok(rows) => rows,
        Err(e) => {
            faults.push(format!("{features_path}: {e}"));
            return Ok(faults);
        }
    };
    let artifact = Artifact::load(&map.source.model)?;
    let rebuilt = assemble(
        &ads,
        &artifact,
        &map.config,
        map.source.clone(),
        map.built_at.clone(),
    )?;
    if rebuilt != map {
        faults.push("the rebuilt map differs from the shipped map".to_string());
    }
    for (unit, fitted) in &map.units {
        if fitted.rows
            != fitted
                .classes
                .iter()
                .map(|class| class.train)
                .sum::<usize>()
        {
            faults.push(format!(
                "{unit} claims {} rows but its classes hold {}",
                fitted.rows,
                fitted
                    .classes
                    .iter()
                    .map(|class| class.train)
                    .sum::<usize>()
            ));
        }
    }
    Ok(faults)
}

fn read_rows(path: &str) -> io::Result<Vec<FeatureRow>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: FeatureRow = serde_json::from_str(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{path}:{}: {e}", i + 1))
        })?;
        out.push(row);
    }
    if out.is_empty() {
        return Err(bad(&format!("{path} holds no rows")));
    }
    Ok(out)
}

fn jsonl(rows: &[FeatureRow]) -> io::Result<String> {
    let mut out = String::new();
    for row in rows {
        out.push_str(&serde_json::to_string(row).map_err(io::Error::other)?);
        out.push('\n');
    }
    Ok(out)
}

fn round4(value: f64) -> f64 {
    crate::segment::model::round4(value)
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::model::Source as ModelSource;
    use crate::segment::SegmentConfig;
    use crate::segment::{ProfileRow, Segment};

    fn artifact() -> Artifact {
        let segment = |id: usize, label: &str, z: f64| Segment {
            id,
            label: label.to_string(),
            size: 50,
            share: 0.5,
            profile: vec![ProfileRow {
                name: "ads".into(),
                label: "ads".into(),
                family: "volume".into(),
                unit: "ads".into(),
                z,
                mean: 10.0,
            }],
        };
        Artifact {
            name: "t".into(),
            version: 1,
            built_at: String::new(),
            source: ModelSource::default(),
            config: SegmentConfig::default(),
            features: Vec::new(),
            scaling: crate::segment::Scaling {
                mean: Vec::new(),
                sd: Vec::new(),
                clip: 3.0,
            },
            k: 2,
            centroids: Vec::new(),
            segments: vec![segment(0, "left", 1.0), segment(1, "right", -1.0)],
            assignments: BTreeMap::new(),
            metrics: crate::segment::Metrics {
                entities: 0,
                features: 0,
                k: 2,
                inertia: 0.0,
                silhouette: 0.0,
                min_share: 0.0,
                sweep: Vec::new(),
                stability: crate::segment::stability::Stability {
                    bootstraps: 0,
                    mean: 0.0,
                    p10: 0.0,
                    min: 0.0,
                },
                control: crate::segment::stability::Control {
                    runs: 0,
                    mean: 0.0,
                    max: 0.0,
                },
                flat: Vec::new(),
            },
        }
    }

    /// Two classes separated exactly by the first column.
    fn separable(rows: usize) -> Vec<FeatureRow> {
        (0..rows)
            .map(|i| {
                let segment = i % 2;
                let mut x = vec![0.0; columns("ad").len()];
                x[0] = if segment == 0 { -2.0 } else { 2.0 };
                FeatureRow {
                    page_id: format!("page-{i:03}"),
                    segment,
                    split: if i % 5 == 0 {
                        "test".into()
                    } else {
                        "train".into()
                    },
                    x,
                }
            })
            .collect()
    }

    fn cfg() -> TextConfig {
        TextConfig {
            iterations: 200,
            controls: 2,
            min_rows: 10,
            ..TextConfig::default()
        }
    }

    #[test]
    fn a_separable_panel_is_recovered_and_cleared() {
        let rows = separable(80);
        let fitted = fit_unit(
            "ad",
            &rows,
            &artifact()
                .segments
                .iter()
                .map(|s| ClassRow {
                    id: s.id,
                    label: s.label.clone(),
                    share: s.share,
                    train: 0,
                    recall: 0.0,
                })
                .collect::<Vec<_>>(),
            &artifact(),
            &cfg(),
            0.25,
        )
        .expect("fit");
        assert!(fitted.balanced > 0.95, "balanced {}", fitted.balanced);
        assert!(fitted.usable);
        assert_eq!(fitted.confusion.len(), 2);
        assert_eq!(fitted.rows, 80);
    }

    #[test]
    fn random_labels_do_not_clear_the_gate() {
        let mut rows = separable(80);
        // Break the tie between the column and the label.
        for (i, row) in rows.iter_mut().enumerate() {
            row.segment = (i / 4) % 2;
            row.x[0] = ((i * 37) % 5) as f64;
        }
        let fitted = fit_unit(
            "ad",
            &rows,
            &artifact()
                .segments
                .iter()
                .map(|s| ClassRow {
                    id: s.id,
                    label: s.label.clone(),
                    share: s.share,
                    train: 0,
                    recall: 0.0,
                })
                .collect::<Vec<_>>(),
            &artifact(),
            &cfg(),
            0.5,
        )
        .expect("fit");
        assert!(!fitted.usable);
        assert!(fitted.note.contains("withheld"));
    }

    #[test]
    fn folds_hold_whole_advertisers() {
        let rows = separable(40);
        let ids: Vec<&str> = rows.iter().map(|r| r.page_id.as_str()).collect();
        let fold = fold_of(ids[3], 5);
        assert_eq!(fold, fold_of(ids[3], 5));
        assert!(fold < 5);
    }

    #[test]
    fn brand_rows_average_the_ads_of_one_advertiser() {
        let mut rows = separable(4);
        for row in &mut rows {
            row.page_id = "one".into();
        }
        let brands = brand_rows(&rows);
        assert_eq!(brands.len(), 1);
        assert_eq!(brands[0].x.len(), columns("brand").len());
        assert!((brands[0].x[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn a_map_round_trips_through_its_file() {
        let rows = separable(80);
        let map =
            assemble(&rows, &artifact(), &cfg(), Source::default(), String::new()).expect("map");
        let path = std::env::temp_dir().join(format!("textmap-{}.json", std::process::id()));
        map.save(path.to_str().expect("path")).expect("save");
        let back = TextMap::load(path.to_str().expect("path")).expect("load");
        assert_eq!(back, map);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn an_unknown_column_width_is_refused() {
        let mut rows = separable(40);
        rows[0].x.pop();
        let specs: Vec<ClassRow> = artifact()
            .segments
            .iter()
            .map(|s| ClassRow {
                id: s.id,
                label: s.label.clone(),
                share: s.share,
                train: 0,
                recall: 0.0,
            })
            .collect();
        assert!(fit_unit("ad", &rows, &specs, &artifact(), &cfg(), 0.25).is_err());
    }

    #[test]
    fn the_confusion_matrix_counts_every_row_once() {
        let pred = vec![0, 1, 1, 0];
        let labels = vec![0, 1, 0, 0];
        let eval = score(&pred, &labels, 2);
        assert_eq!(eval.confusion[0][0], 2);
        assert_eq!(eval.confusion[0][1], 1);
        assert_eq!(eval.confusion[1][1], 1);
        assert!((eval.accuracy - 0.75).abs() < 1e-9);
    }

    #[test]
    fn a_tampered_row_file_is_a_fault_before_anything_is_refitted() {
        let rows = separable(80);
        let map =
            assemble(&rows, &artifact(), &cfg(), Source::default(), String::new()).expect("map");
        let dir = std::env::temp_dir().join(format!("textmap-check-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        let mut tampered = map.clone();
        tampered.source.fnv64 = "deadbeefdeadbeef".into();
        tampered
            .save(dir.join(TEXTMAP).to_str().expect("path"))
            .expect("save");
        fs::write(dir.join(TEXT_FEATURES), "{}\n").expect("rows");
        let faults = check(dir.to_str().expect("path")).expect("check");
        assert!(
            faults
                .iter()
                .any(|fault| fault.contains("does not match the map digest")),
            "{faults:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_flat_column_standardises_without_a_nan() {
        let (mean, sd) = standardise(&[vec![2.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(mean[0], 2.0);
        assert!(sd[0] > 0.0);
        assert!(sd[1] > 0.0);
    }
}
