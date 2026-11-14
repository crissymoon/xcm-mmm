//! Extract, fit, ship and check the behavioural segmentation.
//!
//! The extraction step reads the ad panel once and writes the entities to
//! `features.jsonl`, which is the file that ships. The fit reads only that file,
//! so a rebuild needs nothing from the machine that built it and the panel
//! itself never has to be in the repository.
//!
//! The fold count is chosen, not assumed: every k in the declared range is
//! fitted and scored, and the winner is the highest silhouette among the
//! partitions that keep every group above the declared share. Two further
//! questions decide whether the winner ships at all: does it beat a partition
//! found in the same columns with their pairing destroyed, and do the groups
//! survive resampling. A fit that cannot answer both is refused rather than
//! written.

use std::fs;
use std::io;
use std::path::Path;

use super::ad::{self, Entity};
use super::card;
use super::feature::{self, Matrix};
use super::kmeans::{self, Fit, Sweep};
use super::model::{self, Artifact, FileStamp, Manifest, SegmentStamp, Source};
use super::stability::{self, Control, Stability};
use super::{SegmentConfig, CARD, FEATURES, MANIFEST, MODEL};
use crate::text::hash::{fnv1a, hex64};

/// Fold the panel into entities and write them where the fit reads them.
pub fn extract(panel_path: &str, out: &str) -> io::Result<Vec<Entity>> {
    let rows = ad::read_rows(panel_path)?;
    let entities = ad::entities(&rows)?;
    if let Some(parent) = Path::new(out).parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    for entity in &entities {
        let line = serde_json::to_string(entity).map_err(io::Error::other)?;
        text.push_str(&line);
        text.push('\n');
    }
    fs::write(out, text)?;
    Ok(entities)
}

/// Read the shipped entities back.
pub fn read_entities(path: &str) -> io::Result<Vec<Entity>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let entity: Entity = serde_json::from_str(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{path}:{}: {e}", i + 1))
        })?;
        out.push(entity);
    }
    if out.is_empty() {
        return Err(bad(&format!("{path} holds no entities")));
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Fit the artifact from the entities file and check it against its own gates.
pub fn fit(features_path: &str, cfg: &SegmentConfig) -> io::Result<Artifact> {
    let entities = read_entities(features_path)?;
    if entities.len() < cfg.min_entities {
        return Err(bad(&format!(
            "{} entities is below the declared minimum of {}",
            entities.len(),
            cfg.min_entities
        )));
    }
    let specs = feature::specs();
    let matrix = Matrix::build(&entities, &specs, cfg.clip)?;
    let rows = &matrix.z;

    let k_high = cfg
        .k_max
        .min(entities.len().saturating_sub(1))
        .max(cfg.k_min);
    if cfg.k_min < 2 || entities.len() <= cfg.k_min {
        return Err(bad("the k range cannot hold a segmentation"));
    }
    let mut sweep = Vec::new();
    let mut fits: Vec<Fit> = Vec::new();
    for k in cfg.k_min..=k_high {
        let fit = kmeans::fit(rows, k, cfg.restarts, cfg.iterations, cfg.seed);
        let sizes = kmeans::sizes(&fit.labels, k);
        let share = sizes.iter().copied().min().unwrap_or(0) as f64 / rows.len() as f64;
        sweep.push(Sweep {
            k,
            silhouette: kmeans::silhouette(rows, &fit.labels, k),
            inertia: fit.inertia,
            sizes,
            min_share: share,
        });
        fits.push(fit);
    }

    let eligible: Vec<usize> = (0..sweep.len())
        .filter(|&i| sweep[i].min_share + 1e-12 >= cfg.min_share)
        .collect();
    if eligible.is_empty() {
        let widest = sweep.iter().map(|s| s.min_share).fold(0.0f64, f64::max);
        return Err(bad(&format!(
            "no k keeps every group above {:.1} percent; the widest smallest group is {:.1} percent",
            cfg.min_share * 100.0,
            widest * 100.0
        )));
    }
    let chosen = eligible
        .iter()
        .copied()
        .max_by(|&a, &b| {
            sweep[a]
                .silhouette
                .partial_cmp(&sweep[b].silhouette)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| sweep[b].k.cmp(&sweep[a].k))
        })
        .expect("a non empty eligible set");
    let best = &sweep[chosen];
    if best.silhouette < cfg.min_silhouette {
        return Err(bad(&format!(
            "k {} scores a silhouette of {:.3}, below the declared floor of {:.3}",
            best.k, best.silhouette, cfg.min_silhouette
        )));
    }
    let k = best.k;
    let fit = fits.swap_remove(chosen);

    let control = stability::control(
        rows,
        k,
        cfg.controls,
        cfg.restarts,
        cfg.iterations,
        cfg.seed,
    );
    if best.silhouette < control.mean + cfg.margin {
        return Err(bad(&format!(
            "silhouette {:.3} is within {:.3} of the shuffled control at {:.3}",
            best.silhouette, cfg.margin, control.mean
        )));
    }
    let boot = stability::bootstrap(
        rows,
        k,
        &fit,
        cfg.bootstraps,
        cfg.restarts,
        cfg.iterations,
        cfg.seed,
    );
    if boot.mean < cfg.min_stability {
        return Err(bad(&format!(
            "the partition survives resampling at {:.3}, below the declared floor of {:.3}",
            boot.mean, cfg.min_stability
        )));
    }

    let segments = model::segments_of(&fit, &specs, &matrix);
    let mut assignments = std::collections::BTreeMap::new();
    for (i, entity) in entities.iter().enumerate() {
        assignments.insert(entity.id.clone(), fit.labels[i]);
    }
    let bytes = fs::read(features_path)?;
    let shipped = Path::new(features_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| FEATURES.to_string());
    let source = Source {
        path: shipped,
        entities: entities.len(),
        ads: entities.iter().map(|e| e.ads).sum(),
        test_ads: entities.iter().map(|e| e.test_ads).sum(),
        first_day: entities
            .iter()
            .map(|e| e.first.clone())
            .min()
            .unwrap_or_default(),
        last_day: entities
            .iter()
            .map(|e| e.last.clone())
            .max()
            .unwrap_or_default(),
        fnv64: hex64(fnv1a(&bytes)),
    };
    let metrics = model::Metrics {
        entities: entities.len(),
        features: specs.len(),
        k,
        inertia: model::round4(fit.inertia),
        silhouette: model::round4(best.silhouette),
        min_share: model::round4(best.min_share),
        sweep: sweep
            .iter()
            .map(|s| Sweep {
                k: s.k,
                silhouette: model::round4(s.silhouette),
                inertia: model::round4(s.inertia),
                sizes: s.sizes.clone(),
                min_share: model::round4(s.min_share),
            })
            .collect(),
        stability: Stability {
            bootstraps: boot.bootstraps,
            mean: model::round4(boot.mean),
            p10: model::round4(boot.p10),
            min: model::round4(boot.min),
        },
        control: Control {
            runs: control.runs,
            mean: model::round4(control.mean),
            max: model::round4(control.max),
        },
        flat: matrix.flat.clone(),
    };
    Ok(Artifact {
        name: cfg.name.clone(),
        version: 1,
        built_at: model::now_iso8601(),
        source,
        config: cfg.clone(),
        features: specs,
        scaling: matrix.scaling.clone(),
        k,
        centroids: fit.centroids.clone(),
        segments,
        assignments,
        metrics,
    })
}

/// Fit and write a folder: the entities, the model, the manifest and the card.
pub fn build(features_path: &str, out: &str, cfg: &SegmentConfig) -> io::Result<Artifact> {
    let artifact = fit(features_path, cfg)?;
    fs::create_dir_all(out)?;
    place(features_path, out)?;
    artifact.save(&format!("{out}/{MODEL}"))?;
    let manifest = manifest_of(&artifact, out)?;
    fs::write(format!("{out}/{CARD}"), card::render(&artifact, &manifest))?;
    write_json(&format!("{out}/{MANIFEST}"), &manifest)?;
    Ok(artifact)
}

/// Read a folder back and report every way it disagrees with itself.
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
                "{} is named in the manifest and missing",
                stamp.name
            )),
        }
    }
    let artifact = Artifact::load(&format!("{dir}/{MODEL}"))?;
    let features_path = format!("{dir}/{FEATURES}");
    let rebuilt = match fit(&features_path, &manifest.config) {
        Ok(artifact) => artifact,
        Err(e) => {
            faults.push(format!("the fit no longer rebuilds: {e}"));
            return Ok(faults);
        }
    };
    let mut comparable = rebuilt;
    comparable.built_at = artifact.built_at.clone();
    if comparable != artifact {
        faults.push("the rebuilt fit differs from the shipped model".to_string());
    }
    if artifact.assignments.len() != artifact.metrics.entities {
        faults.push(format!(
            "{} assignments for {} entities",
            artifact.assignments.len(),
            artifact.metrics.entities
        ));
    }
    let mut counted = vec![0usize; artifact.k];
    for (entity, &segment) in &artifact.assignments {
        if segment >= artifact.k {
            faults.push(format!("entity {entity} sits in segment {segment}"));
        } else {
            counted[segment] += 1;
        }
    }
    for segment in &artifact.segments {
        if counted.get(segment.id) != Some(&segment.size) {
            faults.push(format!(
                "segment {} claims {} members and holds {}",
                segment.id,
                segment.size,
                counted.get(segment.id).copied().unwrap_or(0)
            ));
        }
    }
    if artifact.segments.len() != artifact.k {
        faults.push(format!(
            "{} segment profiles for k = {}",
            artifact.segments.len(),
            artifact.k
        ));
    }
    Ok(faults)
}

fn manifest_of(artifact: &Artifact, out: &str) -> io::Result<Manifest> {
    let mut files = Vec::new();
    for name in [FEATURES, MODEL] {
        if let Ok(bytes) = fs::read(format!("{out}/{name}")) {
            files.push(FileStamp {
                name: name.to_string(),
                rows: if name == FEATURES {
                    artifact.metrics.entities
                } else {
                    artifact.k
                },
                bytes: bytes.len() as u64,
                fnv64: hex64(fnv1a(&bytes)),
            });
        }
    }
    let mut checks = vec![
        format!(
            "silhouette {:.3} beats the shuffled control at {:.3}",
            artifact.metrics.silhouette, artifact.metrics.control.mean
        ),
        format!(
            "resampling keeps the partition at {:.3} mean agreement",
            artifact.metrics.stability.mean
        ),
        format!(
            "every group holds at least {:.1} percent of the entities",
            artifact.metrics.min_share * 100.0
        ),
    ];
    if artifact.metrics.flat.is_empty() {
        checks.push("every feature varies across the panel".to_string());
    } else {
        checks.push(format!(
            "flat features: {}",
            artifact.metrics.flat.join(", ")
        ));
    }
    Ok(Manifest {
        name: artifact.name.clone(),
        version: artifact.version,
        built_at: artifact.built_at.clone(),
        files,
        config: artifact.config.clone(),
        k: artifact.k,
        segments: artifact
            .segments
            .iter()
            .map(|s| SegmentStamp {
                id: s.id,
                label: s.label.clone(),
                size: s.size,
            })
            .collect(),
        checks,
    })
}

fn place(path: &str, out: &str) -> io::Result<()> {
    let source = Path::new(path);
    let name = source
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| FEATURES.to_string());
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

fn write_json<T: serde::Serialize>(path: &str, value: &T) -> io::Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    fs::write(path, format!("{text}\n"))
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}
