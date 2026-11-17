//! Extract, ship and check the behavioural segmentation.
//!
//! The extraction step reads the ad panel once and writes the entities to
//! `features.jsonl`, which is the file that ships. The fit reads only that file,
//! so a rebuild needs nothing from the machine that built it and the panel
//! itself never has to be in the repository.
//!
//! Fitting lives in [`super::pipeline`] and the choice it makes in
//! [`super::select`]; this module is the file surface around them: place the
//! entities, stamp the manifest, and rebuild the folder from its own rows to see
//! whether it still agrees with itself.
//!
//! The manifest carries the choice in words as well as numbers. A reader who
//! opens only `manifest.json` should learn which likelihood the panel was read
//! through and what it gained over the k-means fit the folder used to ship, not
//! only that some digests matched.

use std::fs;
use std::io;
use std::path::Path;

use super::ad::{self, Entity};
use super::card;
use super::model::{Artifact, FileStamp, Manifest, Metrics, SegmentStamp};
use super::pipeline;
use super::stability;
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

/// Fit and write a folder: the entities, the model, the manifest and the card.
pub fn build(features_path: &str, out: &str, cfg: &SegmentConfig) -> io::Result<Artifact> {
    let artifact = pipeline::fit(features_path, cfg)?;
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
    let rebuilt = match pipeline::fit(&features_path, &manifest.config) {
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
    if artifact.components.len() != artifact.k {
        faults.push(format!(
            "{} components for k = {}",
            artifact.components.len(),
            artifact.k
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
    let metrics = &artifact.metrics;
    let silhouette = if metrics.silhouette > metrics.control.mean {
        format!(
            "silhouette {:.3} beats the shuffled control at {:.3}",
            metrics.silhouette, metrics.control.mean
        )
    } else {
        format!(
            "silhouette {:.3} sits under its shuffled control at {:.3}, because a partition read through a covariance is not a sphere; the likelihood gate carries that reading instead",
            metrics.silhouette, metrics.control.mean
        )
    };
    let mut checks = vec![
        format!(
            "read through a {} covariance at k = {}, chosen on {:.3} nats per entity held out",
            metrics.family.name(),
            metrics.k,
            metrics.heldout
        ),
        format!(
            "{:.3} nats per entity better than the k-means partition the folder used to ship at {:.3}",
            metrics.gain, metrics.baseline.heldout
        ),
        format!(
            "held-out likelihood {:.3} stands clear of the same fit on shuffled columns at {:.3}",
            metrics.heldout, metrics.likelihood_control.mean
        ),
        format!(
            "a fold model places its own held-out rows at {:.3} agreement with the shipped partition",
            metrics.agreement
        ),
        silhouette,
        format!(
            "resampling keeps the partition at {:.3} mean agreement",
            metrics.stability.mean
        ),
        group_check(metrics),
        format!(
            "every group holds at least {:.1} percent of the entities",
            metrics.min_share * 100.0
        ),
    ];
    for audit in &metrics.gates {
        checks.push(format!(
            "the {} gate gives the observed value {:+.3} nats over the Gaussian conditional law, and calls the edge right on {:.1} percent of held-out rows against {:.1} percent for the panel rate",
            audit.name,
            audit.gain(),
            audit.accuracy * 100.0,
            audit.rate * 100.0
        ));
    }
    let sharp = &metrics.sharpness;
    if sharp.rows > 0 {
        checks.push(format!(
            "the mean winning posterior is {:.4} and falls to {:.4} once no component may use a flattened axis, with {} of {} rows changing group",
            sharp.mean_posterior, sharp.cushioned_posterior, sharp.cushioned_moved, sharp.rows
        ));
        checks.push(format!(
            "a model fitted without each row's fold keeps {:.1} percent of them in the group this card gives",
            sharp.held_out_kept * 100.0
        ));
    }
    if metrics.flat.is_empty() {
        checks.push("every feature varies across the panel".to_string());
    } else {
        checks.push(format!("flat features: {}", metrics.flat.join(", ")));
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

/// One manifest line for the per-group reading.
///
/// A partition-level agreement can pass while a small group of it does not come
/// back, so the checks name the groups that hold and the groups that do not.
fn group_check(metrics: &Metrics) -> String {
    let floor = stability::CLUSTER_FLOOR;
    let total = metrics.stability.cluster.len();
    if total == 0 {
        return "the partition was not read group by group".to_string();
    }
    let flagged: Vec<String> = metrics
        .stability
        .cluster
        .iter()
        .filter(|row| row.flagged)
        .map(|row| format!("group {} at {:.3}", row.cluster, row.mean))
        .collect();
    let held = total - flagged.len();
    if flagged.is_empty() {
        format!(
            "every one of the {total} groups clears the {floor:.1} Jaccard floor under resampling"
        )
    } else {
        format!(
            "{held} of {total} groups clear the {floor:.1} Jaccard floor under resampling; {} below it, each read as a region rather than a membership",
            flagged.join(", ")
        )
    }
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}
