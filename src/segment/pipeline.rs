//! Fit one panel, run every gate, and assemble the artifact.
//!
//! The steps are in the order a reader would ask them. Choose the belief and
//! read its reasoning from [`super::select`]. Then three gates that a likelihood
//! alone cannot pass: the partition must be separated in the Euclidean view, it
//! must beat the same protocol on columns shuffled independently, and it must
//! survive resampling of the entities. Anything that fails is refused rather
//! than written, and the refusal says which gate and which reading.
//!
//! The k-means readings are kept beside the likelihood on purpose. A reader who
//! has the old card in front of them should be able to see what changed without
//! running the old code: same panel, same sweep, a different belief.

use std::fs;
use std::io;
use std::path::Path;

use super::feature::{self, Matrix};
use super::gate;
use super::kmeans::{self, Fit};
use super::model::{self, Artifact, Metrics, Source};
use super::stability::{self, Control};
use super::{select, SegmentConfig, FEATURES};
use crate::text::hash::{fnv1a, hex64};

/// Fit the artifact from the entities file and check it against its own gates.
pub fn fit(features_path: &str, cfg: &SegmentConfig) -> io::Result<Artifact> {
    let entities = super::train::read_entities(features_path)?;
    if entities.len() < cfg.min_entities {
        return Err(bad(&format!(
            "{} entities is below the declared minimum of {}",
            entities.len(),
            cfg.min_entities
        )));
    }
    let specs = feature::specs();
    let matrix = Matrix::build(&entities, &specs, cfg.clip, cfg.gate_share)?;
    let rows = &matrix.z;
    let choice = select::choose(rows, &matrix, cfg)?;

    let chosen = choice
        .candidates
        .iter()
        .find(|candidate| candidate.family == choice.family && candidate.k == choice.k)
        .ok_or_else(|| bad("the chosen candidate is missing from its own table"))?;
    if chosen.silhouette < cfg.min_silhouette {
        return Err(bad(&format!(
            "k {} in a {} covariance scores a silhouette of {:.3}, below the declared floor of {:.3}",
            chosen.k,
            chosen.family.name(),
            chosen.silhouette,
            cfg.min_silhouette
        )));
    }

    let folds = select::folds_for(entities.len(), cfg);
    let control = select::shuffle_control(rows, choice.k, choice.family, &folds, cfg);
    if choice.heldout < control.mean + cfg.min_gain {
        return Err(bad(&format!(
            "held-out likelihood {:.3} is within {:.3} nats per entity of the same fit on shuffled columns at {:.3}",
            choice.heldout, cfg.min_gain, control.mean
        )));
    }

    let mixture = choice.mixture.clone();
    if mixture.components.is_empty() {
        return Err(bad("the chosen family could not be fitted to the panel"));
    }
    let labels = choice.labels.clone();
    let column_names: Vec<String> = specs.iter().map(|spec| spec.name.clone()).collect();
    let people: Vec<(String, String)> = entities
        .iter()
        .map(|entity| (entity.id.clone(), entity.name.clone()))
        .collect();
    let sharpness =
        super::sharp::measure(rows, &mixture, &labels, &column_names, &people, &folds, cfg)
            .rounded();
    let boot = stability::mixture_bootstrap(
        rows,
        &labels,
        &stability::Plan {
            k: choice.k,
            family: choice.family,
            runs: cfg.bootstraps,
            restarts: cfg.em_restarts.clamp(1, 2),
            iterations: cfg.iterations,
            seed: cfg.seed,
        },
    );
    if boot.mean < cfg.min_stability {
        return Err(bad(&format!(
            "the partition survives resampling at {:.3}, below the declared floor of {:.3}",
            boot.mean, cfg.min_stability
        )));
    }

    let euclidean = stability::control(
        rows,
        choice.k,
        cfg.controls,
        cfg.restarts,
        cfg.iterations,
        cfg.seed,
    );
    let inertia: f64 = labels
        .iter()
        .enumerate()
        .map(|(i, &label)| kmeans::dist2(&rows[i], &mixture.components[label].mean))
        .sum();
    let partition = Fit {
        k: choice.k,
        centroids: mixture.means(),
        labels: labels.clone(),
        inertia,
    };
    let mut segments = model::segments_of(&partition, &specs, &matrix);
    // The gate is fitted on the chosen partition rather than inside it: see the
    // module note on why a likelihood with an atom and one without are not
    // comparable, and what `audit` reads instead.
    let columns: Vec<usize> = (0..specs.len())
        .filter(|c| !matrix.gates.iter().any(|gate| gate.column == *c))
        .collect();
    let origin: Vec<f64> = columns
        .iter()
        .map(|c| rows.iter().map(|row| row[*c]).sum::<f64>() / rows.len() as f64)
        .collect();
    let resp: Vec<Vec<f64>> = rows.iter().map(|row| mixture.posterior(row)).collect();
    let gates = gate::bars(rows, &resp, &matrix.gates, &columns, &origin);
    let shares = gate::segment_shares(
        rows,
        &labels,
        choice.k,
        &matrix.gates,
        &specs,
        &matrix.scaling,
    );
    for (id, segment) in segments.iter_mut().enumerate() {
        let Some(list) = shares.get(id) else { continue };
        for (name, edge, inside) in list {
            if let Some(row) = segment.profile.iter_mut().find(|row| &row.name == name) {
                row.atom = Some(model::round4(*edge));
                row.inside = Some(model::round4(*inside));
            }
        }
    }
    let ads: Vec<f64> = entities.iter().map(|e| e.ads as f64).collect();
    let audited = gate::audit(
        rows,
        &ads,
        &mixture.components,
        &matrix.gates,
        &matrix.scaling,
        cfg,
    );
    let mut assignments = std::collections::BTreeMap::new();
    for (i, entity) in entities.iter().enumerate() {
        assignments.insert(entity.id.clone(), labels[i]);
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
    let metrics = Metrics {
        entities: entities.len(),
        features: specs.len(),
        k: choice.k,
        family: choice.family,
        heldout: model::round4(choice.heldout),
        heldout_sd: model::round4(choice.heldout_sd),
        agreement: model::round4(choice.agreement),
        folds: choice.folds,
        parameters: chosen.parameters,
        gain: model::round4(choice.heldout - choice.baseline.heldout),
        choice: choice.reason.clone(),
        candidates: choice
            .candidates
            .iter()
            .map(|candidate| super::Candidate {
                heldout: model::round4(candidate.heldout),
                heldout_sd: model::round4(candidate.heldout_sd),
                agreement: model::round4(candidate.agreement),
                agreement_sd: model::round4(candidate.agreement_sd),
                bic: model::round4(candidate.bic),
                icl: model::round4(candidate.icl),
                silhouette: model::round4(candidate.silhouette),
                min_share: model::round4(candidate.min_share),
                ..candidate.clone()
            })
            .collect(),
        baseline: super::Baseline {
            heldout: model::round4(choice.baseline.heldout),
            silhouette: model::round4(choice.baseline.silhouette),
            ..choice.baseline.clone()
        },
        likelihood_control: super::select::Shuffle {
            mean: model::round4(control.mean),
            max: model::round4(control.max),
            ..control
        },
        dependence: choice.dependence.clone(),
        inertia: model::round4(inertia),
        silhouette: model::round4(chosen.silhouette),
        min_share: model::round4(chosen.min_share),
        sweep: choice
            .sweep
            .iter()
            .map(|point| super::Sweep {
                silhouette: model::round4(point.silhouette),
                inertia: model::round4(point.inertia),
                min_share: model::round4(point.min_share),
                ..point.clone()
            })
            .collect(),
        stability: stability::Stability {
            bootstraps: boot.bootstraps,
            mean: model::round4(boot.mean),
            p10: model::round4(boot.p10),
            min: model::round4(boot.min),
            cluster: boot
                .cluster
                .iter()
                .map(|row| stability::ClusterStability {
                    mean: model::round4(row.mean),
                    min: model::round4(row.min),
                    ..row.clone()
                })
                .collect(),
        },
        control: Control {
            runs: euclidean.runs,
            mean: model::round4(euclidean.mean),
            max: model::round4(euclidean.max),
        },
        flat: matrix.flat.clone(),
        gates: audited.clone(),
        sharpness: sharpness.clone(),
    };
    Ok(Artifact {
        name: cfg.name.clone(),
        version: 2,
        built_at: model::now_iso8601(),
        source,
        config: cfg.clone(),
        features: specs,
        scaling: matrix.scaling.clone(),
        k: choice.k,
        family: choice.family,
        components: mixture.components.clone(),
        gates,
        centroids: mixture.means(),
        segments,
        assignments,
        metrics,
    })
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}
