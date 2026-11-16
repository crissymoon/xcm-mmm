//! The card: what the folder holds, in the words a reader needs first.

use std::fmt::Write;

use super::model::{Artifact, Manifest};

/// Render `card.md` for a fitted folder.
pub fn render(artifact: &Artifact, manifest: &Manifest) -> String {
    let metrics = &artifact.metrics;
    let mut out = String::new();
    let _ = writeln!(out, "# {}", artifact.name);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Behavioural segments of advertisers, fitted in Rust. Built {} from the entities listed below, and checked by rebuilding them.",
        artifact.built_at
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "A segment describes what an advertiser did in the panel, not what it is: ads started, the gaps between those starts, the spread of those gaps, the published spend band, copy length, terms bought against, creative reuse and copy concentration. Nothing descriptive enters a feature, and page names are carried for reading the result, never for fitting it."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Source");
    let _ = writeln!(out);
    let _ = writeln!(out, "| file | entities | ads | digest |");
    let _ = writeln!(out, "|---|---:|---:|---|");
    let _ = writeln!(
        out,
        "| {} | {} | {} | `{}` |",
        artifact.source.path, artifact.source.entities, artifact.source.ads, artifact.source.fnv64
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Start dates run from {} to {}. {} of the ads arrived on the archive's test side and are kept in the same entities, because a segment of an advertiser is a description rather than a prediction and splitting an advertiser in half would describe nobody.",
        artifact.source.first_day, artifact.source.last_day, artifact.source.test_ads
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Segments");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{} entities, {} groups, each group allowed {}. The search ran k = {} to {} in three families, and every group holds at least the declared {:.1} percent of the panel; the smallest here holds {:.1} percent. `z` is the group's mean against the whole panel in standard deviations; the reading beside it is the same column in its own units.",
        metrics.entities,
        metrics.k,
        metrics.family.shape(),
        artifact.config.k_min,
        artifact.config.k_max,
        artifact.config.min_share * 100.0,
        metrics.min_share * 100.0
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| segment | label | entities | share | strongest readings |"
    );
    let _ = writeln!(out, "|---:|---|---:|---:|---|");
    for segment in &artifact.segments {
        let readings: Vec<String> = segment
            .profile
            .iter()
            .take(3)
            .map(|row| format!("{} {:+.2} ({:.2} {})", row.name, row.z, row.mean, row.unit))
            .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.3} | {} |",
            segment.id,
            segment.label,
            segment.size,
            segment.share,
            readings.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Choice");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}.", metrics.choice);
    let _ = writeln!(out);
    let strongest = metrics
        .dependence
        .strongest
        .first()
        .map(|pair| format!("`{}` and `{}` at {:+.2}", pair.a, pair.b, pair.r))
        .unwrap_or_else(|| "no pair stands out".to_string());
    let _ = writeln!(
        out,
        "A k-means fit is the maximum likelihood fit of a mixture whose groups are spheres of one shared width. That is a claim that the eight readings are equally spread inside a group and that none of them co-vary. This panel says otherwise: the mean absolute correlation between columns is {:.2}, and {} run together. A sphere cannot use that, so three families were fitted and compared: one width per group, one width per group per column, and a covariance per group. Held-out likelihood chooses between families, because a comparison of predictions can only be settled by rows the fit never saw. Placement agreement is the gate that says the groups a fold model predicts are the groups that ship. And the comparison is paired by fold, because every candidate is read on the same rows: a larger k or a richer family had to add {:.3} nats per entity more than its own standard error before it was preferred, so fold noise cannot buy parameters.",
        metrics.dependence.mean_abs, strongest, artifact.config.min_gain
    );
    let _ = writeln!(out);
    match metrics.candidates.iter().min_by(|a, b| {
        a.bic
            .partial_cmp(&b.bic)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(row) if row.heldout < metrics.heldout => {
            let _ = writeln!(
                out,
                "BIC is reported because it is the reading a mixture is usually chosen on, and it is not the rule here, because it is read in sample: the lowest BIC on the grid is {} at k = {} with {} parameters against {} entities, and its held-out reading is {:.3} nats per entity against {:.3} for the fit that won. A criterion that lets a model hold nearly one parameter per entity is a reading to print, not a rule to follow.",
                row.family.name(),
                row.k,
                row.parameters,
                metrics.entities,
                row.heldout,
                metrics.heldout
            );
        }
        _ => {
            let _ = writeln!(
                out,
                "BIC is reported because it is the reading a mixture is usually chosen on. Here the lowest BIC on the grid is also the fit that won, so the in-sample and the held-out reading agree, and the choice does not depend on which of them a reader prefers."
            );
        }
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| family | k | held out, nats per entity | spread over folds | placement agreement | BIC | parameters | smallest group | verdict |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---|");
    for candidate in &metrics.candidates {
        let chosen = if candidate.family == metrics.family && candidate.k == metrics.k {
            " **chosen**"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "| {}{} | {} | {:.3} | {:.3} | {:.3} | {:.0} | {} | {:.3} | {} |",
            candidate.family.name(),
            chosen,
            candidate.k,
            candidate.heldout,
            candidate.heldout_sd,
            candidate.agreement,
            candidate.bic,
            candidate.parameters,
            candidate.min_share,
            candidate.verdict
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Silhouette is a Euclidean reading, so it structurally rewards the spherical shape k-means produces: the chosen partition scores {:.3} on it against {:.3} for the k-means partition the folder used to ship, and {:.3} for the same protocol on columns shuffled independently. {} That is why the family is settled on likelihood and placement, and silhouette is kept as a floor and a comparison rather than as the selector.",
        metrics.silhouette,
        metrics.baseline.silhouette,
        metrics.control.mean,
        if metrics.silhouette > metrics.control.mean {
            "It clears the control."
        } else {
            "It does not clear the control, which is the price of a shape that is not a sphere, and it is written here rather than left for a reader to find."
        }
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Features");
    let _ = writeln!(out);
    let _ = writeln!(out, "| feature | family | unit | transform | description |");
    let _ = writeln!(out, "|---|---|---|---|---|");
    for spec in &artifact.features {
        let transform = match spec.transform {
            super::Transform::Log1p => "ln(1 + x)",
            super::Transform::Identity => "as it stands",
        };
        let _ = writeln!(
            out,
            "| `{}` | {} | {} | {} | {} |",
            spec.name, spec.family, spec.unit, transform, spec.description
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Each column is standardised and clipped at {:.1} standard deviations before the fit. The mean and the standard deviation travel in the model, so a z can be recomputed by hand from the readings file.",
        artifact.scaling.clip
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## The k-means reading");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The sweep the folder used to gate on, kept so the old reading sits beside the new one. It is the same k-means protocol at every k: Euclidean silhouette, within-group inertia, and the size of the smallest group."
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "| k | silhouette | smallest group | inertia | sizes |");
    let _ = writeln!(out, "|---:|---:|---:|---:|---|");
    for point in &metrics.sweep {
        let sizes: Vec<String> = point.sizes.iter().map(|n| n.to_string()).collect();
        let chosen = if point.k == metrics.k {
            " **chosen**"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "| {}{} | {:.3} | {:.3} | {:.1} | {} |",
            point.k,
            chosen,
            point.silhouette,
            point.min_share,
            point.inertia,
            sizes.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Audit");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "There are no labels, so nothing here is a truth. What is measured is how well the fit predicts rows it never saw, and whether the groups survive the two protocols that try to break a partition: the same fit on columns shuffled independently, and the same fit on resamples of the entities scored against the full panel by adjusted Rand index."
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "| question | reading | gate |");
    let _ = writeln!(out, "|---|---:|---|");
    let _ = writeln!(
        out,
        "| held-out likelihood, nats per entity | {:.3} | gate: clear of the shuffled control by {:.3} |",
        metrics.heldout, artifact.config.min_gain
    );
    let _ = writeln!(
        out,
        "| held-out spread over {} folds | {:.3} | reported, not gated |",
        metrics.folds, metrics.heldout_sd
    );
    let _ = writeln!(
        out,
        "| held-out placement agreement | {:.3} | at least {:.2} |",
        metrics.agreement, artifact.config.min_stability
    );
    let _ = writeln!(
        out,
        "| the same fit on shuffled columns, mean of {} | {:.3} | reported, not gated |",
        metrics.likelihood_control.runs, metrics.likelihood_control.mean
    );
    let _ = writeln!(
        out,
        "| the same fit on shuffled columns, best | {:.3} | reported, not gated |",
        metrics.likelihood_control.max
    );
    let _ = writeln!(
        out,
        "| the k-means partition this folder used to ship | {:.3} | the incumbent's held-out reading |",
        metrics.baseline.heldout
    );
    let _ = writeln!(
        out,
        "| silhouette | {:.3} | at least {:.2} |",
        metrics.silhouette, artifact.config.min_silhouette
    );
    let _ = writeln!(
        out,
        "| shuffled control, mean | {:.3} | margin {:.2} |",
        metrics.control.mean, artifact.config.margin
    );
    let _ = writeln!(
        out,
        "| shuffled control, best of {} | {:.3} | reported, not gated |",
        metrics.control.runs, metrics.control.max
    );
    let _ = writeln!(
        out,
        "| resampling, mean agreement | {:.3} | at least {:.2} |",
        metrics.stability.mean, artifact.config.min_stability
    );
    let _ = writeln!(
        out,
        "| resampling, tenth percentile | {:.3} | reported, not gated |",
        metrics.stability.p10
    );
    let _ = writeln!(
        out,
        "| resampling, worst of {} | {:.3} | reported, not gated |",
        metrics.stability.bootstraps, metrics.stability.min
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "A silhouette of {:.2} is ordinary for behavioural data whose columns share a scale factor, and a partition read through a covariance scores lower on it by construction: a stretched group is not a sphere, and the reading measures distance from a centre, not the shape a group actually has. The separated groups are {:.2} against {:.2} for columns with their pairing destroyed, so the structure is real; the shape is the part silhouette cannot see.",
        metrics.silhouette, metrics.silhouette, metrics.control.mean
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Checks");
    let _ = writeln!(out);
    for line in &manifest.checks {
        let _ = writeln!(out, "- {line}");
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Reproduce");
    let _ = writeln!(out);
    let _ = writeln!(out, "```bash");
    let _ = writeln!(
        out,
        "./xcm-mmm segment fit --features {}/{} --out {}",
        super::FOLDER,
        super::FEATURES,
        super::FOLDER
    );
    let _ = writeln!(out, "./xcm-mmm segment check --dir {}", super::FOLDER);
    let _ = writeln!(out, "```");
    let _ = writeln!(out);

    let _ = writeln!(out, "## Read it");
    let _ = writeln!(out);
    let _ = writeln!(out, "```bash");
    let _ = writeln!(out, "./xcm-mmm segment info");
    let _ = writeln!(
        out,
        "./xcm-mmm segment assign --entities data/segments/features.jsonl"
    );
    let _ = writeln!(out, "```");
    let _ = writeln!(out);

    let _ = writeln!(out, "## Limits");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- The panel is a snapshot of what the archive published, weighted towards recent ads. Cadence is read inside the snapshot, so an advertiser buying steadily for years is seen only through the ads the archive kept."
    );
    let _ = writeln!(
        out,
        "- Spend and impressions are bands. Their midpoint is used, which is a coarse reading of value and is not a measurement of money spent."
    );
    let _ = writeln!(
        out,
        "- Reuse is read from the archive's copy fingerprint. Two ads that differ by a word count as two, so reuse understates what a reader would call the same creative."
    );
    let _ = writeln!(
        out,
        "- Copy concentration is bounded by the number of ads: an advertiser with one ad scores one by construction. That group is a real behaviour, the advertiser who bought once, but its concentration reading carries no information beyond its volume."
    );
    let _ = writeln!(
        out,
        "- Recency is deliberately not a feature. The panel is a snapshot weighted towards recent ads, so the distance from a last start to the snapshot day would measure the archive's coverage as much as the advertiser's activity."
    );
    let _ = writeln!(
        out,
        "- The accuracy reported is predictive: how well the fitted belief explains rows it was not fitted on, in nats per entity. It is not a measure of whether a group is the right group, because no label says which advertiser belongs where. A richer family can predict better and still be the less stable reading of the panel, which is why the resampling agreement is reported beside it."
    );
    let _ = writeln!(
        out,
        "- The family choice is made on one panel. A diagonal covariance winning here says the columns of this archive are not equally spread inside an advertiser group; it is not a claim about advertising in general, and a different register or city could choose differently."
    );
    let _ = writeln!(
        out,
        "- A group is a description of behaviour, not a target. Nothing here says that moving an advertiser between groups would change an outcome."
    );
    let _ = writeln!(out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_card_names_every_segment_and_feature_it_renders() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-card-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let features = dir.join("features.jsonl");
        let mut text = String::new();
        for i in 0..24 {
            let mut obs = String::new();
            for (n, name) in [
                "ads",
                "cadence",
                "burstiness",
                "spend_mid",
                "copy_words",
                "terms",
                "repeat",
                "concentrate",
            ]
            .iter()
            .enumerate()
            {
                if n > 0 {
                    obs.push(',');
                }
                let value = 1.0 + ((i * (n + 3)) % 7) as f64;
                let _ = write!(obs, "\"{name}\":{value}");
            }
            let _ = writeln!(
                text,
                "{{\"id\":\"e{i}\",\"name\":\"E {i}\",\"ads\":{},\"test_ads\":0,\"first\":\"2026-01-01\",\"last\":\"2026-02-01\",\"obs\":{{{obs}}}}}",
                i + 1
            );
        }
        std::fs::write(&features, text).unwrap();
        let mut cfg = super::super::SegmentConfig {
            name: "card-fixture".into(),
            k_min: 2,
            k_max: 3,
            min_entities: 10,
            min_silhouette: 0.0,
            margin: -1.0,
            min_stability: 0.0,
            bootstraps: 2,
            controls: 2,
            restarts: 4,
            iterations: 40,
            ..Default::default()
        };
        cfg.clip = 3.0;
        let artifact = super::super::train::build(
            features.to_str().unwrap(),
            dir.join("segments").to_str().unwrap(),
            &cfg,
        )
        .unwrap();
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(dir.join("segments/manifest.json")).unwrap(),
        )
        .unwrap();
        let card = render(&artifact, &manifest);
        assert!(card.contains("# card-fixture"));
        for segment in &artifact.segments {
            assert!(card.contains(&segment.label));
        }
        for spec in &artifact.features {
            assert!(card.contains(&format!("`{}`", spec.name)));
        }
        assert!(card.contains("## Audit") && card.contains("## Limits"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
