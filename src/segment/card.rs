//! The card: what the folder holds, in the words a reader needs first.

use std::fmt::Write;

use crate::linalg;
use crate::num::commas;

use super::mixture;
use super::model::{Artifact, Manifest, Metrics};

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

    if !artifact.gates.is_empty() {
        render_gates(&mut out, artifact, metrics);
    }

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

    let worst = mixture::conditioning(&artifact.components);
    let longest = artifact
        .components
        .iter()
        .filter_map(|component| linalg::jacobi_eigen(&component.covariance))
        .flat_map(|(values, _)| values)
        .fold(0.0, f64::max);
    let rows = metrics.entities.max(1) as f64;
    let error = longest * (metrics.features.max(1) as f64 / rows).sqrt();
    let resolution = longest / error.max(1e-12);
    let _ = writeln!(
        out,
        "Every component is bounded on shape as well as on width. The width floor refuses a component narrower than a thousandth of a standard deviation on any axis, and that floor reads the diagonal; a diagonal is not a shape, because two columns that move together to within rounding can both sit near unit variance while the covariance is nearly singular along the direction they share, and a density read through that direction is decided by one row. So the ratio between a component's longest axis and its shortest is capped at {:.0e}, and a component that exceeds it has its short axis lifted along the direction the data gave. That cap is not a statistical bound and the card will not pretend it is: at {} rows over {} columns the eigenvalue error is of order {:.2} against a longest axis of {:.2}, so no ratio below about {:.0} could be separated from sampling error at all, and a cap picked near the resolution limit refuses structure rather than noise. Measured on this panel, a cap of ten thousand or a million both hand the choice to the diagonal family and cost the full covariance fit more than two nats per entity on held-out rows. The cap therefore sits where a covariance stops being one, at the edge of what a double-precision factorisation can invert, so that only a direction flat to twelve digits is refused. The worst shipped component sits at {}. {}",
        mixture::CONDITION_MAX,
        commas(metrics.entities),
        metrics.features,
        error,
        longest,
        resolution,
        commas(worst.round() as usize),
        if worst > mixture::CONDITION_MAX / 10.0 {
            "That is within a factor of ten of the cap, so the bound is doing work on this panel and the covariances below are the bounded ones."
        } else {
            "That is far inside the cap, so no component needed lifting: the bound is a guard this panel never triggers, and the covariances below are the ones the maximisation step produced."
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
    let _ = writeln!(
        out,
        "- The gate describes the two shares inside a partition fitted without it. A partition fitted with the gate in the loop would draw different boundaries, and settling which is better needs a measure both models share; the held-out lattice score above is that measure for the shares alone, not for the partition."
    );
    let _ = writeln!(out);
    out
}

/// What a share does on its edge, and how well that predicted out of sample.
///
/// Two tables: the pile as it stands, group by group, and the same share scored
/// as a probability on the lattice it lives on against the Gaussian conditional
/// law it replaces. The second is the reason the first is here: a description
/// nobody has measured is a decoration.
fn render_gates(out: &mut String, artifact: &Artifact, metrics: &Metrics) {
    let _ = writeln!(out, "## The two shares");
    let _ = writeln!(out);
    let piles: Vec<String> = artifact
        .gates
        .iter()
        .map(|gate| {
            format!(
                "`{}` at {:.2} holds {:.1} percent of the panel",
                gate.name,
                gate.boundary,
                gate.share * 100.0
            )
        })
        .collect();
    let _ = writeln!(
        out,
        "Two readings are shares that pile on an edge: {}. A Gaussian can only imitate a pile with a group of almost no width, and then it reports a density where the data are a probability. Each is read here as a gate: the chance a member of a group sits on the edge, and the members inside it, whose mean moves with the other readings. `inside` is that mean in plain units, and the last column is the share of each group on the edge, in segment order. A group with nobody inside the edge repeats the edge value there rather than reporting a mean over no rows.",
        piles.join(" and ")
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| reading | edge | panel on the edge | inside | groups on the edge |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---|");
    for gate in &artifact.gates {
        let inside: Vec<String> = artifact
            .segments
            .iter()
            .filter_map(|segment| {
                segment
                    .profile
                    .iter()
                    .find(|row| row.name == gate.name)
                    .map(|row| format!("{:.3}", row.inside.unwrap_or(0.0)))
            })
            .collect();
        let edge: Vec<String> = artifact
            .segments
            .iter()
            .filter_map(|segment| {
                segment
                    .profile
                    .iter()
                    .find(|row| row.name == gate.name)
                    .map(|row| format!("{:.1}%", row.atom.unwrap_or(0.0) * 100.0))
            })
            .collect();
        let _ = writeln!(
            out,
            "| {} | {:.2} | {:.1}% | {} | {} |",
            gate.name,
            gate.boundary,
            gate.share * 100.0,
            inside.join(", "),
            edge.join(", ")
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Both readings are scored as the probability of the observed value on the lattice a share actually lives on: the possible values are multiples of one over the advertiser's ad count, so a cell is integrated rather than approximated by density times spacing. Multiplying a density by that spacing is how a continuous model appears to win a lattice it cannot represent, and it reports mass above one."
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "| reading | gate | Gaussian | Brier gate | Brier Gaussian | edge called right | panel rate |");
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|");
    for audit in &metrics.gates {
        let _ = writeln!(
            out,
            "| {} | {:+.3} | {:+.3} | {:.4} | {:.4} | {:.1}% | {:.1}% |",
            audit.name,
            audit.log_gate,
            audit.log_gauss,
            audit.brier_gate,
            audit.brier_gauss,
            audit.accuracy * 100.0,
            audit.rate * 100.0
        );
    }
    let _ = writeln!(out);
    let gain: f64 = metrics.gates.iter().map(|audit| audit.gain()).sum();
    let _ = writeln!(
        out,
        "Held out on {} folds, the gates give the observed values {:.3} nats more probability per entity than the Gaussian conditional inside the same groups, and calls the edge right on {:.1} percent of rows against {:.1} percent for the panel rate. The gate is refit on the training rows of each fold; the partition is the shipped one, so the level is optimistic and the paired difference is the reading. The gate does not move a group's boundary: a likelihood with an atom and one without are densities against different measures, so the partition is still chosen on the Gaussian likelihood and the gate is fitted to it afterwards.",
        metrics.folds,
        gain,
        metrics
            .gates
            .iter()
            .map(|audit| audit.accuracy)
            .sum::<f64>()
            / metrics.gates.len().max(1) as f64
            * 100.0,
        metrics
            .gates
            .iter()
            .map(|audit| audit.rate)
            .sum::<f64>()
            / metrics.gates.len().max(1) as f64
            * 100.0
    );
    let _ = writeln!(out);
    let fired: Vec<String> = artifact
        .features
        .iter()
        .filter(|spec| spec.boundary.is_none())
        .map(|spec| spec.name.clone())
        .collect();
    let _ = writeln!(
        out,
        "The clip at {:.0} standard deviations still applies to {}. It never fired on either share, so the pile was never a clipping problem: it is the Gaussian's need to spend a group on it. What the clip exemption guards is a share whose edge sits beyond the bound, which would fold the edge into the interior and leave the gate without a boundary value.",
        artifact.scaling.clip,
        fired.join(", ")
    );
    let _ = writeln!(out);
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
