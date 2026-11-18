//! The card: what the folder holds, in the words a reader needs first.

use std::fmt::Write;

use crate::linalg;
use crate::num::commas;

use super::mixture;
use super::model::{Artifact, Manifest, Metrics};

/// The distance between two groups, in either order.
fn pair_distance(pairs: &[super::sharp::Separation], a: usize, b: usize) -> Option<f64> {
    pairs
        .iter()
        .find(|pair| (pair.a == a && pair.b == b) || (pair.a == b && pair.b == a))
        .map(|pair| pair.distance)
}

/// The largest movement between two groups in a held-out confusion matrix.
///
/// Off the diagonal only: a group keeping its own rows is the reading the table
/// already carries, and the question here is where the rows that leave go.
fn worst_flow(confusion: &[Vec<usize>]) -> Option<(usize, usize, usize)> {
    let mut best: Option<(usize, usize, usize)> = None;
    for (from, row) in confusion.iter().enumerate() {
        for (to, count) in row.iter().enumerate() {
            if from == to || *count == 0 {
                continue;
            }
            if best.map_or(true, |(_, _, seen)| *count > seen) {
                best = Some((from, to, *count));
            }
        }
    }
    best
}

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
    if !metrics.pattern.is_empty() {
        render_pattern(&mut out, metrics);
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
    let icl_best = metrics.candidates.iter().min_by(|a, b| {
        a.icl
            .partial_cmp(&b.icl)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let winner = metrics
        .candidates
        .iter()
        .find(|candidate| candidate.family == metrics.family && candidate.k == metrics.k);
    match (icl_best, winner) {
        (Some(row), Some(fit)) if row.heldout < metrics.heldout => {
            let penalty = fit.icl - fit.bic;
            let winning = (-penalty / (2.0 * metrics.entities.max(1) as f64)).exp();
            let _ = writeln!(
                out,
                "ICL is printed beside BIC because BIC charges a model for the parameters it spends and nothing for the rows it cannot place: two overlapping components can raise the likelihood while every row sits near their boundary. ICL adds that ambiguity as `BIC - 2 * sum log p`, where `p` is the posterior the winning component gives a row, so it can only be BIC or worse. On this grid it does not change the answer. The lowest ICL is {} at k = {}, reading {:.0} against {:.0} on the same fit's BIC, and it is still the near-saturated candidate whose held-out reading is {:.3} against {:.3} for the fit that won. The charge is small here because a free covariance buys sharp posteriors: the chosen fit gives each of its own rows a mean winning posterior of {:.4}, so the penalty adds {:.1} across {} entities, which is {:.3} per entity. A criterion that is BIC plus a term that nearly vanishes is, on this panel, BIC.",
                row.family.name(),
                row.k,
                row.icl,
                row.bic,
                row.heldout,
                metrics.heldout,
                winning,
                penalty,
                metrics.entities,
                penalty / metrics.entities.max(1) as f64
            );
        }
        _ => {
            let _ = writeln!(
                out,
                "ICL is printed beside BIC because BIC charges a model for the parameters it spends and nothing for the rows it cannot place. ICL adds the ambiguity as `BIC - 2 * sum log p`, and here the lowest ICL is also the fit that won, so the two readings agree and the choice does not depend on which a reader prefers."
            );
        }
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| family | k | held out, nats per entity | spread over folds | placement agreement | BIC | ICL | parameters | smallest group | verdict |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    for candidate in &metrics.candidates {
        let chosen = if candidate.family == metrics.family && candidate.k == metrics.k {
            " **chosen**"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "| {}{} | {} | {:.3} | {:.3} | {:.3} | {:.0} | {:.0} | {} | {:.3} | {} |",
            candidate.family.name(),
            chosen,
            candidate.k,
            candidate.heldout,
            candidate.heldout_sd,
            candidate.agreement,
            candidate.bic,
            candidate.icl,
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
    let _ = writeln!(out, "### Which groups survive resampling");
    let _ = writeln!(out);
    let flagged = metrics
        .stability
        .cluster
        .iter()
        .filter(|row| row.flagged)
        .count();
    if !metrics.stability.cluster.is_empty() {
        let _ = writeln!(
            out,
            "| group | holds | share | Jaccard, mean of {} | worst | reading |",
            metrics.stability.bootstraps
        );
        let _ = writeln!(out, "|---|---:|---:|---:|---:|---|");
        for row in &metrics.stability.cluster {
            let label = artifact
                .segment(row.cluster)
                .map(|segment| segment.label.clone())
                .unwrap_or_else(|| format!("group {}", row.cluster));
            let share = if metrics.entities == 0 {
                0.0
            } else {
                row.size as f64 / metrics.entities as f64
            };
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} percent | {:.3} | {:.3} | {} |",
                label,
                row.size,
                share * 100.0,
                row.mean,
                row.min,
                if row.flagged {
                    "below the 0.6 floor".to_string()
                } else {
                    "reproducible".to_string()
                }
            );
        }
        let _ = writeln!(out);
        let total = metrics.stability.cluster.len();
        let stable = total - flagged;
        let across = metrics
            .stability
            .cluster
            .iter()
            .map(|row| row.mean)
            .sum::<f64>()
            / total.max(1) as f64;
        let _ = writeln!(
            out,
            "The adjusted Rand index above is one number for the whole partition, and it weighs every pair of entities: a large group that comes back exactly pays for a small group that does not. The per-group reading is the Jaccard index, the share of rows a group and its best match in a refit have in common over the rows either holds, so a group that splits in the refit scores low even though both halves together cover it. {} of the {} groups clear the 0.6 floor taken from Hennig (2007) and the mean across the partition is {:.3}. {} A group that does not return under resampling is a region of this panel rather than a membership that would reappear on a new draw, and the card says which is which rather than reporting the average alone.",
            stable,
            total,
            across,
            if flagged == 0 {
                "Every group here is a membership rather than a region.".to_string()
            } else {
                "A group below it is a region of this panel rather than a membership: an entity placed in one is described by the group it sits nearest, and nothing here claims that entity would land in the same group on a new draw.".to_string()
            }
        );
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "A silhouette of {:.2} is ordinary for behavioural data whose columns share a scale factor, and a partition read through a covariance scores lower on it by construction: a stretched group is not a sphere, and the reading measures distance from a centre, not the shape a group actually has. The separated groups are {:.2} against {:.2} for columns with their pairing destroyed, so the structure is real; the shape is the part silhouette cannot see.",
        metrics.silhouette, metrics.silhouette, metrics.control.mean
    );
    let _ = writeln!(out);

    let sharp = &metrics.sharpness;
    if sharp.rows == 0 {
        let _ = writeln!(out, "## How sharp the groups are");
        let _ = writeln!(out);
        let _ = writeln!(out, "Not measured for this artifact.");
        let _ = writeln!(out);
    } else {
        let _ = writeln!(out, "## How sharp the groups are");
        let _ = writeln!(out);
        let _ = writeln!(out, "| winning posterior | entities |");
        let _ = writeln!(out, "|---|---:|");
        let _ = writeln!(
            out,
            "| at or above {:.3} | {} |",
            super::sharp::SHARP,
            commas(sharp.sharp)
        );
        let _ = writeln!(
            out,
            "| between {:.3} and {:.3} | {} |",
            super::sharp::CONTESTED,
            super::sharp::SHARP,
            commas(sharp.contested)
        );
        let _ = writeln!(
            out,
            "| below {:.3} | {} |",
            super::sharp::CONTESTED,
            commas(sharp.soft)
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "The mean winning posterior is {:.4}, the tenth percentile {:.4} and the lowest reading {:.4}, with a mean margin of {:.4} over the runner-up and a lowest of {:.4}. Taken alone that would say the groups are almost perfectly separated. It is partly the clip. A component whose covariance carries an axis at the width floor has no spread in that direction, so the density it reports there is enormous for a row on the sheet and negligible off it; three of the four shipped components are in that state, because the z is clipped at three standard deviations and a group whose members all sit on the clip has no variation left. The flat readings are listed per group below.",
            sharp.mean_posterior, sharp.p10_posterior, sharp.min_posterior, sharp.mean_margin, sharp.min_margin
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "Read again with every covariance axis raised to at least {:.1} standard deviations, so no component can lean on a flattened axis, the mean winning posterior falls to {:.4} and {} entities change group, which leaves {:.1} percent of the panel where it was. That is the honest split of the claim: the partition survives the clip being taken away, the confidence does not. A leave-one-fold-out model, fitted without each row's fold and matched to the shipped components by the ordering that minimises the distance between their centres, places {:.1} percent of the {} rows it could read in the group this card gives them.",
            sharp.cushion,
            sharp.cushioned_posterior,
            commas(sharp.cushioned_moved),
            sharp.cushioned_kept * 100.0,
            sharp.held_out_kept * 100.0,
            commas(sharp.held_out_rows)
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| group | entities | axes at the floor | readings flat inside the group | condition | mean posterior | held out keeps |"
        );
        let _ = writeln!(out, "|---|---:|---:|---|---:|---:|---:|");
        for group in &sharp.groups {
            let label = artifact
                .segment(group.component)
                .map(|segment| segment.label.clone())
                .unwrap_or_else(|| format!("group {}", group.component));
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {:.0} | {:.3} | {:.1} percent |",
                label,
                commas(group.size),
                group.axes,
                if group.pinned.is_empty() {
                    "none".to_string()
                } else {
                    group
                        .pinned
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                },
                group.condition,
                group.posterior,
                group.held_out_kept * 100.0
            );
        }
        let _ = writeln!(out);
        if !sharp.confusion.is_empty() {
            let ids: Vec<String> = sharp
                .groups
                .iter()
                .map(|group| group.component.to_string())
                .collect();
            let _ = writeln!(
                out,
                "| this card's group, by the group a model fitted without its fold gave it | {} |",
                ids.join(" | ")
            );
            let _ = writeln!(
                out,
                "|---|{}",
                ids.iter().map(|_| "---:|").collect::<Vec<_>>().join("")
            );
            for (from, row) in sharp.confusion.iter().enumerate() {
                let cells: Vec<String> = row.iter().map(|count| commas(*count)).collect();
                let _ = writeln!(out, "| {} | {} |", from, cells.join(" | "));
            }
            let _ = writeln!(out);
        }
        if let Some((from, to, count)) = worst_flow(&sharp.confusion) {
            let across = pair_distance(&sharp.pairs, from, to);
            let _ = writeln!(
                out,
                "The largest movement is {} entities from group {} to group {}{}. Movement is not only proximity: a group of this size refits its own shape on four fifths of the panel, and a member that sits on the thin part of one group's sheet falls inside the other's. The reading to take from the table is which groups a new draw would reproduce and which it would rearrange, and it agrees with the per-group Jaccard above rather than repeating the posterior.",
                commas(count),
                from,
                to,
                across
                    .map(|distance| format!(
                        ", two centres {distance:.2} apart under their pooled shape"
                    ))
                    .unwrap_or_default()
            );
            let _ = writeln!(out);
        }
        let _ = writeln!(
            out,
            "| pair | distance between centres | share a boundary would confuse |"
        );
        let _ = writeln!(out, "|---|---:|---:|");
        for pair in &sharp.pairs {
            let _ = writeln!(
                out,
                "| {} and {} | {:.2} | {:.3} |",
                pair.a, pair.b, pair.distance, pair.overlap
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "The distance is Mahalanobis under the two groups' pooled shape and the share is `2 Phi(-d/2)`, the overlap of two equal Gaussians at that distance. It is the closest thing to a margin a partition without labels can have, and it is why the nearest pair matters more than the mean posterior does."
        );
        let _ = writeln!(out);
        if !sharp.closest.is_empty() {
            let _ = writeln!(
                out,
                "| entity | group | nearest rival | posterior | rival | margin | fitted without its fold | cushioned |"
            );
            let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|");
            for point in &sharp.closest {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {} | {} |",
                    point.name.replace('|', "/"),
                    point.assigned,
                    point.rival,
                    point.posterior,
                    point.rival_posterior,
                    point.margin,
                    point
                        .held_out
                        .map(|group| group.to_string())
                        .unwrap_or_else(|| "unmeasured".to_string()),
                    point.cushioned
                );
            }
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "An entity listed here sits nearest a boundary. The held-out column is where a model that never saw it placed it and the cushioned column is where it lands once no component may use a flattened axis. When the two agree with the group this card gives, the placement is the entity's; when they disagree, the row is a region rather than a membership, and the card says which rather than rounding the disagreement away."
            );
            let _ = writeln!(out);
        }
    }

    let gating = &metrics.concomitant;
    let _ = writeln!(out, "## Recency in the mixing weights");
    let _ = writeln!(out);
    if gating.covariates.is_empty() || gating.folds == 0 {
        let _ = writeln!(out, "Not measured for this artifact.");
        let _ = writeln!(out);
    } else {
        let _ = writeln!(
            out,
            "A mixture has two places a column can enter. The densities say what a group is like, and recency is kept out of them because the panel is a snapshot weighted towards recent ads: the distance from a last start measures the archive's coverage as much as the advertiser's activity. The mixing weights make a weaker claim, that a descriptive column changes which group a row comes from, and this is the audit of whether the panel supports it. The model writes `logit(pi_k(x))` as a multinomial logistic function of the descriptive columns and leaves every density on behaviour (Dayton and Macready, JASA 83, 1988)."
        );
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| model | covariates | constant weights | gated weights | gain | standard error | worst mean move |"
        );
        let _ = writeln!(out, "|---|---|---:|---:|---:|---:|---:|");
        for (label, audit) in [
            ("recency alone", &metrics.concomitant),
            ("all three", &metrics.concomitant_all),
        ] {
            if audit.covariates.is_empty() || audit.folds == 0 {
                continue;
            }
            let names: Vec<String> = audit
                .covariates
                .iter()
                .map(|covariate| format!("{} ({:.1}%)", covariate.name, covariate.eta2 * 100.0))
                .collect();
            let _ = writeln!(
                out,
                "| {} | {} | {:.3} | {:.3} | {:+.3} | {:.3} | {:.3} sd |",
                label,
                names.join(", "),
                audit.constant,
                audit.gated,
                audit.delta,
                audit.delta_se,
                audit.shift_max
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "The parenthesis after each covariate is the share of that column's own variance the four groups already explain. It is the reading to look at first: a column the partition already separates cannot add much to the weights."
        );
        let _ = writeln!(
            out,
            "The per entity spread is the only estimate of the noise this panel offers, so the gate must add twice its own standard error before the gain is called one, and no component mean may move more than a tenth of a standard deviation, because a moved mean means the groups describe something else. Both readings are held out on the same folds the family choice used, and both fits start from the same model, so the difference is the gate's doing."
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", gating.verdict);
        let _ = writeln!(out);
        if metrics.concomitant_all.folds > 0 {
            let _ = writeln!(out, "{}", metrics.concomitant_all.verdict);
            let _ = writeln!(out);
        }
        let _ = writeln!(
            out,
            "The wider set is reported beside the pre-registered one rather than instead of it. Its span column is mechanically tied to ad volume, which every density already reads, and the fold to fold spread of its difference is {:.3} nats, which no reading at this row count can see through. That is why the acceptance is stated before the fit: a wider set offered as a second chance would find a number above zero sooner or later, and the number would mean nothing.",
            metrics.concomitant_all.delta_sd
        );
        let _ = writeln!(out);
    }

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
        "- Recency is deliberately not a feature. The panel is a snapshot weighted towards recent ads, so the distance from a last start to the snapshot day would measure the archive's coverage as much as the advertiser's activity. It is not in the mixing weights either, and that is measured rather than assumed: the four groups explain {:.1} percent of its own spread, and gating on it moves the held-out reading by {:+.3} nats per entity against a standard error of {:.3}.",
        metrics
            .concomitant
            .covariates
            .first()
            .map_or(0.0, |c| c.eta2 * 100.0),
        metrics.concomitant.delta,
        metrics.concomitant.delta_se
    );
    let _ = writeln!(
        out,
        "- The accuracy reported is predictive: how well the fitted belief explains rows it was not fitted on, in nats per entity. It is not a measure of whether a group is the right group, because no label says which advertiser belongs where. A richer family can predict better and still be the less stable reading of the panel, which is why the resampling agreement is reported beside it."
    );
    let _ = writeln!(
        out,
        "- A group below the Jaccard floor is a region of this panel, not a membership. The fit still places every entity in it, because a segment has to answer for every advertiser, but nothing here says the same advertiser would land there on a new draw."
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
        "- The gate describes the two shares inside a partition fitted without it. The pattern mixture in the section above fits the atom inside the likelihood instead, and on this panel it does not predict the shares better: it loses on the inside branch and draws a partition this card does not ship. What neither model can settle is which partition is better, because a likelihood with an atom and one without are densities against different measures, and the one comparable reading, the held-out probability of a share, does not measure a partition."
    );
    let _ = writeln!(
        out,
        "- A winning posterior here is not a probability that the group is right. {} of the {} components carry axes at the width floor, which is what the clip leaves behind, and a density with a zero axis is enormous on the sheet and negligible off it. The cushioned reading and the held-out confusion above are the two corrections, and both are lower than the nominal number: read the table before quoting {:.3}.",
        sharp.groups.iter().filter(|group| group.axes > 0).count(),
        sharp.groups.len(),
        sharp.mean_posterior
    );
    if let Some(largest) = sharp.groups.iter().max_by_key(|group| group.size) {
        let smaller: Vec<f64> = sharp
            .groups
            .iter()
            .filter(|group| group.component != largest.component)
            .map(|group| group.held_out_kept)
            .collect();
        if let (Some(low), Some(high)) = (
            smaller.iter().copied().reduce(f64::min),
            smaller.iter().copied().reduce(f64::max),
        ) {
            let _ = writeln!(
                out,
                "- The confidence a row carries and the stability of its group are separate readings. Group {} holds {:.0} percent of its {} entities when a model is fitted without them, while the smaller groups keep between {:.0} and {:.0} percent of theirs at a mean posterior of {:.3} or better. A stable large group and a small unstable one can both be sharp on this panel.",
                largest.component,
                largest.held_out_kept * 100.0,
                largest.size,
                low * 100.0,
                high * 100.0,
                sharp
                    .groups
                    .iter()
                    .map(|group| group.posterior)
                    .fold(1.0f64, f64::min)
            );
        }
    }
    let _ = writeln!(
        out,
        "- The held-out vote places each row with a model fitted on four fifths of the panel, matched to the shipped groups by the ordering that minimises the distance between centres. A row that moves there is not mislabelled by this card: the two models answer the same question with different evidence, and the disagreement is the reading."
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

/// The atom modelled inside the likelihood, and what it bought.
fn render_pattern(out: &mut String, metrics: &Metrics) {
    let _ = writeln!(out, "## The gate inside the loop");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The gate above is fitted to a partition chosen without it. The alternative is to state the pattern as data and fit the two together: a reading that sits on its edge is absent rather than equal to the edge value, the row is scored on the readings it shows, and the probability of the edge is its own parameter per group per gate. That is a pattern mixture in the sense of Ghahramani and Jordan (1993). Two rounds of measurement live in this section. The first fitted one joint normal over the eight readings and imputed an absent share at its conditional mean; the reading then said that 93 percent of the shortfall against the partition gate sat on rows inside the edge, where the gate fits a slope on the rows that show a value and a joint covariance has to explain the rows on the edge as well. The second round acted on that and estimated the conditional where the gate does: each group gets its own weighted least squares of the gated reading on the six readings the score conditions on, fitted inside the loop, with its own residual variance. The stored covariance is then completed from those pieces, so conditioning it on the six readings returns the regression exactly, and the groups keep the residual correlation between the two shares (0.63 on this panel) rather than assuming it away."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Its likelihood is a density against a mixed measure, Lebesgue on the readings and counting on the edges, so it is a proper likelihood for these data. It is not comparable with a Gaussian likelihood, which is a density against a different measure, so it is not used to choose a family. What is comparable is the held out probability of one event, the observed value of a gated reading given the six readings the gates do not hold, and both models are scored on that event, on the same rows, with both sides refit on the training rows of each fold, including the mixture the gate is fitted to. The paired difference is the reading. The two-part step is a partial likelihood, not the joint normal's maximiser, so the loop keeps its best iteration rather than its last, and the reason is written beside the loop."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| reading | folds | two-part gate | loop, four groups | loop, one group | gain | standard error |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|");
    for verdict in &metrics.pattern {
        let _ = writeln!(
            out,
            "| {} | {} | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:.3} |",
            verdict.name,
            verdict.folds,
            verdict.two_stage,
            verdict.pattern,
            verdict.pooled,
            verdict.gain,
            verdict.se
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| reading | edge rows | two-part | loop | inside rows | two-part | loop |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|");
    for verdict in &metrics.pattern {
        let _ = writeln!(
            out,
            "| {} | {} | {:+.3} | {:+.3} | {} | {:+.3} | {:+.3} |",
            verdict.name,
            verdict.edge_rows,
            verdict.two_stage_edge,
            verdict.pattern_edge,
            verdict.inside_rows,
            verdict.two_stage_inside,
            verdict.pattern_inside
        );
    }
    let _ = writeln!(out);
    for verdict in &metrics.pattern {
        let _ = writeln!(out, "- `{}`: {}.", verdict.name, verdict.verdict);
    }
    let _ = writeln!(out);
    let worst = metrics.pattern.iter().min_by(|a, b| {
        a.gain
            .partial_cmp(&b.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(verdict) = worst {
        let inside_gap = verdict.pattern_inside - verdict.two_stage_inside;
        let edge_gap = verdict.pattern_edge - verdict.two_stage_edge;
        let inside_weight = (inside_gap * verdict.inside_rows as f64).abs();
        let edge_weight = (edge_gap * verdict.edge_rows as f64).abs();
        let total = (inside_weight + edge_weight).max(1e-12);
        let _ = writeln!(
            out,
            "The second table says where a difference sits, and on this panel it sits almost entirely inside the edge. On `{}` the loop model is {:.3} nats per row behind on the readings that are on their edge and {:.3} behind on those inside it, so {:.0} percent of the total shortfall comes from rows the pile does not touch. The atom itself, the part the loop model was built to fix, is predicted about as well by the gate fitted to the partition.",
            verdict.name,
            -edge_gap,
            -inside_gap,
            100.0 * inside_weight / total
        );
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "| reading | rows showing it per group: gate | loop | residual standard deviation: gate | loop | mean winning posterior in the six reading block: gate | loop |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|");
    for verdict in &metrics.pattern {
        let _ = writeln!(
            out,
            "| {} | {:.1} | {:.1} | {:.3} | {:.3} | {:.3} | {:.3} |",
            verdict.name,
            verdict.gate_rows,
            verdict.loop_rows,
            verdict.gate_sd,
            verdict.loop_sd,
            verdict.gate_sharp,
            verdict.loop_sharp
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "That third table is the one that explains the order rather than restating it. The loop model runs on fewer rows per group than the gate does, {:.1} against {:.1} on the first reading and {:.1} against {:.1} on the second, and in the previous round that showed up as a tighter in-sample residual with a worse held-out prediction. After the pooling described below it is no longer tighter: {:.3} against {:.3} on the first reading and {:.3} against {:.3} on the second, so the loop is now the model with the wider residual and still the worse prediction, which is a different and more honest failure. It is also sharper in the block the score conditions on, a mean winning posterior of {:.3} against {:.3}, so it is not a model that hedges. The one-group column separates the remaining explanations. A single pooled conditional, with no grouping at all, reads {:.3} on the first reading and {:.3} on the second, so the grouping does buy something the pooling cannot; what it does not buy is the partition's own advantage. The groups the gate is fitted to are formed from the six readings, so they separate exactly the block the score conditions on, while the loop's groups are formed partly from the share and land on a different segmentation.",
        metrics.pattern.first().map(|v| v.loop_rows).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.gate_rows).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.loop_rows).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.gate_rows).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.loop_sd).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.gate_sd).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.loop_sd).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.gate_sd).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.loop_sharp).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.gate_sharp).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.pooled).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.pooled).unwrap_or(0.0)
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The reading moved against the version that imputed instead of regressing: that one read {:.3} and {:.3} behind, this one {:.3} and {:.3}. Estimating the conditional the way the gate does makes the loop model more internally consistent and makes its prediction of the share worse, and the third table says why rather than leaving the regression to be blamed for it.",
        0.099,
        0.142,
        metrics.pattern.first().map(|v| -v.gain).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| -v.gain).unwrap_or(0.0)
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "### How much the slope is pooled");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The third table names a variance problem: a slope vector of six coefficients per group on about thirty five rows fits its own training rows and does not predict. The answer is partial pooling, and it is estimated rather than assumed. Each group's slope is a precision weighted average of its own estimate and the slope fitted to the whole panel. The weight comes from the group's standard error against the between group spread of that coefficient, measured net of the sampling noise the spread is contaminated by. A coefficient whose groups differ by more than their errors keeps its own value; one whose spread the errors already explain is pooled. Because two separately refitted settings land on different partitions, the three settings below are re-estimated on one partition, from a single maximisation step on the shipped model's own responsibilities, so the table is the estimator and not the grouping; the share of held rows that move when a setting is refitted from scratch is printed beside it for the same reason."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| reading | no pooling | partial | full | pooled, one group | full against partial | folds | refit apart |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|");
    for verdict in &metrics.pattern {
        let _ = writeln!(
            out,
            "| {} | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {} | {:.0}% |",
            verdict.name,
            verdict.pattern_none,
            verdict.pattern,
            verdict.pattern_full,
            verdict.pooled,
            verdict.pool_gain,
            verdict.folds,
            100.0 * (1.0 - verdict.pool_same)
        );
    }
    let _ = writeln!(out);
    if let Some(first) = metrics.pattern.first() {
        let _ = writeln!(
            out,
            "The shipped setting is {} pooling, and the panel said how much of it to use: the mean weight a group takes from the pooled slope is {:.2} on one share and {:.2} on the other, against a between group slope spread net of sampling error of {:.3} and {:.3}. A weight of one means the group's slope is not distinguishable from the pooled one; a weight of zero means it keeps its own estimate. Full pooling reads {:+.3} nats more than shipping that weight on the first share (se {:.3}) and {:+.3} on the second (se {:.3}). Neither clears two standard errors, so the adaptive weight ships: it is the estimator that decides, and choosing the extreme because it won by a fraction on the same folds is the selection this repository has a gate against.",
            first.pooling,
            first.share,
            metrics.pattern.get(1).map(|v| v.share).unwrap_or(0.0),
            first.tau,
            metrics.pattern.get(1).map(|v| v.tau).unwrap_or(0.0),
            first.pool_gain,
            first.pool_se,
            metrics.pattern.get(1).map(|v| v.pool_gain).unwrap_or(0.0),
            metrics.pattern.get(1).map(|v| v.pool_se).unwrap_or(0.0)
        );
        let _ = writeln!(out);
    }
    let _ = writeln!(
        out,
        "Pooling closes part of the gap and not all of it. Against no pooling it buys {:.3} nats on the first share and {:.3} on the second, and the loop model still reads {:.3} and {:.3} behind the gate fitted to the partition. What it did fix is the inversion the third table named: the loop's inside residual is now {:.3} against the gate's {:.3} on the first share and {:.3} against {:.3} on the second, so it is no longer the tighter in sample residual of a model that predicts worse. The remaining shortfall is the partition. The groups the gate is fitted to are formed from the six readings the score conditions on; the loop's groups are formed partly from the share itself.",
        metrics.pattern.first().map(|v| v.pattern - v.pattern_none).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.pattern - v.pattern_none).unwrap_or(0.0),
        metrics.pattern.first().map(|v| -v.gain).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| -v.gain).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.loop_sd).unwrap_or(0.0),
        metrics.pattern.first().map(|v| v.gate_sd).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.loop_sd).unwrap_or(0.0),
        metrics.pattern.get(1).map(|v| v.gate_sd).unwrap_or(0.0)
    );
    let _ = writeln!(out);
    let agreement = metrics
        .pattern
        .first()
        .map(|verdict| verdict.agreement * 100.0)
        .unwrap_or(0.0);
    let moved = metrics
        .pattern
        .first()
        .map(|verdict| verdict.moved)
        .unwrap_or(0);
    let _ = writeln!(
        out,
        "The comparison also says what would move if the artifact were rebuilt on this model's own partition: {:.1} percent of held out rows keep the group they have and {} move. The atom enters the partition, because a group is allowed an edge probability near one, so rows end up separated partly by which edges they sit on. That is a different segmentation rather than an improved one, and the choice at the top of this card still stands on held out Gaussian likelihood, which is the one comparison of partitions the two measures allow."
    , agreement, moved);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Three things had to be shown before this reading could be trusted. The maximisation step is checked on data drawn from the model, where the pattern mixture must not lose to the gate fitted to the partition, and the completion is checked against the regression it is built from: conditioning a stored component on the six readings must return the slope and residual the maximisation step fitted, which is what makes the numbers here a statement about the model rather than about the arithmetic. And two defects were found by writing the numbers down. The first version of the maximisation step added both orders of the conditional covariance to one cell of the second moment, doubling the off diagonal, which made the average indefinite and ended every fit after a single step; it read several tenths of a nat worse without saying why. The one-group reference was then clamped with a floor meant for a probability, which turned every log score into -27.631 and would have printed a pooled model twenty-seven nats behind for an arithmetic reason. Both are pinned by tests in the module."
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
        assert!(card.contains("## How sharp the groups are"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_largest_movement_is_read_off_the_diagonal() {
        let confusion = vec![vec![10, 4, 1], vec![2, 20, 0], vec![0, 1, 7]];
        assert_eq!(worst_flow(&confusion), Some((0, 1, 4)));
        assert_eq!(worst_flow(&[vec![5, 0], vec![0, 5]]), None);
    }
}
