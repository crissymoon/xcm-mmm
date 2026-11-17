//! How sharply each entity sits in its group, and whether the sharpness is earned.
//!
//! A mixture reports a posterior for every row, and the mean winning posterior on
//! this panel is 0.9994. Read as confidence that would say the four groups are
//! separated almost perfectly, and the previous cards did read it that way. It is
//! not what it looks like. Three of the four shipped components carry between one
//! and five axes at the variance floor, because the z is clipped at three
//! standard deviations and a group of advertisers who all sit on the clip has
//! zero within-group spread on that column. A covariance with a zero axis gives
//! a density that is enormous on the sheet and negligible off it, so the
//! posterior is decided by a handful of coordinates that the clipping flattened.
//!
//! This module therefore reads three things rather than one.
//!
//! * The nominal posterior, the number a reader sees today, with the count of
//!   rows above [`SHARP`], between it and [`CONTESTED`], and below.
//! * The same posterior under a cushion: every covariance axis is raised to at
//!   least [`CUSHION`] standard deviations before the rows are placed. A component
//!   that can no longer lean on a flattened axis has to place rows on the coarse
//!   structure, which is what the grouping is supposed to be. The share of rows
//!   that keep their group under the cushion is the part of the confidence that
//!   survives the clipping being taken away.
//! * A held-out vote. Every row is placed by a model fitted without its fold,
//!   aligned to the shipped components by nearest mean, so a row is scored by a
//!   model that never saw it and never saw the shipped partition.
//!
//! The closest rows are listed with their nearest rival and all three readings,
//! because "which group is this entity nearly in" is the question a table of
//! centroids cannot answer. Pair separation is reported as a Mahalanobis distance
//! and the share of rows a boundary between the two groups would be expected to
//! confuse, `2 * Phi(-d / 2)`, which is the overlap of two equal Gaussians and the
//! honest analogue of a margin for a two-group comparison.
//!
//! Nothing here changes the fit. It is the reading that says how much of a sharp
//! posterior is structure and how much is the clip, and it is printed beside the
//! claims rather than left for a reader to discover.

use serde::{Deserialize, Serialize};

use crate::linalg;
use crate::num::round4;
use crate::stats::normal_cdf;

use super::mixture::{self, Family, Mixture};
use super::SegmentConfig;

/// A winning posterior at or above this is called sharp.
pub const SHARP: f64 = 0.9;
/// Between this and [`SHARP`] a row is contested; below it, soft.
pub const CONTESTED: f64 = 0.6;
/// The floor every covariance axis is raised to for the cushioned reading, in z.
pub const CUSHION: f64 = 0.5;
/// Rows listed in the closest table.
pub const LISTED: usize = 6;
/// A reading whose within-group spread is below this is pinned by the clip.
pub const PINNED: f64 = 1e-9;
/// A leave-one-fold-out model is given the restarts the shipped fit used.
///
/// Not the reduced count the selection loop uses while it is comparing twelve
/// candidates: an audit that searches less hard than the fit did charges the
/// placement for optimizer noise and reports it as sampling noise. The cost of
/// running the full search five times is a second on this panel.
fn fold_restarts(cfg: &SegmentConfig) -> usize {
    cfg.em_restarts.max(1)
}

/// One group's shape, as a reader needs it: how many axes are at the floor,
/// which readings are flat inside the group, and how confidently it holds rows.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Group {
    pub component: usize,
    pub size: usize,
    /// Covariance axes sitting at the mixture's width floor.
    pub axes: usize,
    /// Readings with no variation inside the group: the clip, or a constant.
    pub pinned: Vec<String>,
    /// Longest axis over shortest.
    pub condition: f64,
    /// Mean winning posterior over the group's own members.
    pub posterior: f64,
    /// Share of the group a leave-one-fold-out model keeps in it.
    #[serde(default)]
    pub held_out_kept: f64,
}

/// How far apart two groups sit, and what that implies for a boundary.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Separation {
    pub a: usize,
    pub b: usize,
    /// Mahalanobis distance between the two centres under their pooled shape.
    pub distance: f64,
    /// `2 * Phi(-d / 2)`: the share of rows a boundary between the two would
    /// confuse if the groups were equal Gaussians.
    pub overlap: f64,
}

/// One entity that sits nearest a boundary, with every reading of where it goes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: String,
    pub name: String,
    pub assigned: usize,
    pub rival: usize,
    pub posterior: f64,
    pub rival_posterior: f64,
    /// Winning minus runner-up posterior.
    pub margin: f64,
    /// Where the model fitted without this row's fold placed it, if it could.
    #[serde(default)]
    pub held_out: Option<usize>,
    /// Where the cushioned model placed it.
    pub cushioned: usize,
}

/// The sharpness audit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sharpness {
    pub rows: usize,
    /// Rows at or above [`SHARP`], in `[CONTESTED, SHARP)`, and below.
    pub sharp: usize,
    pub contested: usize,
    pub soft: usize,
    pub mean_posterior: f64,
    pub p10_posterior: f64,
    pub min_posterior: f64,
    pub mean_margin: f64,
    pub min_margin: f64,
    /// The cushion used for the second reading, in z standard deviations.
    pub cushion: f64,
    pub cushioned_posterior: f64,
    /// Share of rows the cushioned model keeps in their group.
    pub cushioned_kept: f64,
    pub cushioned_moved: usize,
    /// Share of rows a leave-one-fold-out model places in their group.
    pub held_out_kept: f64,
    pub held_out_rows: usize,
    /// Rows by shipped group against the group a fold model placed them in.
    /// Empty when no fold could be read.
    #[serde(default)]
    pub confusion: Vec<Vec<usize>>,
    pub groups: Vec<Group>,
    pub pairs: Vec<Separation>,
    pub closest: Vec<Point>,
}

/// A covariance with every axis raised to at least `floor`.
///
/// The eigenvectors are kept and only the short axes move, exactly as in the
/// shipped shape bound, so the direction a group is stretched along is the
/// direction the data gave and not a re-fit.
pub fn cushioned(covariance: &[Vec<f64>], floor: f64) -> Vec<Vec<f64>> {
    let dim = covariance.len();
    let Some((values, vectors)) = linalg::jacobi_eigen(covariance) else {
        return covariance.to_vec();
    };
    if values.iter().all(|value| *value >= floor) {
        return covariance.to_vec();
    }
    let mut out = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in i..dim {
            let value: f64 = (0..dim)
                .map(|m| {
                    let bounded = values[m].max(floor);
                    vectors[i][m] * bounded * vectors[j][m]
                })
                .sum();
            out[i][j] = value;
            out[j][i] = value;
        }
    }
    out
}

/// Mahalanobis distance between two centres under their pooled shape.
pub fn separation(components: &[mixture::Component], a: usize, b: usize) -> f64 {
    let (Some(left), Some(right)) = (components.get(a), components.get(b)) else {
        return 0.0;
    };
    let dim = left.mean.len();
    if right.mean.len() != dim {
        return 0.0;
    }
    let mut pooled = vec![vec![0.0; dim]; dim];
    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        for j in 0..dim {
            pooled[i][j] = 0.5 * (left.covariance[i][j] + right.covariance[i][j]);
        }
    }
    let mut factor = pooled;
    if linalg::cholesky(&mut factor).is_none() {
        return 0.0;
    }
    let delta: Vec<f64> = left
        .mean
        .iter()
        .zip(&right.mean)
        .map(|(x, y)| x - y)
        .collect();
    let solved = linalg::cholesky_solve(&factor, &delta);
    let distance: f64 = solved.iter().zip(&delta).map(|(a, b)| a * b).sum();
    distance.max(0.0).sqrt()
}

/// The share two equal Gaussians at distance `d` would confuse.
pub fn overlap(distance: f64) -> f64 {
    2.0 * normal_cdf(-distance / 2.0)
}

/// Per-component shape: axes at the floor, readings flat inside the group.
pub fn groups(
    rows: &[Vec<f64>],
    components: &[mixture::Component],
    labels: &[usize],
    columns: &[String],
    posteriors: &[f64],
) -> Vec<Group> {
    let k = components.len();
    let mut out = Vec::with_capacity(k);
    for (id, component) in components.iter().enumerate() {
        let members: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, label)| **label == id)
            .map(|(at, _)| at)
            .collect();
        let axes = linalg::jacobi_eigen(&component.covariance)
            .map(|(values, _)| {
                values
                    .iter()
                    .filter(|value| **value <= mixture::FLOOR * 1.001)
                    .count()
            })
            .unwrap_or(0);
        let mut pinned = Vec::new();
        if members.len() > 1 {
            for (column, name) in columns.iter().enumerate() {
                let mean =
                    members.iter().map(|&at| rows[at][column]).sum::<f64>() / members.len() as f64;
                let variance = members
                    .iter()
                    .map(|&at| (rows[at][column] - mean) * (rows[at][column] - mean))
                    .sum::<f64>()
                    / members.len() as f64;
                if variance.sqrt() < PINNED {
                    pinned.push(name.clone());
                }
            }
        }
        let posterior = if members.is_empty() {
            0.0
        } else {
            members.iter().map(|&at| posteriors[at]).sum::<f64>() / members.len() as f64
        };
        out.push(Group {
            component: id,
            size: members.len(),
            axes,
            pinned,
            condition: mixture::condition_number(&component.covariance),
            posterior,
            held_out_kept: 0.0,
        });
    }
    out
}

/// Place every row with the posterior, then again with every axis cushioned.
///
/// Returns the posterior matrix, the labels and the confidence of each row under
/// both readings.
fn readings(mixture: &Mixture, rows: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<usize>, Vec<usize>) {
    let nominal: Vec<Vec<f64>> = rows.iter().map(|row| mixture.posterior(row)).collect();
    let mut cushioned_components = mixture.components.clone();
    for component in cushioned_components.iter_mut() {
        component.covariance = cushioned(&component.covariance, CUSHION);
    }
    let soft = Mixture {
        family: mixture.family,
        components: cushioned_components,
        loglik: mixture.loglik,
        iterations: mixture.iterations,
    };
    let labels: Vec<usize> = nominal.iter().map(|row| best(row).0).collect();
    let soft_labels: Vec<usize> = if soft.components.len() == mixture.components.len() {
        rows.iter()
            .map(|row| best(&soft.posterior(row)).0)
            .collect()
    } else {
        labels.clone()
    };
    (nominal, labels, soft_labels)
}

/// Best component and the runner-up in one posterior row.
fn best(posterior: &[f64]) -> (usize, usize) {
    let mut order: Vec<usize> = (0..posterior.len()).collect();
    order.sort_by(|a, b| {
        posterior[*b]
            .partial_cmp(&posterior[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    let first = order.first().copied().unwrap_or(0);
    let second = order.get(1).copied().unwrap_or(first);
    (first, second)
}

/// The ten percent quantile of a sample, on the same definition as the audits.
fn p10(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = ((sorted.len() - 1) as f64 * 0.1).round() as usize;
    sorted[at.min(sorted.len() - 1)]
}

/// Walk every ordering of the fold components and keep the cheapest.
///
/// An exact search rather than a greedy nearest match. With four groups there
/// are twenty-four orderings, so the exhaustive answer costs nothing, and a
/// greedy walk can take a near mean early and then be forced into a worse
/// assignment for the rest. A wrongly aligned fold model scores correct
/// placements as disagreements, which is the one error this audit cannot afford:
/// it would report instability where the refit was right.
fn permute(
    order: &mut Vec<usize>,
    at: usize,
    best: &mut (f64, Vec<usize>),
    shipped: &[mixture::Component],
    fold: &[mixture::Component],
) {
    if at == order.len() {
        let cost: f64 = order
            .iter()
            .enumerate()
            .map(|(f, s)| {
                fold[f]
                    .mean
                    .iter()
                    .zip(&shipped[*s].mean)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
            })
            .sum();
        if cost < best.0 {
            best.0 = cost;
            best.1 = order.clone();
        }
        return;
    }
    for slot in at..order.len() {
        order.swap(at, slot);
        permute(order, at + 1, best, shipped, fold);
        order.swap(at, slot);
    }
}

/// Align a fold model's components to the shipped ones by nearest mean.
///
/// Returns, for each fold component, the shipped index it stands for. A fold
/// model numbers its groups by the order the seeds found them, so comparing a
/// placement without this step would score a correct answer as a disagreement
/// whenever the numbering happened to differ.
pub(super) fn align(shipped: &[mixture::Component], fold: &[mixture::Component]) -> Vec<usize> {
    if fold.len() == shipped.len() && fold.len() <= 7 {
        let mut order: Vec<usize> = (0..fold.len()).collect();
        let mut best = (f64::INFINITY, order.clone());
        permute(&mut order, 0, &mut best, shipped, fold);
        return best.1;
    }
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (f, component) in fold.iter().enumerate() {
        for (s, reference) in shipped.iter().enumerate() {
            let distance: f64 = component
                .mean
                .iter()
                .zip(&reference.mean)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            pairs.push((distance, s, f));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut taken_shipped = vec![false; shipped.len()];
    let mut taken_fold = vec![false; fold.len()];
    let mut out = vec![usize::MAX; fold.len()];
    for (_, s, f) in pairs {
        if taken_shipped[s] || taken_fold[f] {
            continue;
        }
        out[f] = s;
        taken_shipped[s] = true;
        taken_fold[f] = true;
    }
    for (f, slot) in out.iter_mut().enumerate() {
        if *slot == usize::MAX {
            *slot = f.min(shipped.len().saturating_sub(1));
        }
    }
    out
}

/// Place each row with a model fitted without its fold, aligned to the shipped ids.
fn held_out_votes(
    rows: &[Vec<f64>],
    shipped: &Mixture,
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
) -> Vec<Option<usize>> {
    let mut out = vec![None; rows.len()];
    let k = shipped.k();
    let family: Family = shipped.family;
    for (index, fold) in folds.iter().enumerate() {
        let mut held = vec![false; rows.len()];
        for &at in fold {
            held[at] = true;
        }
        let train: Vec<Vec<f64>> = rows
            .iter()
            .enumerate()
            .filter(|(at, _)| !held[*at])
            .map(|(_, row)| row.clone())
            .collect();
        if train.len() <= k || fold.is_empty() {
            continue;
        }
        let model = Mixture::fit(
            &train,
            k,
            family,
            fold_restarts(cfg),
            cfg.iterations,
            cfg.seed.wrapping_add(index as u64),
        );
        if model.components.len() != k {
            continue;
        }
        let map = align(&shipped.components, &model.components);
        for &at in fold {
            let placed = model.assign(&[rows[at].clone()]);
            out[at] = placed.first().map(|label| map[*label]);
        }
    }
    out
}

/// Measure the sharpness of a partition against its own rows.
///
/// `people` carries the entity id and name for each row, in row order, so the
/// closest table can name what it lists.
pub fn measure(
    rows: &[Vec<f64>],
    mixture: &Mixture,
    labels: &[usize],
    columns: &[String],
    people: &[(String, String)],
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
) -> Sharpness {
    let (posterior, nominal, cushioned_labels) = readings(mixture, rows);
    let confidence: Vec<f64> = posterior
        .iter()
        .map(|row| row.iter().copied().fold(0.0f64, f64::max))
        .collect();
    let votes = held_out_votes(rows, mixture, folds, cfg);
    let cushion_confidence: Vec<f64> = {
        let mut components = mixture.components.clone();
        for component in components.iter_mut() {
            component.covariance = cushioned(&component.covariance, CUSHION);
        }
        let soft_model = Mixture {
            family: mixture.family,
            components,
            loglik: mixture.loglik,
            iterations: mixture.iterations,
        };
        rows.iter()
            .map(|row| soft_model.posterior(row).into_iter().fold(0.0f64, f64::max))
            .collect()
    };
    let margins: Vec<f64> = posterior
        .iter()
        .map(|row| {
            let (first, second) = best(row);
            row[first] - row[second]
        })
        .collect();
    let kept = nominal
        .iter()
        .zip(&cushioned_labels)
        .filter(|(a, b)| a == b)
        .count();
    let voted = votes.iter().filter(|vote| vote.is_some()).count();
    let agreed = votes
        .iter()
        .enumerate()
        .filter(|(at, vote)| vote.is_some_and(|group| group == labels[*at]))
        .count();
    let k = mixture.components.len();
    let mut confusion = vec![vec![0usize; k]; k];
    for (at, vote) in votes.iter().enumerate() {
        if let Some(group) = vote {
            if *group < k && labels[at] < k {
                confusion[labels[at]][*group] += 1;
            }
        }
    }
    let mut pairs = Vec::new();
    for a in 0..mixture.components.len() {
        for b in (a + 1)..mixture.components.len() {
            let distance = separation(&mixture.components, a, b);
            pairs.push(Separation {
                a,
                b,
                distance,
                overlap: overlap(distance),
            });
        }
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|a, b| {
        confidence[*a]
            .partial_cmp(&confidence[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let closest = order
        .iter()
        .take(LISTED)
        .map(|&at| {
            let (first, second) = best(&posterior[at]);
            let (id, name) = people
                .get(at)
                .cloned()
                .unwrap_or_else(|| (format!("row {at}"), String::new()));
            Point {
                id,
                name,
                assigned: labels[at],
                rival: second,
                posterior: confidence[at],
                rival_posterior: posterior[at][second],
                margin: posterior[at][first] - posterior[at][second],
                held_out: votes[at],
                cushioned: cushioned_labels[at],
            }
        })
        .collect();
    let mut groups = groups(rows, &mixture.components, labels, columns, &confidence);
    for group in groups.iter_mut() {
        let total: usize = confusion
            .get(group.component)
            .map(|row| row.iter().sum())
            .unwrap_or(0);
        group.held_out_kept = if total == 0 {
            0.0
        } else {
            confusion[group.component][group.component] as f64 / total as f64
        };
    }
    Sharpness {
        rows: rows.len(),
        sharp: confidence.iter().filter(|value| **value >= SHARP).count(),
        contested: confidence
            .iter()
            .filter(|value| **value >= CONTESTED && **value < SHARP)
            .count(),
        soft: confidence
            .iter()
            .filter(|value| **value < CONTESTED)
            .count(),
        mean_posterior: confidence.iter().sum::<f64>() / confidence.len().max(1) as f64,
        p10_posterior: p10(&confidence),
        min_posterior: confidence.iter().copied().fold(1.0f64, f64::min),
        mean_margin: margins.iter().sum::<f64>() / margins.len().max(1) as f64,
        min_margin: margins.iter().copied().fold(1.0f64, f64::min),
        cushion: CUSHION,
        cushioned_posterior: cushion_confidence.iter().sum::<f64>()
            / cushion_confidence.len().max(1) as f64,
        cushioned_kept: kept as f64 / rows.len().max(1) as f64,
        cushioned_moved: rows.len().saturating_sub(kept),
        held_out_kept: agreed as f64 / voted.max(1) as f64,
        held_out_rows: voted,
        confusion,
        groups,
        pairs,
        closest,
    }
}

impl Sharpness {
    /// Every stored number rounded to the precision the artifacts keep.
    pub fn rounded(mut self) -> Self {
        self.mean_posterior = round4(self.mean_posterior);
        self.p10_posterior = round4(self.p10_posterior);
        self.min_posterior = round4(self.min_posterior);
        self.mean_margin = round4(self.mean_margin);
        self.min_margin = round4(self.min_margin);
        self.cushioned_posterior = round4(self.cushioned_posterior);
        self.cushioned_kept = round4(self.cushioned_kept);
        self.held_out_kept = round4(self.held_out_kept);
        for group in self.groups.iter_mut() {
            group.condition = round4(group.condition);
            group.posterior = round4(group.posterior);
            group.held_out_kept = round4(group.held_out_kept);
        }
        for pair in self.pairs.iter_mut() {
            pair.distance = round4(pair.distance);
            pair.overlap = round4(pair.overlap);
        }
        for point in self.closest.iter_mut() {
            point.posterior = round4(point.posterior);
            point.rival_posterior = round4(point.rival_posterior);
            point.margin = round4(point.margin);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::mixture::Component;

    fn component(weight: f64, mean: [f64; 2], sd: f64, rho: f64) -> Component {
        Component {
            weight,
            mean: mean.to_vec(),
            covariance: vec![vec![sd * sd, rho * sd * sd], vec![rho * sd * sd, sd * sd]],
        }
    }

    fn two_groups() -> (Mixture, Vec<Vec<f64>>) {
        let mixture = Mixture {
            family: Family::Full,
            components: vec![
                component(0.5, [-3.0, 0.0], 1.0, 0.0),
                component(0.5, [3.0, 0.0], 1.0, 0.0),
            ],
            loglik: 0.0,
            iterations: 0,
        };
        let rows = vec![
            vec![-3.0, 0.0],
            vec![-2.9, 0.1],
            vec![3.0, 0.0],
            vec![0.0, 0.0],
        ];
        (mixture, rows)
    }

    #[test]
    fn a_row_on_a_centre_is_sharp_and_a_row_between_is_not() {
        let (mixture, rows) = two_groups();
        let labels = mixture.assign(&rows);
        let columns = vec!["a".to_string(), "b".to_string()];
        let people: Vec<(String, String)> = (0..rows.len())
            .map(|at| (format!("{at}"), format!("row {at}")))
            .collect();
        let measured = measure(
            &rows,
            &mixture,
            &labels,
            &columns,
            &people,
            &[],
            &SegmentConfig::default(),
        );
        assert_eq!(measured.sharp, 3);
        assert_eq!(measured.contested, 0);
        assert_eq!(measured.soft, 1);
        assert_eq!(measured.closest.len(), 4);
        assert_eq!(measured.closest[0].posterior, 0.5);
        assert_eq!(measured.held_out_rows, 0);
        assert!(measured
            .confusion
            .iter()
            .all(|row| row.iter().all(|n| *n == 0)));
    }

    #[test]
    fn cushioning_keeps_separated_rows_and_opens_a_narrow_one() {
        let narrow = vec![vec![1e-6, 0.0], vec![0.0, 0.0]];
        let out = cushioned(&narrow, 0.5);
        let (values, _) = linalg::jacobi_eigen(&out).expect("a symmetric matrix");
        assert!(values.iter().all(|value| *value >= 0.5 - 1e-12));
        let same = cushioned(&[vec![1.0, 0.0], vec![0.0, 1.0]], 0.5);
        assert_eq!(same[1][1], 1.0);
    }

    #[test]
    fn separation_falls_as_groups_approach() {
        let near = vec![
            component(0.5, [0.0, 0.0], 1.0, 0.0),
            component(0.5, [1.0, 0.0], 1.0, 0.0),
        ];
        let far = vec![
            component(0.5, [0.0, 0.0], 1.0, 0.0),
            component(0.5, [6.0, 0.0], 1.0, 0.0),
        ];
        assert!((separation(&near, 0, 1) - 1.0).abs() < 1e-9);
        assert!((separation(&far, 0, 1) - 6.0).abs() < 1e-9);
        assert!(overlap(1.0) > overlap(6.0));
        assert!((overlap(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_flat_reading_is_reported_as_pinned() {
        let components = vec![
            component(0.5, [0.0, 0.0], 1.0, 0.0),
            component(0.5, [3.0, 0.0], 1.0, 0.0),
        ];
        let rows = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![3.0, 0.0]];
        let labels = vec![0, 0, 1];
        let columns = vec!["flat".to_string(), "free".to_string()];
        let out = groups(&rows, &components, &labels, &columns, &[1.0; 3]);
        assert_eq!(out[0].pinned, vec!["flat".to_string()]);
        assert_eq!(out[0].size, 2);
        assert!(out[1].pinned.is_empty());
    }

    #[test]
    fn a_fold_model_is_aligned_before_it_is_scored() {
        let shipped = vec![
            component(0.5, [0.0, 0.0], 1.0, 0.0),
            component(0.5, [4.0, 0.0], 1.0, 0.0),
        ];
        let reordered = vec![
            component(0.5, [4.1, 0.0], 1.0, 0.0),
            component(0.5, [-0.1, 0.0], 1.0, 0.0),
        ];
        assert_eq!(align(&shipped, &reordered), vec![1, 0]);
    }

    #[test]
    fn rounding_does_not_change_a_rounded_audit() {
        let (mixture, rows) = two_groups();
        let labels = mixture.assign(&rows);
        let measured = measure(
            &rows,
            &mixture,
            &labels,
            &["a".to_string(), "b".to_string()],
            &[],
            &[],
            &SegmentConfig::default(),
        )
        .rounded();
        assert_eq!(measured.clone().rounded(), measured);
    }
}
