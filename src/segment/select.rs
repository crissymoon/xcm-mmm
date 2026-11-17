//! Which belief this panel is read through, and the reason for that one.
//!
//! The folder used to pick k by silhouette and let the shape of a group be
//! whatever k-means implies: a sphere of one shared width. That is a choice, not
//! the absence of one, and the previous card never wrote it down. Silhouette is
//! a Euclidean reading, so it also cannot fairly compare a spherical split with
//! a stretched one: the criterion itself rewards the shape k-means produces.
//!
//! This module chooses per panel, and every criterion is read on rows the fit
//! did not see:
//!
//! * **Held-out likelihood**, in nats per entity, is the accuracy of the belief.
//!   It is the only number here that a richer family cannot raise by stretching,
//!   and it is comparable across panel sizes.
//! * **Held-out agreement** is the accuracy of the partition: assign the rows a
//!   fold held out with the model that never saw them, and compare that
//!   placement with the shipped one by adjusted Rand index. A family that
//!   predicts density well and places rows differently every fold is refused by
//!   this gate, which is the failure a likelihood alone would let through.
//! * **A paired margin** decides between two candidates. The folds are the same
//!   rows each time, so the comparison is paired by fold: the mean difference
//!   must clear both the declared margin and its own standard error. A gain
//!   smaller than the fold noise does not buy parameters.
//!
//! Two hard gates run first, and both are about a group being a group: every
//! group must hold the declared share of the panel, and the partition must keep
//! its placement under held-out scoring. The chosen fit then has to beat the
//! same family on columns shuffled independently, which is the gate a diagonal
//! or full covariance can fail by stretching to fit noise.
//!
//! Nothing is declared per panel in code. The declared family is an override for
//! a reader who wants the shape held fixed; the shipped folder chooses.

use std::collections::BTreeMap;
use std::io;

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

use super::feature::Matrix;
use super::kmeans::{self, Sweep};
use super::mixture::{Family, Mixture};
use super::stability::ari;
use super::SegmentConfig;

/// Restarts spent on a fold. A fold needs a fair optimum, not the best one, and
/// the full fit keeps the whole budget for the model that actually ships.
const FOLD_RESTARTS: usize = 2;

/// One (family, k) the search considered, with its reading and its verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub family: Family,
    pub k: usize,
    /// Mean log likelihood per entity on the held-out fold, in nats.
    pub heldout: f64,
    /// Spread of the fold readings about that mean, in nats.
    pub heldout_sd: f64,
    /// Mean adjusted Rand index between the fold's placement and the shipped
    /// placement of the same rows.
    #[serde(default)]
    pub agreement: f64,
    #[serde(default)]
    pub agreement_sd: f64,
    pub bic: f64,
    /// BIC plus the entropy of the placement, in the same units. Read in sample,
    /// like BIC, so it is printed rather than selected on.
    #[serde(default)]
    pub icl: f64,
    pub silhouette: f64,
    pub sizes: Vec<usize>,
    pub min_share: f64,
    pub parameters: usize,
    pub verdict: String,
}

/// How well two columns move together, which is what a sphere assumes away.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pair {
    pub a: String,
    pub b: String,
    pub r: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Dependence {
    pub mean_abs: f64,
    pub strongest: Vec<Pair>,
}

/// The chosen fit's held-out reading on columns shuffled independently.
///
/// A freer family can stretch to fit noise, so the only honest partner to a
/// likelihood is the likelihood of the same protocol on columns with their
/// pairing destroyed. Every marginal is kept exactly; only the joint structure
/// goes, which is what the richer families claim to use.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Shuffle {
    pub runs: usize,
    pub mean: f64,
    pub max: f64,
}

/// The k-means partition the folder used to ship, scored on the same scale.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub k: usize,
    pub heldout: f64,
    pub silhouette: f64,
    pub sizes: Vec<usize>,
}

/// What the panel was read through, and every row the search considered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub family: Family,
    pub k: usize,
    /// The fitted model the choice refers to, so nothing has to fit it twice.
    pub mixture: Mixture,
    /// The component each row landed in, by posterior.
    pub labels: Vec<usize>,
    pub heldout: f64,
    pub heldout_sd: f64,
    pub agreement: f64,
    pub reason: String,
    pub candidates: Vec<Candidate>,
    pub baseline: Baseline,
    pub sweep: Vec<Sweep>,
    pub dependence: Dependence,
    pub folds: usize,
}

/// Folds that hold out whole rows, drawn once and shared by every candidate.
fn folds_of(n: usize, folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let folds = folds.clamp(2, n.max(2));
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng::stream(seed, 0x5eed_f01d);
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let mut out = vec![Vec::new(); folds];
    for (at, &row) in order.iter().enumerate() {
        out[at % folds].push(row);
    }
    out
}

/// The folds this configuration reads on, so an audit can rebuild them.
pub fn folds_for(n: usize, cfg: &SegmentConfig) -> Vec<Vec<usize>> {
    folds_of(n, cfg.folds, cfg.seed)
}

fn split(rows: &[Vec<f64>], fold: &[usize]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut out = vec![false; rows.len()];
    for &i in fold {
        out[i] = true;
    }
    let train: Vec<Vec<f64>> = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| !out[*i])
        .map(|(_, row)| row.clone())
        .collect();
    let test: Vec<Vec<f64>> = fold.iter().map(|&i| rows[i].clone()).collect();
    (train, test)
}

/// Per-fold held-out likelihood and per-fold placement agreement.
///
/// The two are read in one pass because they come from the same fold model: the
/// likelihood says how well the belief predicts, the agreement says whether the
/// groups it predicts are the groups that ship.
fn fold_scores(
    rows: &[Vec<f64>],
    labels: &[usize],
    k: usize,
    family: Family,
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
) -> (Vec<f64>, Vec<f64>) {
    let restarts = cfg.em_restarts.clamp(1, FOLD_RESTARTS);
    let mut likelihood = Vec::with_capacity(folds.len());
    let mut agreement = Vec::with_capacity(folds.len());
    for (index, fold) in folds.iter().enumerate() {
        let (train, test) = split(rows, fold);
        if train.len() <= k || test.is_empty() {
            continue;
        }
        let model = Mixture::fit(
            &train,
            k,
            family,
            restarts,
            cfg.iterations,
            cfg.seed.wrapping_add(index as u64),
        );
        if model.components.is_empty() {
            continue;
        }
        likelihood.push(model.total_loglik(&test) / test.len() as f64);
        let placed = model.assign(&test);
        let shipped: Vec<usize> = fold.iter().map(|&i| labels[i]).collect();
        agreement.push(ari(&placed, &shipped));
    }
    (likelihood, agreement)
}

/// The incumbent's held-out reading: k-means, scored through its own sphere.
fn held_out_incumbent(
    rows: &[Vec<f64>],
    k: usize,
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
) -> f64 {
    let mut scores = Vec::with_capacity(folds.len());
    for (index, fold) in folds.iter().enumerate() {
        let (train, test) = split(rows, fold);
        if train.len() <= k || test.is_empty() {
            continue;
        }
        let start = kmeans::fit(
            &train,
            k,
            cfg.restarts.clamp(1, FOLD_RESTARTS),
            cfg.iterations,
            cfg.seed.wrapping_add(index as u64),
        );
        let model = Mixture::plugged_in(&train, &start.labels, k);
        if model.components.is_empty() {
            continue;
        }
        scores.push(model.total_loglik(&test) / test.len() as f64);
    }
    stats(&scores).0
}

/// Score the chosen (family, k) on columns shuffled independently.
pub fn shuffle_control(
    rows: &[Vec<f64>],
    k: usize,
    family: Family,
    folds: &[Vec<usize>],
    cfg: &SegmentConfig,
) -> Shuffle {
    let runs = cfg.controls.max(1);
    let dim = rows.first().map(|row| row.len()).unwrap_or(0);
    let mut scores = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut rng = Rng::stream(cfg.seed ^ 0x9e37_79b9, run as u64 + 1);
        let mut shuffled = rows.to_vec();
        for column in 0..dim {
            for i in (1..shuffled.len()).rev() {
                let j = rng.below(i + 1);
                let value = shuffled[i][column];
                shuffled[i][column] = shuffled[j][column];
                shuffled[j][column] = value;
            }
        }
        let restarts = cfg.em_restarts.clamp(1, FOLD_RESTARTS);
        let mut fold_scores = Vec::new();
        for (index, fold) in folds.iter().enumerate() {
            let (train, test) = split(&shuffled, fold);
            if train.len() <= k || test.is_empty() {
                continue;
            }
            let model = Mixture::fit(
                &train,
                k,
                family,
                restarts,
                cfg.iterations,
                cfg.seed.wrapping_add(index as u64),
            );
            if model.components.is_empty() {
                continue;
            }
            fold_scores.push(model.total_loglik(&test) / test.len() as f64);
        }
        scores.push(stats(&fold_scores).0);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Shuffle { runs, mean, max }
}

fn stats(scores: &[f64]) -> (f64, f64) {
    if scores.is_empty() {
        return (f64::NEG_INFINITY, 0.0);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / scores.len() as f64;
    (mean, variance.sqrt())
}

/// Mean and standard error of the per-fold difference, challenger minus champion.
fn paired(challenger: &[f64], champion: &[f64]) -> (f64, f64) {
    let n = challenger.len().min(champion.len());
    if n == 0 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let diffs: Vec<f64> = (0..n).map(|i| challenger[i] - champion[i]).collect();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, f64::INFINITY);
    }
    let variance = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1) as f64;
    (mean, (variance / n as f64).sqrt())
}

/// A challenger is preferred only when its paired gain clears both the declared
/// margin and its own standard error, so fold noise cannot buy parameters.
fn clears(challenger: &[f64], champion: &[f64], min_gain: f64) -> bool {
    let (gain, error) = paired(challenger, champion);
    gain > min_gain && gain > error
}

/// How far apart the columns actually run, which is the argument for a shape
/// richer than a sphere.
pub fn dependence(rows: &[Vec<f64>], names: &[String]) -> Dependence {
    let dim = rows.first().map(|r| r.len()).unwrap_or(0);
    if dim < 2 {
        return Dependence::default();
    }
    let mut pairs = Vec::new();
    let mut total = 0.0;
    let mut counted = 0usize;
    for a in 0..dim {
        for b in (a + 1)..dim {
            let x: Vec<f64> = rows.iter().map(|row| row[a]).collect();
            let y: Vec<f64> = rows.iter().map(|row| row[b]).collect();
            let r = crate::linalg::pearson(&x, &y);
            total += r.abs();
            counted += 1;
            pairs.push(Pair {
                a: names.get(a).cloned().unwrap_or_else(|| a.to_string()),
                b: names.get(b).cloned().unwrap_or_else(|| b.to_string()),
                r,
            });
        }
    }
    pairs.sort_by(|x, y| {
        y.r.abs()
            .partial_cmp(&x.r.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.a.cmp(&y.a))
    });
    pairs.truncate(3);
    Dependence {
        mean_abs: if counted > 0 {
            total / counted as f64
        } else {
            0.0
        },
        strongest: pairs,
    }
}

/// Choose the belief, and write down why every other one lost.
pub fn choose(rows: &[Vec<f64>], matrix: &Matrix, cfg: &SegmentConfig) -> io::Result<Choice> {
    let n = rows.len();
    let dim = rows.first().map(|row| row.len()).unwrap_or(0);
    if n < cfg.min_entities || dim == 0 {
        return Err(bad("there are not enough rows to choose a segmentation"));
    }
    let names: Vec<String> = matrix.specs.iter().map(|s| s.name.clone()).collect();
    let k_high = cfg.k_max.min(n.saturating_sub(1)).max(cfg.k_min);
    let folds = folds_of(n, cfg.folds, cfg.seed);

    // The incumbent: a silhouette sweep, kept because it is the reading the
    // folder used to gate on and a reader may want to see it beside the new one.
    let mut sweep = Vec::new();
    for k in cfg.k_min..=k_high {
        let fit = kmeans::fit(rows, k, cfg.restarts, cfg.iterations, cfg.seed);
        let sizes = kmeans::sizes(&fit.labels, k);
        let share = sizes.iter().copied().min().unwrap_or(0) as f64 / n as f64;
        sweep.push(Sweep {
            k,
            silhouette: kmeans::silhouette(rows, &fit.labels, k),
            inertia: fit.inertia,
            sizes,
            min_share: share,
        });
    }
    let eligible_sweep: Vec<&Sweep> = sweep
        .iter()
        .filter(|point| point.min_share + 1e-12 >= cfg.min_share)
        .collect();
    if eligible_sweep.is_empty() {
        let widest = sweep.iter().map(|s| s.min_share).fold(0.0f64, f64::max);
        return Err(bad(&format!(
            "no k keeps every group above {:.1} percent; the widest smallest group is {:.1} percent",
            cfg.min_share * 100.0,
            widest * 100.0
        )));
    }
    let incumbent = eligible_sweep
        .iter()
        .copied()
        .max_by(|a, b| {
            a.silhouette
                .partial_cmp(&b.silhouette)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.k.cmp(&a.k))
        })
        .expect("a non empty eligible set");
    let baseline = Baseline {
        k: incumbent.k,
        heldout: held_out_incumbent(rows, incumbent.k, &folds, cfg),
        silhouette: incumbent.silhouette,
        sizes: incumbent.sizes.clone(),
    };

    // Every (family, k), fitted on the whole panel and scored on folds.
    let mut candidates = Vec::new();
    let mut models: Vec<(Family, usize, Mixture, Vec<usize>)> = Vec::new();
    let mut per_fold: Vec<Vec<f64>> = Vec::new();
    for family in Family::ALL {
        for k in cfg.k_min..=k_high {
            let mixture = Mixture::fit(rows, k, family, cfg.em_restarts, cfg.iterations, cfg.seed);
            if mixture.components.is_empty() {
                continue;
            }
            let labels = mixture.assign(rows);
            let sizes = kmeans::sizes(&labels, k);
            let (likelihood, agreement) = fold_scores(rows, &labels, k, family, &folds, cfg);
            let (heldout, heldout_sd) = stats(&likelihood);
            let (agree, agree_sd) = stats(&agreement);
            candidates.push(Candidate {
                family,
                k,
                heldout,
                heldout_sd,
                agreement: if agree.is_finite() { agree } else { 0.0 },
                agreement_sd: if agree_sd.is_finite() { agree_sd } else { 0.0 },
                bic: mixture.bic(n, dim),
                icl: mixture.icl(rows, dim),
                silhouette: kmeans::silhouette(rows, &labels, k),
                sizes,
                min_share: sizes_at(&labels, k, n),
                parameters: family.parameters(k, dim),
                verdict: String::new(),
            });
            per_fold.push(likelihood);
            models.push((family, k, mixture, labels));
        }
    }
    if candidates.is_empty() {
        return Err(bad("no candidate could be fitted"));
    }

    let declared = cfg.family;
    let allowed = |candidate: &Candidate| {
        candidate.min_share + 1e-12 >= cfg.min_share
            && candidate.agreement + 1e-12 >= cfg.min_stability
            && declared.map_or(true, |family| family == candidate.family)
    };
    if candidates.iter().all(|c| !allowed(c)) {
        let widest = candidates
            .iter()
            .map(|c| c.min_share)
            .fold(0.0f64, f64::max);
        let placed = candidates
            .iter()
            .map(|c| c.agreement)
            .fold(f64::NEG_INFINITY, f64::max);
        return Err(bad(&format!(
            "no candidate keeps every group above {:.1} percent and its placement above {:.2}; the widest smallest group is {:.1} percent and the best held-out agreement is {:.3}",
            cfg.min_share * 100.0,
            cfg.min_stability,
            widest * 100.0,
            placed
        )));
    }

    // Inside a family the larger k wins only when it clears the same paired
    // margin, so k is not bought on a difference smaller than fold noise.
    let mut best: BTreeMap<Family, usize> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !allowed(candidate) {
            continue;
        }
        match best.get(&candidate.family).copied() {
            None => {
                best.insert(candidate.family, index);
            }
            Some(current) => {
                // k is walked in ascending order, so the champion of a family
                // always has the smaller k and the challenger is the larger one.
                if clears(&per_fold[index], &per_fold[current], cfg.min_gain) {
                    best.insert(candidate.family, index);
                }
            }
        }
    }

    // Walk from the simplest family upward, taking a richer one only when it
    // predicts better by the declared margin and its own error.
    let mut order: Vec<Family> = Family::ALL.to_vec();
    if let Some(family) = declared {
        order.retain(|candidate| *candidate == family);
    }
    let mut champion = *best.keys().next().expect("a non empty eligible set");
    for family in order.iter().copied() {
        let Some(&index) = best.get(&family) else {
            continue;
        };
        if family == champion {
            continue;
        }
        if clears(&per_fold[index], &per_fold[best[&champion]], cfg.min_gain) {
            champion = family;
        }
    }
    let winner = best[&champion];
    let chosen = candidates[winner].clone();
    let across = choice_gain(&per_fold, &best, winner, cfg.min_gain);

    let mut reason = format!(
        "{} covariance at k = {}: {:.3} nats per entity on rows it did not see, spread {:.3} across folds, against {:.3} for the k-means partition this folder used to ship, a gain of {:.3}. A fold model places its own held-out rows at {:.3} agreement with it, and it costs {} parameters",
        chosen.family.name(),
        chosen.k,
        chosen.heldout,
        chosen.heldout_sd,
        baseline.heldout,
        chosen.heldout - baseline.heldout,
        chosen.agreement,
        chosen.parameters
    );
    if let Some((family, gain, error, index)) = across {
        reason.push_str(&format!(
            ". {family} was on the grid at a paired gain of {:.3} plus or minus {:.3} nats per entity, which does not clear the {:.3} margin, so its {} parameters were not bought",
            gain, error, cfg.min_gain, candidates[index].parameters
        ));
    }
    if declared.is_some() {
        reason.push_str(". The family was declared for this run rather than chosen");
    }

    // Every row gets a verdict: the table is an account, not a ranking. The
    // facts are read out first, because the loop below holds candidates mutably.
    let facts: Vec<Row> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| Row {
            family: candidate.family,
            k: candidate.k,
            heldout: candidate.heldout,
            agreement: candidate.agreement,
            min_share: candidate.min_share,
            is_best: best.get(&candidate.family) == Some(&index),
        })
        .collect();
    let champion_row = &facts[winner];
    for (index, candidate) in candidates.iter_mut().enumerate() {
        let row = &facts[index];
        candidate.verdict = if declared.is_some_and(|family| family != row.family) {
            format!(
                "set aside: {} was declared for this run, so the panel did not choose",
                row.family.name()
            )
        } else if row.min_share + 1e-12 < cfg.min_share {
            format!(
                "set aside: k {} puts a group at {:.1} percent, below the {:.1} percent floor",
                row.k,
                row.min_share * 100.0,
                cfg.min_share * 100.0
            )
        } else if row.agreement + 1e-12 < cfg.min_stability {
            format!(
                "set aside: a fold model places these rows at {:.3} agreement, below the {:.2} floor",
                row.agreement, cfg.min_stability
            )
        } else if index == winner {
            format!(
                "chosen: {:.3} nats per entity held out at {:.3} placement agreement",
                row.heldout, row.agreement
            )
        } else if !row.is_best {
            let held = facts
                .iter()
                .find(|other| other.family == row.family && other.is_best)
                .expect("every family with a best row has it in the list");
            format!(
                "set aside: k {} reads {:.3} nats against {:.3} at k {}, and the paired gain does not clear the {:.3} margin and its own standard error",
                row.k, row.heldout, held.heldout, held.k, cfg.min_gain
            )
        } else {
            format!(
                "set aside: {:.3} nats per entity against {:.3} for the fit that won, a gap of {:.3}",
                row.heldout,
                champion_row.heldout,
                champion_row.heldout - row.heldout
            )
        };
    }

    let Some((_, _, mixture, labels)) = models
        .into_iter()
        .find(|(family, k, _, _)| *family == chosen.family && *k == chosen.k)
    else {
        return Err(bad("the chosen candidate lost its own fit"));
    };

    Ok(Choice {
        family: chosen.family,
        k: chosen.k,
        mixture,
        labels,
        heldout: chosen.heldout,
        heldout_sd: chosen.heldout_sd,
        agreement: chosen.agreement,
        reason,
        candidates,
        baseline,
        sweep,
        dependence: dependence(rows, &names),
        folds: folds.len(),
    })
}

fn sizes_at(labels: &[usize], k: usize, n: usize) -> f64 {
    kmeans::sizes(labels, k).iter().copied().min().unwrap_or(0) as f64 / n as f64
}

/// The richer family that failed the margin against the winner, if any.
fn choice_gain(
    per_fold: &[Vec<f64>],
    best: &BTreeMap<Family, usize>,
    winner: usize,
    min_gain: f64,
) -> Option<(String, f64, f64, usize)> {
    let champion = per_fold.get(winner)?;
    for family in Family::ALL {
        let Some(&index) = best.get(&family) else {
            continue;
        };
        if index == winner {
            continue;
        }
        let rival = per_fold.get(index)?;
        let (gain, error) = paired(rival, champion);
        if gain.is_finite() && gain > 0.0 && !clears(rival, champion, min_gain) {
            return Some((family.name().to_string(), gain, error, index));
        }
    }
    None
}

/// The candidate facts the verdicts need, read out before the rows are written.
struct Row {
    family: Family,
    k: usize,
    heldout: f64,
    agreement: f64,
    min_share: f64,
    is_best: bool,
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests;
