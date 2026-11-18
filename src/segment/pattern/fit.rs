//! The maximisation step: a two-part estimator nested in a mixture.
//!
//! The previous step in this module's history estimated one joint normal over all
//! the readings and let the missing-data arithmetic fill an absent share at its
//! conditional mean. That is the maximum likelihood answer for a joint normal, and
//! it is the wrong answer here for a reason that is visible in the measurement it
//! produced: 93 percent of the shortfall against the two-part gate sat on rows
//! *inside* the edge, where the gate fits a slope and a variance on exactly the
//! rows that show a value while the joint covariance derived the same conditional
//! from an estimate that also had to explain the rows on the edge. An edge row's
//! share was never drawn from the continuous law, so letting it into the
//! regression pulls the slope toward zero and inflates the residual.
//!
//! This module therefore estimates the model the comparison was always about: a
//! density over the readings the gates do not hold, times, for each gated reading,
//! the probability of its state and the conditional law of its value given the
//! readings it shows. The maximisation step is coordinate ascent on that
//! factorisation, and each coordinate is closed form.
//!
//! * The free block, the readings no gate holds, is estimated from every row: it
//!   is present on all of them, and its marginal is part of the likelihood.
//! * Each gate's inside regression is weighted least squares on the rows that
//!   show that reading, so one edge does not cost another gate its slope. The
//!   slope and the residual variance are the two-part model's, fitted inside the
//!   loop rather than after the partition was chosen.
//! * The residual correlation between two gates needs the rows that show both,
//!   and that count is carried in [`super::Coupling`] rather than left implicit.
//!   On this panel the assumption of conditional independence is false at 0.63, so
//!   the model keeps the correlation instead of assuming it away.
//!
//! The stored covariance is then completed from those pieces so that every reader
//! of a [`Component`] sees the model this step estimated: the free block as it was
//! measured, the cross terms the regression implies, and the gate block with the
//! residual covariance on its diagonal. Marginalising a gate out of that
//! covariance returns the free block exactly, and conditioning on the free block
//! returns the regression exactly, which is why [`super::Pattern::posterior`] and
//! [`super::Pattern::score`] needed no change and why their readings now describe
//! the two-part law.
//!
//! Because this is a partial-likelihood estimator rather than the joint normal's
//! maximiser, the observed-data likelihood is not guaranteed to rise at every
//! step. The loop keeps the best iteration rather than the last, and the reason is
//! written beside the loop rather than left as a surprise: a two-part model trades
//! in-sample fit for the out-of-sample conditional, which is the whole question.

use super::super::gate::{self, GateSpec};
use super::super::mixture::{self, Component, Family, Mixture, FLOOR};
use super::{start_atom, Coupling, GateFit, Pattern, View};

/// Expectation maximisation from one starting assignment, then the shipping bound
/// on shape.
pub(super) fn em(
    rows: &[Vec<f64>],
    views: &[View],
    k: usize,
    specs: &[GateSpec],
    iterations: usize,
    start: &[Vec<f64>],
) -> Pattern {
    let gates: Vec<usize> = specs.iter().map(|spec| spec.column).collect();
    let filled = super::filled(rows, views);
    let components = Mixture::step(&filled, start, k, Family::Full).components;
    if components.is_empty() {
        return Pattern::default();
    }
    let atom = start_atom(rows, specs, start, k);
    let (mut resp, mut loglik) = super::e_step(&components, &atom, &gates, views, rows);
    let mut best = (
        loglik,
        components,
        atom,
        regressions(specs, k),
        couplings(specs, k),
    );

    let mut used = 1;
    for step in 1..iterations.max(1) {
        let (next, next_atom, next_regressions, next_couplings) = m_step(rows, &resp, k, specs);
        if next.is_empty() {
            break;
        }
        let (next_resp, total) = super::e_step(&next, &next_atom, &gates, views, rows);
        let settled = (total - loglik).abs() <= 1e-8 * rows.len() as f64;
        resp = next_resp;
        loglik = total;
        used = step + 1;
        if total > best.0 + 1e-9 {
            best = (
                total,
                next,
                next_atom,
                next_regressions.clone(),
                next_couplings.clone(),
            );
        }
        if settled {
            break;
        }
    }

    let (loglik, components, atom, regressions, coupling) = best;
    let mut pattern = Pattern {
        k,
        components,
        gates: gates.clone(),
        edges: specs.iter().map(|spec| spec.atom).collect(),
        atom,
        regressions,
        coupling,
        loglik,
        iterations: used,
        conditions: 0.0,
    };
    if let Some(bounded) = mixture::bounded_components(&pattern.components) {
        pattern.components = bounded;
        let (_, total) = super::e_step(&pattern.components, &pattern.atom, &gates, views, rows);
        pattern.loglik = total;
    }
    pattern.conditions = mixture::conditioning(&pattern.components);
    pattern
}

/// Zeroed audit readings, so a group that never learns a slope still reports one.
fn regressions(specs: &[GateSpec], k: usize) -> Vec<GateFit> {
    specs.iter().map(|spec| GateFit::empty(spec, k)).collect()
}

fn couplings(specs: &[GateSpec], k: usize) -> Vec<Coupling> {
    let mut out = Vec::new();
    for (a, first) in specs.iter().enumerate() {
        for second in specs.iter().skip(a + 1) {
            out.push(Coupling {
                first: first.name.clone(),
                second: second.name.clone(),
                correlation: vec![0.0; k],
                rows: vec![0; k],
            });
        }
    }
    out
}

/// One maximisation step on the factorised likelihood.
///
/// Returns the components, the atom probabilities, the per-gate regressions and
/// the residual correlations, so a caller can print what the fit decided instead
/// of inferring it from a mean.
pub(super) fn m_step(
    rows: &[Vec<f64>],
    resp: &[Vec<f64>],
    k: usize,
    specs: &[GateSpec],
) -> (Vec<Component>, Vec<Vec<f64>>, Vec<GateFit>, Vec<Coupling>) {
    let n = rows.len();
    let dim = rows[0].len();
    let gates: Vec<usize> = specs.iter().map(|spec| spec.column).collect();
    let free: Vec<usize> = (0..dim).filter(|c| !gates.contains(c)).collect();
    let width = free.len();

    let mut counts = vec![0.0; k];
    for row in resp.iter() {
        for (j, value) in row.iter().enumerate().take(k) {
            counts[j] += *value;
        }
    }

    // The panel, for the origin of the regression and for a group no row claims.
    let mut centre = vec![0.0; dim];
    for row in rows {
        for (d, value) in row.iter().enumerate().take(dim) {
            centre[d] += value;
        }
    }
    for value in centre.iter_mut() {
        *value /= n as f64;
    }
    let mut spread = vec![FLOOR; dim];
    for row in rows {
        for d in 0..dim {
            let gap = row[d] - centre[d];
            spread[d] += gap * gap / n as f64;
        }
    }
    let origin: Vec<f64> = free.iter().map(|&c| centre[c]).collect();

    let mut components = Vec::with_capacity(k);
    let mut atom = vec![vec![0.0; k]; gates.len()];
    let mut fits = regressions(specs, k);
    let mut couplings_out = couplings(specs, k);

    for j in 0..k {
        if counts[j] < 1e-6 * n as f64 {
            components.push(reset(&centre, &spread, &gates));
            for (g, fit) in fits.iter_mut().enumerate() {
                fit.rows[j] = 0;
                fit.sd[j] = spread[gates[g]].max(FLOOR).sqrt();
                fit.inside[j] = centre[gates[g]];
            }
            continue;
        }
        let count = counts[j];

        // The free block, from every row.
        let mut mean = vec![0.0; width];
        for (i, row) in rows.iter().enumerate() {
            let weight = resp[i][j];
            if weight == 0.0 {
                continue;
            }
            for (a, &column) in free.iter().enumerate() {
                mean[a] += weight * row[column];
            }
        }
        for value in mean.iter_mut() {
            *value /= count;
        }
        let mut block = vec![vec![0.0; width]; width];
        for (i, row) in rows.iter().enumerate() {
            let weight = resp[i][j];
            if weight == 0.0 {
                continue;
            }
            let gaps: Vec<f64> = free
                .iter()
                .enumerate()
                .map(|(a, &c)| row[c] - mean[a])
                .collect();
            for (a, slot) in block.iter_mut().enumerate() {
                for (b, cell) in slot.iter_mut().enumerate().skip(a) {
                    *cell += weight * gaps[a] * gaps[b];
                }
            }
        }
        for (a, slot) in block.iter_mut().enumerate() {
            for cell in slot.iter_mut().skip(a) {
                *cell /= count;
            }
            slot[a] = slot[a].max(FLOOR);
        }
        // Mirror the upper triangle into the lower one. The mirror reads the row's
        // own upper half before writing the lower one, so the two cannot alias:
        // this was first written the other way round, which copied raw sums into
        // the lower triangle and left it effectively absent.
        for a in 0..width {
            let (above, below) = block.split_at_mut(a);
            let values: Vec<f64> = above.iter().map(|row| row[a]).collect();
            for (cell, value) in below[0].iter_mut().take(a).zip(values) {
                *cell = value;
            }
        }

        // Each gate's inside regression, on the rows that show it.
        let weights: Vec<f64> = resp
            .iter()
            .map(|r| r.get(j).copied().unwrap_or(0.0))
            .collect();
        let mut betas = vec![Vec::new(); gates.len()];
        let mut variances = vec![1.0; gates.len()];
        for (g, spec) in specs.iter().enumerate() {
            let inside: Vec<usize> = (0..n)
                .filter(|&i| !spec.holds(&rows[i]) && weights[i] > 1e-8)
                .collect();
            fits[g].rows[j] = inside.len();
            if inside.len() < width + 3 {
                // Too few rows to separate a slope from noise. The mean of the
                // inside is reported and the slope stays at zero, which is the
                // same guard the two-part gate uses.
                let (sum, wsum) = inside.iter().fold((0.0, 0.0), |(s, w), &i| {
                    (s + weights[i] * rows[i][spec.column], w + weights[i])
                });
                let inside_mean = if wsum > 0.0 {
                    sum / wsum
                } else {
                    centre[spec.column]
                };
                let mut beta = vec![0.0; width + 1];
                beta[0] = inside_mean;
                let variance = inside.iter().fold(0.0, |acc, &i| {
                    let gap = rows[i][spec.column] - inside_mean;
                    acc + weights[i] * gap * gap
                });
                variances[g] = if wsum > 0.0 {
                    (variance / wsum).max(FLOOR)
                } else {
                    spread[spec.column].max(FLOOR)
                };
                betas[g] = beta;
            } else {
                let (beta, variance) =
                    gate::weighted(rows, &weights, spec.column, &free, &origin, &inside);
                betas[g] = beta;
                variances[g] = variance.max(FLOOR);
            }
            fits[g].sd[j] = variances[g].sqrt();
            fits[g].inside[j] = betas[g][0];
            fits[g].slope[j] = betas[g].iter().skip(1).map(|b| b * b).sum::<f64>().sqrt();
        }

        // The residual correlation between gates, from the rows that show both.
        for (a, first) in specs.iter().enumerate() {
            for (b, second) in specs.iter().enumerate().skip(a + 1) {
                let slot = couplings_out
                    .iter()
                    .position(|pair| pair.first == first.name && pair.second == second.name);
                let Some(slot) = slot else { continue };
                let mut wsum = 0.0;
                let (mut cross, mut left, mut right) = (0.0, 0.0, 0.0);
                for i in 0..n {
                    let weight = weights[i];
                    if weight <= 1e-8 || first.holds(&rows[i]) || second.holds(&rows[i]) {
                        continue;
                    }
                    let ra = rows[i][first.column] - predicted(&betas[a], &free, &origin, &rows[i]);
                    let rb =
                        rows[i][second.column] - predicted(&betas[b], &free, &origin, &rows[i]);
                    cross += weight * ra * rb;
                    left += weight * ra * ra;
                    right += weight * rb * rb;
                    wsum += weight;
                }
                let correlation = if left > 1e-12 && right > 1e-12 {
                    (cross / (left * right).sqrt()).clamp(-0.9, 0.9)
                } else {
                    0.0
                };
                couplings_out[slot].correlation[j] = correlation;
                couplings_out[slot].rows[j] = wsum.round() as usize;
            }
        }

        // The completed component: the free block as measured, the cross terms the
        // regression implies, and the gate block with the residual covariance on
        // its diagonal.
        let mut completed_mean = vec![0.0; dim];
        let mut covariance = vec![vec![0.0; dim]; dim];
        for (a, &ca) in free.iter().enumerate() {
            completed_mean[ca] = mean[a];
            for (b, &cb) in free.iter().enumerate() {
                covariance[ca][cb] = block[a][b];
            }
        }
        for (g, &column) in gates.iter().enumerate() {
            let beta = &betas[g];
            let shift: f64 = (0..width)
                .map(|d| beta[d + 1] * (mean[d] - origin[d]))
                .sum();
            completed_mean[column] = beta[0] + shift;
            for (a, &ca) in free.iter().enumerate() {
                let value: f64 = (0..width).map(|d| block[a][d] * beta[d + 1]).sum();
                covariance[ca][column] = value;
                covariance[column][ca] = value;
            }
        }
        for (g, &column) in gates.iter().enumerate() {
            for (h, &other) in gates.iter().enumerate() {
                let (bg, bh) = (&betas[g], &betas[h]);
                let shared: f64 = (0..width)
                    .map(|a| {
                        (0..width)
                            .map(|d| bg[a + 1] * block[a][d] * bh[d + 1])
                            .sum::<f64>()
                    })
                    .sum();
                let extra = if g == h {
                    variances[g]
                } else {
                    let slot = couplings_out
                        .iter()
                        .find(|pair| {
                            (pair.first == specs[g].name && pair.second == specs[h].name)
                                || (pair.first == specs[h].name && pair.second == specs[g].name)
                        })
                        .map(|pair| pair.correlation[j])
                        .unwrap_or(0.0);
                    slot * variances[g].sqrt() * variances[h].sqrt()
                };
                covariance[column][other] = if g == h {
                    (shared + extra).max(FLOOR)
                } else {
                    shared + extra
                };
            }
        }

        components.push(Component {
            weight: counts[j] / n as f64,
            mean: completed_mean,
            covariance,
        });
    }

    // The edge probability of each gate inside each group, the same estimator the
    // initialization used, now from the fitted responsibilities.
    for (g, spec) in specs.iter().enumerate() {
        for (j, cell) in atom[g].iter_mut().enumerate() {
            let total: f64 = resp.iter().map(|r| r.get(j).copied().unwrap_or(0.0)).sum();
            let edge: f64 = rows
                .iter()
                .enumerate()
                .filter(|(_, row)| spec.holds(row))
                .map(|(i, _)| resp[i].get(j).copied().unwrap_or(0.0))
                .sum();
            *cell = if total > 0.0 {
                (edge / total).clamp(super::EPS, 1.0 - super::EPS)
            } else {
                spec.share.clamp(super::EPS, 1.0 - super::EPS)
            };
        }
    }
    (components, atom, fits, couplings_out)
}

fn predicted(beta: &[f64], free: &[usize], origin: &[f64], row: &[f64]) -> f64 {
    let mut value = beta.first().copied().unwrap_or(0.0);
    for (d, &column) in free.iter().enumerate() {
        value += beta.get(d + 1).copied().unwrap_or(0.0)
            * (row[column] - origin.get(d).copied().unwrap_or(0.0));
    }
    value
}

fn reset(centre: &[f64], spread: &[f64], gates: &[usize]) -> Component {
    let dim = centre.len();
    let mut covariance = vec![vec![0.0; dim]; dim];
    for d in 0..dim {
        covariance[d][d] = spread[d].max(FLOOR);
    }
    let _ = gates;
    Component {
        weight: 0.0,
        mean: centre.to_vec(),
        covariance,
    }
}
