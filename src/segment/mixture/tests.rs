//! The likelihood's own tests, beside the fit they exercise.

use super::*;
use crate::segment::kmeans;

fn blob(cx: f64, cy: f64, n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.37;
            vec![cx + 0.05 * t.sin(), cy + 0.05 * t.cos()]
        })
        .collect()
}

fn blobs() -> Vec<Vec<f64>> {
    let mut all = blob(0.0, 0.0, 20);
    all.extend(blob(6.0, 0.0, 20));
    all.extend(blob(0.0, 6.0, 20));
    all
}

/// Two thin arms that cross at the origin. No straight cut can separate them,
/// which is exactly the case a spherical family is blind to.
fn arms() -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut truth = Vec::new();
    for i in 0..41 {
        let t = -2.2 + 4.4 * i as f64 / 40.0;
        rows.push(vec![t + 0.12 * (t * 7.0).sin(), t + 0.12 * (t * 5.0).cos()]);
        truth.push(0);
    }
    for i in 0..41 {
        let t = -2.2 + 4.4 * i as f64 / 40.0;
        rows.push(vec![
            t + 0.12 * (t * 5.0).cos(),
            -t + 0.12 * (t * 6.0).sin(),
        ]);
        truth.push(1);
    }
    (rows, truth)
}

/// Accuracy up to any relabelling, because a component id is arbitrary.
fn agreement(labels: &[usize], truth: &[usize]) -> f64 {
    crate::segment::stability::ari(labels, truth).max(0.0)
}

#[test]
fn a_full_covariance_recovers_arms_that_k_means_cuts_through() {
    let (rows, truth) = arms();
    let full = Mixture::fit(&rows, 2, Family::Full, 4, 200, 7);
    let shaped = agreement(&full.assign(&rows), &truth);
    let incumbent = kmeans::fit(&rows, 2, 4, 200, 7);
    let cut = agreement(&incumbent.labels, &truth);
    assert!(shaped > 0.9, "full covariance reached {shaped:.3}");
    assert!(
        shaped > cut + 0.15,
        "full {shaped:.3} against k-means {cut:.3}"
    );
}

#[test]
fn the_richer_family_predicts_rows_it_did_not_see_better() {
    let (rows, _) = arms();
    let (even, odd): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
        rows.iter()
            .enumerate()
            .fold((Vec::new(), Vec::new()), |mut acc, (i, row)| {
                if i % 2 == 0 {
                    acc.0.push(row.clone());
                } else {
                    acc.1.push(row.clone());
                }
                acc
            });
    let spherical = Mixture::fit(&even, 2, Family::Spherical, 4, 200, 7);
    let full = Mixture::fit(&even, 2, Family::Full, 4, 200, 7);
    let simple = spherical.total_loglik(&odd) / odd.len() as f64;
    let rich = full.total_loglik(&odd) / odd.len() as f64;
    assert!(
        rich > simple + 1.0,
        "full {rich:.3} nats per row against spherical {simple:.3}"
    );
}

#[test]
fn separated_blobs_are_found_and_a_fit_is_reproducible() {
    let rows = blobs();
    let a = Mixture::fit(&rows, 3, Family::Diagonal, 4, 120, 11);
    let b = Mixture::fit(&rows, 3, Family::Diagonal, 4, 120, 11);
    assert_eq!(a, b);
    assert_eq!(a.k(), 3);
    let labels = a.assign(&rows);
    let truth: Vec<usize> = (0..rows.len()).map(|i| i / 20).collect();
    assert!(agreement(&labels, &truth) > 0.95);
    assert!(a.loglik > 0.0);
}

#[test]
fn the_posterior_sums_to_one_and_the_assignment_follows_it() {
    let rows = blobs();
    let fit = Mixture::fit(&rows, 3, Family::Spherical, 2, 100, 3);
    for row in rows.iter().take(5) {
        let weights = fit.posterior(row);
        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "posterior sums to {total}");
        let top = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(fit.assign(std::slice::from_ref(row))[0], top);
    }
}

#[test]
fn three_identical_rows_do_not_score_an_unbounded_density() {
    let mut rows = blobs();
    rows.push(vec![6.0, 0.0]);
    rows.push(vec![6.0, 0.0]);
    rows.push(vec![6.0, 0.0]);
    let fit = Mixture::fit(&rows, 3, Family::Diagonal, 4, 120, 5);
    for row in rows.iter().take(4) {
        let density = fit.log_density(row);
        assert!(density.is_finite(), "density {density}");
    }
}

#[test]
fn the_plug_in_model_scores_the_labels_it_was_handed() {
    let rows = blobs();
    let incumbent = kmeans::fit(&rows, 3, 4, 100, 2);
    let plug = Mixture::plugged_in(&rows, &incumbent.labels, 3);
    assert_eq!(plug.k(), 3);
    assert!(plug.loglik.is_finite());
    let sizes = kmeans::sizes(&incumbent.labels, 3);
    assert_eq!(sizes.iter().sum::<usize>(), rows.len());
    assert!(plug.components.iter().all(|c| c.weight > 0.0));
}

#[test]
fn a_richer_family_costs_more_parameters_and_pays_for_them() {
    let rows = blobs();
    let rich = Mixture::fit(&rows, 2, Family::Full, 2, 80, 1);
    assert!(Family::Full.parameters(2, 8) > Family::Diagonal.parameters(2, 8));
    assert!(Family::Diagonal.parameters(2, 8) > Family::Spherical.parameters(2, 8));
    // The criterion is the likelihood plus an exact charge for every parameter,
    // so the charge is checked rather than assumed.
    let charge = rich.bic(rows.len(), 8) + 2.0 * rich.loglik;
    let expected = Family::Full.parameters(2, 8) as f64 * (rows.len() as f64).ln();
    assert!(
        (charge - expected).abs() < 1e-9,
        "{charge} against {expected}"
    );
}

#[test]
fn a_family_round_trips_through_its_name() {
    for family in Family::ALL {
        assert_eq!(Family::parse(family.name()), Some(family));
        assert!(!family.shape().is_empty());
    }
    assert_eq!(Family::parse(" mixture "), None);
}

/// Two columns that move together to within rounding, both at unit variance.
/// Every diagonal entry clears the width floor, which is the whole point: the
/// floor reads widths and this matrix is sick in shape.
fn collinear(gap: f64) -> Vec<Vec<f64>> {
    let r = 1.0 - gap;
    vec![vec![1.0, r], vec![r, 1.0]]
}

/// Two columns that agree to thirteen places: the short axis is four orders of
/// magnitude past the cap, which is the case the bound exists for.
fn degenerate() -> Vec<Vec<f64>> {
    collinear(1e-13)
}

#[test]
fn the_condition_number_reads_the_axis_ratio() {
    let flat = degenerate();
    assert!(condition_number(&flat) > CONDITION_MAX);
    let round = vec![vec![0.5, 0.1], vec![0.1, 0.5]];
    assert!((condition_number(&round) - 1.5).abs() < 1e-9);
    // A matrix with a non-positive axis has no ratio to report.
    let broken = vec![vec![0.0, 0.0], vec![0.0, -1.0]];
    assert!(condition_number(&broken).is_infinite());
}

fn single(covariance: Vec<Vec<f64>>) -> Component {
    Component {
        weight: 1.0,
        mean: vec![0.0; covariance.len()],
        covariance,
    }
}

#[test]
fn a_flat_component_is_lifted_to_the_bound() {
    let flat = degenerate();
    assert!(condition_number(&flat) > CONDITION_MAX);
    let (shipped, factors) = ship(&[single(flat)]).unwrap();
    let bounded = shipped[0].covariance.clone();
    let factor = &factors[0];
    // Lifted to exactly the cap, not past it: the short axis ends at
    // longest / CONDITION_MAX, which for this matrix is two trillionths.
    let reached = condition_number(&bounded) / CONDITION_MAX;
    // To a thousandth: at the precision limit the ratio is known to a few
    // digits, because the short axis is at the size of the arithmetic error.
    assert!((reached - 1.0).abs() < 1e-3, "{reached}");
    // Symmetric, as a covariance has to be, and lifted along the axis that was
    // too short: for this matrix that direction is (1, -1)/sqrt(2), so the
    // difference of the diagonal entries is its variance.
    assert_eq!(bounded[0][1], bounded[1][0]);
    let thin: f64 = bounded[0][0] - bounded[0][1];
    assert!((thin - 2e-12).abs() < 1e-13, "{thin}");
    // Below the width floor, which is the point of a ratio rather than an
    // absolute width: the axis is short relative to its own component.
    assert!(thin < FLOOR);
    // The long axis is kept where the data put it: it is the sum of the two
    // entries, because the direction is (1, 1)/sqrt(2).
    assert!((bounded[0][0] + bounded[0][1] - 2.0).abs() < 1e-4);
    // The density is finite and falls off along the long axis, so the component
    // has not become a spike that scores one row and nothing else.
    let centre = factor.log_density(&[0.0, 0.0], &[0.0, 0.0]);
    let out = factor.log_density(&[3.0, 3.0], &[0.0, 0.0]);
    assert!(centre.is_finite() && out < centre, "{centre} {out}");
}

#[test]
fn a_round_component_is_returned_untouched() {
    let round = vec![vec![0.5, 0.1], vec![0.1, 0.5]];
    let (shipped, _) = ship(&[single(round.clone())]).unwrap();
    // Rebuilding from eigenvectors would change the last bit of every entry and
    // an artifact that changes for no reason is a worse artifact.
    assert_eq!(shipped[0].covariance, round);
}

#[test]
fn every_fitted_component_respects_the_bound() {
    // Two groups that each lie on a line with almost no thickness: the case a
    // diagonal floor cannot see, because the variances along both axes are fine.
    let mut rows = Vec::new();
    for i in 0..40 {
        let t = i as f64 / 40.0 * 4.0;
        rows.push(vec![t, t + 1e-7 * (i as f64).sin(), 0.5 * t + 2.0]);
    }
    for i in 0..40 {
        let t = i as f64 / 40.0 * 4.0;
        rows.push(vec![
            8.0 + t,
            8.0 + t + 1e-7 * (i as f64).cos(),
            0.5 * t + 2.0,
        ]);
    }
    let fitted = Mixture::fit(&rows, 2, Family::Full, 1, 60, 1);
    assert_eq!(fitted.k(), 2);
    assert!(
        fitted.conditioning() <= CONDITION_MAX,
        "{}",
        fitted.conditioning()
    );
    assert!(fitted.loglik.is_finite());
}

#[test]
fn a_thin_component_inside_the_cap_is_still_shipped_as_it_is() {
    // The ratio is what is bounded, not thinness itself. This component is
    // thinner than the width floor on one diagonal entry would allow and it is
    // still well inside the cap, which is where the shipped full covariance fit
    // sits at 6,089.
    let r = 0.999_8;
    let thousand = vec![vec![1.0, r], vec![r, 1.0]];
    assert!(condition_number(&thousand) > 1e3 && condition_number(&thousand) < CONDITION_MAX);
    let (shipped, _) = ship(&[single(thousand.clone())]).unwrap();
    assert_eq!(shipped[0].covariance, thousand);
}

#[test]
fn icl_is_bic_plus_the_ambiguity_of_the_placement() {
    // Two components on the same centre: every row gets a posterior of one half,
    // so the ambiguity charge is exactly two nats per row and per component.
    let twin = Component {
        weight: 0.5,
        mean: vec![0.0, 0.0],
        covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    };
    let model = Mixture {
        family: Family::Full,
        components: vec![twin.clone(), twin],
        loglik: 0.0,
        iterations: 1,
    };
    let rows = vec![vec![0.0, 0.0]; 10];
    let charge = 2.0 * 10.0 * 2f64.ln();
    let got = model.icl(&rows, 2) - model.bic(10, 2);
    assert!((got - charge).abs() < 1e-9, "{got} against {charge}");
}

#[test]
fn a_clean_partition_pays_almost_nothing_for_ambiguity() {
    let rows = blobs();
    let fit = Mixture::fit(&rows, 3, Family::Full, 2, 80, 1);
    let bic = fit.bic(rows.len(), 2);
    let icl = fit.icl(&rows, 2);
    assert!(icl >= bic - 1e-9, "ICL {icl} fell below BIC {bic}");
    // Three separated blobs are placed sharply, so ICL is BIC to within a nat.
    assert!(icl - bic < 1.0, "ambiguity charge {:.3}", icl - bic);
}
