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
