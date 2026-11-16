//! The choice's own tests: the rule, the gates and the account it writes.

use super::*;
use crate::segment::feature::{self, Matrix};
use crate::segment::mixture::Family;

/// Two thin arms that cross, plus a third group of a different size off to one
/// side. A sphere cannot describe the arms, and the sizes are unequal so the
/// share floor has something to do.
fn panel() -> (Matrix, Vec<Vec<f64>>) {
    let mut rows = Vec::new();
    for i in 0..10 {
        let t = -2.0 + 4.0 * i as f64 / 9.0;
        rows.push(vec![t, t, 1.0]);
    }
    for i in 0..40 {
        let t = -2.0 + 4.0 * i as f64 / 39.0;
        rows.push(vec![t, -t, 1.0]);
    }
    for i in 0..40 {
        let t = i as f64 * 0.01;
        rows.push(vec![0.5 * t, 0.6 * t, 6.0]);
    }
    let specs = feature::specs().into_iter().take(3).collect::<Vec<_>>();
    let mut z = rows.clone();
    for column in 0..3 {
        let values: Vec<f64> = z.iter().map(|row| row[column]).collect();
        let mu = values.iter().sum::<f64>() / values.len() as f64;
        let sd = (values.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / values.len() as f64)
            .sqrt()
            .max(1e-9);
        for row in z.iter_mut() {
            row[column] = ((row[column] - mu) / sd).clamp(-3.0, 3.0);
        }
    }
    let matrix = Matrix {
        specs,
        raw: rows,
        z: z.clone(),
        scaling: feature::Scaling {
            mean: vec![0.0; 3],
            sd: vec![1.0; 3],
            clip: 3.0,
        },
        flat: Vec::new(),
    };
    (matrix, z)
}

fn config() -> SegmentConfig {
    SegmentConfig {
        k_min: 2,
        k_max: 3,
        min_entities: 30,
        folds: 4,
        restarts: 6,
        em_restarts: 4,
        iterations: 120,
        min_share: 0.05,
        seed: 20261114,
        ..Default::default()
    }
}

#[test]
fn the_choice_finds_the_crossing_arms_a_sphere_cannot_bend_around() {
    let (matrix, rows) = panel();
    let choice = choose(&rows, &matrix, &config()).unwrap();
    assert_eq!(
        choice.family,
        Family::Full,
        "chose {}",
        choice.family.name()
    );
    assert_eq!(choice.k, 3);
    assert!(
        choice.heldout > choice.baseline.heldout + 0.5,
        "{:.3} against a k-means baseline of {:.3}",
        choice.heldout,
        choice.baseline.heldout
    );
    assert_eq!(choice.candidates.len(), 6);
    assert_eq!(choice.folds, 4);
}

#[test]
fn every_candidate_carries_a_verdict_and_exactly_one_says_chosen() {
    let (matrix, rows) = panel();
    let choice = choose(&rows, &matrix, &config()).unwrap();
    assert!(choice.candidates.iter().all(|c| !c.verdict.is_empty()));
    let chosen = choice
        .candidates
        .iter()
        .filter(|c| c.verdict.starts_with("chosen"))
        .count();
    assert_eq!(chosen, 1);
    let winner = choice
        .candidates
        .iter()
        .find(|c| c.verdict.starts_with("chosen"))
        .unwrap();
    assert_eq!((winner.family, winner.k), (choice.family, choice.k));
    assert!(choice.reason.contains(winner.family.name()));
    assert!(choice.reason.contains("nats per entity"));
}

#[test]
fn the_share_floor_sets_a_row_aside_rather_than_letting_it_win() {
    let (matrix, rows) = panel();
    let mut cfg = config();
    // The smallest group here is a tenth of the panel, so two in ten refuses the
    // partitions that keep it and leaves the ones that merge it.
    cfg.min_share = 0.2;
    let choice = choose(&rows, &matrix, &cfg).unwrap();
    let aside = choice
        .candidates
        .iter()
        .filter(|c| c.min_share + 1e-12 < cfg.min_share)
        .count();
    assert!(aside > 0, "no candidate was set aside by the floor");
    for candidate in &choice.candidates {
        if candidate.min_share + 1e-12 < cfg.min_share {
            assert!(
                candidate.verdict.contains("below the"),
                "{}",
                candidate.verdict
            );
        }
    }
    assert!(choice.candidates[0].min_share > 0.0);
}

#[test]
fn a_floor_no_partition_can_meet_is_refused_with_the_reason() {
    let (matrix, rows) = panel();
    let mut cfg = config();
    cfg.min_share = 0.6;
    let err = choose(&rows, &matrix, &cfg).unwrap_err();
    assert!(err.to_string().contains("widest smallest group"), "{err}");
}

#[test]
fn a_declared_family_is_obeyed_and_says_so() {
    let (matrix, rows) = panel();
    let mut cfg = config();
    cfg.family = Some(Family::Spherical);
    let choice = choose(&rows, &matrix, &cfg).unwrap();
    assert_eq!(choice.family, Family::Spherical);
    assert!(choice.reason.contains("declared"));
    assert!(choice
        .candidates
        .iter()
        .filter(|c| c.family != Family::Spherical)
        .all(|c| c.verdict.contains("declared")));
}

#[test]
fn a_margin_wider_than_any_gain_keeps_the_simplest_family() {
    let (matrix, rows) = panel();
    let mut cfg = config();
    cfg.min_gain = 1_000.0;
    let choice = choose(&rows, &matrix, &cfg).unwrap();
    assert_eq!(choice.family, Family::Spherical);
    assert!(choice.reason.contains("margin"));
}

#[test]
fn the_incumbent_is_scored_on_the_same_scale_as_the_candidates() {
    let (matrix, rows) = panel();
    let choice = choose(&rows, &matrix, &config()).unwrap();
    assert!(choice.baseline.heldout.is_finite());
    assert_eq!(choice.baseline.sizes.iter().sum::<usize>(), rows.len());
    assert!(!choice.sweep.is_empty());
    assert!(choice
        .sweep
        .iter()
        .all(|point| point.silhouette.is_finite()));
}

#[test]
fn dependence_names_the_columns_that_move_together() {
    let rows = vec![
        vec![0.0, 1.0, 5.0],
        vec![1.0, 2.0, 5.0],
        vec![2.0, 3.0, 5.0],
        vec![3.0, 4.0, 5.0],
    ];
    let names: Vec<String> = ["ads", "cadence", "terms"]
        .iter()
        .map(|n| n.to_string())
        .collect();
    let got = dependence(&rows, &names);
    // One pair is perfectly aligned and the constant column pairs with nothing.
    assert!(
        (got.mean_abs - 1.0 / 3.0).abs() < 1e-9,
        "{:.3}",
        got.mean_abs
    );
    assert_eq!(got.strongest[0].a, "ads");
    assert_eq!(got.strongest[0].b, "cadence");
    assert!((got.strongest[0].r - 1.0).abs() < 1e-9);
}

#[test]
fn a_panel_too_small_to_choose_is_refused() {
    let (matrix, rows) = panel();
    let mut cfg = config();
    cfg.min_entities = 10_000;
    let err = choose(&rows, &matrix, &cfg).unwrap_err();
    assert!(err.to_string().contains("enough rows"), "{err}");
}
