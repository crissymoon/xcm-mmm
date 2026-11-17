use super::*;
use crate::segment::mixture::Family;
use std::collections::BTreeMap;

fn entity(id: &str, ads: usize, test_ads: usize, first: &str, last: &str) -> Entity {
    Entity {
        id: id.into(),
        name: id.into(),
        ads,
        test_ads,
        first: first.into(),
        last: last.into(),
        obs: BTreeMap::new(),
    }
}

const ALL: [&str; 3] = ["recency", "span", "test_share"];

fn names() -> Vec<String> {
    vec!["a".into(), "b".into(), "c".into()]
}

#[test]
fn recency_is_read_from_the_panel_anchor_and_not_the_clock() {
    let entities = vec![
        entity("e1", 4, 0, "2026-01-01", "2026-01-10"),
        entity("e2", 2, 1, "2026-01-01", "2026-02-20"),
        entity("e3", 1, 0, "2025-12-01", "2025-12-11"),
    ];
    let block = Block::build(&entities, &ALL).unwrap();
    assert_eq!(block.len(), 3);
    assert_eq!(block.covariates[0].name, "recency");
    // The anchor is the latest last day in the panel, 2026-02-20, so the newest
    // entity reads a recency of one day and the oldest reads 71.
    let expected = [(41.0f64 + 1.0).ln(), 1.0f64.ln(), (71.0f64 + 1.0).ln()];
    let mean = expected.iter().sum::<f64>() / 3.0;
    let sd = (expected
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / 3.0)
        .sqrt();
    for (row, value) in block.z.iter().zip(expected) {
        assert!((row[0] - (value - mean) / sd).abs() < 1e-9);
    }
}

#[test]
fn a_column_that_does_not_vary_is_dropped_rather_than_standardised() {
    let entities: Vec<Entity> = (0..5)
        .map(|i| entity(&format!("e{i}"), 4, i % 3, "2026-01-01", "2026-02-01"))
        .collect();
    let block = Block::build(&entities, &ALL).unwrap();
    assert!(!block.covariates.iter().any(|c| c.name == "recency"));
    assert!(!block.covariates.iter().any(|c| c.name == "span"));
    assert!(block.covariates.iter().any(|c| c.name == "test_share"));
    assert!(block.z.iter().all(|row| row.len() == 1));
}

#[test]
fn a_panel_with_no_descriptive_spread_is_refused() {
    let entities: Vec<Entity> = (0..4)
        .map(|i| entity(&format!("e{i}"), 3, 0, "2026-03-01", "2026-03-09"))
        .collect();
    let err = Block::build(&entities, &ALL).unwrap_err();
    assert!(err.to_string().contains("varies"));
}

#[test]
fn a_gate_learns_which_side_of_a_split_a_row_sits_on() {
    let x = vec![
        vec![1.5],
        vec![1.4],
        vec![1.6],
        vec![-1.5],
        vec![-1.4],
        vec![-1.6],
    ];
    let resp: Vec<Vec<f64>> = (0..6)
        .map(|i| {
            if i < 3 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            }
        })
        .collect();
    let gate = fit_gate(&x, &resp, 2, 1);
    let left = gate.weights(&[1.5]);
    let right = gate.weights(&[-1.5]);
    assert!(left[0] > 0.6 && right[0] < 0.4);
    assert!((left[0] + left[1] - 1.0).abs() < 1e-12);
    // The ridge shrinks the slope, so the reading is a direction and a size,
    // not a probability of certainty.
    assert!(gate.slopes()[0] > 0.3);
}

#[test]
fn a_uniform_gate_scores_exactly_like_the_mixture_it_wraps() {
    let rows = planted(24);
    let mixture = Mixture::fit(&rows, 2, Family::Diagonal, 2, 100, 7);
    let gate = Gate::uniform(2, 1);
    let x = vec![vec![0.0]; rows.len()];
    let gated = Gated::score(&mixture, &gate, &rows, &x);
    let plain = mixture.total_loglik(&rows) / rows.len() as f64;
    assert!((gated - plain).abs() < 1e-9);
}

#[test]
fn the_gated_fit_never_scores_below_the_model_it_starts_from() {
    let mut rows = planted(40);
    let mut x = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter_mut().enumerate() {
        // The covariate repeats the group the density already shows, so the gate
        // has nothing left to add and the fit must not lose anything either.
        if i < 20 {
            row[0] += 0.0;
            x.push(vec![1.0]);
        } else {
            x.push(vec![-1.0]);
        }
    }
    let base = Mixture::fit(&rows, 2, Family::Diagonal, 2, 100, 11);
    let plain = base.total_loglik(&rows) / rows.len() as f64;
    let gated = Gated::refine(base, &rows, &x, 100);
    assert!(
        gated.loglik / rows.len() as f64 >= plain - 1e-6,
        "gated {:.6} plain {:.6}",
        gated.loglik / rows.len() as f64,
        plain
    );
    assert!(gated.rounds > 0);
}

#[test]
fn movement_is_zero_when_the_refinement_cannot_change_the_model() {
    let rows = planted(24);
    let before = Mixture::fit(&rows, 2, Family::Full, 2, 100, 3);
    let after = Gated::refine(before.clone(), &rows, &vec![vec![0.0]; rows.len()], 0);
    let (mean, worst, _) = movement(&before, &after.mixture, &names());
    assert_eq!(mean, 0.0);
    assert_eq!(worst, 0.0);
}

#[test]
fn eta2_is_one_when_the_groups_split_the_column_and_zero_when_they_do_not() {
    let block_x = vec![vec![-1.0], vec![-1.0], vec![1.0], vec![1.0]];
    let labels = vec![0, 0, 1, 1];
    assert!((eta2(&block_x, 0, &labels, 2) - 1.0).abs() < 1e-12);
    let flat = vec![vec![0.5], vec![0.5], vec![0.5], vec![0.5]];
    assert!(eta2(&flat, 0, &labels, 2).abs() < 1e-12);
    let narrow = eta2(&[vec![0.0], vec![0.1], vec![0.2], vec![0.3]], 0, &labels, 2);
    assert!((0.0..1.0).contains(&narrow));
}

#[test]
fn the_verdict_says_which_half_of_the_acceptance_failed() {
    let base = Audit {
        covariates: vec![Covariate {
            name: "recency".into(),
            label: "recency".into(),
            unit: "days".into(),
            eta2: 0.02,
        }],
        delta: 0.004,
        delta_se: 0.003,
        shift_max: 0.02,
        added: true,
        steady: true,
        ..Audit::default()
    };
    let mut audit = base.clone();
    audit.verdict = verdict(&audit);
    assert!(audit.verdict.contains("supports it"));

    let mut thin = base.clone();
    thin.added = false;
    thin.delta = 0.001;
    thin.verdict = verdict(&thin);
    assert!(thin.verdict.contains("not a gain"));
    assert!(!thin.honest());

    let mut moved = base;
    moved.shift_max = 0.4;
    moved.steady = false;
    moved.verdict = verdict(&moved);
    assert!(moved.verdict.contains("mean"));
    assert!(!moved.honest());
}

/// Two groups, separated on the first column, with a shared width on the rest.
fn planted(n: usize) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let wobble = (i as f64 * 0.37).sin() * 0.2;
        rows.push(vec![2.0 * side + wobble, 0.5 * side, wobble]);
    }
    rows
}
