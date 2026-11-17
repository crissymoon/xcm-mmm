//! The pattern mixture's own tests, beside the fit they exercise.

use super::*;
use crate::rng::Rng;

fn spec(column: usize, atom: f64, share: f64) -> GateSpec {
    GateSpec {
        name: if column == 0 { "repeat" } else { "other" }.into(),
        column,
        boundary: if atom < 0.0 { 0.0 } else { 1.0 },
        atom,
        share,
    }
}

/// A panel drawn from a pattern mixture: two groups, a correlated inside and an
/// edge whose probability differs by group.
fn drawn(n: usize, seed: u64) -> (Vec<Vec<f64>>, usize, usize) {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(n);
    let (mut edge, mut inside) = (0, 0);
    for i in 0..n {
        let group = i % 2;
        let (shift, rate) = if group == 0 {
            (0.6, 0.55)
        } else {
            (-0.6, 0.25)
        };
        let first = shift + rng.normal(0.0, 1.0);
        let second = -shift + 0.5 * first + rng.normal(0.0, 0.8);
        let on = rng.chance(rate);
        let share = if on {
            edge += 1;
            -0.455
        } else {
            inside += 1;
            0.6 * first + 0.3 * second + rng.normal(0.0, 0.4)
        };
        rows.push(vec![share, first, second, rng.normal(0.0, 0.7)]);
    }
    (rows, edge, inside)
}

/// The maximisation step keeps producing a second moment. The first version
/// added both orders of the conditional covariance to one upper-triangle cell,
/// which doubled the off diagonal, made the average indefinite and ended every
/// fit after a single step without saying so.
#[test]
fn the_fit_runs_until_the_likelihood_settles() {
    let (rows, _, _) = drawn(160, 41);
    let specs = vec![spec(0, -0.455, 0.4)];
    let fitted = Pattern::fit(&rows, &specs, 2, 2, 200, 41);
    assert!(
        fitted.iterations > 5,
        "the fit stopped after {} steps, which is the indefinite-covariance signature",
        fitted.iterations
    );
    let views: Vec<View> = rows.iter().map(|row| View::of(row, &specs, 4)).collect();
    let (_, total) = e_step(
        &fitted.components,
        &fitted.atom,
        &fitted.gates,
        &views,
        &rows,
    );
    assert!((fitted.loglik - total).abs() < 1e-6);
}

/// On a panel drawn from this model, the pattern mixture matches the gate fitted
/// to the partition. If it lost here the comparison on the real panel would be
/// measuring a defect rather than a model difference.
#[test]
fn the_correctly_specified_model_does_not_lose_its_own_data() {
    let (rows, edge, inside) = drawn(200, 53);
    assert!(edge > 20 && inside > 20);
    let specs = vec![spec(0, -0.455, 0.4)];
    let fitted = Pattern::fit(&rows, &specs, 2, 4, 200, 53);
    let mixture = Mixture::fit(&rows, 2, Family::Full, 4, 200, 53);
    let columns = vec![1, 2, 3];
    let origin = vec![0.0, 0.0, 0.5];
    let resp = gate::responsibilities(&rows, &mixture.components, &columns);
    let bars = gate::bars(&rows, &resp, &specs, &columns, &origin);
    let fit = bars.first().expect("a gate");
    let scale = Scaling {
        mean: vec![0.0, 0.0, 0.0, 0.0],
        sd: vec![1.0, 1.0, 1.0, 1.0],
        clip: 3.0,
        clip_exempt: Vec::new(),
    };
    let spacing = gate::lattice(100.0, &specs[0], &scale);
    let mut two = 0.0;
    let mut pattern = 0.0;
    for (i, row) in rows.iter().enumerate() {
        let scored = gate::scored(
            row,
            spacing,
            &specs[0],
            fit,
            &mixture.components,
            &resp[i],
            &columns,
            &scale,
        );
        two += scored.gate.max(1e-12).ln();
        pattern += fitted.score(row, &specs[0], &columns, &scale, spacing);
    }
    let gap = (two - pattern) / rows.len() as f64;
    assert!(
        gap < 0.05,
        "the pattern mixture loses {gap:.3} nats per entity on its own data"
    );
}

/// A panel where one reading piles on an edge and the rest are a tight blob.
fn gated(n: usize, edge: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let on = (i as f64) < edge * n as f64;
            let share = if on { 0.0 } else { rng.uniform(0.2, 0.9) };
            vec![
                share,
                rng.normal(0.0, 1.0),
                rng.normal(0.5, 1.2),
                rng.normal(-0.3, 0.8),
            ]
        })
        .collect()
}

fn scaling(dim: usize) -> Scaling {
    Scaling {
        mean: vec![0.4; dim],
        sd: vec![1.0; dim],
        clip: 3.0,
        clip_exempt: Vec::new(),
    }
}

/// With no gate the maximisation step is the complete-data one, exactly.
#[test]
fn a_panel_with_no_edge_reduces_to_the_standard_step() {
    let rows = gated(60, 0.0, 11);
    let views: Vec<View> = rows.iter().map(|row| View::of(row, &[], 4)).collect();
    let labels: Vec<usize> = (0..rows.len()).map(|i| i % 2).collect();
    let resp = one_hot(&labels, 2);
    let plain = Mixture::step(&rows, &resp, 2, Family::Full);
    let (components, _) = m_step(&rows, &views, &resp, 2, &[], &plain.components);
    assert_eq!(components.len(), plain.components.len());
    for (ours, theirs) in components.iter().zip(&plain.components) {
        assert!((ours.weight - theirs.weight).abs() < 1e-9);
        for (a, b) in ours.mean.iter().zip(&theirs.mean) {
            assert!((a - b).abs() < 1e-9, "{a} against {b}");
        }
        for (row_a, row_b) in ours.covariance.iter().zip(&theirs.covariance) {
            for (a, b) in row_a.iter().zip(row_b) {
                assert!((a - b).abs() < 1e-9, "{a} against {b}");
            }
        }
    }
}

/// An edge row contributes its atom probability and the marginal density over the
/// readings it shows, which is the whole point of the model.
#[test]
fn an_edge_row_is_scored_on_the_readings_it_shows() {
    let rows = gated(40, 0.5, 3);
    let specs = vec![spec(0, 0.0, 0.5)];
    let views: Vec<View> = rows.iter().map(|row| View::of(row, &specs, 4)).collect();
    let components = vec![Component {
        weight: 1.0,
        mean: vec![0.0; 4],
        covariance: {
            let mut cov = vec![vec![0.0; 4]; 4];
            for (d, row) in cov.iter_mut().enumerate() {
                row[d] = 1.0;
            }
            cov
        },
    }];
    let atom = vec![vec![0.6]];
    let (_, total) = e_step(&components, &atom, &[0], &views, &rows);
    let mut expected = 0.0;
    for (row, view) in rows.iter().zip(&views) {
        let mass = if view.missing.is_empty() {
            (1.0f64 - 0.6).ln()
        } else {
            0.6f64.ln()
        };
        expected += mass
            + gate::Marginal::build(&components[0], &view.observed)
                .unwrap()
                .log_density(row);
    }
    assert!(
        (total - expected).abs() < 1e-9,
        "{total} against {expected}"
    );
}

/// The atom parameter is the edge share of the group it belongs to.
#[test]
fn the_fitted_atom_is_the_edge_share() {
    let rows = gated(120, 0.4, 5);
    let specs = vec![spec(0, 0.0, 0.4)];
    let fitted = Pattern::fit(&rows, &specs, 1, 2, 200, 7);
    assert_eq!(fitted.components.len(), 1);
    let atom = fitted.atom[0][0];
    assert!((atom - 0.4).abs() < 0.05, "atom {atom} against 0.4");
}

/// The fitted likelihood is the likelihood of the model it returns: one
/// expectation step at the shipped parameters reproduces the stored number.
#[test]
fn the_stored_likelihood_is_the_models_own() {
    let rows = gated(90, 0.4, 13);
    let specs = vec![spec(0, 0.0, 0.4)];
    let fitted = Pattern::fit(&rows, &specs, 2, 2, 120, 21);
    let views: Vec<View> = rows.iter().map(|row| View::of(row, &specs, 4)).collect();
    let (_, total) = e_step(
        &fitted.components,
        &fitted.atom,
        &fitted.gates,
        &views,
        &rows,
    );
    assert!(
        (fitted.loglik - total).abs() < 1e-6,
        "{} against {total}",
        fitted.loglik
    );
}

/// The model is deterministic from its seed, which is what lets a fold be
/// reproduced from the card.
#[test]
fn the_fit_is_reproducible() {
    let rows = gated(90, 0.4, 17);
    let specs = vec![spec(0, 0.0, 0.4)];
    let one = Pattern::fit(&rows, &specs, 2, 2, 80, 99);
    let two = Pattern::fit(&rows, &specs, 2, 2, 80, 99);
    assert_eq!(one.loglik, two.loglik);
    assert_eq!(one.atom, two.atom);
    assert_eq!(one.components, two.components);
}

/// Both branches of the score are probabilities, so both are non-positive on the
/// log scale and finite.
#[test]
fn the_score_is_a_probability_on_both_branches() {
    let rows = gated(80, 0.5, 23);
    let specs = vec![spec(0, 0.0, 0.5)];
    let fitted = Pattern::fit(&rows, &specs, 2, 2, 120, 3);
    let scale = scaling(4);
    let columns: Vec<usize> = (1..4).collect();
    let mut edge = 0;
    let mut inside = 0;
    for row in &rows {
        let log = fitted.score(row, &specs[0], &columns, &scale, 0.05);
        assert!(log <= 0.0 && log.is_finite(), "log probability {log}");
        if specs[0].holds(row) {
            edge += 1;
        } else {
            inside += 1;
        }
    }
    assert!(
        edge > 0 && inside > 0,
        "the panel has to show both branches"
    );
}

/// A gate that is uncertain says so: the same row scores higher on the edge when
/// the fitted atom is high than when it is low.
#[test]
fn the_score_follows_the_atom() {
    let rows = gated(80, 0.5, 29);
    let specs = vec![spec(0, 0.0, 0.5)];
    let fitted = Pattern::fit(&rows, &specs, 1, 2, 120, 31);
    let scale = scaling(4);
    let columns: Vec<usize> = (1..4).collect();
    let edge_row = rows.iter().find(|row| specs[0].holds(row)).unwrap();
    let mut raised = fitted.clone();
    raised.atom[0][0] = 0.9;
    let mut lowered = fitted.clone();
    lowered.atom[0][0] = 0.1;
    let high = raised.score(edge_row, &specs[0], &columns, &scale, 0.05);
    let low = lowered.score(edge_row, &specs[0], &columns, &scale, 0.05);
    assert!(high > low, "{high} against {low}");
}

/// The conditional mean and covariance follow the textbook formulas.
#[test]
fn the_imputation_is_the_conditional_law() {
    let component = Component {
        weight: 1.0,
        mean: vec![1.0, 2.0],
        covariance: vec![vec![4.0, 1.5], vec![1.5, 1.0]],
    };
    let impute = Impute::build(&component, &[1], &[0]).unwrap();
    let (mean, cov) = impute.apply(&[0.0, 3.0], &[1]);
    // The conditioning block carries the numerical floor, so its variance is
    // 1 + FLOOR: mean = 1 + 1.5 * 1 / 1.001, variance = 4 - 1.5^2 / 1.001.
    let held = 1.0 + FLOOR;
    assert!((mean[0] - (1.0 + 1.5 / held)).abs() < 1e-9, "{}", mean[0]);
    assert!(
        (cov[0][0] - (4.0 - 2.25 / held)).abs() < 1e-9,
        "{}",
        cov[0][0]
    );
}

/// The verdict gate is two standard errors, and a single fold is not measured.
#[test]
fn the_verdict_gate_holds_its_line() {
    let spec = spec(0, 0.0, 0.5);
    let ledger = |two: &[f64], pattern: &[f64]| Ledger {
        two: two.to_vec(),
        pattern: pattern.to_vec(),
        ..Ledger::default()
    };
    let cleared = verdict(
        &spec,
        &ledger(&[-1.2, -1.1, -1.3, -1.15], &[-0.6, -0.45, -0.7, -0.5]),
    );
    assert!(cleared.cleared(), "{}", cleared.verdict);
    let refused = verdict(&spec, &ledger(&[-1.2, -1.1, -1.3], &[-1.1, -1.3, -1.05]));
    assert!(!refused.cleared(), "{}", refused.verdict);
    let single = verdict(&spec, &ledger(&[-1.2], &[-0.6]));
    assert!(!single.cleared());
    assert!(single.verdict.starts_with("not measured"));
}
