//! The known cases both estimators have to reproduce.

use super::*;

fn ids(n: usize) -> Vec<String> {
    (0..n).map(|at| format!("c{at}")).collect()
}

#[test]
fn a_clean_bank_reads_no_noise_and_a_ceiling_of_one() {
    let mut scores = vec![0.99; 50];
    scores.extend(vec![0.01; 50]);
    let mut labels = vec![true; 50];
    labels.extend(vec![false; 50]);
    let reading = audit(&ids(100), &scores, &labels, &AuditConfig::default());
    assert!(reading.usable());
    assert_eq!(reading.suspected, 0);
    assert_eq!(reading.estimated_errors, 0);
    assert!((reading.noise - 0.0).abs() < 1e-9);
    assert!((reading.ceiling - 1.0).abs() < 1e-9);
    assert_eq!(
        reading.confident[confident::NEGATIVE][confident::POSITIVE],
        0
    );
}

#[test]
fn injected_noise_comes_back_at_the_rate_it_was_injected() {
    // 100 rows of each class, all scored confidently and correctly, then every
    // tenth label flipped. The audit should recover a tenth.
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    for at in 0..200 {
        let positive = at < 100;
        let flipped = at % 10 == 0;
        scores.push(if positive { 0.9 } else { 0.1 });
        labels.push(if flipped { !positive } else { positive });
    }
    let reading = audit(&ids(200), &scores, &labels, &AuditConfig::default());
    assert!(reading.usable());
    assert_eq!(reading.suspected, 20);
    assert_eq!(reading.estimated_errors, 20);
    assert!((reading.noise - 0.1).abs() < 1e-9, "{}", reading.noise);
    assert!((reading.noise_positive - 0.1).abs() < 1e-9);
    assert!((reading.noise_negative - 0.1).abs() < 1e-9);
    assert!((reading.ceiling - 0.9).abs() < 1e-9);
    // Worst margin first. Every injected error ties at -0.8: a row labelled
    // positive that the model scores at 0.1, or the mirror. The suspect list is
    // exactly the flipped rows, at the top of the ordering.
    assert_eq!(reading.suspects.len(), AuditConfig::default().top);
    assert!(reading
        .suspects
        .iter()
        .all(|s| (s.margin + 0.8).abs() < 1e-9));
    assert!(reading.suspects.iter().all(|s| {
        s.id.strip_prefix('c')
            .and_then(|at| at.parse::<usize>().ok())
            .is_some_and(|at| at % 10 == 0)
    }));
}

#[test]
fn an_uninformative_model_cannot_audit_a_bank() {
    // Every row scored the same. The confident joint puts all rows in one class
    // and the ceiling collapses to that class's share. This is why the audit is
    // quoted beside the AUC rather than instead of it.
    let scores = vec![0.5; 100];
    let mut labels = vec![true; 50];
    labels.extend(vec![false; 50]);
    let reading = audit(&ids(100), &scores, &labels, &AuditConfig::default());
    assert!(reading.usable());
    assert!((reading.noise_negative - 1.0).abs() < 1e-9);
    assert!((reading.noise_positive - 0.0).abs() < 1e-9);
    assert!((reading.ceiling - 0.5).abs() < 1e-9);
}

#[test]
fn the_joint_is_scaled_back_to_the_labelled_totals() {
    // The positive threshold is the mean of 0.95, 0.9 and 0.5, the negative one
    // the mean of 0.9 and 0.95, so the row at 0.5 and the row at 0.1 clear
    // neither. The joint keeps the class totals of 3 and 2 and leaves both out.
    let scores = vec![0.95, 0.9, 0.5, 0.1, 0.05];
    let labels = vec![true, true, true, false, false];
    let reading = audit(&ids(5), &scores, &labels, &AuditConfig::default());
    assert!(reading.usable(), "{:?}", reading.refused);
    assert_eq!(reading.unconfident, 2);
    let negatives: f64 = reading.joint[confident::NEGATIVE].iter().sum();
    let positives: f64 = reading.joint[confident::POSITIVE].iter().sum();
    assert!((negatives - 2.0).abs() < 1e-6, "{negatives}");
    assert!((positives - 3.0).abs() < 1e-6, "{positives}");
    assert!(reading.estimated_errors >= reading.suspected);
}

#[test]
fn a_margin_is_not_a_probability_and_is_refused() {
    let scores = vec![0.2, 1.7, -3.0, 0.4];
    let labels = vec![true, false, true, false];
    let reading = audit(&ids(4), &scores, &labels, &AuditConfig::default());
    assert!(!reading.usable());
    assert!(reading
        .refused
        .as_deref()
        .is_some_and(|why| why.contains("probability")));
}

#[test]
fn one_class_and_mismatched_lengths_are_refused_by_name() {
    let single = audit(
        &ids(4),
        &[0.2, 0.3, 0.4, 0.5],
        &[true; 4],
        &AuditConfig::default(),
    );
    assert!(!single.usable());
    assert!(single
        .refused
        .as_deref()
        .is_some_and(|why| why.contains("no threshold")));

    let mixed = audit(
        &ids(2),
        &[0.2, 0.3, 0.4],
        &[true, false],
        &AuditConfig::default(),
    );
    assert!(!mixed.usable());
    assert!(mixed
        .refused
        .as_deref()
        .is_some_and(|why| why.contains("3 scores")));
}

#[test]
fn the_hanley_mcneil_error_matches_a_hand_computation() {
    // A = 0.867, 50 positives, 200 negatives: q1 = 0.76515, q2 = 0.80524, and
    // the variance is 0.001143, so the standard error is 0.0338.
    let se = power::hanley_mcneil(0.867, 50, 200);
    assert!((se - 0.0338).abs() < 5e-4, "{se}");
    // A perfect reading has no variance left to estimate.
    assert!(power::hanley_mcneil(1.0, 50, 200) < 0.01);
    assert_eq!(power::hanley_mcneil(0.7, 0, 200), 0.0);
}

#[test]
fn the_detectable_gain_falls_as_the_square_root_of_the_rows() {
    let small = detectable_share(0.85, 20, 0.03, 0.80, 0.95);
    let large = detectable_share(0.85, 80, 0.03, 0.80, 0.95);
    assert!((small.detectable / large.detectable - 2.0).abs() < 0.05);
    assert!(!small.resolvable);
    // Resolving a three point gain at this accuracy takes more than a hundred
    // times the rows in hand, which is the number the card quotes.
    assert!(small.rows_for_floor > 2_000, "{}", small.rows_for_floor);
}

#[test]
fn the_quoted_row_count_actually_resolves_the_floor() {
    let reading = detectable_share(0.85, 20, 0.03, 0.80, 0.95);
    let rerun = detectable_share(0.85, reading.rows_for_floor, 0.03, 0.80, 0.95);
    assert!(rerun.resolvable, "{} at {}", rerun.detectable, rerun.rows);
    assert!(
        (rerun.detectable - 0.03).abs() < 1e-3,
        "{}",
        rerun.detectable
    );
}

#[test]
fn a_large_bank_resolves_the_floor_and_says_so() {
    let reading = detectable_auc(0.867, 1869, 5174, 0.03, 0.80, 0.95);
    assert!(reading.resolvable);
    assert!(reading.detectable < 0.03);
    assert!(reading.se < 0.01);
    assert!(reading.note.contains("sits above it"));
}
