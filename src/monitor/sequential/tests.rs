use super::*;

fn config(delta: f64) -> Config {
    Config {
        bound: Bound::Variance,
        delta,
        min_sub: 4,
        max_buckets: 64,
    }
}

#[test]
fn a_constant_series_never_cuts_and_keeps_its_mean() {
    let mut adwin = Adwin::new(&config(1e-6));
    for _ in 0..500 {
        assert!(adwin.push(0.4).expect("in range").is_none());
    }
    assert_eq!(adwin.width(), 500);
    assert!((adwin.mean() - 0.4).abs() < 1e-12);
    assert_eq!(adwin.cuts(), 0);
}

#[test]
fn a_step_is_cut_and_the_sides_read_as_the_step() {
    let mut adwin = Adwin::new(&config(1e-3));
    let mut found = None;
    for at in 0..2000 {
        let value = if at < 1000 { 0.20 } else { 0.60 };
        if let Some(cut) = adwin.push(value).expect("in range") {
            found = found.or(Some(cut));
        }
    }
    let cut = found.expect("a step of 0.4 is a change");
    assert!((cut.after - 0.60).abs() < 0.05, "{cut:?}");
    assert!((cut.before - 0.20).abs() < 0.10, "{cut:?}");
    assert!(cut.difference() > 0.0);
    assert_eq!(cut.dropped + cut.kept, cut.at);
    assert_eq!(cut.start(), cut.at - cut.dropped + 1);
}

#[test]
fn the_bound_needs_a_rate_in_the_unit_interval() {
    let mut adwin = Adwin::new(&config(1e-6));
    assert!(adwin.push(1.5).is_err());
    assert!(adwin.push(-0.1).is_err());
    assert!(adwin.push(f64::NAN).is_err());
    assert!(adwin.push(0.5).is_ok());
}

#[test]
fn the_range_bound_is_blind_where_the_variance_bound_is_not() {
    // The lesson the module documents: on a rare rate the known range is
    // the wrong range, so the same step is invisible to Hoeffding at a
    // length where the variance bound sees it.
    let mut narrow = Adwin::new(&Config {
        bound: Bound::Range,
        ..config(0.05)
    });
    let mut wide = Adwin::new(&Config {
        bound: Bound::Variance,
        ..config(0.05)
    });
    for at in 0..4000 {
        let value = if at < 2000 { 0.02 } else { 0.06 };
        narrow.push(value).expect("in range");
        wide.push(value).expect("in range");
    }
    assert!(wide.cuts() > 0);
    assert_eq!(narrow.cuts(), 0);
}

#[test]
fn a_stationary_stream_holds_the_budget() {
    let config = CalibrationConfig {
        stationary: Stream {
            rate: 0.05,
            batch_rows: 200,
            batches: 300,
            trials: 20,
        },
        shift: None,
        grid: vec![0.001, 0.05, 0.5],
        // Twenty trials hold a 0.16 upper bound at best, so a budget of
        // 0.05 could not be established by this measurement. The budget
        // here is set where twenty trials can speak to it.
        target_share: 0.2,
        permutations: 20,
        ..CalibrationConfig::default()
    };
    let calibration = calibrate(&config);
    assert_eq!(calibration.points.len(), 3);
    let chosen = &calibration.points[calibration.chosen];
    assert!(chosen.high <= config.target_share, "{chosen:?}");
    assert!(chosen.delta <= config.target_share, "{chosen:?}");
    for point in &calibration.points {
        assert_eq!(point.raised, 0, "{point:?}");
        assert_eq!(point.share, 0.0);
    }
}

#[test]
fn a_shift_is_detected_and_a_stationary_series_is_not_more_structured() {
    let config = CalibrationConfig {
        stationary: Stream {
            rate: 0.05,
            batch_rows: 200,
            batches: 600,
            trials: 20,
        },
        shift: Some(Shift {
            to: 0.40,
            at: 400,
            batches: 1200,
            trials: 10,
        }),
        grid: vec![0.001, 0.05],
        target_share: 0.05,
        permutations: 40,
        ..CalibrationConfig::default()
    };
    let calibration = calibrate(&config);
    let point = &calibration.points[calibration.chosen];
    assert!(point.detected > 0.5, "{point:?}");
    assert!(point.delay.is_some());
}

#[test]
fn the_refusal_prices_a_per_event_alarm() {
    let out = refusal(0.02, 200, 0.25, 0.05, 0.80);
    assert!((out.events - 4.0).abs() < 1e-9);
    // One batch of two hundred at two percent: sqrt(0.98 / 4) as a share
    // of the level, which is 49.5 percent. One label: sqrt(0.98 / 0.02),
    // which is 700 percent, and that is why a per-event rule is refused.
    assert!((out.relative_se - 0.495).abs() < 0.001, "{out:?}");
    assert!((out.event_se - 7.0).abs() < 0.01, "{out:?}");
    assert!(out.batches > 60.0 && out.batches < 70.0, "{out:?}");
    assert!(out.verdict.contains("refused"));
}

#[test]
fn shuffle_separates_an_ordered_series_from_its_own_shuffles() {
    let ordered: Vec<f64> = (0..1600)
        .map(|at| if at < 800 { 0.90 } else { 0.10 })
        .collect();
    let flat = vec![0.2; 1600];
    let out = shuffle(&ordered, &config(1e-3), 40, 7).expect("in range");
    assert!(out.observed > out.high, "{out:?}");
    assert!(out.p_value < 0.05, "{out:?}");
    let quiet = shuffle(&flat, &config(1e-3), 40, 7).expect("in range");
    assert_eq!(quiet.observed, 0);
    assert_eq!(quiet.p_value, 1.0);
}

#[test]
fn a_short_series_is_refused_with_the_length_it_needs() {
    let err = reading(&[0.1, 0.2, 0.3], &config(1e-6)).unwrap_err();
    assert!(err.contains("at least 8"), "{err}");
}

#[test]
fn a_report_carries_its_calibration_and_its_refusal() {
    let series: Vec<f64> = (0..100)
        .map(|at| if at < 50 { 0.30 } else { 0.05 })
        .collect();
    let config = CalibrationConfig {
        stationary: Stream {
            rate: 0.05,
            batch_rows: 200,
            batches: 200,
            trials: 10,
        },
        shift: Some(Shift {
            to: 0.15,
            at: 100,
            batches: 400,
            trials: 10,
        }),
        grid: vec![0.001, 0.05],
        target_share: 0.05,
        permutations: 20,
        ..CalibrationConfig::default()
    };
    let out = report(&series, &config).expect("in range");
    assert!(!out.calibration.points.is_empty());
    assert_eq!(out.bound, "variance");
    // A hundred points is too short for a uniform bound at any level, and
    // the note says so rather than pretending the absence of a cut is
    // evidence of stationarity.
    assert!(out.note.contains("batches"), "{}", out.note);
    assert!(out.refusal.batches > 0.0);
    let text = serde_json::to_string(&out).expect("serialises");
    assert!(text.contains("\"delta\""));
}
