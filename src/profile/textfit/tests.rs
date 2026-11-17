//! The learned copy map's own tests, beside the fit they exercise.

use super::*;
use crate::segment::model::Source as ModelSource;
use crate::segment::SegmentConfig;
use crate::segment::{ProfileRow, Segment};

fn artifact() -> Artifact {
    let segment = |id: usize, label: &str, z: f64| Segment {
        id,
        label: label.to_string(),
        size: 50,
        share: 0.5,
        profile: vec![ProfileRow {
            name: "ads".into(),
            label: "ads".into(),
            family: "volume".into(),
            unit: "ads".into(),
            z,
            mean: 10.0,
            atom: None,
            inside: None,
        }],
    };
    Artifact {
        name: "t".into(),
        version: 1,
        built_at: String::new(),
        source: ModelSource::default(),
        config: SegmentConfig::default(),
        features: Vec::new(),
        scaling: crate::segment::Scaling {
            mean: Vec::new(),
            sd: Vec::new(),
            clip: 3.0,
            clip_exempt: Vec::new(),
        },
        k: 2,
        family: crate::segment::Family::Spherical,
        components: Vec::new(),
        gates: Vec::new(),
        centroids: Vec::new(),
        segments: vec![segment(0, "left", 1.0), segment(1, "right", -1.0)],
        assignments: BTreeMap::new(),
        metrics: crate::segment::Metrics {
            entities: 0,
            features: 0,
            k: 2,
            family: crate::segment::Family::Spherical,
            heldout: 0.0,
            heldout_sd: 0.0,
            agreement: 0.0,
            folds: 2,
            parameters: 0,
            gain: 0.0,
            choice: String::new(),
            candidates: Vec::new(),
            baseline: crate::segment::Baseline {
                k: 2,
                heldout: 0.0,
                silhouette: 0.0,
                sizes: Vec::new(),
            },
            likelihood_control: crate::segment::Shuffle::default(),
            dependence: crate::segment::Dependence::default(),
            inertia: 0.0,
            silhouette: 0.0,
            min_share: 0.0,
            sweep: Vec::new(),
            stability: crate::segment::stability::Stability {
                bootstraps: 0,
                mean: 0.0,
                p10: 0.0,
                min: 0.0,
                cluster: Vec::new(),
            },
            control: crate::segment::stability::Control {
                runs: 0,
                mean: 0.0,
                max: 0.0,
            },
            flat: Vec::new(),
            gates: Vec::new(),
            sharpness: crate::segment::Sharpness::default(),
            concomitant: crate::segment::ConcomitantAudit::default(),
            concomitant_all: crate::segment::ConcomitantAudit::default(),
        },
    }
}

/// Two classes separated exactly by the first column.
fn separable(rows: usize) -> Vec<FeatureRow> {
    (0..rows)
        .map(|i| {
            let segment = i % 2;
            let mut x = vec![0.0; columns("ad").len()];
            x[0] = if segment == 0 { -2.0 } else { 2.0 };
            FeatureRow {
                page_id: format!("page-{i:03}"),
                segment,
                split: if i % 5 == 0 {
                    "test".into()
                } else {
                    "train".into()
                },
                x,
            }
        })
        .collect()
}

fn cfg() -> TextConfig {
    TextConfig {
        iterations: 200,
        controls: 2,
        min_rows: 10,
        ..TextConfig::default()
    }
}

#[test]
fn a_separable_panel_is_recovered_and_cleared() {
    let rows = separable(80);
    let fitted = fit_unit(
        "ad",
        &rows,
        &artifact()
            .segments
            .iter()
            .map(|s| ClassRow {
                id: s.id,
                label: s.label.clone(),
                share: s.share,
                train: 0,
                recall: 0.0,
            })
            .collect::<Vec<_>>(),
        &artifact(),
        &cfg(),
        0.25,
    )
    .expect("fit");
    assert!(fitted.balanced > 0.95, "balanced {}", fitted.balanced);
    assert!(fitted.usable);
    assert_eq!(fitted.confusion.len(), 2);
    assert_eq!(fitted.rows, 80);
}

#[test]
fn random_labels_do_not_clear_the_gate() {
    let mut rows = separable(80);
    // Break the tie between the column and the label.
    for (i, row) in rows.iter_mut().enumerate() {
        row.segment = (i / 4) % 2;
        row.x[0] = ((i * 37) % 5) as f64;
    }
    let fitted = fit_unit(
        "ad",
        &rows,
        &artifact()
            .segments
            .iter()
            .map(|s| ClassRow {
                id: s.id,
                label: s.label.clone(),
                share: s.share,
                train: 0,
                recall: 0.0,
            })
            .collect::<Vec<_>>(),
        &artifact(),
        &cfg(),
        0.5,
    )
    .expect("fit");
    assert!(!fitted.usable);
    assert!(fitted.note.contains("withheld"));
}

#[test]
fn folds_hold_whole_advertisers() {
    let rows = separable(40);
    let ids: Vec<&str> = rows.iter().map(|r| r.page_id.as_str()).collect();
    let fold = fold_of(ids[3], 5);
    assert_eq!(fold, fold_of(ids[3], 5));
    assert!(fold < 5);
}

#[test]
fn brand_rows_average_the_ads_of_one_advertiser() {
    let mut rows = separable(4);
    for row in &mut rows {
        row.page_id = "one".into();
    }
    let brands = brand_rows(&rows);
    assert_eq!(brands.len(), 1);
    assert_eq!(brands[0].x.len(), columns("brand").len());
    assert!((brands[0].x[0] - 0.0).abs() < 1e-9);
}

#[test]
fn a_map_round_trips_through_its_file() {
    let rows = separable(80);
    let map = assemble(&rows, &artifact(), &cfg(), Source::default(), String::new()).expect("map");
    let path = std::env::temp_dir().join(format!("textmap-{}.json", std::process::id()));
    map.save(path.to_str().expect("path")).expect("save");
    let back = TextMap::load(path.to_str().expect("path")).expect("load");
    assert_eq!(back, map);
    let _ = fs::remove_file(path);
}

#[test]
fn an_unknown_column_width_is_refused() {
    let mut rows = separable(40);
    rows[0].x.pop();
    let specs: Vec<ClassRow> = artifact()
        .segments
        .iter()
        .map(|s| ClassRow {
            id: s.id,
            label: s.label.clone(),
            share: s.share,
            train: 0,
            recall: 0.0,
        })
        .collect();
    assert!(fit_unit("ad", &rows, &specs, &artifact(), &cfg(), 0.25).is_err());
}

#[test]
fn the_confusion_matrix_counts_every_row_once() {
    let pred = vec![0, 1, 1, 0];
    let labels = vec![0, 1, 0, 0];
    let eval = score(&pred, &labels, 2);
    assert_eq!(eval.confusion[0][0], 2);
    assert_eq!(eval.confusion[0][1], 1);
    assert_eq!(eval.confusion[1][1], 1);
    assert!((eval.accuracy - 0.75).abs() < 1e-9);
}

#[test]
fn a_tampered_row_file_is_a_fault_before_anything_is_refitted() {
    let rows = separable(80);
    let map = assemble(&rows, &artifact(), &cfg(), Source::default(), String::new()).expect("map");
    let dir = std::env::temp_dir().join(format!("textmap-check-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("dir");
    let mut tampered = map.clone();
    tampered.source.fnv64 = "deadbeefdeadbeef".into();
    tampered
        .save(dir.join(TEXTMAP).to_str().expect("path"))
        .expect("save");
    fs::write(dir.join(TEXT_FEATURES), "{}\n").expect("rows");
    let faults = check(dir.to_str().expect("path")).expect("check");
    assert!(
        faults
            .iter()
            .any(|fault| fault.contains("does not match the map digest")),
        "{faults:?}"
    );
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn a_flat_column_standardises_without_a_nan() {
    let (mean, sd) = standardise(&[vec![2.0, 1.0], vec![2.0, 3.0]]);
    assert_eq!(mean[0], 2.0);
    assert!(sd[0] > 0.0);
    assert!(sd[1] > 0.0);
}
