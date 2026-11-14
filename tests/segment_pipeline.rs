//! The segmentation folder end to end: extract it, fit it, check it, break it.
//!
//! The fixture is three kinds of advertiser on purpose: one that repeats a
//! single creative on a weekly cadence, one that buys in bursts with a fresh
//! creative each time, and one that bought once. A fit that cannot separate
//! those has nothing to say about a real panel, so the test asserts the groups
//! are found and that a tampered copy is a fault.

use std::path::{Path, PathBuf};

use xcm_mmm::segment::{self, Artifact, SegmentConfig};

fn folder(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-mmm-segment-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config() -> SegmentConfig {
    SegmentConfig {
        name: "fixture-segments".into(),
        k_min: 3,
        k_max: 4,
        min_entities: 30,
        restarts: 8,
        iterations: 80,
        bootstraps: 6,
        controls: 3,
        ..Default::default()
    }
}

fn ad(
    page: &str,
    name: &str,
    day: &str,
    words: usize,
    fingerprint: &str,
    terms: &[&str],
) -> serde_json::Value {
    serde_json::json!({
        "id": format!("{page}-{day}-{fingerprint}"),
        "page_id": page,
        "page_name": name,
        "start": day,
        "spend": ["100", "199"],
        "impressions": ["1000", "1999"],
        "words": words,
        "sentences": 3,
        "terms": terms,
        "fingerprint": fingerprint,
        "split": "train",
    })
}

/// A panel of three behaviours, one file with a row per ad.
fn panel(path: &Path) {
    let mut rows = Vec::new();
    for i in 0..20 {
        let page = format!("repeat{i:03}");
        for week in 0..5 {
            let day = format!("2026-03-{:02}", 1 + week * 7);
            rows.push(ad(&page, "Repeat Buyer", &day, 60, "one", &["housing"]));
        }
    }
    for i in 0..20 {
        let page = format!("burst{i:03}");
        for k in 0..12 {
            let day = format!("2026-{:02}-{:02}", 3 + k / 6, 1 + k % 6);
            let creative = format!("burst-{k}");
            rows.push(ad(
                &page,
                "Burst Buyer",
                &day,
                140,
                &creative,
                &["city", "issue"],
            ));
        }
    }
    for i in 0..20 {
        let page = format!("once{i:03}");
        let day = format!("2026-06-{:02}", 1 + i % 20);
        rows.push(ad(&page, "One Off", &day, 300, "only", &["services"]));
    }
    let mut text = String::new();
    for row in rows {
        text.push_str(&row.to_string());
        text.push('\n');
    }
    std::fs::write(path, text).unwrap();
}

#[test]
fn the_pipeline_extracts_fits_assigns_and_checks() {
    let dir = folder("pipeline");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    let out = dir.join("segments");
    panel(&panel_path);

    let entities =
        segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    assert_eq!(entities.len(), 60);
    assert!(entities.iter().all(|e| e.ads >= 1));

    let artifact =
        segment::build(features.to_str().unwrap(), out.to_str().unwrap(), &config()).unwrap();
    assert_eq!(artifact.k, 3);
    assert_eq!(artifact.assignments.len(), 60);
    assert_eq!(artifact.segments.len(), 3);
    assert_eq!(artifact.metrics.stability.bootstraps, 6);
    assert!(artifact.metrics.control.runs == 3);
    // Three behaviours, so the three largest groups hold the whole panel.
    let sizes: Vec<usize> = artifact.segments.iter().map(|s| s.size).collect();
    assert_eq!(sizes.iter().sum::<usize>(), 60);
    assert!(sizes.iter().all(|&n| n >= 10), "sizes {sizes:?}");

    let faults = segment::check(out.to_str().unwrap()).unwrap();
    assert!(faults.is_empty(), "{faults:?}");

    let loaded = Artifact::load(out.join("model.json").to_str().unwrap()).unwrap();
    let placed: Vec<usize> = entities
        .iter()
        .map(|e| loaded.classify(&e.obs).unwrap().0)
        .collect();
    assert_eq!(placed.len(), 60);
    assert!(placed.iter().all(|&s| s < artifact.k));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fit_is_reproducible_from_its_own_entities() {
    let dir = folder("reproducible");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    panel(&panel_path);
    segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    let first = segment::fit(features.to_str().unwrap(), &config()).unwrap();
    let second = segment::fit(features.to_str().unwrap(), &config()).unwrap();
    assert_eq!(first.centroids, second.centroids);
    assert_eq!(first.assignments, second.assignments);
    assert_eq!(first.metrics, second.metrics);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_model_is_a_fault() {
    let dir = folder("tamper");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    let out = dir.join("segments");
    panel(&panel_path);
    segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    segment::build(features.to_str().unwrap(), out.to_str().unwrap(), &config()).unwrap();

    let path = out.join("model.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    value["centroids"][0][0] = serde_json::json!(9.5);
    std::fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();

    let faults = segment::check(out.to_str().unwrap()).unwrap();
    assert!(
        faults.iter().any(|f| f.contains("digest")),
        "no digest fault in {faults:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_panel_below_the_declared_minimum_is_refused() {
    let dir = folder("small");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    panel(&panel_path);
    segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    let mut cfg = config();
    cfg.min_entities = 500;
    let err = segment::fit(features.to_str().unwrap(), &cfg).unwrap_err();
    assert!(
        err.to_string().contains("below the declared minimum"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_silhouette_below_its_floor_is_refused() {
    let dir = folder("floor");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    panel(&panel_path);
    segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    let mut cfg = config();
    // Above any attainable silhouette, so the floor is what stops the fit
    // rather than the shape of the fixture.
    cfg.min_silhouette = 1.01;
    let err = segment::fit(features.to_str().unwrap(), &cfg).unwrap_err();
    assert!(err.to_string().contains("silhouette"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_entity_missing_a_reading_is_refused() {
    let dir = folder("missing");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    panel(&panel_path);
    segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&features).unwrap();
    let first = text.lines().next().unwrap();
    let mut value: serde_json::Value = serde_json::from_str(first).unwrap();
    value["obs"].as_object_mut().unwrap().remove("terms");
    let rest: Vec<&str> = text.lines().skip(1).collect();
    let mut patched = value.to_string();
    patched.push('\n');
    patched.push_str(&rest.join("\n"));
    patched.push('\n');
    std::fs::write(&features, patched).unwrap();
    let err = segment::fit(features.to_str().unwrap(), &config()).unwrap_err();
    assert!(err.to_string().contains("terms"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unreadable_start_date_is_refused_before_it_becomes_a_cadence() {
    let dir = folder("date");
    let panel_path = dir.join("panel.jsonl");
    let features = dir.join("features.jsonl");
    let mut rows = Vec::new();
    for i in 0..40 {
        let day = if i == 7 { "2026-02-30" } else { "2026-02-10" };
        rows.push(ad(&format!("p{i:03}"), "Page", day, 40, "one", &["city"]));
    }
    let mut text = String::new();
    for row in rows {
        text.push_str(&row.to_string());
        text.push('\n');
    }
    std::fs::write(&panel_path, text).unwrap();
    let err =
        segment::extract(panel_path.to_str().unwrap(), features.to_str().unwrap()).unwrap_err();
    assert!(err.to_string().contains("start date"), "{err}");
    assert!(!features.exists());
    let _ = std::fs::remove_dir_all(&dir);
}
