//! From an ad panel to one behavioural row per advertiser.
//!
//! The panel is a flat list of ads: who ran it, when it started, the spend and
//! impression bands the archive published, how long the copy is, which terms it
//! was bought against and a fingerprint of the copy. Segmentation is at the
//! advertiser, so this module folds the ads back up and keeps only what an
//! advertiser did: how often it started an ad, how far apart those starts were,
//! what it paid and how concentrated the copy is.
//!
//! Every reading here is behaviour observed in the panel. Nothing descriptive
//! about the advertiser enters a feature: the page name is carried for reading
//! the result, never for fitting it.

use std::collections::BTreeMap;
use std::fmt;
use std::io;

use serde::{Deserialize, Serialize};

/// One row of the ad panel, as the archive exports it.
#[derive(Clone, Debug, Deserialize)]
pub struct AdRow {
    #[serde(default)]
    pub id: String,
    pub page_id: String,
    #[serde(default)]
    pub page_name: String,
    pub start: String,
    #[serde(default)]
    pub spend: Vec<String>,
    #[serde(default)]
    pub impressions: Vec<String>,
    #[serde(default)]
    pub words: usize,
    #[serde(default)]
    pub sentences: usize,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub fingerprint: String,
    #[serde(default)]
    pub split: String,
}

/// A civil date as a count of days since 1970-01-01.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Day(pub i64);

impl Day {
    /// Parse `YYYY-MM-DD`. Anything else is refused rather than guessed at,
    /// because a bad date silently changes every cadence reading.
    pub fn parse(text: &str) -> Option<Self> {
        let bytes = text.as_bytes();
        if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
            return None;
        }
        let number = |range: std::ops::Range<usize>| -> Option<i64> {
            let mut value = 0i64;
            for &b in &bytes[range] {
                if !b.is_ascii_digit() {
                    return None;
                }
                value = value * 10 + (b - b'0') as i64;
            }
            Some(value)
        };
        let (y, m, d) = (
            number(0..4)? as i32,
            number(5..7)? as u32,
            number(8..10)? as u32,
        );
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) || y < 1900 {
            return None;
        }
        // A round trip catches the days a month does not have, so 2026-02-30
        // is refused rather than rolled into March.
        let day = Day(days_from_civil(y, m, d));
        if civil_from_days(day.0) != (y, m, d) {
            return None;
        }
        Some(day)
    }
}

impl fmt::Display for Day {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (y, m, d) = civil_from_days(self.0);
        write!(f, "{y:04}-{m:02}-{d:02}")
    }
}

/// Howard Hinnant's `days_from_civil`, the inverse of `civil_from_days`.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((m as i64) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of `days_from_civil`.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((if m <= 2 { y + 1 } else { y }) as i32, m, d)
}

/// One advertiser and what it did in the panel.
///
/// `obs` holds readings in their own units, before any transform: ads counted,
/// days between starts, dollars at the band midpoint, words per ad. The model
/// applies the transform its feature table declares, so the file a reader opens
/// is in units a reader recognises.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub ads: usize,
    pub test_ads: usize,
    pub first: String,
    pub last: String,
    pub obs: BTreeMap<String, f64>,
}

/// Fold a panel into entities, sorted by advertiser id.
///
/// Refusals are explicit: an empty panel, an ad without an advertiser, and a
/// start date that is not a date. Each of those would quietly change a cadence
/// reading, so none of them is skipped.
pub fn entities(rows: &[AdRow]) -> io::Result<Vec<Entity>> {
    if rows.is_empty() {
        return Err(bad("the panel holds no ads"));
    }
    let mut grouped: BTreeMap<&str, Vec<&AdRow>> = BTreeMap::new();
    for row in rows {
        if row.page_id.trim().is_empty() {
            return Err(bad(&format!("ad {} has no advertiser id", label_of(row))));
        }
        if Day::parse(&row.start).is_none() {
            return Err(bad(&format!(
                "ad {} has an unreadable start date {:?}",
                label_of(row),
                row.start
            )));
        }
        grouped.entry(row.page_id.as_str()).or_default().push(row);
    }
    let mut out = Vec::with_capacity(grouped.len());
    for (id, ads) in grouped {
        let mut starts: Vec<Day> = ads.iter().filter_map(|r| Day::parse(&r.start)).collect();
        starts.sort();
        starts.dedup();
        let gaps: Vec<f64> = starts
            .windows(2)
            .map(|w| (w[1].0 - w[0].0).max(0) as f64)
            .collect();
        let mean_gap = mean(&gaps);
        let burstiness = if gaps.len() > 1 && mean_gap > 0.0 {
            (spread(&gaps) / mean_gap).min(3.0)
        } else {
            0.0
        };

        let mut spend = Vec::new();
        let mut words = Vec::new();
        let mut terms = Vec::new();
        let mut copies: BTreeMap<&str, usize> = BTreeMap::new();
        for ad in &ads {
            if let Some(mid) = band_mid(&ad.spend) {
                spend.push(mid);
            }
            words.push(ad.words as f64);
            terms.push(ad.terms.len() as f64);
            if !ad.fingerprint.is_empty() {
                *copies.entry(ad.fingerprint.as_str()).or_default() += 1;
            }
        }
        let repeated: usize = copies.values().filter(|&&n| n > 1).sum();
        let largest = copies.values().copied().max().unwrap_or(0);
        let count = ads.len().max(1);

        let mut obs = BTreeMap::new();
        obs.insert("ads".to_string(), ads.len() as f64);
        obs.insert("cadence".to_string(), median(&mut gaps.clone()));
        obs.insert("burstiness".to_string(), burstiness);
        obs.insert("spend_mid".to_string(), mean(&spend));
        obs.insert("copy_words".to_string(), median(&mut words.clone()));
        obs.insert("terms".to_string(), mean(&terms));
        obs.insert("repeat".to_string(), repeated as f64 / count as f64);
        obs.insert("concentrate".to_string(), largest as f64 / count as f64);

        out.push(Entity {
            id: id.to_string(),
            name: common_name(&ads),
            ads: ads.len(),
            test_ads: ads.iter().filter(|r| r.split == "test").count(),
            first: starts[0].to_string(),
            last: starts[starts.len() - 1].to_string(),
            obs,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Read a panel from JSON Lines.
pub fn read_rows(path: &str) -> io::Result<Vec<AdRow>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: AdRow = serde_json::from_str(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{path}:{}: {e}", i + 1))
        })?;
        rows.push(row);
    }
    Ok(rows)
}

/// Midpoint of a band such as `["100", "199"]`, or nothing when the archive
/// published no band at all. Unreported is not zero and is never read as zero.
pub fn band_mid(band: &[String]) -> Option<f64> {
    let low: f64 = band.first()?.trim().parse().ok()?;
    let high: f64 = band
        .get(1)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(low);
    Some(0.5 * (low + high))
}

/// Arithmetic mean, zero for an empty slice.
pub fn mean(values: &[f64]) -> f64 {
    crate::linalg::mean(values)
}

/// Population standard deviation, zero for a slice shorter than two.
pub fn spread(values: &[f64]) -> f64 {
    crate::linalg::std_dev(values)
}

/// Median, averaging the two middle values on an even count. The slice is
/// reordered, which is why the caller passes a copy.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// The name an advertiser used most often, so a renamed page still reads.
fn common_name(ads: &[&AdRow]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for ad in ads {
        let name = ad.page_name.trim();
        if !name.is_empty() {
            *counts.entry(name).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(name, _)| name.to_string())
        .unwrap_or_default()
}

fn label_of(row: &AdRow) -> String {
    if row.id.is_empty() {
        "<no id>".to_string()
    } else {
        row.id.clone()
    }
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ad(id: &str, page: &str, start: &str, words: usize, terms: &[&str]) -> AdRow {
        AdRow {
            id: id.into(),
            page_id: page.into(),
            page_name: "A Page".into(),
            start: start.into(),
            spend: vec!["100".into(), "199".into()],
            impressions: vec!["1000".into(), "1999".into()],
            words,
            sentences: 3,
            terms: terms.iter().map(|t| t.to_string()).collect(),
            fingerprint: id.into(),
            split: "train".into(),
        }
    }

    #[test]
    fn dates_round_trip_through_the_civil_conversion() {
        for text in ["1970-01-01", "2000-02-29", "2026-11-10", "1900-01-01"] {
            let day = Day::parse(text).unwrap();
            assert_eq!(day.to_string(), text);
        }
    }

    #[test]
    fn a_bad_date_is_refused_rather_than_guessed() {
        assert!(Day::parse("2026-13-01").is_none());
        assert!(Day::parse("10/11/2026").is_none());
        assert!(Day::parse("2026-11-1").is_none());
    }

    #[test]
    fn an_unreported_band_is_missing_not_zero() {
        assert_eq!(band_mid(&["100".into(), "199".into()]), Some(149.5));
        assert_eq!(band_mid(&[]), None);
        assert_eq!(band_mid(&["n/a".into()]), None);
    }

    #[test]
    fn cadence_and_reuse_come_from_what_the_advertiser_did() {
        let rows = vec![
            ad("a", "p1", "2026-01-01", 10, &["x"]),
            ad("b", "p1", "2026-01-11", 30, &["x", "y"]),
            ad("c", "p1", "2026-02-01", 20, &["x"]),
            ad("d", "p2", "2026-02-01", 50, &["z"]),
        ];
        let got = entities(&rows).unwrap();
        assert_eq!(got.len(), 2);
        let p1 = &got[0];
        assert_eq!(p1.ads, 3);
        assert_eq!(p1.obs["cadence"], 15.5);
        assert!(p1.obs["terms"] > 1.3 && p1.obs["terms"] < 1.4);
        assert_eq!(p1.obs["copy_words"], 20.0);
        assert_eq!(p1.obs["repeat"], 0.0);
        assert_eq!(p1.first, "2026-01-01");
        assert_eq!(p1.last, "2026-02-01");
    }

    #[test]
    fn a_repeated_fingerprint_shows_up_in_reuse_and_concentration() {
        let mut rows = vec![
            ad("a", "p1", "2026-01-01", 10, &["x"]),
            ad("b", "p1", "2026-01-02", 10, &["x"]),
            ad("c", "p1", "2026-01-03", 12, &["x"]),
        ];
        rows[1].fingerprint = "a".into();
        rows[2].fingerprint = "a".into();
        let got = entities(&rows).unwrap();
        assert_eq!(got[0].obs["repeat"], 1.0);
        assert_eq!(got[0].obs["concentrate"], 1.0);
    }

    #[test]
    fn an_empty_panel_or_a_missing_advertiser_is_refused() {
        assert!(entities(&[]).is_err());
        let mut row = ad("a", "", "2026-01-01", 1, &[]);
        row.page_id = " ".into();
        assert!(entities(&[row]).is_err());
        let mut row = ad("a", "p1", "not a date", 1, &[]);
        row.start = "not a date".into();
        assert!(entities(&[row]).is_err());
    }

    #[test]
    fn burstiness_is_capped_and_zero_for_a_single_start() {
        let one = vec![ad("a", "p1", "2026-01-01", 1, &[])];
        assert_eq!(entities(&one).unwrap()[0].obs["burstiness"], 0.0);
        let wide = vec![
            ad("a", "p1", "2026-01-01", 1, &[]),
            ad("b", "p1", "2026-01-02", 1, &[]),
            ad("c", "p1", "2026-06-01", 1, &[]),
        ];
        let got = entities(&wide).unwrap();
        assert!(got[0].obs["burstiness"] <= 3.0);
    }
}
