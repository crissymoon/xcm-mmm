//! What a shipped folder says about itself.
//!
//! The types here are the manifest: one line per file with its digest, one line
//! per split with its day range, and the policy that chose the sample. They are
//! serialised beside the data, so the build time belongs here too rather than in
//! the module that writes the files.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::BuildConfig;

/// One shipped file and the digest that proves it is the one described.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Shipped {
    pub name: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
}

/// What one split holds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SplitCount {
    pub rows: usize,
    pub bytes: u64,
    pub first_day: String,
    pub last_day: String,
    pub tasks: BTreeMap<String, usize>,
}

/// Where the rows came from.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Source {
    pub path: String,
    pub rows: usize,
    pub bytes: u64,
    pub fnv64: String,
    pub first_day: String,
    pub last_day: String,
    pub issuers: usize,
}

/// The manifest a shipped folder carries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub folder: String,
    pub built_at: String,
    pub source: Source,
    pub config: BuildConfig,
    pub rows: usize,
    pub tasks: BTreeMap<String, usize>,
    pub splits: BTreeMap<String, SplitCount>,
    pub years: BTreeMap<String, usize>,
    pub files: Vec<Shipped>,
    pub checks: Vec<String>,
}

/// Civil date from a Unix timestamp, so the manifest carries a readable build
/// time without depending on a calendar crate.
fn iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // Howard Hinnant's civil_from_days, shifted to the March based year.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };

    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Build time as ISO 8601, honouring `SOURCE_DATE_EPOCH` so a rebuild by a
/// packager is not a diff of one line.
pub fn built_at() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });
    iso8601(secs)
}
