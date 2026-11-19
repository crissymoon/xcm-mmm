//! What a shipped binary can say about its own origin.
//!
//! The dataset card and the notebook both list a checksum, a commit and a
//! toolchain. Written by hand those three drift apart on the first rebuild, so
//! the binary carries them itself: [`build`](build) reads the values `build.rs`
//! baked in, and the `provenance` subcommand prints them as text or JSON.
//!
//! The date is formatted here rather than pulled from a date crate. It is one
//! civil-from-days conversion on an epoch second, and a dependency that ships a
//! timezone database to print `YYYY-MM-DDTHH:MM:SSZ` is not a trade worth making
//! in a binary that measures its own size.

use serde::Serialize;

/// Facts about the build, as recorded at compile time.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Build {
    /// Crate version from `Cargo.toml`.
    pub version: &'static str,
    /// Short commit the binary was compiled from, or `unknown`.
    pub commit: &'static str,
    /// Whether the working tree had uncommitted changes at build time. Reported
    /// as `unknown` when the build did not run inside a repository.
    pub dirty: &'static str,
    /// Target triple the artefact was linked for.
    pub target: &'static str,
    /// `rustc --version` of the compiler that produced it.
    pub rustc: &'static str,
    /// Build time as RFC 3339 in UTC, derived from `SOURCE_DATE_EPOCH`.
    pub built: String,
    /// The epoch second `built` was formatted from, kept for scripts.
    pub epoch: i64,
}

impl Build {
    /// A one line identity, for `--version` style output.
    pub fn short(&self) -> String {
        format!(
            "xcm-mmm {} ({} on {}, {})",
            self.version, self.commit, self.target, self.built
        )
    }

    /// Whether the artefact came from a clean tree, so a checksum can be trusted
    /// to describe the commit beside it.
    pub fn reproducible(&self) -> bool {
        self.dirty == "false"
    }
}

/// The build this binary was compiled from.
pub fn build() -> Build {
    let epoch: i64 = env!("XCM_BUILD_EPOCH").parse().unwrap_or(0);
    Build {
        version: env!("CARGO_PKG_VERSION"),
        commit: env!("XCM_COMMIT"),
        dirty: env!("XCM_DIRTY"),
        target: env!("XCM_TARGET"),
        rustc: env!("XCM_RUSTC"),
        built: rfc3339(epoch),
        epoch,
    }
}

/// Format an epoch second as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// Epoch zero is reported as `unknown` rather than `1970-01-01T00:00:00Z`: a
/// source tarball with no git metadata should say so, not claim a 1970 build.
pub fn rfc3339(epoch: i64) -> String {
    if epoch <= 0 {
        return "unknown".into();
    }
    let days = epoch.div_euclid(86_400);
    let secs = epoch.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let (hour, minute, second) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Howard Hinnant's days-to-civil algorithm, valid for the whole `i32` range.
///
/// It is the inverse of `days_from_civil`, and the two are tested against each
/// other over every day from 1970 to 2400 so a leap year cannot slip through.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
        let y = if month <= 2 { year - 1 } else { year };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = if month > 2 { month - 3 } else { month + 9 } as i64;
        let doy = (153 * mp + 2) / 5 + day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    #[test]
    fn known_dates_round_trip() {
        assert_eq!(rfc3339(0), "unknown");
        assert_eq!(rfc3339(1), "1970-01-01T00:00:01Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_767_225_600), "2026-01-01T00:00:00Z");
    }

    #[test]
    fn civil_from_days_inverts_days_from_civil() {
        for day in days_from_civil(1970, 1, 1)..=days_from_civil(2400, 12, 31) {
            let (year, month, date) = civil_from_days(day);
            assert_eq!(
                days_from_civil(year, month, date),
                day,
                "day {day} formatted as {year}-{month}-{date}"
            );
        }
    }

    #[test]
    fn a_binary_reports_its_own_commit() {
        let build = build();
        assert!(!build.version.is_empty());
        assert!(!build.target.is_empty());
        assert!(
            build.commit == "unknown" || build.commit.chars().all(|c| c.is_ascii_hexdigit()),
            "commit {:?} is neither unknown nor hexadecimal",
            build.commit
        );
    }

    #[test]
    fn a_dirty_tree_is_not_called_reproducible() {
        let clean = Build {
            dirty: "false",
            ..build()
        };
        let dirty = Build {
            dirty: "true",
            ..build()
        };
        assert!(clean.reproducible());
        assert!(!dirty.reproducible());
    }
}
