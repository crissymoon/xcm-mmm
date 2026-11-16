//! The observation window: where time zero sits, and what a panel may claim.
//!
//! Every survival model needs a time zero and a window, and both are decisions
//! about the data rather than properties of an estimator. If time zero is the
//! customer's join, tenure month 1 is the first month at risk and no row can
//! sit at month 0. If time zero is instead the campaign start, a customer who
//! joined earlier enters the panel left truncated, and a customer observed for
//! less than the label horizon is right censored: their churn is unknown, not
//! negative. Getting this wrong does not produce an error, it produces a model
//! that scores well because it counted the future as a survivor.
//!
//! This module states the contract the shipped panel satisfies and refuses the
//! readings that would break it. It is deliberately a check on the rendered
//! series rather than a new estimator: the fitted numbers already exist, and
//! what was missing was an assertion that they were measured over an interval
//! that means something.
//!
//! The window here is `[1, last]` tenure months with time zero at the join,
//! which is what the telecom panel supports. A panel whose time zero is a
//! campaign start can adopt the same contract by shifting its first month to
//! one after truncating the rows that entered before it; the checks below are
//! written against month numbers, not against dates, so nothing else changes.

use std::io;

use serde::{Deserialize, Serialize};

use super::csvx;
use super::load::Row;
use super::panel::Month;

/// The two columns a window check needs from a panel row.
///
/// Taking this instead of a whole [`Row`] keeps the check usable on a rendered
/// panel, where the text columns are gone, and keeps the coupling in one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tenure {
    /// Tenure month, counted from time zero.
    pub tenure: u32,
    /// Whether the customer left.
    pub churned: bool,
    /// Whether the customer entered inside the window.
    pub joined: bool,
}

impl Tenure {
    /// Read the two fields out of a source row.
    pub fn from_row(row: &Row) -> Self {
        Self {
            tenure: row.tenure,
            churned: row.churned,
            joined: row.joined,
        }
    }
}

/// A month needs at least this many rows at risk before its hazard is read.
///
/// Below it the hazard is one or two events divided by a handful of customers,
/// which is noise dressed as a rate. The contract reports those months rather
/// than hiding them.
pub const MIN_AT_RISK: usize = 30;

/// What the window check found.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeZero {
    /// Where time zero sits, in words.
    pub declared: String,
    /// First tenure month in the panel.
    pub first_month: u32,
    /// Last observed tenure month.
    pub last_month: u32,
    /// Customers in the panel.
    pub rows: usize,
    /// Customers who left inside the window.
    pub events: usize,
    /// Customers still at risk at the last month, so their outcome is unknown
    /// rather than a survivor.
    pub censored: usize,
    /// Months with fewer than [`MIN_AT_RISK`] rows at risk.
    pub thin_months: Vec<u32>,
    /// Rows that entered inside the window, so they are not left truncated.
    pub entered: usize,
    /// Rows that claim a tenure before time zero.
    pub before_zero: usize,
    /// Months missing from the series between the first and the last.
    pub gaps: Vec<u32>,
    /// Everything that breaks the contract, empty when it holds.
    pub violations: Vec<String>,
}

impl TimeZero {
    /// True when the window holds.
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }

    /// One line for a terminal.
    pub fn summary(&self) -> String {
        format!(
            "time zero at the {}, months {}-{}, {} rows, {} events, {} censored, {} thin months{}{}",
            self.declared,
            self.first_month,
            self.last_month,
            self.rows,
            self.events,
            self.censored,
            self.thin_months.len(),
            if self.gaps.is_empty() {
                String::new()
            } else {
                format!(", {} gaps", self.gaps.len())
            },
            if self.ok() { "" } else { ", contract broken" }
        )
    }
}

/// Check the window the panel describes.
///
/// The series and the panel have to agree on the count of events. That is an
/// exact check and a sharp one: a churn recorded at a tenure outside the window
/// makes the series lose an event, and a month dropped from the series loses
/// every event in it. The last survival reading is deliberately not compared
/// with `1 - events / rows`, because those are different estimators. The series
/// is a hazard over a shrinking risk set, so a row that leaves without churning
/// stops contributing; the crude share keeps dividing by every row at once.
/// They agree only when nobody is censored before the last month, which is
/// almost never the shape a real panel has.
pub fn check(rows: &[Tenure], months: &[Month]) -> TimeZero {
    let mut out = TimeZero {
        declared: "join".to_string(),
        rows: rows.len(),
        ..TimeZero::default()
    };
    let Some(first) = months.first() else {
        out.violations.push("the series is empty".to_string());
        return out;
    };
    out.first_month = first.month;
    out.last_month = months.last().map(|month| month.month).unwrap_or(0);
    out.events = rows.iter().filter(|row| row.churned).count();
    out.entered = rows.iter().filter(|row| row.joined).count();
    out.before_zero = rows.iter().filter(|row| row.tenure < first.month).count();
    // A customer still present at the last month who has not churned has not
    // been observed to fail, so their outcome is unknown past the window.
    out.censored = rows
        .iter()
        .filter(|row| row.tenure >= out.last_month && !row.churned)
        .count();
    if out.before_zero > 0 {
        out.violations.push(format!(
            "{} rows carry a tenure before month {}, so time zero is not the join for them",
            out.before_zero, out.first_month
        ));
    }
    let mut previous: Option<u32> = None;
    for month in months {
        if let Some(last) = previous {
            if month.month != last + 1 {
                out.gaps.push(month.month);
            }
        }
        previous = Some(month.month);
    }
    if !out.gaps.is_empty() {
        out.violations.push(format!(
            "{} months are missing from the series, so the hazard is not measured over a continuous window",
            out.gaps.len()
        ));
    }
    for month in months {
        if month.at_risk < MIN_AT_RISK {
            out.thin_months.push(month.month);
        }
    }
    let decreasing = months
        .windows(2)
        .all(|pair| pair[1].at_risk <= pair[0].at_risk);
    if !decreasing {
        out.violations
            .push("the rows at risk rise from one month to the next".to_string());
    }
    let series_events: usize = months.iter().map(|month| month.churned).sum();
    if series_events != out.events {
        out.violations.push(format!(
            "the panel holds {} churn events but the series counts {series_events}, so a row left or entered the window between them",
            out.events
        ));
    }
    out
}

/// Read the shipped panel and series and check the window they describe.
///
/// This is the entry point a command uses: it reads the two rendered files
/// rather than the source table, so the contract is checked against exactly the
/// numbers a model was fitted on.
pub fn check_files(dir: &str) -> io::Result<TimeZero> {
    let panel = csvx::read(&format!("{dir}/panel.csv"))?;
    let tenure = panel.require("tenure")?;
    let churn = panel.require("churn")?;
    let rows: Vec<Tenure> = panel
        .rows
        .iter()
        .map(|row| Tenure {
            tenure: row[tenure].parse().unwrap_or(0),
            churned: row[churn].parse::<u8>().map(|v| v == 1).unwrap_or(false),
            // The rendered panel drops the join flag, so a rendered row is read
            // as having been present at time zero. A row that entered inside
            // the window would show up as a tenure below the first month, which
            // the check reports.
            joined: false,
        })
        .collect();
    let series = csvx::read(&format!("{dir}/series.csv"))?;
    let month = series.require("month")?;
    let at_risk = series.require("at_risk")?;
    let churned = series.require("churned")?;
    let hazard = series.require("hazard")?;
    let survival = series.require("survival")?;
    let months: Vec<Month> = series
        .rows
        .iter()
        .map(|row| Month {
            month: row[month].parse().unwrap_or(0),
            at_risk: row[at_risk].parse().unwrap_or(0),
            churned: row[churned].parse().unwrap_or(0),
            hazard: row[hazard].parse().unwrap_or(0.0),
            survival: row[survival].parse().unwrap_or(0.0),
        })
        .collect();
    Ok(check(&rows, &months))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::churn::panel::series;

    fn healthy() -> Vec<Tenure> {
        let mut rows = Vec::new();
        for month in 1..=6u32 {
            for index in 0..40 {
                let churned = index < 4 && month < 6;
                rows.push(Tenure {
                    tenure: month,
                    churned,
                    joined: false,
                });
            }
        }
        rows
    }

    #[test]
    fn a_complete_window_holds() {
        let rows = healthy();
        let months = series(&rows);
        let contract = check(&rows, &months);
        assert!(contract.ok(), "{:?}", contract.violations);
        assert_eq!(contract.first_month, 1);
        assert_eq!(contract.last_month, 6);
        assert_eq!(contract.rows, 240);
        assert!(contract.summary().contains("time zero at the join"));
    }

    #[test]
    fn a_row_before_time_zero_breaks_it() {
        let mut rows = healthy();
        rows.push(Tenure {
            tenure: 0,
            churned: false,
            joined: false,
        });
        let months = series(&rows);
        let contract = check(&rows, &months);
        assert!(!contract.ok());
        assert_eq!(contract.before_zero, 1);
        assert!(contract.violations[0].contains("time zero"));
    }

    #[test]
    fn a_gap_in_the_series_is_a_violation() {
        let rows = healthy();
        let mut months = series(&rows);
        months.retain(|month| month.month != 3);
        let contract = check(&rows, &months);
        assert!(!contract.ok());
        assert_eq!(contract.gaps, vec![4]);
    }

    #[test]
    fn survivors_at_the_last_month_are_censored_not_negative() {
        let rows = healthy();
        let months = series(&rows);
        let contract = check(&rows, &months);
        assert_eq!(contract.censored, 40);
        assert!(contract.events < contract.rows);
    }

    #[test]
    fn a_thin_month_is_reported_and_does_not_break_the_contract() {
        let mut rows = healthy();
        rows.retain(|row| row.tenure < 6);
        rows.push(Tenure {
            tenure: 6,
            churned: false,
            joined: false,
        });
        let months = series(&rows);
        let contract = check(&rows, &months);
        assert_eq!(contract.thin_months, vec![6]);
        assert!(contract.ok(), "{:?}", contract.violations);
    }
}
