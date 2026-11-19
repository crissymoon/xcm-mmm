//! The record a repeat window needs.
//!
//! RFC 3834, section 7 illustrates a vacation responder that refuses to answer
//! a message when the same sender "has been sent a response within the previous
//! 7 days". The section is illustration rather than a normative rule, but the
//! window is the only thing standing between an automatic responder and writing
//! to the same person again every time they follow up, so it is worth holding.
//!
//! A single run has no memory, so the history has to come from somewhere
//! durable. It comes from the ledger the previous run wrote: one JSON object per
//! arrival carrying `sender`, `action` and `at`. Reading it is one file read and
//! the ledger is additive, so the record only ever grows.
//!
//! Two readings this module refuses to make:
//!
//! * a reply line without `at` cannot date anything, so it is counted as
//!   undated and reported rather than guessed at, because a guessed timestamp
//!   either silences a real answer or releases a repeat;
//! * a timestamp in the future is read as "answered now", not as "answered
//!   minus three days", because a clock that moved backwards has to fail in the
//!   direction that sends less mail.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The file a run appends to and this module reads back.
pub const LEDGER: &str = "ledger.jsonl";

const SECONDS_PER_DAY: f64 = 86_400.0;

/// Seconds since the Unix epoch, the unit the ledger stores.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// One ledger line, read leniently: a row from an older run carries more fields
/// than this, and a row from an older format may carry fewer.
#[derive(Deserialize)]
struct LedgerLine {
    sender: String,
    action: String,
    #[serde(default)]
    at: Option<u64>,
}

/// What the ledger said, in the counts the report needs.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Facts {
    /// senders with at least one dated answer
    pub senders: usize,
    /// reply lines that carried a timestamp
    pub dated_replies: usize,
    /// reply lines written before the ledger stored a timestamp
    pub undated_replies: usize,
    /// lines that were not a JSON object at all
    pub malformed_lines: usize,
}

/// The latest answer to each sender, read from the ledger.
#[derive(Clone, Debug, Default)]
pub struct History {
    latest: BTreeMap<String, u64>,
    facts: Facts,
}

impl History {
    /// Read the ledger beside the tree. A missing ledger is an empty history,
    /// not an error: the first run of a new tree has nothing behind it.
    pub fn load(out: &Path) -> Result<Self, String> {
        let path = out.join(LEDGER);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(History::default())
            }
            Err(error) => return Err(format!("{}: {error}", path.display())),
        };
        let mut history = History::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<LedgerLine>(line) {
                Ok(row) if row.action == "reply" => match row.at {
                    Some(at) => history.record(&row.sender, at),
                    None => history.facts.undated_replies += 1,
                },
                Ok(_) => {}
                Err(_) => history.facts.malformed_lines += 1,
            }
        }
        Ok(history)
    }

    /// Note that `sender` was answered at `at`.
    pub fn record(&mut self, sender: &str, at: u64) {
        let key = normalise(sender);
        if key.is_empty() {
            return;
        }
        self.facts.dated_replies += 1;
        let slot = self.latest.entry(key).or_insert(at);
        if at > *slot {
            *slot = at;
        }
    }

    /// Days since this sender was last answered, or `None` if the ledger has no
    /// dated answer for them. A future timestamp reads as zero days.
    pub fn reply_age_days(&self, sender: &str, now: u64) -> Option<f64> {
        let at = *self.latest.get(&normalise(sender))?;
        let seconds = now.saturating_sub(at);
        Some(seconds as f64 / SECONDS_PER_DAY)
    }

    pub fn facts(&self) -> Facts {
        Facts {
            senders: self.latest.len(),
            ..self.facts
        }
    }

    /// Senders with a dated answer on record.
    pub fn senders(&self) -> usize {
        self.latest.len()
    }
}

fn normalise(sender: &str) -> String {
    sender.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latest_answer_wins_and_case_does_not_matter() {
        let mut history = History::default();
        history.record("Person@Example.org", 1_000);
        history.record("person@example.org", 5_000);
        assert_eq!(
            history.reply_age_days(" PERSON@example.ORG ", 5_000),
            Some(0.0)
        );
        assert_eq!(history.senders(), 1);
        assert_eq!(history.facts().dated_replies, 2);
    }

    #[test]
    fn a_future_timestamp_reads_as_zero_days() {
        let mut history = History::default();
        history.record("a@b.c", 10_000);
        assert_eq!(history.reply_age_days("a@b.c", 4_000), Some(0.0));
    }

    #[test]
    fn an_empty_sender_is_not_recorded() {
        let mut history = History::default();
        history.record("   ", 1_000);
        assert_eq!(history.senders(), 0);
        assert_eq!(history.facts().dated_replies, 0);
    }
}
