//! Which registers the glossary can read, and which it is blind to.
//!
//! A benchmark that reports one number over a mixed corpus hides the only thing
//! worth knowing about a legal instrument: it was built from what its authors
//! happened to read. The shipped bank is 2,324 sentences, and four sources
//! carry 1,786 of them, so a single balanced accuracy is mostly a score for
//! federal civil procedure plus court self-help pages.
//!
//! This module splits the same bank by the register its source belongs to,
//! scans every row with the shipped glossary, and reports three things per
//! register: how many rows exist, how many of them the glossary touches at all,
//! and which term kinds it fires. A register with no rows is not measured. A
//! register with rows that the glossary never touches is blind, and the fix is
//! words, not code.
//!
//! The verdicts are deliberately coarse, and each one carries its threshold:
//!
//! A register has a role. A target register is drafting the instrument should
//! read, so a hit is the instrument working. A contrast register is copy that
//! should come out plain, so a hit is a false positive and the share is a cost.
//! Scoring both with one word would read a clean negative as a blind spot.
//!
//! * `covered`, at least [`MIN_ROWS`] rows and at least [`MIN_REACH`] of those
//!   rows carrying a hit;
//! * `thin`, rows present but fewer than [`MIN_ROWS`], so nothing is claimed
//!   from them;
//! * `blind`, enough rows to measure, but the glossary reaches fewer than
//!   [`MIN_REACH`] of them;
//! * `clean`, a contrast register the glossary leaves alone;
//! * `leaky`, a contrast register where the glossary fires on more than
//!   [`MAX_LEAK`] of the rows;
//! * `absent`, no rows at all for that register in this bank.
//!
//! `absent` is the interesting column. Contracts, judicial opinions,
//! regulations and agency notices are each absent from the shipped bank, so
//! every claim the card makes stops at the four registers it does hold.
//!
//! The taxonomy is not invented here. LegalBench (Guha et al., 2023, 162 tasks)
//! splits legal reasoning into six IRAC categories rather than one accuracy,
//! and LexGLUE (Chalkidis et al., 2022) evaluates on separate sub-corpora for
//! legislation, contracts and court cases for the same reason. CUAD
//! (Hendrycks et al., 2021) exists because contracts are a register of their
//! own: the parties define their own terms and the clauses are negotiated, so a
//! glossary read off statutes misses them. The list below is those registers
//! reduced to the ones this instrument could plausibly be asked to read.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};

use serde::{Deserialize, Serialize};

use crate::num::round4;

use super::translate::{words, Legal};

/// Rows a register needs before a share over it means anything.
pub const MIN_ROWS: usize = 100;
/// Share of rows the glossary must touch for a target register to count as read.
pub const MIN_REACH: f64 = 0.20;
/// Share of contrast rows a hit may land on before the scan is called leaky.
pub const MAX_LEAK: f64 = 0.10;

/// What a hit in this register means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Drafting the instrument should read: a hit is the instrument working.
    Target,
    /// Copy that should come out plain: a hit is a false positive.
    Contrast,
}

/// One register the instrument might be asked to read.
#[derive(Clone, Debug)]
pub struct Register {
    pub name: &'static str,
    /// Source prefixes that place a bank row in this register.
    pub markers: &'static [&'static str],
    pub role: Role,
    pub note: &'static str,
}

/// The registers a plain language instrument is asked about, in the order a
/// widening effort should read them: the four the bank already has first, then
/// the drafted texts it does not.
pub const REGISTERS: &[Register] = &[
    Register {
        name: "court-rule",
        markers: &["frcp-", "frap-", "fre-", "ca-crc-", "crc-"],
        role: Role::Target,
        note: "rules of procedure and evidence, the densest legalese in the bank",
    },
    Register {
        name: "statute",
        markers: &[
            "usc-", "ca-ccp-", "ca-civ-", "ca-gov-", "ca-lamc-", "ca-lab-", "ca-pen-", "ca-bpc-",
            "ca-edu-", "ca-hs-",
        ],
        role: Role::Target,
        note: "codes and session law, where the definition sections live",
    },
    Register {
        name: "self-help",
        markers: &["selfhelp:"],
        role: Role::Contrast,
        note: "court self-help pages, legal content already written for a lay reader",
    },
    Register {
        name: "ad-copy",
        markers: &["ad"],
        role: Role::Contrast,
        note: "advertising copy, the negative side of the detector, not a legal register",
    },
    Register {
        name: "regulation",
        markers: &["cfr-", "ca-code-regs", "ca-reg-"],
        role: Role::Target,
        note: "administrative rules with incorporation by reference",
    },
    Register {
        name: "contract",
        markers: &["cuad", "contract", "msa", "nda", "lease", "license"],
        role: Role::Target,
        note: "negotiated agreements, where the parties write their own defined terms",
    },
    Register {
        name: "opinion",
        markers: &["opinion", "ca-opinion", "court-opinion", "holding"],
        role: Role::Target,
        note: "judicial reasoning, which argues in prose rather than clauses",
    },
    Register {
        name: "agency-notice",
        markers: &["fedreg", "notice", "bulletin", "circular"],
        role: Role::Target,
        note: "notices, bulletins and forms that tell a person what to do next",
    },
    Register {
        name: "plain-summary",
        markers: &["plain"],
        role: Role::Contrast,
        note: "plain restatements, the reference side of a pair",
    },
];

/// One bank row, narrowed to what the audit needs.
#[derive(Clone, Debug, Deserialize)]
pub struct BankRow {
    #[serde(default)]
    pub axis: String,
    #[serde(default, deserialize_with = "flag")]
    pub label: bool,
    #[serde(default)]
    pub source: String,
    pub text: String,
}

/// The bank writes its label as `1` and `0`, and a hand written extension may
/// write `true` and `false`. Both are read rather than one silently dropped.
fn flag<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Written {
        Bool(bool),
        Number(i64),
    }
    Ok(match Option::<Written>::deserialize(deserializer)? {
        Some(Written::Bool(value)) => value,
        Some(Written::Number(value)) => value != 0,
        None => false,
    })
}

/// One register, audited.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Row {
    pub register: String,
    pub role: String,
    pub note: String,
    pub rows: usize,
    pub legal: usize,
    pub plain: usize,
    pub words: usize,
    /// Rows where at least one glossary term fired.
    pub reached: usize,
    pub hits: usize,
    pub hits_per_100: f64,
    /// Term kinds the glossary fired here, with counts.
    pub kinds: BTreeMap<String, usize>,
    pub verdict: String,
}

/// The bank, split by register.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Coverage {
    pub rows: Vec<Row>,
    pub bank: usize,
    /// Registers with no rows in this bank, in declaration order.
    pub absent: Vec<String>,
    pub blind: Vec<String>,
    pub thin: Vec<String>,
    /// Contrast registers the glossary fires on too often.
    pub leaky: Vec<String>,
    /// The share of target rows the glossary touches. The instrument's reach
    /// over drafting, with the negative side left out of the denominator.
    pub reach: f64,
    /// The share of contrast rows the glossary touches: the false positive
    /// cost of scanning copy that is already plain.
    pub leak: f64,
}

/// Place a source in a register. The first marker that matches wins, so the
/// order of [`REGISTERS`] is the tie break, and a source that matches nothing
/// is reported as `unclassified` rather than folded into a neighbour.
pub fn classify(source: &str) -> &'static str {
    let source = source.trim();
    for register in REGISTERS {
        for marker in register.markers {
            if marked(source, marker) {
                return register.name;
            }
        }
    }
    "unclassified"
}

/// True when `source` is the marker or continues it after a separator. The
/// separator matters: `ad` names ad copy, `advisory-note` does not, and a bare
/// `starts_with` would fold the second into the first.
fn marked(source: &str, marker: &str) -> bool {
    let stem = marker.trim_end_matches(['-', ':']);
    match source.strip_prefix(stem) {
        Some("") => true,
        Some(rest) => rest.starts_with(&['-', ':', '/'][..]),
        None => false,
    }
}

fn note(register: &str) -> &'static str {
    REGISTERS
        .iter()
        .find(|r| r.name == register)
        .map(|r| r.note)
        .unwrap_or("a source the register table does not know")
}

/// Read the bank. A malformed line is a refusal with its number: a register
/// count that silently lost rows would read as a gap in the language.
pub fn read_bank(path: &str) -> Result<Vec<BankRow>, String> {
    let file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut rows = Vec::new();
    for (number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{}: {e}", number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: BankRow =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{}: {e}", number + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

impl Coverage {
    /// Split the bank and scan every row with the shipped glossary.
    pub fn audit(legal: &Legal, bank_path: &str) -> Result<Self, String> {
        let bank = read_bank(bank_path)?;
        let mut buckets: BTreeMap<&'static str, Row> = BTreeMap::new();
        for register in REGISTERS {
            buckets.insert(
                register.name,
                Row {
                    register: register.name.to_string(),
                    role: role_name(register.role).to_string(),
                    note: register.note.to_string(),
                    verdict: "absent".to_string(),
                    ..Default::default()
                },
            );
        }
        buckets.insert(
            "unclassified",
            Row {
                register: "unclassified".to_string(),
                role: "target".to_string(),
                note: note("unclassified").to_string(),
                verdict: "absent".to_string(),
                ..Default::default()
            },
        );

        for entry in &bank {
            let name = classify(&entry.source);
            let row = buckets.entry(name).or_default();
            row.rows += 1;
            if entry.label {
                row.legal += 1;
            } else {
                row.plain += 1;
            }
            row.words += words(&entry.text);
            let scan = legal.scan(&entry.text);
            if !scan.terms.is_empty() {
                row.reached += 1;
            }
            row.hits += scan.terms.len();
            for term in &scan.terms {
                *row.kinds.entry(term.kind.clone()).or_default() += 1;
            }
        }

        let mut rows: Vec<Row> = buckets.into_values().collect();
        for row in &mut rows {
            if row.words > 0 {
                row.hits_per_100 = round4(row.hits as f64 * 100.0 / row.words as f64);
            }
            row.verdict = verdict(row.rows, row.reached, &row.role).to_string();
        }
        let (reached, total) = tally(&rows, "target");
        let (leaked, contrast) = tally(&rows, "contrast");
        Ok(Self {
            absent: collect(&rows, "absent"),
            blind: collect(&rows, "blind"),
            thin: collect(&rows, "thin"),
            leaky: collect(&rows, "leaky"),
            reach: round4(reached as f64 / total.max(1) as f64),
            leak: round4(leaked as f64 / contrast.max(1) as f64),
            bank: total + contrast,
            rows,
        })
    }

    /// What a widening effort should write next, in the order that buys the
    /// most: registers with no text, then registers with too little, then
    /// registers the glossary cannot see.
    pub fn widen(&self) -> Vec<String> {
        let mut out = Vec::new();
        for name in self.absent.iter().filter(|name| is_target(name)) {
            out.push(format!(
                "add {name} text: no bank row exists, so nothing is claimed ({})",
                note(name)
            ));
        }
        for name in &self.blind {
            out.push(format!(
                "add {name} words: the glossary reaches under {:.0} percent of its rows",
                MIN_REACH * 100.0
            ));
        }
        for name in &self.thin {
            out.push(format!(
                "grow {name} text: fewer than {MIN_ROWS} rows, so no share is reported"
            ));
        }
        for name in &self.leaky {
            out.push(format!(
                "tighten the glossary on {name}: it fires on more than {:.0} percent of copy that is already plain",
                MAX_LEAK * 100.0
            ));
        }
        out
    }

    /// Target registers this bank cannot speak to, in the order worth fixing.
    /// A contrast register with no rows is not on the list: the plain side of a
    /// pair is the reference, not a register the instrument has to read.
    pub fn gaps(&self) -> Vec<String> {
        self.absent
            .iter()
            .chain(self.blind.iter())
            .chain(self.thin.iter())
            .filter(|name| is_target(name))
            .cloned()
            .collect()
    }

    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{} bank rows, {} registers declared, glossary reaches {:.1} percent of target rows and fires on {:.1} percent of contrast rows",
            self.bank,
            REGISTERS.len(),
            self.reach * 100.0,
            self.leak * 100.0
        ));
        lines.push(format!(
            "{:<14} {:>8} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7} {:>11}",
            "register", "role", "rows", "legal", "plain", "reached", "hits", "per100", "verdict"
        ));
        for row in &self.rows {
            lines.push(format!(
                "{:<14} {:>8} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7.2} {:>11}",
                row.register,
                row.role,
                row.rows,
                row.legal,
                row.plain,
                row.reached,
                row.hits,
                row.hits_per_100,
                row.verdict
            ));
        }
        for advice in self.widen() {
            lines.push(format!("widen: {advice}"));
        }
        let classes: Vec<String> = self
            .rows
            .iter()
            .filter(|row| !row.kinds.is_empty())
            .map(|row| {
                let mut kinds: Vec<String> = row
                    .kinds
                    .iter()
                    .map(|(kind, count)| format!("{kind} {count}"))
                    .collect();
                kinds.sort();
                format!("{}: {}", row.register, kinds.join(", "))
            })
            .collect();
        for line in classes {
            lines.push(format!("  {line}"));
        }
        lines.join("\n")
    }
}

/// The verdict for one register: how many rows it holds, how many of those the
/// glossary reached, and what a hit there means.
pub fn verdict(rows: usize, reached: usize, role: &str) -> &'static str {
    if rows == 0 {
        return "absent";
    }
    if rows < MIN_ROWS {
        return "thin";
    }
    let share = reached as f64 / rows as f64;
    match role {
        "contrast" => {
            if share > MAX_LEAK {
                "leaky"
            } else {
                "clean"
            }
        }
        _ => {
            if share < MIN_REACH {
                "blind"
            } else {
                "covered"
            }
        }
    }
}

/// True when a register name is one the instrument is meant to read. A name
/// the table does not know is treated as a target, so an unclassified source is
/// reported as a gap rather than excused.
fn is_target(name: &str) -> bool {
    REGISTERS
        .iter()
        .find(|register| register.name == name)
        .map(|register| register.role == Role::Target)
        .unwrap_or(true)
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::Target => "target",
        Role::Contrast => "contrast",
    }
}

/// Reached rows and total rows for one role.
fn tally(rows: &[Row], role: &str) -> (usize, usize) {
    rows.iter()
        .filter(|row| row.role == role)
        .fold((0, 0), |(reached, total), row| {
            (reached + row.reached, total + row.rows)
        })
}

fn collect(rows: &[Row], verdict: &str) -> Vec<String> {
    rows.iter()
        .filter(|row| row.verdict == verdict)
        .map(|row| row.register.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rule_source_is_a_court_rule() {
        assert_eq!(classify("frcp-23"), "court-rule");
        assert_eq!(classify("ca-crc-8.104"), "court-rule");
    }

    #[test]
    fn a_state_code_is_a_statute() {
        assert_eq!(classify("ca-ccp-1005"), "statute");
        assert_eq!(classify("usc-42-2000d"), "statute");
    }

    #[test]
    fn the_bare_ad_source_is_ad_copy_and_not_a_prefix_accident() {
        assert_eq!(classify("ad"), "ad-copy");
        assert_eq!(classify("ad/laundry"), "ad-copy");
        assert_eq!(classify("advisory-note"), "unclassified");
    }

    #[test]
    fn a_self_help_page_keeps_its_url_in_the_marker() {
        assert_eq!(
            classify("selfhelp:selfhelp.courts.ca.gov/fee-waiver"),
            "self-help"
        );
    }

    #[test]
    fn an_unknown_source_is_named_rather_than_folded_in() {
        assert_eq!(classify(""), "unclassified");
        assert_eq!(classify("random-crawl"), "unclassified");
    }

    #[test]
    fn a_register_with_no_rows_is_absent_before_anything_else() {
        assert_eq!(verdict(0, 0, "target"), "absent");
        assert_eq!(verdict(0, 0, "contrast"), "absent");
    }

    #[test]
    fn a_register_below_the_row_floor_is_thin_rather_than_scored() {
        assert_eq!(verdict(MIN_ROWS - 1, MIN_ROWS - 1, "target"), "thin");
        assert_eq!(verdict(MIN_ROWS - 1, 0, "contrast"), "thin");
    }

    #[test]
    fn a_target_register_is_blind_when_the_glossary_rarely_reaches_it() {
        assert_eq!(verdict(1000, 100, "target"), "blind");
        assert_eq!(verdict(1000, MIN_ROWS * 2, "target"), "covered");
    }

    #[test]
    fn a_contrast_register_is_clean_when_the_glossary_leaves_it_alone() {
        assert_eq!(verdict(1000, 10, "contrast"), "clean");
        assert_eq!(verdict(1000, 500, "contrast"), "leaky");
    }

    fn row(name: &str, role: &str, verdict: &str) -> Row {
        Row {
            register: name.to_string(),
            role: role.to_string(),
            verdict: verdict.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn the_gap_list_holds_target_registers_and_not_the_plain_side() {
        let coverage = Coverage {
            rows: vec![
                row("contract", "target", "absent"),
                row("court-rule", "target", "covered"),
                row("plain-summary", "contrast", "absent"),
                row("ad-copy", "contrast", "clean"),
            ],
            absent: vec!["contract".to_string(), "plain-summary".to_string()],
            ..Default::default()
        };
        assert_eq!(coverage.gaps(), vec!["contract".to_string()]);
        let advice = coverage.widen();
        assert_eq!(advice.len(), 1, "{advice:?}");
        assert!(advice[0].contains("contract"), "{advice:?}");
    }
}
