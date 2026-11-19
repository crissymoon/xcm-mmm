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
//! register with rows that the glossary never touches has one of two faults,
//! and the report separates them: the word list is missing its vocabulary, or
//! the rows are single sentences and most of them carry no wordy phrasing at
//! all. The first is fixed with words, the second is not a fault in the
//! instrument.
//!
//! The verdicts are deliberately coarse, and each one carries its threshold:
//!
//! A register has a role. A target register is drafting the instrument should
//! read, so a hit is the instrument working. A contrast register is copy that
//! should come out plain, so a hit is a false positive and the share is a cost.
//! Scoring both with one word would read a clean negative as a blind spot.
//!
//! * `covered`, at least [`MIN_ROWS`] rows from at least [`MIN_DOCS`] distinct
//!   sources and at least [`MIN_REACH`] of those rows carrying a hit;
//! * `thin`, rows present but fewer than [`MIN_ROWS`], so nothing is claimed
//!   from them;
//! * `narrow`, enough rows, but fewer than [`MIN_DOCS`] sources carry them, so
//!   the share describes those instruments and not the register;
//! * `sparse`, a wide enough sample the glossary reaches fewer than
//!   [`MIN_REACH`] of, but at least [`MIN_AVOIDABLE`] avoidable terms per 100
//!   words, so the text does carry drafting the instrument can remove and the
//!   row share is low because most rows are sentences with none, not because
//!   the words are missing;
//! * `blind`, a wide enough sample where both the row share is under
//!   [`MIN_REACH`] and the avoidable density is under [`MIN_AVOIDABLE`], so the
//!   glossary has nothing to say about the register at all;
//! * `clean`, a contrast register the glossary leaves alone;
//! * `leaky`, a contrast register where the glossary fires on more than
//!   [`MAX_LEAK`] of the rows;
//! * `absent`, no rows at all for that register in this bank.
//!
//! `absent` is the interesting column. Contracts, judicial opinions,
//! regulations and agency notices are each absent from the shipped bank, so
//! every claim the card makes stops at the four registers it does hold. The
//! widening pass fills them in a separate file, and then `narrow` takes over
//! the same job: a register can hold 249 rows and still be four documents.
//!
//! The taxonomy is not invented here. LegalBench (Guha et al., 2023, 162 tasks)
//! splits legal reasoning into six IRAC categories rather than one accuracy,
//! and LexGLUE (Chalkidis et al., 2022) evaluates on separate sub-corpora for
//! legislation, contracts and court cases for the same reason. CUAD
//! (Hendrycks et al., 2021) exists because contracts are a register of their
//! own: the parties define their own terms and the clauses are negotiated, so a
//! glossary read off statutes misses them. The list below is those registers
//! reduced to the ones this instrument could plausibly be asked to read.
//!
//! A plain restatement register was declared in an earlier draft and is gone.
//! No source in the bank begins with `plain`, so it could only ever report
//! `absent`, and a contrast register that no corpus can ever fill takes a line
//! in the table without measuring anything. The plain side is the ad copy and
//! the court self-help pages, which hold real text.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};

use serde::{Deserialize, Serialize};

use crate::num::round4;

use super::translate::{words, Legal};

/// Rows a register needs before a share over it means anything.
pub const MIN_ROWS: usize = 100;
/// Distinct sources a target register needs before a share over it describes
/// the register. A hundred rows cut from four opinions is a share over four
/// opinions, and the sentence count hides that while the source count does not.
pub const MIN_DOCS: usize = 20;
/// Share of rows the glossary must touch for a target register to count as read.
pub const MIN_REACH: f64 = 0.20;
/// Avoidable terms per 100 words below which drafting is not there to remove.
///
/// The row share answers "does the glossary see this register at all", and it
/// is the right question for a bank of clauses, where a clause without a term
/// is unusual. It is the wrong question for a bank of single sentences: a
/// sentence of judicial reasoning often holds no wordy phrasing, so the share
/// falls with the sentence style rather than with the word list. Density over
/// words is not sensitive to that. The two contrast registers, copy and legal
/// content already written for a lay reader, both measure 0.09; every drafting
/// register measures at or above 0.78, so three times the plain baseline is a
/// floor that separates the plain side from the drafting side with room to
/// spare.
pub const MIN_AVOIDABLE: f64 = 0.20;
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
        note: "rules of procedure and evidence, where every obligation has a deadline",
    },
    Register {
        name: "statute",
        markers: &[
            "usc-", "ca-ccp-", "ca-civ-", "ca-gov-", "ca-lamc-", "ca-lab-", "ca-pen-", "ca-bpc-",
            "ca-edu-", "ca-hs-", "ca-lac-",
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
        markers: &[
            "cap",
            "courtlistener",
            "opinion",
            "ca-opinion",
            "court-opinion",
            "holding",
        ],
        role: Role::Target,
        note: "judicial reasoning, which argues in prose rather than clauses",
    },
    Register {
        name: "agency-notice",
        markers: &["fedreg", "notice", "bulletin", "circular"],
        role: Role::Target,
        note: "notices, bulletins and forms that tell a person what to do next",
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
    /// Distinct sources the rows come from. A register can be long and still
    /// be one document, and this is the column that shows it.
    pub documents: usize,
    pub legal: usize,
    pub plain: usize,
    pub words: usize,
    /// Rows where at least one glossary term fired.
    pub reached: usize,
    pub hits: usize,
    pub hits_per_100: f64,
    /// Hits the rewrite would actually remove: avoidable terms outside a
    /// quotation or a citation. A citation is not wordiness, and a rule inside
    /// a quoted holding is not the court's own drafting.
    pub avoidable_hits: usize,
    pub avoidable_per_100: f64,
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
    /// Target registers the glossary reads at a normal density while touching
    /// few rows, because their rows are single sentences and most carry no
    /// drafting. Not a gap: the words are there to remove.
    pub sparse: Vec<String>,
    pub thin: Vec<String>,
    /// Registers with enough rows but too few distinct sources to speak for
    /// themselves. Their rows still count in the bank and in `reach`, and the
    /// share over them describes those sources.
    pub narrow: Vec<String>,
    /// Target registers with enough rows and enough sources behind them.
    pub held: Vec<String>,
    /// Contrast registers the glossary fires on too often.
    pub leaky: Vec<String>,
    /// The share of target rows the glossary touches. The instrument's reach
    /// over drafting, with the negative side left out of the denominator.
    /// Rows in `narrow` registers are counted here, because the glossary did
    /// read them; what they cannot carry is a claim about the register.
    pub reach: f64,
    /// The share of contrast rows the glossary touches: the false positive
    /// cost of scanning copy that is already plain.
    pub leak: f64,
    /// Sources no marker matched, sorted. These rows are counted in the bank
    /// total and in no register, so they are a fault in the table rather than a
    /// gap in the language, and they are reported as one.
    pub unmapped: Vec<String>,
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
        Self::audit_many(legal, &[bank_path])
    }

    /// Split one or more banks and scan every row with the shipped glossary.
    ///
    /// Several paths are what makes the widening pass visible: the shipped
    /// bank is reproducible and hashed, so text collected for the registers it
    /// lacks lands in a separate file, and the two are scored together here
    /// rather than by rewriting the shipped bank.
    pub fn audit_many(legal: &Legal, bank_paths: &[&str]) -> Result<Self, String> {
        let mut bank = Vec::new();
        for path in bank_paths {
            bank.extend(read_bank(path)?);
        }
        Self::audit_rows(legal, &bank)
    }

    fn audit_rows(legal: &Legal, bank: &[BankRow]) -> Result<Self, String> {
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
                role: "unmapped".to_string(),
                note: note("unclassified").to_string(),
                verdict: "none".to_string(),
                ..Default::default()
            },
        );

        let mut unmapped: std::collections::BTreeSet<String> = Default::default();
        let mut sources: BTreeMap<&'static str, std::collections::BTreeSet<&str>> =
            Default::default();
        for entry in bank {
            let name = classify(&entry.source);
            if name == "unclassified" {
                unmapped.insert(entry.source.trim().to_string());
            }
            sources.entry(name).or_default().insert(entry.source.trim());
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
            row.avoidable_hits += scan
                .terms
                .iter()
                .filter(|term| term.avoidable && !term.in_citation)
                .count();
            for term in &scan.terms {
                *row.kinds.entry(term.kind.clone()).or_default() += 1;
            }
        }

        let mut rows: Vec<Row> = buckets.into_values().collect();
        for row in &mut rows {
            row.documents = sources
                .get(row.register.as_str())
                .map(|set| set.len())
                .unwrap_or(0);
            if row.words > 0 {
                row.hits_per_100 = round4(row.hits as f64 * 100.0 / row.words as f64);
                row.avoidable_per_100 =
                    round4(row.avoidable_hits as f64 * 100.0 / row.words as f64);
            }
            row.verdict = verdict(
                row.rows,
                row.documents,
                row.reached,
                row.avoidable_per_100,
                &row.role,
            )
            .to_string();
        }
        // An empty mapping bucket is not a register with no text: it is a table
        // with nothing to complain about, and it should not take a line.
        let unmapped_rows: usize = rows
            .iter()
            .filter(|row| row.role == "unmapped")
            .map(|row| row.rows)
            .sum();
        rows.retain(|row| row.verdict != "none");
        let (reached, total) = tally(&rows, "target");
        let (leaked, contrast) = tally(&rows, "contrast");
        let held: Vec<String> = rows
            .iter()
            .filter(|row| row.role == "target" && row.verdict == "covered")
            .map(|row| row.register.clone())
            .collect();
        Ok(Self {
            absent: collect(&rows, "absent"),
            blind: collect(&rows, "blind"),
            sparse: collect(&rows, "sparse"),
            thin: collect(&rows, "thin"),
            narrow: collect(&rows, "narrow"),
            held,
            leaky: collect(&rows, "leaky"),
            reach: round4(reached as f64 / total.max(1) as f64),
            leak: round4(leaked as f64 / contrast.max(1) as f64),
            bank: total + contrast + unmapped_rows,
            unmapped: unmapped.into_iter().collect(),
            rows,
        })
    }

    /// What a widening effort should write next, in the order that buys the
    /// most: registers with no text, then registers with too little, then
    /// registers the glossary cannot see.
    pub fn widen(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.unmapped.is_empty() {
            out.push(format!(
                "map {} source{} the register table does not know: {}",
                self.unmapped.len(),
                if self.unmapped.len() == 1 { "" } else { "s" },
                self.unmapped.join(", ")
            ));
        }
        for name in self.absent.iter().filter(|name| is_target(name)) {
            out.push(format!(
                "add {name} text: no bank row exists, so nothing is claimed ({})",
                note(name)
            ));
        }
        for name in &self.narrow {
            let count = self.pick(name).map(|row| row.documents).unwrap_or(0);
            let rows = self.pick(name).map(|row| row.rows).unwrap_or(0);
            out.push(format!(
                "add {name} sources: {count} source{} carry all {rows} rows, so the share describes those documents",
                if count == 1 { "" } else { "s" }
            ));
        }
        for name in &self.blind {
            out.push(format!(
                "add {name} words: the glossary reaches under {:.0} percent of its rows and under {MIN_AVOIDABLE:.2} avoidable terms per 100 words",
                MIN_REACH * 100.0
            ));
        }
        for name in &self.sparse {
            let row = self.pick(name);
            let density = row.map(|row| row.avoidable_per_100).unwrap_or(0.0);
            let share = row
                .map(|row| row.reached as f64 / row.rows.max(1) as f64)
                .unwrap_or(0.0);
            out.push(format!(
                "read {name} sentences: {density:.2} avoidable terms per 100 words and {:.0} percent of rows, so the rows are single sentences rather than a gap in the word list",
                share * 100.0
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
            .chain(self.thin.iter())
            .chain(self.narrow.iter())
            .chain(self.blind.iter())
            .filter(|name| is_target(name))
            .cloned()
            .collect()
    }

    fn pick(&self, register: &str) -> Option<&Row> {
        self.rows.iter().find(|row| row.register == register)
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
            "{:<14} {:>8} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7} {:>7} {:>11}",
            "register",
            "role",
            "rows",
            "docs",
            "legal",
            "plain",
            "reached",
            "hits",
            "per100",
            "av/100",
            "verdict"
        ));
        for row in &self.rows {
            lines.push(format!(
                "{:<14} {:>8} {:>5} {:>5} {:>5} {:>5} {:>7} {:>6} {:>7.2} {:>7.2} {:>11}",
                row.register,
                row.role,
                row.rows,
                row.documents,
                row.legal,
                row.plain,
                row.reached,
                row.hits,
                row.hits_per_100,
                row.avoidable_per_100,
                row.verdict
            ));
        }
        let targets = REGISTERS
            .iter()
            .filter(|register| register.role == Role::Target)
            .count();
        let narrow: Vec<String> = self
            .narrow
            .iter()
            .map(|name| {
                format!(
                    "{name} {}",
                    self.pick(name).map(|row| row.documents).unwrap_or(0)
                )
            })
            .collect();
        lines.push(format!(
            "sample: {} of {targets} target registers {} at least {MIN_DOCS} sources{}",
            self.held.len(),
            if self.held.len() == 1 {
                "carries"
            } else {
                "carry"
            },
            if narrow.is_empty() {
                String::new()
            } else {
                format!("; narrow: {}", narrow.join(", "))
            }
        ));
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

/// The verdict for one register: how many rows it holds, how many distinct
/// sources carry them, how many rows the glossary reached, and what a hit there
/// means.
///
/// `narrow` outranks `blind` for a target register. A share over four opinions
/// cannot say whether the glossary is blind to judicial reasoning or whether
/// those four opinions are unusual, so the honest next step is more opinions
/// rather than more words. Once the sample is wide, `blind` means what it says.
pub fn verdict(
    rows: usize,
    documents: usize,
    reached: usize,
    avoidable_per_100: f64,
    role: &str,
) -> &'static str {
    // A mapping bucket is never a register, so the row floor says nothing
    // about it: it is a table fault when it holds rows and absent when it does.
    if role == "unmapped" {
        return if rows == 0 { "none" } else { "unmapped" };
    }
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
            if documents < MIN_DOCS {
                "narrow"
            } else if share < MIN_REACH {
                if avoidable_per_100 < MIN_AVOIDABLE {
                    "blind"
                } else {
                    "sparse"
                }
            } else {
                "covered"
            }
        }
    }
}

/// True when a register name is one the instrument is meant to read. A target
/// register is a gap when it is absent, thin or blind; a contrast register is
/// the reference side, and `unclassified` is a fault in the table rather than a
/// register, so neither is reported as a language gap.
fn is_target(name: &str) -> bool {
    REGISTERS
        .iter()
        .find(|register| register.name == name)
        .map(|register| register.role == Role::Target)
        .unwrap_or(false)
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
    fn a_collected_row_lands_in_the_register_it_was_fetched_for() {
        assert_eq!(classify("cuad-00200"), "contract");
        assert_eq!(classify("cfr-29-1604"), "regulation");
        assert_eq!(classify("fedreg-2026-23515"), "agency-notice");
        assert_eq!(classify("cap-f3d-200-4233516"), "opinion");
    }

    #[test]
    fn the_audit_reads_every_bank_it_is_given() {
        let legal = Legal::load("data/legal").unwrap();
        let shipped = Coverage::audit(&legal, "data/legal/bank.jsonl").unwrap();
        let both = Coverage::audit_many(
            &legal,
            &["data/legal/bank.jsonl", "data/legal/bank-ext.jsonl"],
        )
        .unwrap();
        assert!(
            both.bank > shipped.bank,
            "{} vs {}",
            both.bank,
            shipped.bank
        );
        // The collected text is what turns the absent registers into measured
        // ones; a merge that silently dropped a file would leave them absent.
        for register in ["contract", "opinion", "regulation", "agency-notice"] {
            let row = both.rows.iter().find(|r| r.register == register).unwrap();
            assert!(row.rows >= 100, "{register}: {}", row.rows);
            assert_ne!(row.verdict, "absent", "{register}");
        }
    }

    #[test]
    fn a_register_with_no_rows_is_absent_before_anything_else() {
        assert_eq!(verdict(0, 0, 0, 0.0, "target"), "absent");
        assert_eq!(verdict(0, 0, 0, 0.0, "contrast"), "absent");
    }

    #[test]
    fn a_register_below_the_row_floor_is_thin_rather_than_scored() {
        assert_eq!(
            verdict(MIN_ROWS - 1, MIN_DOCS, MIN_ROWS - 1, 0.0, "target"),
            "thin"
        );
        assert_eq!(verdict(MIN_ROWS - 1, 1, 0, 0.0, "contrast"), "thin");
    }

    #[test]
    fn a_mapping_bucket_is_judged_by_the_table_and_not_by_the_row_floor() {
        assert_eq!(verdict(0, 0, 0, 0.0, "unmapped"), "none");
        assert_eq!(verdict(3, 1, 2, 0.0, "unmapped"), "unmapped");
    }

    #[test]
    fn an_unmapped_source_is_a_table_fault_and_not_a_language_gap() {
        let coverage = Coverage {
            rows: vec![row("unclassified", "unmapped", "unmapped", 1)],
            unmapped: vec!["random-crawl".to_string()],
            ..Default::default()
        };
        assert!(coverage.gaps().is_empty(), "{:?}", coverage.gaps());
        let advice = coverage.widen();
        assert_eq!(advice.len(), 1, "{advice:?}");
        assert!(advice[0].contains("random-crawl"), "{advice:?}");
    }

    #[test]
    fn a_target_register_is_blind_when_the_glossary_rarely_reaches_it() {
        assert_eq!(verdict(1000, MIN_DOCS, 100, 0.0, "target"), "blind");
        assert_eq!(
            verdict(1000, MIN_DOCS, MIN_ROWS * 2, 0.0, "target"),
            "covered"
        );
    }

    #[test]
    fn a_register_of_single_sentences_is_sparse_when_the_density_is_normal() {
        // 250 opinion sentences, 74 sources, 45 reached, 0.78 avoidable terms
        // per 100 words: court rules measure 0.78 too, so the words are there
        // and the row share is low because most sentences carry none.
        assert_eq!(verdict(250, 74, 45, 0.78, "target"), "sparse");
        // The same rows with the density of the self-help pages are a gap in
        // the word list, not a sentence style.
        assert_eq!(verdict(250, 74, 45, 0.06, "target"), "blind");
    }

    #[test]
    fn a_contrast_register_is_clean_when_the_glossary_leaves_it_alone() {
        assert_eq!(verdict(1000, 1, 10, 0.0, "contrast"), "clean");
        assert_eq!(verdict(1000, 1, 500, 0.0, "contrast"), "leaky");
    }

    fn row(name: &str, role: &str, verdict: &str, documents: usize) -> Row {
        Row {
            register: name.to_string(),
            role: role.to_string(),
            verdict: verdict.to_string(),
            documents,
            ..Default::default()
        }
    }

    #[test]
    fn a_target_register_is_narrow_when_few_sources_carry_its_rows() {
        // Four F.3d opinions, 249 sentences, 62 reached: a real share over four
        // documents, and not a claim about judicial reasoning.
        assert_eq!(verdict(249, 4, 62, 0.0, "target"), "narrow");
        // 137 rules, 674 sentences, 172 reached: the sample can speak.
        assert_eq!(verdict(674, 137, 172, 0.0, "target"), "covered");
    }

    #[test]
    fn narrow_outranks_blind_because_few_documents_cannot_show_which_fault_it_is() {
        assert_eq!(verdict(249, 4, 0, 0.0, "target"), "narrow");
        assert_eq!(verdict(249, MIN_DOCS, 0, 0.0, "target"), "blind");
    }

    #[test]
    fn the_gap_list_holds_target_registers_and_not_the_plain_side() {
        let mut contract = row("contract", "target", "narrow", 13);
        contract.rows = 247;
        let coverage = Coverage {
            rows: vec![
                contract,
                row("court-rule", "target", "covered", 137),
                row("ad-copy", "contrast", "absent", 0),
            ],
            absent: vec!["ad-copy".to_string()],
            narrow: vec!["contract".to_string()],
            ..Default::default()
        };
        assert_eq!(coverage.gaps(), vec!["contract".to_string()]);
        let advice = coverage.widen();
        assert_eq!(advice.len(), 1, "{advice:?}");
        assert!(advice[0].contains("contract"), "{advice:?}");
        assert!(advice[0].contains("13 sources"), "{advice:?}");
    }
}
