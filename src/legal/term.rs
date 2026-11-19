//! The glossary a legal word detector reads, and the bands it reports in.
//!
//! Every row is a term a person could have written plainly, with the plain
//! wording beside it. Two fields carry the argument of the module:
//!
//! * `avoidable` says whether the term can be replaced without losing
//!   meaning. "pursuant to" is avoidable; "res judicata" is a term of art that
//!   names a doctrine, and a translator that replaces it has not simplified the
//!   text, it has destroyed it. A density that counts both together punishes
//!   the second for the fault of the first, so they are counted apart.
//! * `kind` groups the rows by what kind of legal English they are: an archaic
//!   deictic (`herein`), a modal (`shall`), a Latin borrowing, a doublet, a
//!   nominalisation, a procedural term. The distribution over kinds is the
//!   reading; a single total is not.
//!
//! Matching is literal and case insensitive. Data supplies words, not regular
//! expressions, so a term cannot smuggle a pattern into the detector: `et seq.`
//! and `and/or` are escaped before they are compiled. Boundaries are checked by
//! hand because `herein` must not fire inside `hereinunder` and the regex crate
//! has no look-around to say so in one pattern.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use super::guard::Span;

fn yes() -> bool {
    true
}

/// One glossary row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Term {
    pub id: String,
    pub term: String,
    pub plain: String,
    pub kind: String,
    /// Whether the term can be replaced without losing meaning.
    pub avoidable: bool,
    /// Whether the term is reported as advice rather than rewritten. Only a row
    /// that says so may have a rule that does not apply, and `check` enforces it.
    #[serde(default)]
    pub advisory: bool,
    /// Whether a term of art is glossed in place. False for citation shorthand,
    /// where a gloss would interrupt the citation it belongs to.
    #[serde(default = "yes")]
    pub explain: bool,
    /// The authority that gives the term its operative force, when there is one.
    #[serde(default)]
    pub law: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub variants: Vec<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

impl Term {
    /// The term and every spelling of it.
    pub fn spellings(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.term.as_str()).chain(self.variants.iter().map(|v| v.as_str()))
    }

    /// A pointer rather than a word: `id.`, `supra`, `infra`, `cf.`, `et seq.`.
    ///
    /// The rows are in the glossary because a reader stops at them, and outside
    /// the definition list because a gloss in place would split the citation.
    /// The brief reads them once, at the end, so they are neither glossed,
    /// replaced nor reported as a rare word in the middle of a clause.
    pub fn shorthand(&self) -> bool {
        !self.explain
    }
}

/// One occurrence of a term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// Index into [`Lexicon::terms`].
    pub index: usize,
    pub span: Span,
    pub text: String,
}

/// How rare in general English a word has to be before a reader is assumed not
/// to know it.
///
/// It lives here rather than in [`super::freq`] because it is a judgment about
/// readers, not a fact about a corpus, and the corpus is only the probe. The
/// note travels with the threshold for the same reason every other threshold in
/// this data carries one: a cut a person cannot argue with is a cut nobody can
/// check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rarity {
    /// Occurrences per million tokens of general English below which a word is
    /// reported as one the reader may not have met.
    pub max_per_million: f64,
    #[serde(default)]
    pub note: String,
}

/// A density band from `settings.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Band {
    pub name: String,
    pub max_per_100: f64,
    #[serde(default)]
    pub note: String,
}

/// The word lists a brief is read with, from `settings.json`.
///
/// They live in the data for the same reason the citation patterns do: a cue a
/// person can read and edit is a published convention of the corpus, not a
/// constant buried in the compiler. Order does not matter; the matcher always
/// takes the earliest cue, and the longest at a shared start, so `shall not`
/// wins over `shall` without depending on which line comes first.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClauseWords {
    #[serde(default)]
    pub duty: Vec<String>,
    #[serde(default)]
    pub prohibition: Vec<String>,
    #[serde(default)]
    pub permission: Vec<String>,
    #[serde(default)]
    pub condition: Vec<String>,
    #[serde(default)]
    pub deadline: Vec<String>,
    #[serde(default)]
    pub negation: Vec<String>,
    /// Negated subjects that state a limit rather than forbid an act, such as
    /// "Nothing in this section shall ...". Such a sentence is a statement.
    #[serde(default)]
    pub statement: Vec<String>,
    /// Shortenings whose full stop does not end a sentence.
    #[serde(default)]
    pub abbreviations: Vec<String>,
    /// Irregular past participles. A regular one ends in `ed` and needs no
    /// list; these do not, and without them "shall be paid" reads as an
    /// adjectival "shall be liable".
    #[serde(default)]
    pub participles: Vec<String>,
    /// Names a contract defines for its own cast, which are not terms of art:
    /// `the Buyer`, `the Agreement`, `the Term`. `legal explain --check` stays
    /// quiet about these, because the glossary rightly carries no row for them.
    #[serde(default)]
    pub defined_ignore: Vec<String>,
}

/// The settings beside the glossary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub version: u32,
    /// The notice every brief carries, so the text and the packaging cards
    /// cannot drift apart.
    #[serde(default)]
    pub disclaimer: String,
    pub bands: Vec<Band>,
    /// Citation patterns the rewrite must not touch.
    #[serde(default)]
    pub protect: Vec<String>,
    /// The cues `legal explain` sorts a sentence by.
    #[serde(default)]
    pub clauses: ClauseWords,
    /// The frequency cut the rarity reading is reported at. Absent means the
    /// instrument can still scan and rewrite; it cannot say which of its own
    /// terms a general reader has never met, and asking it to is an error rather
    /// than an empty answer.
    #[serde(default)]
    pub rarity: Option<Rarity>,
}

impl Settings {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let settings: Settings = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        if settings.bands.is_empty() {
            return Err(format!("{path}: no bands declared"));
        }
        if settings
            .bands
            .windows(2)
            .any(|w| w[0].max_per_100 > w[1].max_per_100)
        {
            return Err(format!("{path}: bands are not in ascending order"));
        }
        // A rarity cut of zero reports nothing and a negative one reports
        // everything, and both look like a working reading in the output. The
        // threshold has to be a real rate.
        if let Some(rarity) = &settings.rarity {
            if rarity.max_per_million <= 0.0 || !rarity.max_per_million.is_finite() {
                return Err(format!(
                    "{path}: rarity.max_per_million must be above zero, not {}",
                    rarity.max_per_million
                ));
            }
        }
        Ok(settings)
    }

    /// The band a density falls in. The last band is open ended, so a very dense
    /// text is named rather than defaulting to the middle.
    pub fn band(&self, per_100: f64) -> &Band {
        self.bands
            .iter()
            .find(|b| per_100 <= b.max_per_100)
            .unwrap_or_else(|| self.bands.last().expect("bands are never empty"))
    }
}

/// The compiled glossary.
#[derive(Clone, Debug)]
pub struct Lexicon {
    terms: Vec<Term>,
    patterns: Vec<Vec<Regex>>,
}

impl Lexicon {
    /// Read `terms.jsonl`. A line that does not parse is a refusal with its
    /// number, not a silently shorter glossary.
    pub fn load(path: &str) -> Result<Self, String> {
        let file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut terms = Vec::new();
        for (number, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{path}: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let term: Term =
                serde_json::from_str(&line).map_err(|e| format!("{path}:{}: {e}", number + 1))?;
            terms.push(term);
        }
        Self::new(terms)
    }

    /// Validate and compile.
    pub fn new(terms: Vec<Term>) -> Result<Self, String> {
        let mut ids = BTreeSet::new();
        for term in &terms {
            if term.id.trim().is_empty() || term.term.trim().is_empty() {
                return Err(format!("a row has an empty id or term: {term:?}"));
            }
            if term.plain.trim().is_empty() {
                return Err(format!("{}: no plain reading", term.id));
            }
            if term.kind.trim().is_empty() {
                return Err(format!("{}: no kind", term.id));
            }
            if !ids.insert(term.id.clone()) {
                return Err(format!("duplicate term id {}", term.id));
            }
        }
        let mut patterns = Vec::with_capacity(terms.len());
        for term in &terms {
            let mut compiled = Vec::new();
            for spelling in term.spellings() {
                let regex = RegexBuilder::new(&tolerant(spelling))
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| format!("{}: {e}", term.id))?;
                compiled.push(regex);
            }
            patterns.push(compiled);
        }
        Ok(Self { terms, patterns })
    }

    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn term(&self, hit: &Hit) -> &Term {
        &self.terms[hit.index]
    }

    /// Whether a glossary row carries `phrase`, in any spelling and whatever the
    /// case. Used by the gap check, which asks the glossary a question the
    /// matcher cannot: does a term this document defines have a row at all.
    pub fn knows(&self, phrase: &str) -> bool {
        let wanted = normalise(phrase);
        !wanted.is_empty()
            && self.terms.iter().any(|term| {
                term.spellings()
                    .any(|spelling| normalise(spelling) == wanted)
            })
    }

    /// Every occurrence, longest match first at a shared start, with no term
    /// reported inside another.
    pub fn find(&self, text: &str) -> Vec<Hit> {
        let mut hits = Vec::new();
        for (index, patterns) in self.patterns.iter().enumerate() {
            for pattern in patterns {
                for found in pattern.find_iter(text) {
                    if !bounded(text, found.start(), found.end()) {
                        continue;
                    }
                    hits.push(Hit {
                        index,
                        span: Span::new(found.start(), found.end()),
                        text: found.as_str().to_string(),
                    });
                }
            }
        }
        hits.sort_by(|a, b| {
            a.span
                .start
                .cmp(&b.span.start)
                .then(b.span.len().cmp(&a.span.len()))
                .then(a.index.cmp(&b.index))
        });
        let mut kept: Vec<Hit> = Vec::with_capacity(hits.len());
        let mut cursor = 0;
        for hit in hits {
            if hit.span.start >= cursor {
                cursor = hit.span.end;
                kept.push(hit);
            }
        }
        kept
    }

    /// Rows per kind, for the report that shows what sort of legal English a
    /// text is made of rather than only how much.
    pub fn kinds(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for term in &self.terms {
            *out.entry(term.kind.clone()).or_insert(0) += 1;
        }
        out
    }
}

/// A literal spelling with any run of whitespace allowed between its words.
///
/// Fetched page text is hard wrapped, so a phrase arrives split across a line
/// as often as not. A pattern built on one space would miss those occurrences
/// and the reader would report a term the rewrite never saw.
fn tolerant(spelling: &str) -> String {
    spelling
        .split_whitespace()
        .map(regex::escape)
        .collect::<Vec<_>>()
        .join(r"\s+")
}

/// A spelling reduced to what a lookup compares: lower case, one space between
/// words, no punctuation a sentence wraps around it.
fn normalise(spelling: &str) -> String {
    spelling
        .trim_matches(|c: char| !c.is_alphanumeric())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Whether a match stands on its own: a word character before or after it means
/// the match is part of a longer word and is not a term.
fn bounded(text: &str, start: usize, end: usize) -> bool {
    let first = text[start..end].chars().next();
    let last = text[start..end].chars().next_back();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let before = text[..start].chars().next_back().map(word).unwrap_or(false);
    let after = text[end..].chars().next().map(word).unwrap_or(false);
    if first.map(word).unwrap_or(false) && before {
        return false;
    }
    if last.map(word).unwrap_or(false) && after {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(id: &str, term: &str, kind: &str, avoidable: bool) -> Term {
        Term {
            id: id.into(),
            term: term.into(),
            plain: "plain".into(),
            kind: kind.into(),
            avoidable,
            advisory: false,
            explain: true,
            law: String::new(),
            note: String::new(),
            variants: Vec::new(),
            sources: Vec::new(),
        }
    }

    fn lexicon() -> Lexicon {
        Lexicon::new(vec![
            term("herein", "herein", "deictic", true),
            term("hereinunder", "hereinunder", "deictic", true),
            term("null-and-void", "null and void", "doublet", true),
            term("void", "void", "operative", false),
            term("and-or", "and/or", "doublet", true),
        ])
        .unwrap()
    }

    #[test]
    fn a_row_that_is_not_glossed_in_place_is_a_shorthand() {
        let mut pointer = term("supra", "supra", "citation", false);
        pointer.explain = false;
        let art = term("laches", "laches", "doctrine", false);
        assert!(pointer.shorthand());
        assert!(!art.shorthand());
    }

    #[test]
    fn a_term_does_not_fire_inside_a_longer_word() {
        let found = lexicon().find("the rules hereinunder stated");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "hereinunder");
    }

    #[test]
    fn the_longer_term_wins_at_the_same_start() {
        let found = lexicon().find("the lease is null and void today");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "null and void");
    }

    #[test]
    fn a_phrase_survives_a_line_break() {
        let found = lexicon().find("the lease is null and\nvoid today");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "null and\nvoid");
    }

    #[test]
    fn punctuation_inside_a_term_is_literal() {
        let found = lexicon().find("a landlord and/or tenant may repair");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "and/or");
    }

    #[test]
    fn a_duplicate_id_is_refused() {
        let err = Lexicon::new(vec![
            term("a", "herein", "deictic", true),
            term("a", "thereof", "deictic", true),
        ])
        .unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
    }

    #[test]
    fn a_row_without_a_plain_reading_is_refused() {
        let mut row = term("a", "herein", "deictic", true);
        row.plain = " ".into();
        assert!(Lexicon::new(vec![row]).unwrap_err().contains("plain"));
    }

    #[test]
    fn bands_are_read_as_ceiling_values() {
        let settings = Settings {
            version: 1,
            disclaimer: String::new(),
            clauses: ClauseWords::default(),
            bands: vec![
                Band {
                    name: "plain".into(),
                    max_per_100: 1.0,
                    note: String::new(),
                },
                Band {
                    name: "dense".into(),
                    max_per_100: 100.0,
                    note: String::new(),
                },
            ],
            protect: Vec::new(),
            rarity: None,
        };
        assert_eq!(settings.band(0.4).name, "plain");
        assert_eq!(settings.band(3.0).name, "dense");
        assert_eq!(settings.band(1e6).name, "dense");
    }
}
