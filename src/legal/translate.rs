//! Reading a document for legal English, and writing it again in plain words.
//!
//! Three readings come out of one pass:
//!
//! * the terms found, with their offsets, so a person can see *which* words
//!   caused the verdict rather than only the verdict;
//! * a density of avoidable terms per hundred words, placed in a band that
//!   travels in `settings.json`, so the threshold is a published convention of
//!   this corpus and not a constant buried in code;
//! * the plain rewrite, with every edit traced to a rule and a reason.
//!
//! Two honesty rules shape the numbers. A term inside a quotation or a citation
//! is reported but not counted as avoidable, because the writer is naming a
//! statute, not drafting like one. And a term of art contributes to `necessary`
//! rather than to the density, because "res judicata" cannot be replaced by a
//! plain word without becoming a different claim.
//!
//! `explain` is the second half of the translator: a term of art is left in
//! place and glossed on first use, which is what a reader needs, while an
//! avoidable phrase is replaced, which is what a writer needs.

use std::collections::BTreeMap;
use std::fs;

use serde::Serialize;

use super::cues;
use super::freq::Freq;
use super::guard::{Guard, Span};
use super::rewrite::{Change, Rules};
use super::term::{Hit, Lexicon, Rarity, Settings, Term};

/// One term found in a text.
#[derive(Clone, Debug, Serialize)]
pub struct TermHit {
    pub id: String,
    pub term: String,
    pub plain: String,
    pub kind: String,
    pub avoidable: bool,
    /// Reported as advice rather than rewritten.
    pub advisory: bool,
    pub span: Span,
    /// True when the term sits inside a quotation or a citation.
    pub in_citation: bool,
    /// The word in the term that general English rarely uses, when there is
    /// one, and the rate it was measured at.
    pub rare: Option<String>,
    pub rare_per_million: f64,
}

/// The counts a reading is made of.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub words: usize,
    pub terms: usize,
    pub avoidable: usize,
    /// Avoidable terms that are reported as advice rather than rewritten.
    pub advisory: usize,
    pub necessary: usize,
    pub in_citation: usize,
    /// Hits whose term contains a word general English rarely uses: the words a
    /// reader has to stop at, which a count of drafting words alone cannot see.
    pub rare: usize,
    pub avoidable_per_100: f64,
    pub terms_per_100: f64,
    pub kinds: BTreeMap<String, usize>,
    pub band: String,
    pub band_note: String,
}

/// A document read for legal English.
#[derive(Clone, Debug, Serialize)]
pub struct Scan {
    pub words: usize,
    pub stats: Stats,
    pub terms: Vec<TermHit>,
}

/// A gloss inserted after a term of art.
#[derive(Clone, Debug, Serialize)]
pub struct Gloss {
    pub id: String,
    pub term: String,
    pub plain: String,
    /// Where the term sat in the text the gloss was added to.
    pub span: Span,
}

/// A document translated.
#[derive(Clone, Debug, Serialize)]
pub struct Reading {
    pub text: String,
    pub words_in: usize,
    pub words_out: usize,
    pub stats: Stats,
    /// The same reading taken of the output, so a run can show what is left.
    pub after: Stats,
    pub changes: Vec<Change>,
    pub suggestions: Vec<Change>,
    pub glosses: Vec<Gloss>,
    /// Avoidable terms that survived, by id.
    pub remaining: Vec<String>,
}

/// The loaded instrument: glossary, rules, guard and bands.
#[derive(Clone, Debug)]
pub struct Legal {
    pub lexicon: Lexicon,
    pub rules: Rules,
    pub guard: Guard,
    pub settings: Settings,
    /// How often each word occurs in general English. The probe behind the
    /// rarity reading, and the only part of the instrument measured off a corpus
    /// rather than written by hand.
    pub freq: Freq,
    /// The clause cues, compiled once. A register audit reads thousands of
    /// rows, and compiling fifty patterns per row is the difference between a
    /// second and a minute.
    cues: Option<cues::Cues>,
}

impl Legal {
    /// Load a folder holding `terms.jsonl`, `rewrites.jsonl` and
    /// `settings.json`, plus the frequency table `freq.json` when the folder has
    /// one.
    ///
    /// The table is optional because a folder someone wrote by hand holds a
    /// glossary and no corpus counts, and refusing to scan it would make the
    /// instrument unusable for the case it was built for. A table that is there
    /// and broken is still a refusal: the difference between "not measured" and
    /// "measured wrongly" is the whole point of the file.
    pub fn load(dir: &str) -> Result<Self, String> {
        let lexicon = Lexicon::load(&format!("{dir}/terms.jsonl"))?;
        let rules = Rules::load(&format!("{dir}/rewrites.jsonl"))?;
        let settings = Settings::load(&format!("{dir}/settings.json"))?;
        let guard = Guard::new(&settings.protect)?;
        let path = format!("{dir}/freq.json");
        let freq = if std::path::Path::new(&path).exists() {
            Freq::load(&path)?
        } else {
            Freq::default()
        };
        Self::new(lexicon, rules, guard, settings, freq)
    }

    pub fn new(
        lexicon: Lexicon,
        rules: Rules,
        guard: Guard,
        settings: Settings,
        freq: Freq,
    ) -> Result<Self, String> {
        if lexicon.is_empty() {
            return Err("the glossary is empty".into());
        }
        // A glossary can be loaded without clause cues; only a brief needs
        // them, so a missing cue table is reported when it is asked for rather
        // than turning every scan into a failure.
        let cues = cues::Cues::new(&settings.clauses).ok();
        Ok(Self {
            lexicon,
            rules,
            guard,
            settings,
            freq,
            cues,
        })
    }

    /// The clause cues, compiled once at load.
    pub fn cues(&self) -> Result<&cues::Cues, String> {
        self.cues.as_ref().ok_or_else(|| {
            "settings.json declares no clause cues: `clauses.duty` and \
             `clauses.permission` are required for a brief"
                .to_string()
        })
    }

    /// The frequency cut the rarity reading is reported at. A glossary without
    /// one can still be scanned and rewritten; what it cannot do is say which of
    /// its own terms a general reader has never met.
    pub fn rarity(&self) -> Result<&Rarity, String> {
        self.settings.rarity.as_ref().ok_or_else(|| {
            "settings.json declares no rarity cut: `rarity.max_per_million` is \
             required for a rarity reading"
                .to_string()
        })
    }

    /// The word that makes a glossary row unknown to a general reader. None
    /// when the folder carries no frequency table: an empty table would report
    /// every word as unmeasured and every term as rare.
    pub fn rare_word(&self, term: &Term) -> Option<(String, f64)> {
        if self.freq.per_million.is_empty() {
            return None;
        }
        let below = self.settings.rarity.as_ref()?.max_per_million;
        let (word, rate) = self.freq.rarest(&term.term)?;
        if rate < below {
            Some((word, rate))
        } else {
            None
        }
    }

    /// Read a text for legal English.
    pub fn scan(&self, text: &str) -> Scan {
        let protected = self.guard.spans(text);
        let words = words(text);
        let hits = self.lexicon.find(text);
        let terms: Vec<TermHit> = hits
            .iter()
            .map(|hit| self.hit(hit, text, &protected))
            .collect();
        let stats = self.stats(&terms, words);
        Scan {
            words,
            stats,
            terms,
        }
    }

    /// Translate a text. `explain` glosses terms of art in place; the rewrite
    /// always runs, and advisory rules are always reported rather than applied.
    pub fn translate(&self, text: &str, explain: bool) -> Reading {
        let protected = self.guard.spans(text);
        let applied = self.rules.apply(text, &protected);

        let mut output = applied.text;
        let mut glosses = Vec::new();
        if explain {
            let (glossed, added) = self.gloss(&output);
            output = glossed;
            glosses = added;
        }

        let scan_in = self.scan(text);
        let scan_out = self.scan(&output);
        let remaining = scan_out
            .terms
            .iter()
            .filter(|t| t.avoidable)
            .map(|t| t.id.clone())
            .collect();

        Reading {
            words_in: scan_in.words,
            words_out: scan_out.words,
            text: output,
            stats: scan_in.stats,
            after: scan_out.stats,
            changes: applied.changes,
            suggestions: applied.suggestions,
            glosses,
            remaining,
        }
    }

    /// The glossary as rows, for a listing.
    pub fn terms(&self) -> &[Term] {
        self.lexicon.terms()
    }

    fn hit(&self, hit: &Hit, text: &str, protected: &[Span]) -> TermHit {
        let term = self.lexicon.term(hit);
        let rare = self.rare_word(term);
        TermHit {
            id: term.id.clone(),
            term: text[hit.span.start..hit.span.end].to_string(),
            plain: term.plain.clone(),
            kind: term.kind.clone(),
            avoidable: term.avoidable,
            advisory: term.advisory,
            span: hit.span,
            in_citation: protected.iter().any(|s| s.intersects(&hit.span)),
            rare_per_million: rare.as_ref().map(|(_, rate)| *rate).unwrap_or(0.0),
            rare: rare.map(|(word, _)| word),
        }
    }

    fn stats(&self, terms: &[TermHit], words: usize) -> Stats {
        let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
        let (mut avoidable, mut advisory, mut necessary, mut quoted, mut rare) = (0, 0, 0, 0, 0);
        for term in terms {
            if term.in_citation {
                quoted += 1;
                continue;
            }
            *kinds.entry(term.kind.clone()).or_insert(0) += 1;
            if term.rare.is_some() {
                rare += 1;
            }
            if term.avoidable {
                avoidable += 1;
                if term.advisory {
                    advisory += 1;
                }
            } else {
                necessary += 1;
            }
        }
        let per = |count: usize| {
            if words == 0 {
                0.0
            } else {
                round2(count as f64 * 100.0 / words as f64)
            }
        };
        let avoidable_per_100 = per(avoidable);
        let band = self.settings.band(avoidable_per_100).clone();
        Stats {
            words,
            terms: terms.len(),
            avoidable,
            advisory,
            necessary,
            in_citation: quoted,
            rare,
            avoidable_per_100,
            terms_per_100: per(terms.len()),
            kinds,
            band: band.name,
            band_note: band.note,
        }
    }

    /// Gloss each necessary term once, where it first appears outside a
    /// quotation, unless a parenthesis already follows it.
    fn gloss(&self, text: &str) -> (String, Vec<Gloss>) {
        let protected = self.guard.spans(text);
        let hits = self.lexicon.find(text);
        let mut seen = BTreeMap::new();
        let mut inserts: Vec<(usize, String, Gloss)> = Vec::new();
        for hit in &hits {
            let term = self.lexicon.term(hit);
            if term.avoidable || !term.explain {
                continue;
            }
            if protected.iter().any(|s| s.intersects(&hit.span)) {
                continue;
            }
            if seen.insert(term.id.clone(), ()).is_some() {
                continue;
            }
            let next = text[hit.span.end..].trim_start().chars().next();
            if next == Some('(') {
                continue;
            }
            inserts.push((
                hit.span.end,
                format!(" ({})", term.plain),
                Gloss {
                    id: term.id.clone(),
                    term: hit.text.clone(),
                    plain: term.plain.clone(),
                    span: hit.span,
                },
            ));
        }
        let mut out = text.to_string();
        for (at, insert, _) in inserts.iter().rev() {
            out.insert_str(*at, insert);
        }
        (
            out,
            inserts.into_iter().map(|(_, _, gloss)| gloss).collect(),
        )
    }

    /// The rules that have no glossary row behind them, and the avoidable terms
    /// that no applying rule reaches. Both are reported by `check` rather than
    /// assumed away.
    ///
    /// A `phrase` rule and a `clause` rule are exempt from the orphan check on
    /// purpose: a phrase rule is a spelling of a glossary row, and a clause rule
    /// rewrites a construction the glossary cannot hold, so neither is expected
    /// to carry an id the glossary knows.
    pub fn coverage(&self) -> (Vec<String>, Vec<String>) {
        let ids: std::collections::BTreeSet<&str> =
            self.lexicon.terms().iter().map(|t| t.id.as_str()).collect();
        let orphans = self
            .rules
            .rules()
            .iter()
            .filter(|r| !ids.contains(r.id.as_str()) && r.kind != "phrase" && r.kind != "clause")
            .map(|r| r.id.clone())
            .collect();
        let applied: std::collections::BTreeSet<&str> =
            self.rules.applying().map(|r| r.id.as_str()).collect();
        let gaps = self
            .lexicon
            .terms()
            .iter()
            .filter(|t| t.avoidable && !t.advisory && !applied.contains(t.id.as_str()))
            .map(|t| t.id.clone())
            .collect();
        (orphans, gaps)
    }
}

/// Words, counted as runs of text holding at least one letter or digit.
pub fn words(text: &str) -> usize {
    text.split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_alphanumeric()))
        .count()
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Read a text from a file, so a document is one command rather than a pasted
/// paragraph.
pub fn read_text(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legal::rewrite::{Case, Rule};
    use crate::legal::term::{Band, ClauseWords};

    /// A rarity cut and a frequency table just large enough to tell a drafting
    /// word from an ordinary one, so a test never depends on the shipped corpus.
    fn rarity() -> Rarity {
        Rarity {
            max_per_million: 1.0,
            note: "a test cut".into(),
        }
    }

    fn freq() -> Freq {
        let mut per_million = std::collections::BTreeMap::new();
        for (word, rate) in [
            ("in", 9000.0),
            ("this", 4000.0),
            ("document", 400.0),
            ("under", 600.0),
            ("herein", 0.4),
            ("hereunder", 0.0),
        ] {
            per_million.insert(word.to_string(), rate);
        }
        Freq {
            source: "a test corpus".into(),
            tokens: 1_000_000,
            per_million,
        }
    }

    fn term(id: &str, word: &str, plain: &str, kind: &str, avoidable: bool) -> Term {
        Term {
            id: id.into(),
            term: word.into(),
            plain: plain.into(),
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

    fn rule(id: &str, pattern: &str, replace: &str) -> Rule {
        Rule {
            id: id.into(),
            pattern: pattern.into(),
            replace: replace.into(),
            kind: "phrase".into(),
            why: "plain word".into(),
            applies: true,
            priority: 0,
            case: Case::Preserve,
        }
    }

    fn legal() -> Legal {
        let lexicon = Lexicon::new(vec![
            term("herein", "herein", "in this document", "deictic", true),
            term("pursuant-to", "pursuant to", "under", "phrase", true),
            term(
                "res-judicata",
                "res judicata",
                "a matter already decided",
                "latin",
                false,
            ),
        ])
        .unwrap();
        let rules = Rules::new(vec![
            rule("herein", r"\bherein\b", "in this document"),
            rule("pursuant-to", r"\bpursuant to\b", "under"),
        ])
        .unwrap();
        let settings = Settings {
            version: 1,
            disclaimer: String::new(),
            clauses: ClauseWords::default(),
            bands: vec![
                Band {
                    name: "plain".into(),
                    max_per_100: 1.0,
                    note: "no drafting words".into(),
                },
                Band {
                    name: "dense".into(),
                    max_per_100: 1e9,
                    note: "drafted".into(),
                },
            ],
            protect: vec![r"§+ ?[0-9.]+".into()],
            rarity: Some(rarity()),
        };
        let guard = Guard::new(&settings.protect).unwrap();
        Legal::new(lexicon, rules, guard, settings, freq()).unwrap()
    }

    #[test]
    fn an_avoidable_phrase_is_replaced_and_counted() {
        let reading = legal().translate(
            "Pursuant to the lease, notices herein must be written.",
            false,
        );
        assert_eq!(
            reading.text,
            "Under the lease, notices in this document must be written."
        );
        assert_eq!(reading.changes.len(), 2);
        assert_eq!(reading.stats.avoidable, 2);
        assert!(reading.remaining.is_empty());
    }

    #[test]
    fn a_term_of_art_is_counted_apart_and_glossed_once() {
        let reading = legal().translate(
            "The plea is barred by res judicata. Res judicata applies.",
            true,
        );
        assert!(reading
            .text
            .starts_with("The plea is barred by res judicata (a matter already decided)."));
        assert_eq!(reading.glosses.len(), 1);
        assert_eq!(reading.stats.necessary, 2);
        assert_eq!(reading.stats.avoidable, 0);
        assert_eq!(reading.stats.band, "plain");
    }

    #[test]
    fn a_term_inside_a_citation_is_not_avoidable() {
        let scan = legal().scan("Under § 1940.2 the tenant may repair pursuant to the lease.");
        let cited: Vec<_> = scan.terms.iter().filter(|t| t.in_citation).collect();
        assert!(cited.is_empty(), "the citation holds no glossary term");
        assert_eq!(scan.stats.avoidable, 1);
    }

    #[test]
    fn a_quoted_term_survives_the_rewrite() {
        let reading = legal().translate(
            "The notice read “pursuant to the lease” and nothing else.",
            false,
        );
        assert!(reading.text.contains("“pursuant to the lease”"));
        assert!(reading.changes.is_empty());
    }

    #[test]
    fn the_band_follows_the_declared_ceilings() {
        let reading = legal().translate("herein herein herein herein herein herein", false);
        assert_eq!(reading.stats.avoidable_per_100, 100.0);
        assert_eq!(reading.stats.band, "dense");
    }

    #[test]
    fn a_citation_shorthand_is_not_glossed() {
        let mut lexicon = Lexicon::new(vec![
            term("supra", "supra", "above", "citation", false),
            term(
                "res-judicata",
                "res judicata",
                "a matter already decided",
                "latin",
                false,
            ),
        ])
        .unwrap();
        let mut rows = lexicon.terms().to_vec();
        rows[0].explain = false;
        lexicon = Lexicon::new(rows).unwrap();
        let rules = Rules::new(Vec::new()).unwrap();
        let settings = Settings {
            version: 1,
            disclaimer: String::new(),
            clauses: ClauseWords::default(),
            bands: vec![Band {
                name: "plain".into(),
                max_per_100: 1e9,
                note: String::new(),
            }],
            protect: Vec::new(),
            rarity: Some(rarity()),
        };
        let guard = Guard::new(&[]).unwrap();
        let legal = Legal::new(lexicon, rules, guard, settings, freq()).unwrap();
        let reading = legal.translate("See supra at 7; res judicata applies.", true);
        assert!(reading.text.contains("supra at 7"), "{}", reading.text);
        assert!(reading
            .text
            .contains("res judicata (a matter already decided)"));
    }

    #[test]
    fn coverage_names_the_gaps_and_the_orphans() {
        let lexicon = Lexicon::new(vec![
            term("herein", "herein", "in this document", "deictic", true),
            term("thereof", "thereof", "of it", "deictic", true),
        ])
        .unwrap();
        let rules = Rules::new(vec![rule("herein", r"\bherein\b", "in this document")]).unwrap();
        let settings = Settings {
            version: 1,
            disclaimer: String::new(),
            clauses: ClauseWords::default(),
            bands: vec![Band {
                name: "plain".into(),
                max_per_100: 1e9,
                note: String::new(),
            }],
            protect: Vec::new(),
            rarity: Some(rarity()),
        };
        let guard = Guard::new(&[]).unwrap();
        let legal = Legal::new(lexicon, rules, guard, settings, freq()).unwrap();
        let (orphans, gaps) = legal.coverage();
        assert!(orphans.is_empty());
        assert_eq!(gaps, vec!["thereof".to_string()]);
    }
}
