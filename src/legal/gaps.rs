//! The defined terms a brief cannot answer for.
//!
//! A brief is built from the glossary, so a term the document defines and the
//! glossary does not carry does not fail loudly. It simply never reaches the
//! definitions section, and a reader cannot tell an empty answer from a missing
//! one. This module reads the definitional cues the document itself writes and
//! names what the glossary has no row for:
//!
//! * a quoted defined term, `(the "Software")` or `(the "Holdback")`;
//! * a term followed by a cue, `Confidential Information means ...`,
//!   `the Fee shall mean ...`, `the Premises refers to ...`.
//!
//! Only the words next to the cue are taken, and only when they are capitalised
//! the way a defined term is, so `This means` and `which means` are not read as
//! definitions. A cue inside a protected citation is skipped for the same reason
//! a rewrite skips it: the glossary does not gloss the name of a statute.
//!
//! A name a contract defines for its own cast is not a term of art. `the Buyer`,
//! `the Agreement`, `the Term` are defined, and the glossary rightly has no row
//! for them, so `settings.json` under `clauses.defined_ignore` lists them and the
//! check stays quiet about them. The list is data, so a corpus with a different
//! cast can change it without a build.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;

use super::translate::Legal;

/// At most this many words make up a defined term. Longer runs are a sentence,
/// not a name.
const MAX_WORDS: usize = 5;

/// How far before a cue a defined term may start, in bytes.
const WINDOW: usize = 120;

/// Words that may sit inside a defined term without being capitalised.
const CONNECTORS: [&str; 6] = ["of", "the", "and", "for", "to", "in"];

/// Words that open a sentence, never a defined term.
const OPENERS: [&str; 9] = [
    "this", "that", "these", "those", "it", "which", "such", "he", "she",
];

/// A defined term the glossary does not carry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Gap {
    /// The term as the document writes it.
    pub term: String,
    /// Byte offset of the definition.
    pub offset: usize,
}

/// Every defined term in `text` that the glossary has no row for, in document
/// order and without repeats.
pub fn find(legal: &Legal, text: &str) -> Vec<Gap> {
    let ignore: BTreeSet<String> = legal
        .settings
        .clauses
        .defined_ignore
        .iter()
        .map(|name| name.trim().to_lowercase())
        .collect();
    let mut out: Vec<Gap> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (term, offset) in candidates(text) {
        let key = term.to_lowercase();
        if ignore.contains(&key) || legal.lexicon.knows(&term) || !seen.insert(key) {
            continue;
        }
        out.push(Gap { term, offset });
    }
    out.sort_by_key(|gap| gap.offset);
    out
}

/// Every place the text looks like it defines a term.
fn candidates(text: &str) -> Vec<(String, usize)> {
    let mut out = quoted(text);
    out.extend(named(text));
    out.sort_by_key(|(_, offset)| *offset);
    out
}

/// `(the "Software")` and `("Confidential Information")`.
fn quoted(text: &str) -> Vec<(String, usize)> {
    let quoted = Regex::new(
        r#"\(\s*(?:(?i:the|a|an)\s+)?["\u{201c}]([^"\u{201d}\n]{2,60})["\u{201d}]\s*\)"#,
    )
    .expect("the quoted-term pattern is a constant");
    quoted
        .captures_iter(text)
        .filter_map(|cap| {
            let group = cap.get(1)?;
            let term = named_party(group.as_str());
            if term.is_empty() {
                None
            } else {
                Some((term, group.start()))
            }
        })
        .collect()
}

/// `Confidential Information means ...` and its variants.
fn named(text: &str) -> Vec<(String, usize)> {
    let cue = Regex::new(r"(?i)\b(?:means|shall mean|refers to|has the meaning)\b")
        .expect("the definition cue is a constant");
    cue.find_iter(text)
        .filter_map(|m| preceding(text, m.start()))
        .collect()
}

/// The capitalised run of words immediately before `at`, if it reads as a name.
///
/// The run is matched with a pattern rather than walked word by word, because a
/// walk crosses punctuation: in `The Fee (the "Service Fee") means` a walk takes
/// `Fee the Service Fee` as one name, while the pattern stops at the bracket.
fn preceding(text: &str, at: usize) -> Option<(String, usize)> {
    let from = floor_boundary(text, at.saturating_sub(WINDOW));
    let window = &text[from..at];
    let found = term_pattern().captures(window)?;
    let matched = found.name("term")?;
    let term = named_party(matched.as_str());
    if term.is_empty() || OPENERS.contains(&term.to_lowercase().as_str()) {
        return None;
    }
    Some((term, from + matched.start()))
}

/// The largest slice of `text` no longer than `at` that starts on a char
/// boundary.
fn floor_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// `Term of Credit`, `Confidential Information`, `Fee`, with an optional leading
/// article, ending immediately before the cue.
fn term_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let word = r"\p{Lu}[\p{L}\p{N}'\u{2019}-]*";
        let connector = CONNECTORS.join("|");
        let tail = format!(
            r"(?:\s+(?:(?:{connector})\s+)?{word}){{0,{}}}",
            MAX_WORDS - 1
        );
        Regex::new(&format!(r"(?:(?i:the|a|an)\s+)?(?P<term>{word}{tail})\s*$"))
            .expect("the defined-term pattern is built from constants")
    })
}

/// A quoted name with the article the writer left inside the quotation marks
/// removed: `the "Act"` is `Act`.
fn named_party(quoted: &str) -> String {
    let term = clean(quoted);
    let lower = term.to_lowercase();
    for article in ["the ", "an ", "a "] {
        if lower.starts_with(article) {
            let rest = &term[article.len()..];
            if rest.chars().next().is_some_and(char::is_uppercase) {
                return rest.to_string();
            }
        }
    }
    term
}

/// The word without the punctuation a sentence wraps around it.
fn clean(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-' && c != '’')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaps(text: &str) -> Vec<String> {
        let legal = Legal::load("data/legal").expect("the shipped instrument");
        find(&legal, text).into_iter().map(|gap| gap.term).collect()
    }

    #[test]
    fn a_quoted_defined_term_is_reported_when_the_glossary_lacks_it() {
        let found = gaps(r#"The Licensee (the "Confidential Information") shall store it."#);
        assert_eq!(found, vec!["Confidential Information".to_string()]);
    }

    #[test]
    fn a_term_before_a_means_cue_is_reported() {
        let found = gaps("Confidential Information means any data the Discloser labels as such.");
        assert_eq!(found, vec!["Confidential Information".to_string()]);
    }

    #[test]
    fn a_term_the_glossary_carries_is_not_a_gap() {
        assert!(gaps("Laches means a claim lost by waiting too long.").is_empty());
    }

    #[test]
    fn a_sentence_opener_is_not_a_defined_term() {
        assert!(gaps("This means the Buyer may reject the goods.").is_empty());
        assert!(gaps("Which means the fee is due.").is_empty());
    }

    #[test]
    fn a_name_the_contract_defines_for_its_cast_is_not_a_term_of_art() {
        assert!(gaps(r#"The party buying (the "Buyer") shall pay."#).is_empty());
        assert!(gaps(r#"Section 5 ("the Act") applies."#).is_empty());
    }

    #[test]
    fn the_same_term_is_reported_once() {
        let found = gaps(
            r#"The Software (the "Software") is licensed. The Software (the "Software") is not sold."#,
        );
        assert_eq!(found, vec!["Software".to_string()]);
    }

    #[test]
    fn a_connector_inside_a_name_is_kept() {
        let found = gaps("The Term of Credit means the period stated in the schedule.");
        assert_eq!(found, vec!["Term of Credit".to_string()]);
    }
}
