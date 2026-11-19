//! The cue words a brief classifies a sentence by, compiled once.
//!
//! A cue is a phrase such as "shall not" or "provided that", and a phrase is
//! matched across any run of whitespace and without regard to case. The list is
//! declared in `settings.json` under `clauses`, so a cue can be added without a
//! build. Sentence boundaries live here too, because the abbreviations that
//! decide whether a full stop ends a sentence are part of the same vocabulary.

use std::collections::BTreeSet;

use regex::Regex;

use super::guard::Span;
use super::term::ClauseWords;

/// The cue words, compiled. A phrase matches across any run of whitespace and
/// is case insensitive, so "No later  than" is the same cue as "no later than".
#[derive(Clone, Debug)]
pub struct Cues {
    pub(crate) duty: Vec<Regex>,
    pub(crate) prohibition: Vec<Regex>,
    pub(crate) permission: Vec<Regex>,
    pub(crate) condition: Vec<Regex>,
    pub(crate) deadline: Vec<Regex>,
    pub(crate) negation: Vec<Regex>,
    pub(crate) statement: Vec<Regex>,
    pub(crate) abbreviations: BTreeSet<String>,
    /// Irregular past participles, lower cased.
    pub(crate) participles: BTreeSet<String>,
}

impl Cues {
    pub fn new(words: &ClauseWords) -> Result<Self, String> {
        if words.duty.is_empty() || words.permission.is_empty() {
            return Err(
                "settings.json declares no clause cues: `clauses.duty` and `clauses.permission` \
                 are required for a brief"
                    .into(),
            );
        }
        Ok(Self {
            duty: compile(&words.duty, "duty")?,
            prohibition: compile(&words.prohibition, "prohibition")?,
            permission: compile(&words.permission, "permission")?,
            condition: compile(&words.condition, "condition")?,
            deadline: compile(&words.deadline, "deadline")?,
            negation: compile(&words.negation, "negation")?,
            statement: compile(&words.statement, "statement")?,
            abbreviations: words
                .abbreviations
                .iter()
                .map(|a| a.trim_end_matches('.').to_string())
                .collect(),
            participles: words.participles.iter().map(|p| p.to_lowercase()).collect(),
        })
    }

    /// The first cue of a class, and the words that matched it.
    pub(crate) fn find(list: &[Regex], text: &str) -> Option<(Span, String)> {
        let mut best: Option<(Span, String)> = None;
        for pattern in list {
            let Some(found) = pattern.find(text) else {
                continue;
            };
            let span = Span::new(found.start(), found.end());
            let better = match &best {
                None => true,
                Some((held, _)) => {
                    span.start < held.start || (span.start == held.start && span.end > held.end)
                }
            };
            if better {
                best = Some((span, text[span.start..span.end].to_string()));
            }
        }
        best
    }
}

fn compile(phrases: &[String], class: &str) -> Result<Vec<Regex>, String> {
    let mut out = Vec::with_capacity(phrases.len());
    for phrase in phrases {
        let body = phrase
            .split_whitespace()
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join(r"\s+");
        if body.is_empty() {
            continue;
        }
        let pattern = format!(r"(?i)\b{body}\b");
        out.push(Regex::new(&pattern).map_err(|e| format!("clause cue {class} {phrase:?}: {e}"))?);
    }
    Ok(out)
}

/// Split a text into sentences without cutting a citation, an abbreviation or a
/// decimal. A newline is always a boundary; a full stop or semicolon is one only
/// when the next character is whitespace and the stop is not inside a protected
/// span.
pub(crate) fn sentences(
    text: &str,
    protected: &[Span],
    abbreviations: &BTreeSet<String>,
) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < text.len() {
        let c = text[i..].chars().next().expect("a char at a boundary");
        let mut cut: Option<usize> = None;
        if c == '\n' {
            cut = Some(i + 1);
        } else if c == '.' || c == ';' {
            let next = text[i + 1..].chars().next();
            let prev = text[..i].chars().next_back().unwrap_or(' ');
            let ends_word = prev.is_alphanumeric() || prev == ')' || prev == '"' || prev == '\'';
            let spaced = next.map(|n| n.is_whitespace()).unwrap_or(true);
            if ends_word && spaced && !inside(protected, i) && !abbreviated(text, i, abbreviations)
            {
                cut = Some(i + 1);
            }
        }
        if let Some(end) = cut {
            push_trimmed(text, start, end, &mut out);
            start = end;
        }
        i += c.len_utf8();
    }
    push_trimmed(text, start, text.len(), &mut out);
    out
}

pub(crate) fn inside(spans: &[Span], at: usize) -> bool {
    spans.iter().any(|s| at >= s.start && at < s.end)
}

fn abbreviated(text: &str, at: usize, set: &BTreeSet<String>) -> bool {
    let token: String = text[..at]
        .chars()
        .rev()
        .take_while(|c| !c.is_whitespace())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    token.contains('.')
        || set.contains(token.trim_end_matches('.'))
        || (token.len() == 1 && token.chars().all(|c| c.is_ascii_uppercase()))
        || (token.len() == 1
            && token.chars().all(|c| c.is_ascii_digit())
            && list_number(text, at - token.len()))
}

/// Whether the digits ending at `at` open a line, which makes them a list
/// marker rather than the end of a sentence.
///
/// `1. FEES. The Licensee shall pay` is one numbered clause, so the stop after
/// `1` must not split it. `See Schedule 1. The Buyer shall pay` is two
/// sentences, and a digit before a full stop in the middle of a line ends the
/// first: a schedule number is not an abbreviation.
fn list_number(text: &str, start: usize) -> bool {
    text[..start]
        .chars()
        .rev()
        .find(|c| !c.is_whitespace())
        .map_or(true, |c| c == '\n')
}

fn push_trimmed(text: &str, start: usize, end: usize, out: &mut Vec<Span>) {
    let piece = &text[start..end];
    let lead = piece.len() - piece.trim_start().len();
    let tail = start + piece.trim_end().len();
    if tail > start + lead {
        out.push(Span::new(start + lead, tail));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn split(text: &str) -> Vec<String> {
        sentences(text, &[], &BTreeSet::new())
            .into_iter()
            .map(|span| span.slice(text).to_string())
            .collect()
    }

    #[test]
    fn a_schedule_number_ends_the_sentence() {
        assert_eq!(
            split("See Schedule 1. The Buyer shall pay."),
            vec!["See Schedule 1.", "The Buyer shall pay."]
        );
    }

    #[test]
    fn a_list_number_at_the_start_of_a_line_opens_its_clause() {
        assert_eq!(
            split("1. FEES. The Licensee shall pay."),
            vec!["1. FEES.", "The Licensee shall pay."]
        );
    }

    #[test]
    fn a_decimal_is_not_a_sentence_end() {
        assert_eq!(
            split("The rate is 4.5 percent. The Buyer shall pay."),
            vec!["The rate is 4.5 percent.", "The Buyer shall pay."]
        );
    }
}
