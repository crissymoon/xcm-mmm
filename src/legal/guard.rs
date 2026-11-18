//! The spans a rewrite must leave alone.
//!
//! A translator that rewrites "pursuant to" into "under" inside *the name of a
//! statute* has changed a citation, not a sentence, and the change is wrong even
//! though the words are legal. Two kinds of text are therefore protected before
//! any rule runs:
//!
//! * quotations, between paired curly or straight quotes, because rewriting
//!   quoted material misquotes the source it came from;
//! * citations, matched by the patterns that travel in `settings.json`, so a
//!   court, code, chapter or reporter cite is never edited.
//!
//! The patterns live in data rather than here: a house that adds "Admin. Code"
//! to its citations should edit a file, not a match arm. The quote pairs are
//! structural, so they stay in code.

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

/// A half open byte range in a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether the two ranges share at least one byte.
    pub fn intersects(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// The text this span covers, or the empty string when it is off the ends.
    pub fn slice<'a>(&self, text: &'a str) -> &'a str {
        text.get(self.start..self.end).unwrap_or("")
    }
}

/// Paired quote marks that open and close a protected region.
const QUOTES: [(char, char); 4] = [('“', '”'), ('"', '"'), ('«', '»'), ('‘', '’')];

/// The protected regions of a text.
#[derive(Clone, Debug)]
pub struct Guard {
    patterns: Vec<Regex>,
}

impl Guard {
    /// Compile the citation patterns. An empty list is allowed: it protects
    /// quotations only, and `check` reports that citations are unguarded.
    pub fn new(patterns: &[String]) -> Result<Self, String> {
        let mut compiled = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            let regex = RegexBuilder::new(pattern)
                .case_insensitive(false)
                .build()
                .map_err(|e| format!("guard pattern {pattern:?}: {e}"))?;
            compiled.push(regex);
        }
        Ok(Self { patterns: compiled })
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Every protected span, sorted and merged so an overlap test is a scan
    /// rather than a comparison against each pattern.
    pub fn spans(&self, text: &str) -> Vec<Span> {
        let mut spans = quoted(text);
        for pattern in &self.patterns {
            for m in pattern.find_iter(text) {
                if m.end() > m.start() {
                    spans.push(Span::new(m.start(), m.end()));
                }
            }
        }
        merge(spans)
    }
}

/// Sort and merge overlapping or touching spans.
pub fn merge(mut spans: Vec<Span>) -> Vec<Span> {
    spans.retain(|s| !s.is_empty());
    spans.sort();
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for span in spans {
        match out.last_mut() {
            Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
            _ => out.push(span),
        }
    }
    out
}

/// Whether a span falls inside any protected span.
pub fn is_protected(spans: &[Span], span: &Span) -> bool {
    spans.iter().any(|s| s.contains(span))
}

/// The regions between paired quote marks.
///
/// An opening mark with no closer before the end of the text is not a
/// quotation, so it protects nothing: a stray apostrophe must not silently
/// disable the rest of the document. A quote nesting the other kind is allowed,
/// which is what happens when a sentence quoted from a contract itself carries
/// a quoted term.
fn quoted(text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut stack: Vec<(char, usize)> = Vec::new();
    for (index, ch) in text.char_indices() {
        let next = index + ch.len_utf8();
        if let Some(&(open, start)) = stack.last() {
            // A straight quote closes the straight quote it opened; a curly one
            // closes only with its own partner.
            let closer = QUOTES.iter().find(|(o, _)| *o == open).map(|(_, c)| *c);
            if closer == Some(ch) {
                stack.pop();
                spans.push(Span::new(start, next));
                continue;
            }
        }
        if QUOTES.iter().any(|(o, _)| *o == ch) {
            stack.push((ch, index));
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_citation_pattern_protects_its_span() {
        let guard = Guard::new(&[r"§+ ?[0-9]+(?:\.[0-9]+)?".to_string()]).unwrap();
        let text = "Under Civil Code § 1940.2 a tenant may repair.";
        let spans = guard.spans(text);
        assert!(spans.iter().any(|s| s.slice(text) == "§ 1940.2"));
    }

    #[test]
    fn a_quotation_is_protected_and_an_unclosed_one_is_not() {
        let guard = Guard::new(&[]).unwrap();
        let text = "The notice said “pay within three days” and nothing more.";
        let spans = guard.spans(text);
        assert_eq!(spans.len(), 1);
        assert!(spans[0].slice(text).starts_with('“'));
        let stray = "The tenant's notice and the landlord's answer.";
        assert!(guard.spans(stray).is_empty());
    }

    #[test]
    fn two_citations_merge_when_they_touch() {
        let spans = merge(vec![Span::new(0, 5), Span::new(5, 9), Span::new(20, 22)]);
        assert_eq!(spans, vec![Span::new(0, 9), Span::new(20, 22)]);
    }

    #[test]
    fn protection_is_containment_not_overlap() {
        let guarded = vec![Span::new(10, 20)];
        assert!(is_protected(&guarded, &Span::new(12, 15)));
        assert!(!is_protected(&guarded, &Span::new(15, 25)));
    }
}
