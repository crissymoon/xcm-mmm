//! The substitutions, and the order they run in.
//!
//! A rule is one pattern and one replacement, with the reason it exists. Two
//! properties are deliberate:
//!
//! * a rule may be advisory. `applies: false` reports the suggestion and an
//!   explanation without editing the sentence, which is what `said` and `such`
//!   need: in "said vehicle" the word means "the", and in "he said" it does not,
//!   and no honest pattern tells those apart.
//! * a rule may be outranked. Several rules can match the same words
//!   (`unless and until` and `unless`), so candidates are resolved by priority,
//!   then by length, then by position, and overlap is decided once rather than
//!   by the order the file happens to be written in.
//!
//! The pass is single. A replacement is never rewritten again, so the output of
//! a run is a function of the input and the file, not of how many rounds it was
//! given, and every change carries the offsets of the text it came from.

use std::collections::BTreeSet;
use std::fs;
use std::io::{BufRead, BufReader};

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use super::guard::{is_protected, Span};

fn yes() -> bool {
    true
}

/// How a replacement takes its capitalisation from the text it replaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Case {
    /// Copy the first letter's case from the match.
    #[default]
    Preserve,
    /// Always lower case, for mid-sentence substitutions.
    Lower,
    /// Always capitalised.
    Title,
}

/// One substitution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub pattern: String,
    pub replace: String,
    pub kind: String,
    pub why: String,
    /// Whether the rewrite is performed. False makes the rule advisory.
    #[serde(default = "yes")]
    pub applies: bool,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub case: Case,
}

/// One edit, or one suggestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Change {
    pub rule: String,
    /// Offsets into the text the rule was applied to.
    pub span: Span,
    pub from: String,
    pub to: String,
    pub why: String,
    pub applied: bool,
}

/// The whole pass.
#[derive(Clone, Debug, Default)]
pub struct Applied {
    pub text: String,
    pub changes: Vec<Change>,
    pub suggestions: Vec<Change>,
}

/// The compiled rule table.
#[derive(Clone, Debug)]
pub struct Rules {
    rules: Vec<Rule>,
    patterns: Vec<Regex>,
}

impl Rules {
    pub fn load(path: &str) -> Result<Self, String> {
        let file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut rules = Vec::new();
        for (number, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{path}: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let rule: Rule =
                serde_json::from_str(&line).map_err(|e| format!("{path}:{}: {e}", number + 1))?;
            rules.push(rule);
        }
        Self::new(rules)
    }

    pub fn new(rules: Vec<Rule>) -> Result<Self, String> {
        let mut ids = BTreeSet::new();
        let mut patterns = Vec::with_capacity(rules.len());
        for rule in &rules {
            if rule.id.trim().is_empty() || rule.pattern.trim().is_empty() {
                return Err(format!("a rule has an empty id or pattern: {rule:?}"));
            }
            if rule.why.trim().is_empty() {
                return Err(format!("{}: no reason given", rule.id));
            }
            if !ids.insert(rule.id.clone()) {
                return Err(format!("duplicate rule id {}", rule.id));
            }
            let regex = RegexBuilder::new(&rule.pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| format!("{}: {e}", rule.id))?;
            patterns.push(regex);
        }
        Ok(Self { rules, patterns })
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Rules that edit the text, as opposed to those that only advise.
    pub fn applying(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().filter(|r| r.applies)
    }

    /// Run every rule once.
    pub fn apply(&self, text: &str, protected: &[Span]) -> Applied {
        struct Candidate {
            priority: i32,
            span: Span,
            rule: usize,
            matched: String,
        }

        let mut candidates = Vec::new();
        for (index, rule) in self.rules.iter().enumerate() {
            for found in self.patterns[index].find_iter(text) {
                let span = Span::new(found.start(), found.end());
                if span.is_empty() || is_protected(protected, &span) {
                    continue;
                }
                candidates.push(Candidate {
                    priority: rule.priority,
                    span,
                    rule: index,
                    matched: found.as_str().to_string(),
                });
            }
        }

        // Higher priority first, then the longer match, then the earlier one.
        candidates.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(b.span.len().cmp(&a.span.len()))
                .then(a.span.start.cmp(&b.span.start))
                .then(a.rule.cmp(&b.rule))
        });

        let mut taken: Vec<Span> = Vec::new();
        let mut accepted: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if taken.iter().any(|s| s.intersects(&candidate.span)) {
                continue;
            }
            taken.push(candidate.span);
            accepted.push(candidate);
        }

        accepted.sort_by_key(|c| c.span.start);

        let mut changes = Vec::new();
        let mut suggestions = Vec::new();
        let mut edits = Vec::new();
        for candidate in accepted {
            let rule = &self.rules[candidate.rule];
            let to = render(rule, &candidate.matched);
            let change = Change {
                rule: rule.id.clone(),
                span: candidate.span,
                from: candidate.matched,
                to,
                why: rule.why.clone(),
                applied: rule.applies,
            };
            if rule.applies {
                edits.push((change.span, change.to.clone()));
                changes.push(change);
            } else {
                suggestions.push(change);
            }
        }

        let mut out = text.to_string();
        for (span, replacement) in edits.iter().rev() {
            out.replace_range(span.start..span.end, replacement);
        }

        Applied {
            text: tidy(&out),
            changes,
            suggestions,
        }
    }
}

/// Take the case of the matched text into the replacement.
fn render(rule: &Rule, matched: &str) -> String {
    let upper = matched
        .chars()
        .find(|c| c.is_alphabetic())
        .map(|c| c.is_uppercase())
        .unwrap_or(false);
    match rule.case {
        Case::Lower => rule.replace.to_lowercase(),
        Case::Title => capitalise(&rule.replace),
        Case::Preserve if upper => capitalise(&rule.replace),
        Case::Preserve => rule.replace.clone(),
    }
}

fn capitalise(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Close the gaps a deletion leaves without flattening the document.
///
/// A deletion leaves two spaces where one word was; a rewrite ending in a comma
/// leaves a space before it. Both are repaired. A line break is not repaired,
/// because paragraphs are the document's own structure: a run holding newlines
/// keeps at most one blank line and is never joined into a single line.
pub fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending = false;
    let mut newlines = 0;
    for ch in text.chars() {
        if ch == '\n' {
            newlines += 1;
            pending = true;
            continue;
        }
        if ch == ' ' || ch == '\t' || ch == '\r' {
            pending = true;
            continue;
        }
        if pending {
            if newlines > 0 {
                for _ in 0..newlines.min(2) {
                    out.push('\n');
                }
            } else if !out.is_empty()
                && !matches!(ch, ',' | '.' | ';' | ':' | ')' | '!' | '?')
                && !matches!(
                    out.chars().next_back(),
                    Some('(') | Some('\u{201c}') | Some('\n')
                )
            {
                out.push(' ');
            }
            pending = false;
            newlines = 0;
        }
        out.push(ch);
    }
    let trimmed =
        out.trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ':' | ';' | ','));
    trimmed.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(id: &str, pattern: &str, replace: &str, priority: i32) -> Rule {
        Rule {
            id: id.into(),
            pattern: pattern.into(),
            replace: replace.into(),
            kind: "phrase".into(),
            why: "test".into(),
            applies: true,
            priority,
            case: Case::Preserve,
        }
    }

    #[test]
    fn the_outranking_rule_wins_and_the_other_is_dropped() {
        let rules = Rules::new(vec![
            rule("unless", r"\bunless\b", "except if", 0),
            rule("unless-and-until", r"\bunless and until\b", "unless", 10),
        ])
        .unwrap();
        let applied = rules.apply("The lease continues unless and until notice is given.", &[]);
        assert_eq!(applied.text, "The lease continues unless notice is given.");
        assert_eq!(applied.changes.len(), 1);
        assert_eq!(applied.changes[0].rule, "unless-and-until");
    }

    #[test]
    fn a_protected_span_is_never_edited() {
        let rules = Rules::new(vec![rule("pursuant", r"\bpursuant to\b", "under", 0)]).unwrap();
        let text = "Pursuant to Civil Code § 1940.2 the tenant may repair.";
        let protected = vec![Span::new(0, 15)];
        let applied = rules.apply(text, &protected);
        assert_eq!(applied.text, text);
        assert!(applied.changes.is_empty());
    }

    #[test]
    fn capitalisation_follows_the_source() {
        let rules = Rules::new(vec![rule("pursuant", r"\bpursuant to\b", "under", 0)]).unwrap();
        assert_eq!(
            rules.apply("Pursuant to the lease", &[]).text,
            "Under the lease"
        );
        assert_eq!(
            rules.apply("pursuant to the lease", &[]).text,
            "under the lease"
        );
    }

    #[test]
    fn an_advisory_rule_is_reported_but_not_performed() {
        let mut advice = rule("said", r"\bsaid\b", "the", 0);
        advice.applies = false;
        let rules = Rules::new(vec![advice]).unwrap();
        let applied = rules.apply("said vehicle was towed", &[]);
        assert_eq!(applied.text, "said vehicle was towed");
        assert_eq!(applied.suggestions.len(), 1);
        assert!(applied.changes.is_empty());
    }

    #[test]
    fn a_deletion_leaves_one_space_not_two() {
        let rules = Rules::new(vec![rule("witnesseth", r"\bwitnesseth:\s*", "", 0)]).unwrap();
        let applied = rules.apply("Witnesseth:  The parties agree.", &[]);
        assert_eq!(applied.text, "The parties agree.");
    }

    #[test]
    fn a_line_break_survives_a_rewrite() {
        let rules = Rules::new(vec![rule("herein", r"\bherein\b", "in this document", 0)]).unwrap();
        let applied = rules.apply("Notice herein\n\nThe tenant may reply.", &[]);
        assert_eq!(
            applied.text,
            "Notice in this document\n\nThe tenant may reply."
        );
    }

    #[test]
    fn offsets_are_the_offsets_of_the_input() {
        let rules = Rules::new(vec![rule("herein", r"\bherein\b", "in this document", 0)]).unwrap();
        let text = "Notices herein shall be written.";
        let applied = rules.apply(text, &[]);
        let change = &applied.changes[0];
        assert_eq!(change.span.slice(text), "herein");
    }
}
