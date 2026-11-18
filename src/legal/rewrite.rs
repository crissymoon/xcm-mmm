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
//!
//! A replacement may name the groups of its own pattern: `$1`, `$2`, or
//! `${name}`, with `$$` for a literal dollar. That is what separates a
//! substitution from a rewrite. "shall not be entitled to payment" cannot be
//! repaired word by word, because the words are in the wrong order; a rule that
//! captures the modal and the noun can put them back as "shall not be paid for".
//! Group references are checked when the table loads, so a `$3` that names a
//! group the pattern does not have is refused instead of quietly deleting a
//! phrase at run time.
//!
//! A bare reference is one digit, because `$500` in a legal text is money. A
//! group numbered ten or higher is written `${10}`.

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
            check_references(&rule, &regex)?;
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
            let slice = &text[candidate.span.start..candidate.span.end];
            let caps = self.patterns[candidate.rule].captures(slice);
            let to = render(rule, &candidate.matched, caps.as_ref());
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
fn render(rule: &Rule, matched: &str, caps: Option<&regex::Captures>) -> String {
    let upper = matched
        .chars()
        .find(|c| c.is_alphabetic())
        .map(|c| c.is_uppercase())
        .unwrap_or(false);
    let body = expand(&rule.replace, caps);
    match rule.case {
        Case::Lower => body.to_lowercase(),
        Case::Title => capitalise(&body),
        Case::Preserve if upper => capitalise(&body),
        Case::Preserve => body,
    }
}

/// Every group name or number a replacement refers to.
///
/// `$$` is a literal dollar and is skipped. A lone `$` followed by anything else
/// is left as written, because it is more likely a currency than a mistake.
fn references(replace: &str) -> Vec<String> {
    let bytes = replace.as_bytes();
    let mut found = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'$' || index + 1 >= bytes.len() {
            index += 1;
            continue;
        }
        match bytes[index + 1] {
            b'$' => index += 2,
            b'{' => match replace[index + 2..].find('}') {
                Some(close) => {
                    found.push(replace[index + 2..index + 2 + close].to_string());
                    index += close + 3;
                }
                None => index += 1,
            },
            b'0'..=b'9' => {
                // One digit is a group; a longer run of digits is money, and a
                // group of two or more digits is written ${10}.
                if index + 2 < bytes.len() && bytes[index + 2].is_ascii_digit() {
                    index += 1;
                    continue;
                }
                found.push(((bytes[index + 1] - b'0') as usize).to_string());
                index += 2;
            }
            _ => index += 1,
        }
    }
    found
}

/// Refuse a rule whose replacement names a group its pattern does not have.
fn check_references(rule: &Rule, regex: &Regex) -> Result<(), String> {
    let groups = regex.captures_len();
    let names: Vec<&str> = regex.capture_names().flatten().collect();
    for name in references(&rule.replace) {
        let known = if name.chars().all(|c| c.is_ascii_digit()) {
            name.parse::<usize>().map(|n| n < groups).unwrap_or(false)
        } else {
            names.iter().any(|known| *known == name.as_str())
        };
        if !known {
            return Err(format!(
                "{}: the replacement names group ${name}, which the pattern does not have",
                rule.id
            ));
        }
    }
    Ok(())
}

/// Write the replacement, filling `$1`, `${name}` and `$$` from the match.
fn expand(replace: &str, caps: Option<&regex::Captures>) -> String {
    let Some(caps) = caps else {
        return replace.replace("$$", "$");
    };
    let bytes = replace.as_bytes();
    let mut out = String::with_capacity(replace.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'$' && index + 1 < bytes.len() {
            match bytes[index + 1] {
                b'$' => {
                    out.push('$');
                    index += 2;
                    continue;
                }
                b'{' => {
                    if let Some(close) = replace[index + 2..].find('}') {
                        let name = &replace[index + 2..index + 2 + close];
                        if let Some(group) = caps.name(name) {
                            out.push_str(group.as_str());
                        }
                        index += close + 3;
                        continue;
                    }
                }
                b'0'..=b'9' => {
                    if index + 2 < bytes.len() && bytes[index + 2].is_ascii_digit() {
                        out.push('$');
                        index += 1;
                        continue;
                    }
                    let number = (bytes[index + 1] - b'0') as usize;
                    if let Some(group) = caps.get(number) {
                        out.push_str(group.as_str());
                    }
                    index += 2;
                    continue;
                }
                _ => {}
            }
        }
        let ch = replace[index..].chars().next().unwrap_or('$');
        out.push(ch);
        index += ch.len_utf8();
    }
    out
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

    #[test]
    fn a_replacement_can_reorder_what_it_captures() {
        // The words cannot be repaired one at a time: "shall not be entitled to
        // payment" keeps its meaning only if the modal comes back in front of a
        // verb, which is a rewrite rather than a substitution.
        let mut clause = rule(
            "not-be-entitled-to-payment",
            r"\b(shall|will|must)\s+not\s+be\s+entitled\s+to\s+(payment|reimbursement)(?:\s+(?:or|and)\s+(?:payment|reimbursement))?\s+for\b",
            r"$1 not be paid for",
            7,
        );
        clause.kind = "clause".into();
        let rules = Rules::new(vec![
            rule("shall-not", r"\bshall not\b", "must not", 2),
            clause,
        ])
        .unwrap();
        let applied = rules.apply(
            "The Contractor shall not be entitled to payment or reimbursement for any task.",
            &[],
        );
        assert_eq!(
            applied.text, "The Contractor shall not be paid for any task.",
            "the clause rule must outrank the two-word substitution"
        );
        assert_eq!(applied.changes.len(), 1);
    }

    #[test]
    fn a_named_group_and_a_literal_dollar_both_expand() {
        let rules = Rules::new(vec![rule(
            "within-receipt",
            r"\bwithin\s+(?P<days>\w+(?:\s*\(\d+\))?)\s+days\s+of\s+receipt\s+of\b",
            "within ${days} days of receiving",
            6,
        )])
        .unwrap();
        assert_eq!(
            rules
                .apply("within ten (10) days of receipt of the notice", &[])
                .text,
            "within ten (10) days of receiving the notice"
        );
    }

    #[test]
    fn a_group_the_pattern_does_not_have_is_refused() {
        let error = Rules::new(vec![rule("gone", r"\bherein\b", "$3", 0)]).unwrap_err();
        assert!(error.contains("$3"), "{error}");
    }

    #[test]
    fn a_dollar_that_is_not_a_group_is_left_alone() {
        let rules = Rules::new(vec![rule("fee", r"\bfee\b", "a $500 charge", 0)]).unwrap();
        assert_eq!(rules.apply("fee", &[]).text, "a $500 charge");
    }
}
