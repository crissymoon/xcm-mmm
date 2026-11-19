//! An explanation of a document rather than a rewrite of it.
//!
//! `scan` names the words that make a text legal English and `translate`
//! rewrites them. A person who is handed a notice wants a third thing: what the
//! document asks of whom. This module reads the same bytes a third way, one
//! sentence at a time, and sorts every sentence by the verb that carries it:
//!
//! * a duty ("shall pay"), a prohibition ("shall not assign"), a permission
//!   ("may terminate") or a condition ("if the fee is late");
//! * with the actor that carries it, the condition that limits it and the
//!   deadline that dates it;
//! * and a definition for every term of art the sentence uses, drawn from
//!   `terms.jsonl` so the explanation and the scan cannot disagree.
//!
//! Two honesty rules hold here as they do in the translator. A sentence with no
//! cue is a statement, not a rule, and is counted rather than forced into a
//! category. And nothing is paraphrased: the clause text is the sentence after
//! the rule table has run, so a reader can check every line of the brief against
//! the document in front of them.
//!
//! Every word this module classifies is declared in `settings.json` under
//! `clauses`, so a cue can be added or removed without a build.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use regex::Regex;
use serde::Serialize;

use super::cues::{inside, sentences, Cues};
use super::guard::Span;
use super::translate::{Legal, Stats};

/// The notice a brief falls back to when `settings.json` carries none. The
/// shipped instrument carries one, so this is only a guard against an empty
/// file, never the usual source.
pub const DEFAULT_DISCLAIMER: &str = "Research project, not legal advice. This is a reading of \
     the words in the document, not a legal opinion, and it is no substitute for a lawyer.";

/// What a sentence does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Something must be done.
    Duty,
    /// Something must not be done.
    Prohibition,
    /// Something may be done.
    Permission,
    /// A sentence that only states a condition.
    Condition,
    /// A sentence with no operative cue.
    Statement,
}

impl Kind {
    pub fn label(&self) -> &'static str {
        match self {
            Kind::Duty => "duty",
            Kind::Prohibition => "prohibition",
            Kind::Permission => "permission",
            Kind::Condition => "condition",
            Kind::Statement => "statement",
        }
    }
}

/// One sentence, with everything the brief says about it.
#[derive(Clone, Debug, Serialize)]
pub struct Clause {
    pub kind: Kind,
    /// The sentence after the rule table has run.
    pub text: String,
    /// The sentence as the document has it.
    pub original: String,
    /// Who carries the verb, as the document names them.
    pub actor: String,
    /// A duty or prohibition written in the passive with no agent: the reader
    /// learns what happens, not who must do it.
    pub agentless: bool,
    /// The cue that decided the kind, as it appears.
    pub cue: String,
    /// The condition that limits the clause, when there is one.
    pub condition: Option<String>,
    /// Time limits quoted from the sentence.
    pub deadlines: Vec<String>,
    /// Glossary ids used in the sentence that the definition list carries.
    pub terms: Vec<String>,
    /// Words in this sentence that general English rarely uses, in the order
    /// they appear. A clause can be light on drafting and still stop a reader
    /// twice.
    pub rare: Vec<String>,
    /// A short all-capitals line, such as `2. FEES.` or `LICENSE GRANT.`
    pub heading: bool,
    pub span: Span,
}

/// One term of art the document relies on.
#[derive(Clone, Debug, Serialize)]
pub struct Definition {
    pub id: String,
    /// The spelling the document used.
    pub term: String,
    pub plain: String,
    pub kind: String,
    pub note: String,
    pub law: String,
    pub sources: Vec<String>,
    /// How many times the term appears outside a citation.
    pub uses: usize,
    /// Byte offset of the first use.
    pub offset: usize,
}

/// The counts the summary is built from.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Counts {
    pub duties: usize,
    pub prohibitions: usize,
    pub permissions: usize,
    pub conditions: usize,
    pub statements: usize,
    /// Clauses of any kind that carry a condition.
    pub limited: usize,
    /// Duties and prohibitions whose actor the text leaves out.
    pub agentless: usize,
    /// Words the text uses that general English does not. Counted per
    /// occurrence, because a reader stops at each one.
    pub rare: usize,
    pub definitions: usize,
}

/// A document explained.
#[derive(Clone, Debug, Serialize)]
pub struct Brief {
    pub words: usize,
    /// One paragraph, built from the counts and from nothing else.
    pub summary: String,
    pub counts: Counts,
    pub clauses: Vec<Clause>,
    pub definitions: Vec<Definition>,
    /// Defined terms the glossary has no row for. The brief cannot answer for
    /// them, so they are named instead of being left out quietly.
    pub gaps: Vec<super::gaps::Gap>,
    pub stats: Stats,
    /// What the brief could not find, said rather than implied.
    pub warnings: Vec<String>,
    /// The notice that travels with the brief in text and in JSON.
    pub disclaimer: String,
}

impl Brief {
    /// Read a text for what it requires, permits and forbids.
    pub fn build(legal: &Legal, text: &str) -> Result<Self, String> {
        let cues = legal.cues()?;
        let protected = legal.guard.spans(text);
        let hits = legal.lexicon.find(text);
        let spans = sentences(text, &protected, &cues.abbreviations);

        let mut clauses = Vec::with_capacity(spans.len());
        for span in spans {
            // One sentence can carry two rules: "..., and the Licensee shall be
            // liable" is a duty after a permission, and reporting the sentence
            // under one verb would lose the other.
            for part in split(text, span, cues, &protected) {
                clauses.push(read_clause(legal, cues, text, part, &hits, &protected));
            }
        }

        let definitions = define(legal, &hits, &protected);
        let gaps = super::gaps::find(legal, text);
        let counts = count(&clauses, definitions.len());
        let scan = legal.scan(text);
        let summary = summarise(&counts, scan.words);
        let warnings = warn(legal, &counts, &definitions, &gaps, scan.words);

        Ok(Self {
            words: scan.words,
            summary,
            counts,
            clauses,
            definitions,
            gaps,
            stats: scan.stats,
            warnings,
            disclaimer: if legal.settings.disclaimer.trim().is_empty() {
                DEFAULT_DISCLAIMER.to_string()
            } else {
                legal.settings.disclaimer.trim().to_string()
            },
        })
    }

    /// The clauses of one kind, in document order.
    pub fn of(&self, kind: Kind) -> impl Iterator<Item = &Clause> {
        self.clauses.iter().filter(move |c| c.kind == kind)
    }
}

/// Classify one sentence and pull out its actor, condition, deadline and terms.
fn read_clause(
    legal: &Legal,
    cues: &Cues,
    text: &str,
    span: Span,
    hits: &[super::term::Hit],
    protected: &[Span],
) -> Clause {
    let original = text[span.start..span.end].to_string();
    let local = protected
        .iter()
        .filter(|p| p.intersects(&span))
        .map(|p| {
            Span::new(
                p.start.max(span.start) - span.start,
                p.end.min(span.end) - span.start,
            )
        })
        .collect::<Vec<_>>();
    let mut local = local;
    let mut adjusted = original.clone();

    let duty = Cues::find(&cues.duty, &original);
    let prohibition = Cues::find(&cues.prohibition, &original);
    let permission = Cues::find(&cues.permission, &original);
    let condition = Cues::find(&cues.condition, &original);

    let mut operative: Option<(Kind, Span, String)> = None;
    for (kind, found) in [
        (Kind::Duty, duty),
        (Kind::Prohibition, prohibition),
        (Kind::Permission, permission),
    ] {
        let Some((at, cue)) = found else { continue };
        let better = match &operative {
            None => true,
            Some((_, held, _)) => {
                at.start < held.start || (at.start == held.start && at.end > held.end)
            }
        };
        if better {
            operative = Some((kind, at, cue));
        }
    }

    let mut kind = match &operative {
        Some((kind, _, _)) => *kind,
        None if condition.is_some() => Kind::Condition,
        None => Kind::Statement,
    };

    // A negation can sit in the subject instead of in the cue: "No person shall
    // assign" forbids the assignment exactly as "shall not" does, and "No
    // person" is not the actor to hand a task to. The subject is what follows
    // the last comma, so "No later than 30 days, the Borrower shall pay" keeps
    // the Borrower as its actor.
    let negated_after = operative
        .as_ref()
        .map(|(_, at, _)| {
            original[at.end..]
                .split_whitespace()
                .take(4)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .is_some_and(|after| cues.negation.iter().any(|re| re.is_match(&after)));
    let negated_subject = operative
        .as_ref()
        .map(|(_, at, _)| subject(&original[..at.start]))
        .filter(|s| {
            let mut words = s.split_whitespace();
            let first = words.next().unwrap_or("");
            words.next().is_some()
                && matches_whole(&cues.negation, first)
                && !cues.deadline.iter().any(|re| re.is_match(s))
        })
        .is_some();

    if negated_subject {
        if operative
            .as_ref()
            .map(|(_, at, _)| subject(&original[..at.start]))
            .is_some_and(|s| begins(&cues.statement, s))
        {
            // "Nothing in this section shall limit ..." states a limit. It puts
            // no duty on a person and forbids no act, so it is a statement.
            kind = Kind::Statement;
        } else if kind == Kind::Duty || kind == Kind::Permission {
            kind = Kind::Prohibition;
        }
    } else if kind == Kind::Duty && negated_after {
        kind = Kind::Prohibition;
    }

    let agentless = is_agentless(
        kind,
        &original,
        operative.as_ref().map(|(_, at, _)| *at),
        &cues.participles,
    );

    // A prohibition carried by its subject needs the permission modal: the rule
    // table turns "shall" into "must", which would print "No person must
    // assign". "No person may assign" is the same rule in the same order, so the
    // cue is swapped before the table runs.
    if kind == Kind::Prohibition && negated_subject {
        if let Some((_, at, _)) = operative.as_ref() {
            let delta = 3 - (at.end - at.start) as isize;
            for span in local.iter_mut() {
                if span.start >= at.end {
                    span.start = shift(span.start, delta);
                }
                if span.end >= at.end {
                    span.end = shift(span.end, delta);
                }
            }
            adjusted = format!("{}{}{}", &original[..at.start], "may", &original[at.end..]);
        }
    }
    let plain = legal.rules.apply(&adjusted, &local);

    // A condition runs from its cue to the comma that closes it, or to the
    // operative verb when the writer left the comma out. Where it ends decides
    // where the actor begins: "If the fee is late, the Licensor may ..." has
    // "the Licensor" as its subject, not the whole sentence.
    let condition = condition.and_then(|(at, _)| {
        let stop = match original[at.start..].find(',') {
            Some(i) => at.start + i,
            // A condition that trails the verb has no comma to stop at and runs
            // to the end of the sentence.
            None => match operative.as_ref() {
                Some((_, held, _)) if held.start > at.start => held.start,
                _ => original.len(),
            },
        };
        let phrase = clip(
            original[at.start..stop]
                .trim()
                .trim_end_matches(['.', ',', ';'])
                .trim_end(),
            120,
        );
        if phrase.is_empty() {
            None
        } else {
            Some((phrase, stop))
        }
    });

    let actor = operative
        .as_ref()
        // A negated subject is not an actor to hand a task to: the clause text
        // already carries "No person" as its subject.
        .filter(|_| !negated_subject)
        .map(|(_, at, _)| {
            let from = match &condition {
                Some((_, stop)) if *stop < at.start => stop + 1,
                _ => fronted(&original[..at.start], cues),
            };
            lead(&original[from..at.start])
        })
        .unwrap_or_default();
    let condition = condition.map(|(phrase, _)| phrase);

    let deadlines = deadlines_in(&original, cues);
    // Only a term the definition list will carry is worth a cross reference: a
    // replaced word has no entry to point at.
    let mut terms: Vec<String> = Vec::new();
    let mut rare: Vec<String> = Vec::new();
    for hit in hits {
        if hit.span.start < span.start || hit.span.end > span.end {
            continue;
        }
        if protected.iter().any(|p| p.intersects(&hit.span)) {
            continue;
        }
        let term = legal.lexicon.term(hit);
        if let Some((word, _)) = legal.rare_word(term) {
            if !rare.contains(&word) {
                rare.push(word);
            }
        }
        if term.avoidable || !term.explain {
            continue;
        }
        if !terms.contains(&term.id) {
            terms.push(term.id.clone());
        }
    }

    let heading = is_heading(&original);
    // A clause cut out of the middle of a sentence starts in lower case. Give it
    // the capital it would have had as a sentence of its own.
    let cut = spans_from(span, text);
    let text_out = if cut {
        capitalise(&plain.text)
    } else {
        plain.text
    };
    let original = if cut { capitalise(&original) } else { original };

    Clause {
        kind,
        text: text_out,
        original,
        actor,
        agentless,
        cue: operative.map(|(_, _, cue)| cue).unwrap_or_default(),
        condition,
        deadlines,
        terms,
        rare,
        heading,
        span,
    }
}

/// Whether a duty or prohibition has left its actor out.
///
/// "The notice shall be filed with the court" says what happens to the notice
/// and never says who files it. The verb after the cue is passive and no agent
/// phrase follows, so the reader has no one to hand the task to. This is the
/// fault plain-language guidance names first, and the brief counts it apart
/// from the duties it can attribute.
fn is_agentless(kind: Kind, text: &str, cue: Option<Span>, participles: &BTreeSet<String>) -> bool {
    if !matches!(kind, Kind::Duty | Kind::Prohibition) {
        return false;
    }
    let Some(cue) = cue else {
        return false;
    };
    let tail = text[cue.end..].trim_start();
    passive(tail, participles) && !agent().is_match(tail)
}

/// The passive shape a duty takes when its actor is dropped: `be` followed by a
/// participle. A regular participle ends in `ed`; an irregular one is declared
/// in `settings.json`, because "be liable" and "be open" are adjectives and
/// must not be read as passives.
fn passive(tail: &str, participles: &BTreeSet<String>) -> bool {
    let mut words = tail.split_whitespace().map(|word| {
        word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    });
    if words.next().as_deref() != Some("be") {
        return false;
    }
    let Some(verb) = words.next() else {
        return false;
    };
    verb.ends_with("ed") || participles.contains(&verb)
}

/// Whether an agent phrase follows the verb: "filed by the clerk".
fn agent() -> &'static Regex {
    static BY: OnceLock<Regex> = OnceLock::new();
    BY.get_or_init(|| {
        Regex::new(r"(?i)\bby\s+(?:the|a|an|its|their|his|her|such|any|each|every)\b")
            .expect("the agent phrase")
    })
}

/// Whether a span begins partway through a sentence, which is how a clause cut
/// at `, and` begins.
fn spans_from(span: Span, text: &str) -> bool {
    if span.start == 0 {
        return false;
    }
    let before = text[..span.start].trim_end();
    !(before.ends_with('.')
        || before.ends_with(';')
        || before.ends_with(':')
        || before.ends_with('\n'))
}

fn capitalise(text: &str) -> String {
    let trimmed = text.trim_start();
    let mut chars = trimmed.chars();
    match chars.next() {
        Some(c) if c.is_lowercase() => format!("{}{}", c.to_uppercase(), chars.as_str()),
        _ => trimmed.to_string(),
    }
}

/// A heading is a short line in capitals, such as `2. FEES.` It is part of the
/// document but it is not a sentence that says anything, so the count of
/// statements leaves it out.
fn is_heading(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || words.len() > 8 {
        return false;
    }
    let mut letters = 0usize;
    for word in &words {
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            if c.is_lowercase() {
                return false;
            }
            letters += 1;
        }
    }
    letters >= 3
}

/// Cut a sentence at `, and` when a second operative cue follows, so a
/// permission and the duty that rides behind it are two clauses rather than one.
fn split(text: &str, span: Span, cues: &Cues, protected: &[Span]) -> Vec<Span> {
    let body = &text[span.start..span.end];
    let mut from = 0usize;
    while let Some(found) = body[from..].find(", and ") {
        let comma = from + found;
        let tail = comma + ", and ".len();
        if !inside(protected, span.start + comma) {
            let look: String = body[tail..]
                .split_whitespace()
                .take(12)
                .collect::<Vec<_>>()
                .join(" ");
            let second = [&cues.duty, &cues.prohibition, &cues.permission]
                .iter()
                .any(|list| Cues::find(list, &look).is_some());
            if second {
                let mut out = vec![Span::new(span.start, span.start + comma)];
                out.extend(split(
                    text,
                    Span::new(span.start + tail, span.end),
                    cues,
                    protected,
                ));
                return out;
            }
        }
        from = comma + 1;
    }
    vec![span]
}

/// The subject of a clause: the words before the cue, without a list number, a
/// leading conjunction or a trailing comma.
fn lead(text: &str) -> String {
    let mut phrase = text.trim();
    for word in [
        "and",
        "but",
        "then",
        "also",
        "however",
        "further",
        "furthermore",
        "provided",
    ] {
        if let Some(rest) = phrase.strip_prefix(word) {
            if rest.starts_with(char::is_whitespace) {
                phrase = rest.trim_start();
            }
        }
    }
    // Drop a list marker: "1." or "(2), at the open of a clause.
    let mut skip = 0usize;
    let mut digits = 0usize;
    for (at, c) in phrase.char_indices() {
        if c.is_ascii_digit() || c == '(' {
            digits += 1;
            continue;
        }
        if (c == '.' || c == ')') && digits > 0 {
            skip = at + 1;
        }
        break;
    }
    let phrase = phrase[skip..].trim().trim_end_matches(',').trim();
    clip(phrase, 80)
}

/// Where the actor begins: after a fronted phrase that names a time. A list
/// such as "the Borrower, the Lender and the Guarantor shall pay" has no
/// deadline in its head, so its actor stays the whole subject.
fn fronted(head: &str, cues: &Cues) -> usize {
    match head.rfind(',') {
        Some(at) if cues.deadline.iter().any(|re| re.is_match(&head[..at])) => at + 1,
        _ => 0,
    }
}

/// The subject phrase of a clause: the words after the last comma before the
/// verb, which is where the actor sits when a condition is fronted.
fn subject(text: &str) -> &str {
    match text.rfind(',') {
        Some(at) => &text[at + 1..],
        None => text,
    }
}

/// Whether a cue matches the whole of a word, so "no" is not found inside
/// "notice".
fn matches_whole(list: &[Regex], word: &str) -> bool {
    list.iter().any(|re| {
        re.find(word)
            .is_some_and(|m| m.start() == 0 && m.end() == word.len())
    })
}

/// Whether a cue opens a phrase.
fn begins(list: &[Regex], text: &str) -> bool {
    let text = text.trim_start();
    list.iter()
        .any(|re| re.find(text).is_some_and(|m| m.start() == 0))
}

fn shift(at: usize, delta: isize) -> usize {
    (at as isize + delta).max(0) as usize
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.trim().to_string();
    }
    let cut: String = text.chars().take(limit).collect();
    match cut.rfind(' ') {
        Some(at) => format!("{} ...", cut[..at].trim_end()),
        None => format!("{cut} ..."),
    }
}

/// The time limits quoted from a sentence: from a deadline cue to the end of the
/// phrase it introduces.
fn deadlines_in(text: &str, cues: &Cues) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for pattern in &cues.deadline {
        for found in pattern.find_iter(text) {
            let tail = &text[found.start()..];
            let end = tail
                .find([';', '.', '\n'])
                .map(|i| found.start() + i)
                .unwrap_or(text.len());
            let phrase = clip(text[found.start()..end].trim().trim_end_matches(','), 72);
            if !phrase.is_empty() && !out.iter().any(|p| p == &phrase) {
                out.push(phrase);
            }
        }
    }
    out
}

/// Every term of art the document uses, in order of first use, with the reading
/// the glossary holds for it.
fn define(legal: &Legal, hits: &[super::term::Hit], protected: &[Span]) -> Vec<Definition> {
    let mut out: Vec<Definition> = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for hit in hits {
        let term = legal.lexicon.term(hit);
        if term.avoidable || !term.explain {
            continue;
        }
        if protected.iter().any(|p| p.intersects(&hit.span)) {
            continue;
        }
        if let Some(&index) = seen.get(&term.id) {
            out[index].uses += 1;
            continue;
        }
        seen.insert(term.id.clone(), out.len());
        out.push(Definition {
            id: term.id.clone(),
            term: hit.text.clone(),
            plain: term.plain.clone(),
            kind: term.kind.clone(),
            note: term.note.clone(),
            law: term.law.clone(),
            sources: term.sources.clone(),
            uses: 1,
            offset: hit.span.start,
        });
    }
    out
}

fn count(clauses: &[Clause], definitions: usize) -> Counts {
    let mut counts = Counts {
        definitions,
        ..Counts::default()
    };
    for clause in clauses {
        if clause.heading {
            continue;
        }
        match clause.kind {
            Kind::Duty => counts.duties += 1,
            Kind::Prohibition => counts.prohibitions += 1,
            Kind::Permission => counts.permissions += 1,
            Kind::Condition => counts.conditions += 1,
            Kind::Statement => counts.statements += 1,
        }
        if clause.condition.is_some() {
            counts.limited += 1;
        }
        if clause.agentless {
            counts.agentless += 1;
        }
        counts.rare += clause.rare.len();
    }
    counts
}

fn summarise(counts: &Counts, words: usize) -> String {
    let mut parts = Vec::new();
    if counts.duties > 0 {
        parts.push(plural(counts.duties, "duty", "duties"));
    }
    if counts.prohibitions > 0 {
        parts.push(plural(counts.prohibitions, "prohibition", "prohibitions"));
    }
    if counts.permissions > 0 {
        parts.push(plural(counts.permissions, "permission", "permissions"));
    }
    if counts.conditions > 0 {
        parts.push(plural(counts.conditions, "condition", "conditions"));
    }

    let mut text = if parts.is_empty() {
        format!(
            "{words} words. No duty, prohibition, permission or condition was found: this reads \
             as description, not as an operative clause."
        )
    } else {
        format!("{words} words. This text sets {}.", join(&parts))
    };
    if counts.limited > 0 {
        text.push_str(&format!(
            " {} limited by a condition.",
            plural(counts.limited, "clause is", "clauses are")
        ));
    }
    if counts.agentless > 0 {
        text.push_str(&format!(
            " {} written in the passive with no actor, so {} who must act.",
            plural(counts.agentless, "rule is", "rules are"),
            if counts.agentless == 1 {
                "it does not say"
            } else {
                "they do not say"
            }
        ));
    }
    if counts.statements > 0 {
        text.push_str(&format!(
            " {} {} a statement rather than a rule.",
            plural(counts.statements, "sentence", "sentences"),
            if counts.statements == 1 {
                "makes"
            } else {
                "make"
            }
        ));
    }
    if counts.rare > 0 {
        let (word, clause) = if counts.rare == 1 {
            ("word", "clause")
        } else {
            ("words", "clauses")
        };
        text.push_str(&format!(
            " It uses {} {word} general English rarely uses, flagged against the {clause} below.",
            counts.rare
        ));
    }
    match counts.definitions {
        0 => text.push_str(" It relies on no term of art from the glossary."),
        1 => text.push_str(" It uses one term of art, defined below."),
        2 => text.push_str(" It uses two terms of art, defined below."),
        n => text.push_str(&format!(" It uses {n} terms of art, defined below.")),
    }
    text
}

fn warn(
    legal: &Legal,
    counts: &Counts,
    definitions: &[Definition],
    gaps: &[super::gaps::Gap],
    words: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    // A folder written by hand holds a glossary and no corpus counts. The brief
    // can still be built, and saying what it could not measure is the point of
    // this list.
    if legal.freq.per_million.is_empty() {
        out.push(
            "no frequency table beside the glossary, so no word is reported as one a general \
             reader has not met"
                .to_string(),
        );
    }
    if !gaps.is_empty() {
        let names = gaps
            .iter()
            .map(|gap| gap.term.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        out.push(format!(
            "{} defined term{} {} no glossary entry: {names}",
            gaps.len(),
            if gaps.len() == 1 { "" } else { "s" },
            if gaps.len() == 1 { "has" } else { "have" }
        ));
    }
    if counts.duties + counts.prohibitions + counts.permissions == 0 {
        out.push(
            "no duty, prohibition or permission was found, so there is nothing to comply \
                  with in this text as written"
                .to_string(),
        );
    }
    let rules = counts.duties + counts.prohibitions;
    if rules > 0 && counts.agentless * 2 >= rules {
        out.push(format!(
            "{} of {} rules name no actor: a reader cannot tell who must act",
            counts.agentless, rules
        ));
    }
    if words >= 20 && definitions.is_empty() {
        out.push("no glossary term was found, so the definitions section is empty".to_string());
    }
    if words < 10 {
        out.push(
            "the text is shorter than ten words, so the brief can only report what is there"
                .to_string(),
        );
    }
    out
}

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

fn join(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [one] => one.clone(),
        [head @ .., last] => format!("{} and {last}", head.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legal() -> Legal {
        Legal::load("data/legal").expect("the shipped instrument")
    }

    fn brief(text: &str) -> Brief {
        Brief::build(&legal(), text).expect("a brief")
    }

    #[test]
    fn splits_at_a_stop_but_not_inside_a_citation() {
        let legal = legal();
        let text = "The Act, 17 U.S.C. § 102, protects the work. It does not protect ideas.";
        let protected = legal.guard.spans(text);
        let cues = Cues::new(&legal.settings.clauses).unwrap();
        let spans = sentences(text, &protected, &cues.abbreviations);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert!(text[spans[1].start..].starts_with("It"));
    }

    #[test]
    fn reads_a_duty_with_its_actor_and_deadline() {
        let b =
            brief("The Licensee shall pay the fee within thirty (30) days of the invoice date.");
        assert_eq!(b.clauses.len(), 1);
        let clause = &b.clauses[0];
        assert_eq!(clause.kind, Kind::Duty);
        assert_eq!(clause.actor, "The Licensee");
        assert!(clause.cue.eq_ignore_ascii_case("shall"));
        assert_eq!(clause.deadlines.len(), 1);
        assert!(clause.deadlines[0].starts_with("within thirty (30) days"));
        assert_eq!(b.counts.duties, 1);
    }

    #[test]
    fn a_passive_duty_with_no_agent_is_counted_apart() {
        let b = brief(
            "The notice shall be filed with the court at least nine days before the hearing.",
        );
        assert_eq!(b.clauses[0].kind, Kind::Duty);
        assert!(b.clauses[0].agentless);
        assert_eq!(b.counts.agentless, 1);
        assert!(b.summary.contains("passive with no actor"));
    }

    #[test]
    fn a_passive_duty_that_names_its_agent_is_not_agentless() {
        let b = brief("The report must be submitted by the Borrower within thirty days.");
        assert_eq!(b.clauses[0].kind, Kind::Duty);
        assert!(!b.clauses[0].agentless);
        assert_eq!(b.counts.agentless, 0);
    }

    #[test]
    fn an_active_duty_and_an_adjectival_duty_are_not_agentless() {
        let b =
            brief("The Buyer shall pay the price. The Software shall be available at all times.");
        assert_eq!(b.counts.duties, 2);
        assert_eq!(b.counts.agentless, 0);
    }

    #[test]
    fn an_irregular_participle_is_read_as_a_passive() {
        let b = brief("The fee shall be paid on the closing date.");
        assert!(b.clauses[0].agentless, "{:?}", b.clauses[0]);
        let adjectival = brief("The Software shall be open to inspection.");
        assert!(!adjectival.clauses[0].agentless);
    }

    #[test]
    fn a_majority_of_agentless_rules_is_a_warning() {
        let b = brief(
            "The notice shall be filed with the court. The fee shall be paid on time. The Buyer shall sign.",
        );
        assert_eq!(b.counts.agentless, 2);
        assert!(b.warnings.iter().any(|w| w.contains("name no actor")));
    }

    #[test]
    fn reads_a_prohibition_and_keeps_it_apart_from_a_duty() {
        let b = brief("The Licensee shall not sublicense the Software to any third party.");
        assert_eq!(b.clauses[0].kind, Kind::Prohibition);
        assert_eq!(b.counts.duties, 0);
        assert_eq!(b.counts.prohibitions, 1);
    }

    #[test]
    fn reads_a_permission_limited_by_a_condition() {
        let b = brief(
            "If the Licensee fails to pay, the Licensor may terminate this Agreement on notice.",
        );
        let clause = &b.clauses[0];
        assert_eq!(clause.kind, Kind::Permission);
        assert_eq!(clause.actor, "the Licensor");
        assert!(clause
            .condition
            .as_deref()
            .is_some_and(|c| c.starts_with("If the Licensee fails to pay")));
        assert_eq!(b.counts.limited, 1);
    }

    #[test]
    fn defines_the_terms_of_art_it_finds() {
        let b = brief("The doctrine of laches bars a claim when the claimant has waited too long.");
        let laches = b
            .definitions
            .iter()
            .find(|d| d.id == "laches")
            .expect("laches is in the glossary");
        assert_eq!(laches.plain, "losing a claim by waiting too long");
        assert!(!laches.note.is_empty());
        assert_eq!(laches.uses, 1);
    }

    #[test]
    fn a_plain_statement_is_counted_as_a_statement() {
        let b = brief("This notice describes the software and the support that comes with it.");
        assert_eq!(b.clauses[0].kind, Kind::Statement);
        assert_eq!(b.counts.statements, 1);
        assert!(b
            .warnings
            .iter()
            .any(|w| w.contains("nothing to comply with")));
    }

    #[test]
    fn a_second_rule_after_a_comma_is_its_own_clause() {
        let b = brief(
            "If the fee is late, the Licensor may suspend access, and the Licensee shall pay \
             interest.",
        );
        assert_eq!(b.counts.permissions, 1);
        assert_eq!(b.counts.duties, 1);
        let duty = b.of(Kind::Duty).next().expect("the duty behind the comma");
        assert_eq!(duty.actor, "the Licensee");
        assert!(duty.text.starts_with("The Licensee"), "{}", duty.text);
    }

    #[test]
    fn a_heading_is_not_counted_as_a_statement() {
        let b = brief("2. FEES. The Licensee shall pay the fee.");
        assert_eq!(b.counts.statements, 0);
        assert_eq!(b.counts.duties, 1);
        assert!(b.clauses.iter().any(|c| c.heading));
    }

    #[test]
    fn the_summary_counts_only_what_was_found() {
        let b = brief("The Vendor must deliver the goods. The Buyer may reject them.");
        assert!(b.summary.contains("1 duty"));
        assert!(b.summary.contains("1 permission"));
        assert!(b.summary.starts_with("11 words."), "{}", b.summary);
        assert!(!b.summary.contains("condition"), "{}", b.summary);
    }

    #[test]
    fn a_negated_subject_forbids_and_takes_the_permission_modal() {
        let b = brief("No person shall assign this agreement without written consent.");
        let clause = &b.clauses[0];
        assert_eq!(clause.kind, Kind::Prohibition);
        assert!(
            clause.text.starts_with("No person may assign"),
            "{}",
            clause.text
        );
        assert!(clause.actor.is_empty(), "{:?}", clause.actor);
        assert_eq!(b.counts.prohibitions, 1);
        assert_eq!(b.counts.duties, 0);
    }

    #[test]
    fn a_negated_subject_that_states_a_limit_is_a_statement() {
        let b = brief("Nothing in this section shall limit the liability of either party.");
        assert_eq!(b.clauses[0].kind, Kind::Statement);
        assert_eq!(b.counts.prohibitions, 0);
        assert_eq!(b.counts.duties, 0);
    }

    #[test]
    fn a_fronted_deadline_is_not_a_negated_subject() {
        let b = brief("No later than 30 days before the hearing, the Borrower shall file.");
        assert_eq!(b.clauses[0].kind, Kind::Duty);
        assert_eq!(b.clauses[0].actor, "the Borrower");
    }

    #[test]
    fn a_trailing_condition_loses_its_full_stop() {
        let b = brief("The Lender may accelerate the loan if the Borrower defaults.");
        assert_eq!(
            b.clauses[0].condition.as_deref(),
            Some("if the Borrower defaults")
        );
    }

    #[test]
    fn every_brief_carries_the_disclaimer() {
        let b = brief("The Borrower shall repay the loan.");
        assert!(b.disclaimer.contains("Research project"));
        assert!(b.disclaimer.contains("not legal advice"));
    }

    #[test]
    fn a_clause_carries_the_word_a_general_reader_has_not_met() {
        let b = brief("The fee shall be paid hereunder within thirty days.");
        assert_eq!(b.clauses[0].rare, vec!["hereunder".to_string()]);
        assert_eq!(b.counts.rare, 1);
        assert!(
            b.summary.contains("general English rarely uses"),
            "{}",
            b.summary
        );
    }

    #[test]
    fn ordinary_words_are_not_reported_as_rare() {
        let b = brief("The Buyer shall pay the price under the contract within thirty days.");
        assert!(b.clauses[0].rare.is_empty(), "{:?}", b.clauses[0].rare);
        assert!(!b.summary.contains("rarely uses"), "{}", b.summary);
    }

    #[test]
    fn a_rare_word_is_named_once_per_clause() {
        let b = brief("The rent hereunder and the fee hereunder shall be paid.");
        assert_eq!(b.clauses[0].rare, vec!["hereunder".to_string()]);
    }

    #[test]
    fn one_agentless_rule_is_reported_in_the_singular() {
        let b = brief("The notice shall be filed with the court.");
        assert_eq!(b.counts.agentless, 1);
        assert!(
            b.summary.contains("it does not say who must act"),
            "{}",
            b.summary
        );
    }
}
