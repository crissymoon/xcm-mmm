//! Declared cue indices: the constructs no corpus labels.
//!
//! Five of the seven axes have no labelled bank of advertising. Nobody has
//! published a set of publicity stunts, pump copy, follower boasts, borrowings
//! of a press frame and threat appeals with labels attached, so fitting a
//! logistic model on a rule's own output would produce a model that agrees with
//! its seed and calls the agreement accuracy. These axes instead ship as
//! declared cue sets with the same index shape, and the audit measures the one
//! thing a rule set can be measured on without labels: whether two disjoint
//! halves of its cues pick out the same ads.
//!
//! The cue table is data, not code. It is exported from the Python module that
//! scores the corpus, so the patterns here cannot drift from the patterns the
//! published numbers were produced with.
//!
//! The reading of one text is a weighted mixture of component densities, each
//! capped at twice its ninetieth percentile in the fitting corpus, behind a
//! gate: a construct that names no gated component scores zero, and one that
//! needs several distinct components scores zero until it has them. That gate
//! is why "only 3 left, ends tonight" is retail pressure rather than market
//! manipulation, and why "join us on Saturday" is a call to action rather than
//! an audience claim.

use std::collections::BTreeMap;

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use super::lexicon::{round4, Lexicon};

/// One declared index axis, as it travels in `cues.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexSpec {
    pub tag: String,
    #[serde(default)]
    pub question: String,
    pub components: BTreeMap<String, f64>,
    pub cues: BTreeMap<String, String>,
    #[serde(default)]
    pub gate: Vec<String>,
    #[serde(default = "one")]
    pub min_groups: usize,
    #[serde(default)]
    pub halves: Vec<Vec<String>>,
    #[serde(default)]
    pub checks: Vec<Check>,
    /// Component a learned lexicon density is added to. Absent for the cue
    /// axes, which carry no lexicon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexicon_in: Option<String>,
}

fn one() -> usize {
    1
}

/// A declared example: text that must fire, or must not.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Check {
    pub text: String,
    pub expected: bool,
}

/// A spec with its patterns compiled.
#[derive(Clone, Debug)]
pub struct Index {
    spec: IndexSpec,
    patterns: BTreeMap<String, Regex>,
}

impl Index {
    /// Compile every cue. A pattern that does not build is a refusal, not a
    /// zero, because a silently dead cue group looks exactly like a construct
    /// that never occurs.
    pub fn compile(spec: IndexSpec) -> Result<Self, String> {
        let mut patterns = BTreeMap::new();
        for (name, pattern) in &spec.cues {
            let compiled = RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| format!("cue {name}: {e}"))?;
            patterns.insert(name.clone(), compiled);
        }
        Ok(Self { spec, patterns })
    }

    pub fn spec(&self) -> &IndexSpec {
        &self.spec
    }

    /// Component densities: cue hits per hundred words, and the lexicon density
    /// folded into the component the spec names when one is supplied.
    pub fn parts(
        &self,
        text: &str,
        words: &[String],
        lexicon: Option<&Lexicon>,
    ) -> BTreeMap<String, f64> {
        let count = words.len().max(1) as f64;
        let mut out = BTreeMap::new();
        for name in self.spec.components.keys() {
            let hits = self
                .patterns
                .get(name)
                .map(|p| p.find_iter(text).count())
                .unwrap_or(0);
            out.insert(name.clone(), 100.0 * hits as f64 / count);
        }
        if let (Some(target), Some(lex)) = (self.spec.lexicon_in.as_ref(), lexicon) {
            if let Some(part) = out.get_mut(target) {
                let mass: f64 = words
                    .iter()
                    .filter_map(|w| lex.terms.get(w))
                    .map(|z| z.min(lex.cap))
                    .sum();
                *part += 100.0 * mass / (count + 20.0);
            }
        }
        out
    }

    /// Weighted, capped mixture, zero unless the gate is satisfied.
    pub fn score(
        &self,
        parts: &BTreeMap<String, f64>,
        reference: &BTreeMap<String, Vec<f64>>,
    ) -> f64 {
        let gated = !self.spec.gate.is_empty();
        if gated
            && !self
                .spec
                .gate
                .iter()
                .any(|name| parts.get(name).copied().unwrap_or(0.0) > 0.0)
        {
            return 0.0;
        }
        let distinct = parts.values().filter(|v| **v > 0.0).count();
        if distinct < self.spec.min_groups.max(1) {
            return 0.0;
        }
        let mut score = 0.0;
        for (name, weight) in &self.spec.components {
            let unit = reference
                .get(name)
                .map(|q| scale_of(q))
                .unwrap_or(1.0)
                .max(0.01);
            let part = parts.get(name).copied().unwrap_or(0.0);
            score += weight * (part / unit).min(1.0);
        }
        round4(score)
    }

    /// The phrases that actually fired, in cue order and de-duplicated.
    pub fn hits(&self, text: &str, take: usize) -> Vec<String> {
        let mut found: Vec<String> = Vec::new();
        for pattern in self.patterns.values() {
            for hit in pattern.find_iter(text) {
                let phrase = hit.as_str().to_lowercase();
                let phrase = phrase.trim().to_string();
                if !phrase.is_empty() && !found.contains(&phrase) {
                    found.push(phrase);
                }
            }
        }
        found.truncate(take);
        found
    }

    /// Mean share of one half of the cues, used by the audit.
    pub fn half_score(
        &self,
        parts: &BTreeMap<String, f64>,
        names: &[String],
        reference: &BTreeMap<String, Vec<f64>>,
    ) -> f64 {
        if names.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for name in names {
            let unit = reference
                .get(name)
                .map(|q| scale_of(q))
                .unwrap_or(1.0)
                .max(0.01);
            total += (parts.get(name).copied().unwrap_or(0.0) / unit).min(1.0);
        }
        total / names.len() as f64
    }
}

/// Quantiles of a component, on a fixed grid so the artifact is small and a
/// rebuild writes the same bytes. Twenty one points put the ninetieth
/// percentile of [`scale_of`] on a grid line rather than between two.
pub fn quantiles(values: &[f64], bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0];
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    (0..=bins)
        .map(|i| {
            let index = (i as f64 / bins as f64 * (ordered.len() - 1) as f64).floor() as usize;
            round4(ordered[index.min(ordered.len() - 1)])
        })
        .collect()
}

/// Twice the ninetieth percentile, so one p90 hit counts as half of a cap.
///
/// A component whose p90 is zero is one most ads never use. Falling back to a
/// fixed epsilon there would make the first sighting saturate the component at
/// full weight, which is how a single word became a "strong" reading on a
/// corpus where the cue appears once. The fallback is twice the observed
/// maximum, and one hit per hundred words only if the cue was never observed at
/// all in the fitting split.
pub fn scale_of(reference: &[f64]) -> f64 {
    if reference.is_empty() {
        return 1.0;
    }
    let p90 = reference[((0.90 * (reference.len() - 1) as f64) as usize).min(reference.len() - 1)];
    if p90 > 0.0 {
        return (2.0 * p90).max(0.01);
    }
    let peak = reference.iter().copied().fold(0.0, f64::max);
    if peak > 0.0 {
        (2.0 * peak).max(0.01)
    } else {
        1.0
    }
}

/// Level cuts from the live scores: median, ninetieth, ninety seventh.
///
/// Below five firing rows there is no distribution to read a percentile off,
/// and the declared defaults are honest about it rather than pretending a
/// single ad is a threshold.
pub fn cuts(scores: &[f64]) -> (Vec<f64>, String) {
    let mut live: Vec<f64> = scores.iter().copied().filter(|s| *s > 0.0).collect();
    if live.len() < 5 {
        return (
            vec![0.5, 0.7, 0.85],
            format!("default: {} firing ads below 5", live.len()),
        );
    }
    live.sort_by(f64::total_cmp);
    let points = [0.50_f64, 0.90, 0.97];
    let out: Vec<f64> = points
        .iter()
        .map(|p| {
            let index = ((p * (live.len() - 1) as f64) as usize).min(live.len() - 1);
            round4(live[index])
        })
        .collect();
    (
        out,
        format!("train quantiles over {} firing ads", live.len()),
    )
}

/// none, weak, clear or strong for a cue index.
///
/// A cue index is gated, so a score of zero is the whole "none" case and any
/// firing ad is at least weak. The cuts then separate weak from clear and clear
/// from strong. Reading "none" for a firing ad below the median of firing ads
/// would call a construct absent on the strength of a percentile.
pub fn level_index(score: f64, cuts: &[f64]) -> &'static str {
    const LEVELS: [&str; 4] = ["none", "weak", "clear", "strong"];
    if score <= 0.0 {
        return LEVELS[0];
    }
    let rank = cuts.iter().filter(|cut| score >= **cut).count();
    LEVELS[rank.clamp(1, 3)]
}

/// none, weak, clear or strong for a learned axis.
///
/// Here the first cut is the firing cut chosen on held out data, so a
/// probability below it is genuinely "none": the axis is answering the question
/// it was fitted to answer.
pub fn level_lexicon(probability: f64, cuts: &[f64]) -> &'static str {
    const LEVELS: [&str; 4] = ["none", "weak", "clear", "strong"];
    LEVELS[cuts
        .iter()
        .filter(|cut| probability >= **cut)
        .count()
        .min(3)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::token;

    fn spec() -> IndexSpec {
        IndexSpec {
            tag: "#MarketManipulation".into(),
            question: String::new(),
            components: BTreeMap::from([
                ("pump".to_string(), 0.5),
                ("guarantee".to_string(), 0.5),
                ("scarcity".to_string(), 0.0),
                ("clock".to_string(), 0.0),
            ]),
            cues: BTreeMap::from([
                ("pump".to_string(), r"(?i)\bto the moon\b".to_string()),
                (
                    "guarantee".to_string(),
                    r"(?i)\bguaranteed returns?\b".to_string(),
                ),
                ("scarcity".to_string(), r"(?i)\bonly \d+ left\b".to_string()),
                ("clock".to_string(), r"(?i)\bends tonight\b".to_string()),
            ]),
            gate: vec!["pump".into(), "guarantee".into()],
            min_groups: 1,
            halves: vec![],
            checks: vec![],
            lexicon_in: None,
        }
    }

    fn reference() -> BTreeMap<String, Vec<f64>> {
        BTreeMap::from([
            ("pump".to_string(), vec![0.0, 1.0, 2.0]),
            ("guarantee".to_string(), vec![0.0, 1.0, 2.0]),
            ("scarcity".to_string(), vec![0.0, 0.0, 0.0]),
            ("clock".to_string(), vec![0.0, 0.0, 0.0]),
        ])
    }

    fn read(index: &Index, text: &str) -> f64 {
        let words = token::tokens(text);
        let parts = index.parts(text, &words, None);
        index.score(&parts, &reference())
    }

    #[test]
    fn the_gate_refuses_retail_pressure() {
        let index = Index::compile(spec()).unwrap();
        assert_eq!(
            read(&index, "Only 3 left and this offer ends tonight."),
            0.0
        );
    }

    #[test]
    fn a_gated_component_carries_the_score() {
        let index = Index::compile(spec()).unwrap();
        assert!(read(&index, "Guaranteed returns, before it ends tonight.") > 0.0);
        assert!(read(&index, "This goes to the moon.") > 0.0);
    }

    #[test]
    fn min_groups_needs_distinct_components() {
        let mut declared = spec();
        declared.min_groups = 2;
        let index = Index::compile(declared).unwrap();
        assert_eq!(read(&index, "Guaranteed returns only."), 0.0);
        assert!(read(&index, "Guaranteed returns to the moon.") > 0.0);
    }

    #[test]
    fn a_lexicon_can_be_folded_into_one_component() {
        let mut declared = spec();
        declared.lexicon_in = Some("pump".into());
        declared.gate = vec!["pump".into()];
        let index = Index::compile(declared).unwrap();
        let lexicon = Lexicon {
            terms: BTreeMap::from([("moon".to_string(), 3.0)]),
            cap: 6.0,
            prior_tokens: 20.0,
            report: Default::default(),
        };
        let text = "moon";
        let words = token::tokens(text);
        let parts = index.parts(text, &words, Some(&lexicon));
        assert!(parts["pump"] > 0.0);
        assert!(index.score(&parts, &reference()) > 0.0);
    }

    #[test]
    fn hits_are_the_phrases_not_the_patterns() {
        let index = Index::compile(spec()).unwrap();
        let hits = index.hits("Guaranteed returns and only 2 left", 5);
        assert!(hits.contains(&"guaranteed returns".to_string()), "{hits:?}");
    }

    #[test]
    fn a_firing_index_is_never_none() {
        let cuts = vec![0.5, 0.7, 0.85];
        assert_eq!(level_index(0.0, &cuts), "none");
        assert_eq!(level_index(0.44, &cuts), "weak");
        assert_eq!(level_index(0.6, &cuts), "weak");
        assert_eq!(level_index(0.75, &cuts), "clear");
        assert_eq!(level_index(0.9, &cuts), "strong");
    }

    #[test]
    fn a_learned_axis_fires_above_its_own_cut() {
        let cuts = vec![0.25, 0.7, 0.9];
        assert_eq!(level_lexicon(0.1, &cuts), "none");
        assert_eq!(level_lexicon(0.3, &cuts), "weak");
        assert_eq!(level_lexicon(0.8, &cuts), "clear");
        assert_eq!(level_lexicon(0.95, &cuts), "strong");
    }
}
