//! The inflections English puts on a word, folded back onto a known stem.
//!
//! A seed lexicon lists `happy`, not `happiness`, `happier` and `happily`; a
//! bank of conversation contains all four. Rather than carry every form as its
//! own row, a word is tried against a short ordered list of candidate stems and
//! the first one the lexicon knows wins. The rules are suffix rules only: they
//! are cheap, deterministic, and they cover the regular morphology that makes
//! a 600 word seed read like a dictionary.
//!
//! The order matters. Plurals are tried before the bare `-s` strip so `stories`
//! reaches `story` and not `storie`; `-ing` is tried before `-er`, so `boring`
//! does not become `bor`. Nothing here invents a word: every candidate is
//! checked against the lexicon, so a wrong guess costs a lookup and nothing
//! else.

/// A stem plus the suffix rule that produced it, for a reader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub stem: String,
    pub rule: &'static str,
}

/// Candidate stems of one lowercased word, the word itself first.
pub fn candidates(word: &str) -> Vec<Candidate> {
    let mut out = vec![Candidate {
        stem: word.to_string(),
        rule: "exact",
    }];
    let length = word.chars().count();
    let mut push = |stem: String, rule: &'static str| {
        let letters = stem.chars().count();
        let long_enough = letters >= 3 && (letters >= 4 || length <= 5 || rule == "undoubled");
        if long_enough && !out.iter().any(|c| c.stem == stem) {
            out.push(Candidate { stem, rule });
        }
    };
    for (suffix, rule) in [
        ("ies", "plural"),
        ("ied", "past"),
        ("ing", "progressive"),
        ("est", "superlative"),
        ("ier", "comparative"),
        ("iest", "superlative"),
        ("er", "comparative"),
        ("ed", "past"),
        ("ly", "adverb"),
        ("ness", "noun"),
        ("ment", "noun"),
        ("es", "plural"),
        ("s", "plural"),
    ] {
        let Some(base) = word.strip_suffix(suffix) else {
            continue;
        };
        match suffix {
            "ies" | "ied" | "ier" | "iest" => {
                push(format!("{base}y"), rule);
            }
            "s" if word.ends_with("ss") => {}
            "es" | "s" => push(base.to_string(), rule),
            "ly" => {
                push(base.to_string(), rule);
                if let Some(stem) = base.strip_suffix('i') {
                    push(format!("{stem}y"), "adverb");
                }
                if let Some(stem) = base.strip_suffix("al") {
                    push(stem.to_string(), "adverb");
                }
            }
            "ness" | "ment" => {
                push(base.to_string(), rule);
                if let Some(stem) = base.strip_suffix('i') {
                    push(format!("{stem}y"), rule);
                }
            }
            _ => {
                push(base.to_string(), rule);
                push(format!("{base}e"), "restored e");
                if let Some(stem) = undouble(base) {
                    push(stem, "undoubled");
                }
                if suffix == "ing" {
                    if let Some(stem) = base.strip_suffix('y') {
                        push(format!("{stem}ie"), "progressive");
                    }
                }
            }
        }
    }
    out
}

/// The stem of a word, for character similarity rather than for lookup.
pub fn stem(word: &str) -> String {
    let all = candidates(word);
    all.iter()
        .find(|candidate| candidate.rule == "undoubled")
        .or_else(|| all.iter().find(|candidate| candidate.rule != "exact"))
        .map(|candidate| candidate.stem.clone())
        .unwrap_or_else(|| word.to_string())
}

/// `stopp` to `stop`, `runn` to `run`: one of a doubled final consonant.
fn undouble(base: &str) -> Option<String> {
    let mut chars = base.chars().rev();
    let last = chars.next()?;
    let previous = chars.next()?;
    if last == previous && !"aeiou".contains(last) {
        let mut stem: String = base.chars().collect();
        stem.pop();
        return Some(stem);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stems(word: &str) -> Vec<String> {
        candidates(word).into_iter().map(|c| c.stem).collect()
    }

    #[test]
    fn the_word_itself_is_always_first() {
        assert_eq!(candidates("happy")[0].stem, "happy");
        assert_eq!(candidates("happy")[0].rule, "exact");
    }

    #[test]
    fn plurals_reach_the_singular() {
        assert!(stems("stories").contains(&"story".to_string()));
        assert!(stems("problems").contains(&"problem".to_string()));
        assert!(stems("boxes").contains(&"box".to_string()));
        assert!(!stems("class").contains(&"clas".to_string()));
    }

    #[test]
    fn regular_verb_forms_reach_the_base() {
        assert!(stems("running").contains(&"run".to_string()));
        assert!(stems("stopped").contains(&"stop".to_string()));
        assert!(stems("making").contains(&"make".to_string()));
        assert!(stems("loved").contains(&"love".to_string()));
        assert!(stems("tried").contains(&"try".to_string()));
    }

    #[test]
    fn adverbs_and_nouns_reach_the_adjective() {
        assert!(stems("happily").contains(&"happy".to_string()));
        assert!(stems("happiness").contains(&"happy".to_string()));
        assert!(stems("easily").contains(&"easy".to_string()));
        assert!(stems("angrier").contains(&"angry".to_string()));
    }

    #[test]
    fn no_candidate_is_shorter_than_three_letters_unless_it_is_the_word() {
        for word in ["is", "as", "us", "goes", "does"] {
            for candidate in candidates(word) {
                assert!(
                    candidate.stem.chars().count() >= 3 || candidate.stem == word,
                    "{word} gave {}",
                    candidate.stem
                );
            }
        }
    }

    #[test]
    fn stem_is_the_first_derived_candidate() {
        assert_eq!(stem("running"), "run");
        assert_eq!(stem("happy"), "happy");
    }
}
