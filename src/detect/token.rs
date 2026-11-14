//! Words for a lexicon, and the closed class that never becomes a term.
//!
//! A token here is a run of letters, with an apostrophe or a hyphen kept when it
//! sits inside the run. Digits start no token and end the one they interrupt,
//! so `$2.4` contributes nothing and `quarter million` contributes two words.
//! That is the rule the marketlab corpus was scored with, kept so a lexicon
//! fitted here reads the same text the same way.
//!
//! One deliberate difference: this rule keeps a non ASCII letter inside a word,
//! so `Canada` and `canada` stay whole where an ASCII-only class would split a
//! name into fragments. The fragments were never terms anyway.

/// Words of a text, in order, lowercased.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphabetic() {
            for lower in c.to_lowercase() {
                word.push(lower);
            }
            continue;
        }
        let joiner = c == '\'' || c == '-' || c == '\u{2019}';
        if joiner && !word.is_empty() {
            word.push(if c == '\u{2019}' { '\'' } else { c });
            continue;
        }
        if !word.is_empty() {
            out.push(std::mem::take(&mut word));
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

/// The closed grammatical class, sorted for binary search.
///
/// This is grammar rather than vocabulary: the words English uses to hold
/// content together. A member can still carry meaning in a register, but it
/// names no instrument, no abstraction and no threat, so it is never a lexicon
/// term. The list is sorted; [`FUNCTION`] is asserted sorted in the tests.
pub const FUNCTION: [&str; 114] = [
    "a", "about", "after", "against", "also", "am", "an", "and", "another", "any", "are", "as",
    "at", "be", "been", "before", "being", "between", "but", "by", "can", "could", "did", "do",
    "does", "done", "during", "each", "every", "for", "from", "had", "has", "have", "having", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "many",
    "may", "me", "might", "more", "most", "much", "must", "my", "no", "nor", "not", "of", "on",
    "only", "or", "other", "ought", "our", "over", "own", "per", "same", "shall", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "upon", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "will", "with", "within",
    "without", "would", "you", "your",
];

/// Whether a lowercased word is in the closed class.
pub fn is_function(word: &str) -> bool {
    FUNCTION.binary_search(&word).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_closed_class_is_sorted_and_unique() {
        for pair in FUNCTION.windows(2) {
            assert!(pair[0] < pair[1], "{} then {}", pair[0], pair[1]);
        }
    }

    #[test]
    fn digits_break_words_and_apostrophes_join_them() {
        assert_eq!(
            tokens("$2.4 million, don't you?"),
            ["million", "don't", "you"]
        );
        assert_eq!(tokens("year-on-year 100%"), ["year-on-year"]);
        assert_eq!(tokens("Ca\u{f1}ada"), ["ca\u{f1}ada"]);
        assert!(tokens("  42 ").is_empty());
    }

    #[test]
    fn a_curly_apostrophe_is_folded_to_a_straight_one() {
        assert_eq!(tokens("don\u{2019}t"), ["don't"]);
    }
}
