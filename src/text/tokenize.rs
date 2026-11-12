//! Words and sentences without a tokenizer dependency.
//!
//! Both functions are lossless over the input: they borrow slices of the text
//! rather than rebuilding it, so a caller can report character offsets if it
//! needs to.
//!
//! The word rule is a run of alphanumeric characters, with an apostrophe or a
//! hyphen kept when it sits between two alphanumeric characters. That keeps
//! `don't` and `year-on-year` whole and splits `billion` from `$2`, which is
//! what the counts want. It is not a Unicode word segmenter; it does not need
//! to be, because every downstream stage works on lowercase and on counts.

/// Words of a text, in order.
pub fn words(text: &str) -> Words<'_> {
    Words { text, pos: 0 }
}

/// Iterator over the words of a text, borrowing slices of it.
pub struct Words<'a> {
    text: &'a str,
    pos: usize,
}

fn alnum(c: char) -> bool {
    c.is_alphanumeric()
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let len = self.text.len();
        let mut i = self.pos;
        while i < len {
            let c = self.text[i..].chars().next()?;
            if alnum(c) {
                break;
            }
            i += c.len_utf8();
        }
        if i >= len {
            self.pos = len;
            return None;
        }
        let start = i;
        let mut j = i;
        let mut end = i;
        while j < len {
            let c = self.text[j..].chars().next()?;
            if alnum(c) {
                j += c.len_utf8();
                end = j;
                continue;
            }
            if c == '\'' || c == '-' {
                let after = j + c.len_utf8();
                if after < len {
                    if let Some(next) = self.text[after..].chars().next() {
                        if alnum(next) {
                            j = after;
                            continue;
                        }
                    }
                }
            }
            break;
        }
        self.pos = end;
        Some(&self.text[start..end])
    }
}

/// Word count, in one pass and without allocating a vector.
pub fn count_words(text: &str) -> usize {
    words(text).count()
}

/// Lowercase a word in place of a full Unicode case fold, with a small buffer
/// reused by the counting helpers.
pub fn lowercase_word(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars() {
        for lower in c.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

/// Abbreviations that end with a full stop but do not end a sentence. The list
/// is deliberately short: a missed abbreviation costs one wrong boundary, an
/// over-eager one costs the sentence split entirely.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "ltd", "co", "corp", "llc", "plc",
    "vs", "etc", "al", "fig", "no", "vol", "pp", "approx", "dept", "est", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "e.g", "i.e",
    "a.m", "p.m", "billion", "million", "bln", "mln", "q1", "q2", "q3", "q4",
];

/// Split a text into sentences, keeping any sentence of at least `min_words`
/// words and dropping the rest.
///
/// A boundary is a full stop, a question mark or an exclamation mark followed
/// by whitespace and a character that is not lowercase. A full stop also has to
/// survive the abbreviation list and the single-letter initial rule.
pub fn sentences(text: &str, min_words: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        current.push(c);
        let boundary = matches!(c, '!' | '?') || c == '.';
        if boundary {
            // Look ahead over closing quotes and spaces.
            let mut j = i + 1;
            while j < chars.len() && matches!(chars[j], '"' | '\'' | ')' | ']') {
                j += 1;
            }
            let mut k = j;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            let next_break = k >= chars.len() || !chars[k].is_lowercase();
            let terminal = j >= chars.len() || chars[j].is_whitespace();
            if boundary && next_break && terminal && c == '.' && is_abbreviation(&current) {
                i += 1;
                continue;
            }
            if boundary && next_break && terminal {
                let sentence = current.trim().to_string();
                if count_words(&sentence) >= min_words.max(1) {
                    out.push(sentence);
                }
                current.clear();
                i = k;
                continue;
            }
        }
        i += 1;
    }
    let tail = current.trim().to_string();
    if count_words(&tail) >= min_words.max(1) {
        out.push(tail);
    }
    out
}

fn is_abbreviation(sentence_so_far: &str) -> bool {
    let trimmed = sentence_so_far.trim_end_matches(['.', '"', '\'', ')', ']', ' ']);
    let last: String = trimmed
        .rsplit(|c: char| c.is_whitespace() || c == '(' || c == '"')
        .next()
        .unwrap_or("")
        .trim_end_matches('.')
        .to_lowercase();
    if last.is_empty() {
        return false;
    }
    if last.chars().count() == 1 {
        return true;
    }
    ABBREVIATIONS.contains(&last.as_str())
}

/// Truncate on a word boundary at or below `max_chars`, and mark the cut so a
/// reader can see the text was shortened rather than written that way.
pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars).collect();
    match head.rfind(char::is_whitespace) {
        Some(cut) if cut > max_chars / 2 => format!("{} ...", head[..cut].trim_end()),
        _ => format!("{head} ..."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_keep_internal_punctuation() {
        let got: Vec<&str> = words("The year-on-year fall isn't $2bn.").collect();
        assert_eq!(got, vec!["The", "year-on-year", "fall", "isn't", "2bn"]);
    }

    #[test]
    fn sentences_respect_abbreviations() {
        let text = "Acme Inc. said profit fell. Analysts at U.S. Bank agreed! Did it fall?";
        let got = sentences(text, 3);
        assert_eq!(got.len(), 3, "{got:?}");
        assert!(got[0].ends_with("fell."));
    }

    #[test]
    fn truncate_cuts_on_a_space() {
        let got = truncate("one two three four five", 12);
        assert_eq!(got, "one two ...");
    }
}
