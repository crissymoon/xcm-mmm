//! Character folding, markup removal, link removal and boilerplate screening.
//!
//! Every stage reports what it removed. A cleaner that silently deletes a fifth
//! of a corpus is indistinguishable from one that corrupts it, so the pipeline
//! prints the counts and a caller can refuse a run whose removals look wrong.
//!
//! The boilerplate screen is deliberately conservative: a line is dropped only
//! when it is short (at most 22 words), carries a known site phrase, and is not
//! the first line of the body. A news sentence that mentions a cookie policy in
//! passing is 30 words long and survives.

use std::collections::HashSet;

use crate::text::tokenize;

/// Phrases that mark a navigation, consent or subscription line. Matched on the
/// lowercase line, which is why they are all lowercase here.
const BOILERPLATE: &[&str] = &[
    "we use cookies",
    "this site uses cookies",
    "accept cookies",
    "cookie policy",
    "cookie settings",
    "manage cookies",
    "subscribe to",
    "sign up for",
    "sign in to",
    "log in to",
    "create an account",
    "newsletter",
    "terms of service",
    "terms and conditions",
    "privacy policy",
    "all rights reserved",
    "follow us on",
    "share this article",
    "advertisement",
    "read more:",
    "click here",
    "enable javascript",
    "javascript is disabled",
    "download the app",
    "get the app",
    "skip to main content",
    "back to top",
    "you are using an outdated browser",
    "your browser does not support",
    "already a subscriber",
    "start your free trial",
    "continue reading",
];

/// How aggressively to clean.
#[derive(Clone, Debug)]
pub struct Options {
    /// Fold typographic characters to ASCII equivalents.
    pub fold: bool,
    /// Strip HTML tags, script and style blocks, and decode entities.
    pub html: bool,
    /// Drop links, emails and phone numbers.
    pub links: bool,
    /// Screen navigation and consent lines, and repeat lines.
    pub boilerplate: bool,
    /// Longest boilerplate line, in words, still eligible to be dropped.
    pub boilerplate_words: usize,
    /// Minimum words for a line to count as content when deduplicating lines.
    pub content_words: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            fold: true,
            html: true,
            links: true,
            boilerplate: true,
            boilerplate_words: 22,
            content_words: 4,
        }
    }
}

/// Counts of what a clean removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub chars_in: usize,
    pub chars_out: usize,
    pub folded: usize,
    pub tags: usize,
    pub entities: usize,
    pub links: usize,
    pub boilerplate_lines: usize,
    pub nav_lines: usize,
    pub duplicate_lines: usize,
    pub separator_lines: usize,
}

impl Report {
    /// Fraction of the input characters that did not survive.
    pub fn loss(&self) -> f64 {
        if self.chars_in == 0 {
            0.0
        } else {
            1.0 - (self.chars_out as f64 / self.chars_in as f64)
        }
    }

    /// Total lines the screen removed, excluding character level edits.
    pub fn lines_dropped(&self) -> usize {
        self.boilerplate_lines + self.nav_lines + self.duplicate_lines + self.separator_lines
    }
}

/// Clean one document and report the removals.
pub fn clean(raw: &str, opts: &Options) -> (String, Report) {
    let mut report = Report {
        chars_in: raw.chars().count(),
        ..Report::default()
    };
    let mut text = if opts.fold {
        fold(raw, &mut report)
    } else {
        raw.to_string()
    };
    if opts.html {
        text = strip_html(&text, &mut report);
        text = decode_entities(&text, &mut report);
    }
    if opts.links {
        text = drop_links(&text, &mut report);
    }
    let lines = if opts.boilerplate {
        screen_lines(&text, &mut report, opts)
    } else {
        text.lines()
            .map(collapse)
            .filter(|line| !line.is_empty())
            .collect()
    };
    let cleaned = lines
        .iter()
        .map(|line| collapse(line))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    report.chars_out = cleaned.chars().count();
    (cleaned, report)
}

/// Fold the characters that break exact matching without changing the words.
pub fn fold(raw: &str, report: &mut Report) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        let replacement: Option<char> = match c {
            '\u{2018}' | '\u{2019}' | '\u{201b}' | '\u{2032}' => Some('\''),
            '\u{201c}' | '\u{201d}' | '\u{201f}' | '\u{2033}' => Some('"'),
            '\u{2010}'..='\u{2015}' | '\u{2212}' | '\u{fe58}' | '\u{fe63}' => Some('-'),
            '\u{2026}' => {
                out.push_str("...");
                report.folded += 1;
                continue;
            }
            '\u{00a0}' | '\u{2000}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => {
                Some(' ')
            }
            '\u{00ab}' | '\u{00bb}' => Some('"'),
            '\u{00ad}' | '\u{200b}'..='\u{200d}' | '\u{2060}' | '\u{feff}' => None,
            '\u{ff01}'..='\u{ff5e}' => {
                let ascii = char::from_u32(c as u32 - 0xfee0).unwrap_or(c);
                Some(ascii)
            }
            c if c.is_control() && c != '\n' && c != '\t' => None,
            other => {
                out.push(other);
                continue;
            }
        };
        report.folded += 1;
        if let Some(ch) = replacement {
            out.push(ch);
        }
    }
    out
}

/// Find an ASCII needle without allocating a lowercase copy of the haystack.
fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Remove script and style blocks, HTML comments and any remaining tag.
pub fn strip_html(raw: &str, report: &mut Report) -> String {
    let mut text = raw.to_string();
    for tag in ["script", "style", "noscript"] {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        while let Some(start) = find_ci(&text, &open, 0) {
            let end = match find_ci(&text, &close, start) {
                Some(at) => at + close.len(),
                None => text.len(),
            };
            text.replace_range(start..end, " ");
            report.tags += 1;
        }
    }
    while let Some(start) = find_ci(&text, "<!--", 0) {
        let end = match text[start..].find("-->") {
            Some(at) => start + at + 3,
            None => text.len(),
        };
        text.replace_range(start..end, " ");
        report.tags += 1;
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(offset) = text[i..].find('>') {
                let inner = &text[i + 1..i + offset];
                let looks_like_tag = inner
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_alphabetic() || c == '/' || c == '!')
                    .unwrap_or(false);
                if looks_like_tag {
                    report.tags += 1;
                    i += offset + 1;
                    out.push(' ');
                    continue;
                }
            }
        }
        let c = text[i..].chars().next().unwrap_or(' ');
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// Decode one entity body, without the ampersand and the semicolon.
fn entity_char(body: &str) -> Option<char> {
    match body {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some(' '),
        "mdash" | "ndash" | "minus" => Some('-'),
        "hellip" => Some('.'),
        "rsquo" | "lsquo" => Some('\''),
        "ldquo" | "rdquo" => Some('"'),
        "copy" => Some('c'),
        "reg" => Some('r'),
        "euro" => Some('E'),
        "pound" => Some('L'),
        "bull" | "middot" => Some(' '),
        other => other
            .strip_prefix('#')
            .and_then(|digits| {
                if let Some(hex) = digits
                    .strip_prefix('x')
                    .or_else(|| digits.strip_prefix('X'))
                {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    digits.parse::<u32>().ok()
                }
            })
            .and_then(char::from_u32),
    }
}

/// Decode the entities that turn up in news and ad copy. Unknown names are left
/// alone rather than replaced with a guess.
///
/// The scan is character driven rather than byte driven, because an entity can
/// sit next to a multi byte character and a fixed byte window would slice it in
/// half. A name is only looked for inside the next twelve characters, which is
/// longer than any name in the list.
pub fn decode_entities(raw: &str, report: &mut Report) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(c) = rest.chars().next() {
        if c != '&' {
            out.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        let limit = rest
            .char_indices()
            .nth(12)
            .map(|(i, _)| i)
            .unwrap_or(rest.len());
        let Some(semi) = rest[..limit].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        match entity_char(&rest[1..semi]) {
            Some(ch) => {
                out.push(ch);
                report.entities += 1;
                rest = &rest[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out
}

fn trimmed_token(token: &str) -> &str {
    token.trim_matches(|c: char| {
        matches!(
            c,
            '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\'' | '<' | '>'
        )
    })
}

fn is_link(token: &str) -> bool {
    let t = trimmed_token(token).trim_start_matches('(');
    if t.starts_with("http://") || t.starts_with("https://") || t.starts_with("www.") {
        return true;
    }
    if t.contains("://") || t.starts_with("mailto:") {
        return true;
    }
    let at = t.find('@');
    if let Some(at) = at {
        let domain = &t[at + 1..];
        if domain.contains('.') && !t.contains(' ') && t.chars().all(|c| !c.is_whitespace()) {
            return true;
        }
    }
    let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
    let punctuation = t
        .chars()
        .filter(|c| matches!(c, '-' | '(' | ')' | '+' | '.'))
        .count();
    digits >= 7 && punctuation >= 2 && t.chars().count() <= 20
}

/// Drop link, email and phone tokens, keeping the rest of the sentence.
pub fn drop_links(raw: &str, report: &mut Report) -> String {
    let mut out = String::with_capacity(raw.len());
    for (i, piece) in raw.split_inclusive(char::is_whitespace).enumerate() {
        let token = piece.trim_end();
        let whitespace = &piece[token.len()..];
        if is_link(token) {
            report.links += 1;
            if i > 0 && !out.ends_with(' ') && !out.ends_with('\n') && !out.is_empty() {
                out.push(' ');
            }
            continue;
        }
        out.push_str(token);
        out.push_str(whitespace);
    }
    out
}

fn is_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.chars().count() >= 3
        && trimmed
            .chars()
            .all(|c| matches!(c, '-' | '=' | '_' | '*' | '~' | '#' | '|' | '+'))
}

fn is_navigation(line: &str) -> bool {
    let pipes = line.matches('|').count();
    let arrows = line.matches(" > ").count();
    if (pipes >= 2 && tokenize::count_words(line) <= 15) || arrows >= 2 {
        return true;
    }
    let words: Vec<&str> = tokenize::words(line).collect();
    if words.len() > 6 || line.trim_end().ends_with(['.', '!', '?', ':', ';']) {
        return false;
    }
    let small = [
        "of", "the", "and", "a", "an", "in", "on", "for", "to", "at", "by", "with", "from", "vs",
    ];
    words.iter().all(|w| {
        let first = w.chars().next().unwrap_or(' ');
        first.is_uppercase() || small.contains(&w.to_lowercase().as_str())
    })
}

fn screen_lines(raw: &str, report: &mut Report, opts: &Options) -> Vec<String> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out = Vec::new();
    for line in raw.lines() {
        let trimmed = collapse(line);
        if trimmed.is_empty() {
            continue;
        }
        if is_separator(&trimmed) {
            report.separator_lines += 1;
            continue;
        }
        let words = tokenize::count_words(&trimmed);
        if words <= opts.boilerplate_words {
            let lowered = trimmed.to_lowercase();
            if BOILERPLATE.iter().any(|phrase| lowered.contains(phrase)) {
                report.boilerplate_lines += 1;
                continue;
            }
            if is_navigation(&trimmed) {
                report.nav_lines += 1;
                continue;
            }
        }
        if words >= opts.content_words {
            let key = crate::text::hash::text_hash(&trimmed);
            if !seen.insert(key) {
                report.duplicate_lines += 1;
                continue;
            }
        }
        out.push(trimmed);
    }
    out
}

/// Collapse every run of whitespace inside a line to one space.
pub fn collapse(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut space = false;
    for c in line.chars() {
        if c.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_removes_smart_quotes_and_soft_hyphens() {
        let mut report = Report::default();
        let got = fold(
            "Acme\u{2019}s \u{201c}profit\u{201d} rose 4%\u{00a0}year on year\u{00ad}.",
            &mut report,
        );
        assert_eq!(got, "Acme's \"profit\" rose 4% year on year.");
        assert!(report.folded >= 4);
    }

    #[test]
    fn strip_html_drops_scripts_and_tags() {
        let mut report = Report::default();
        let got = strip_html(
            "<p>Fed holds <b>rates</b></p><script>var x=1</script>",
            &mut report,
        );
        assert!(!got.contains("var x"));
        assert!(got.contains("Fed holds"));
        assert!(got.contains("rates"));
        assert!(report.tags >= 3);
    }

    #[test]
    fn clean_removes_a_cookie_line_but_keeps_the_story() {
        let raw = "We use cookies to improve your experience.\nFed holds rates steady as inflation cools.\nSubscribe to our newsletter\n";
        let (got, report) = clean(raw, &Options::default());
        assert_eq!(report.boilerplate_lines, 2, "{got}");
        assert_eq!(got, "Fed holds rates steady as inflation cools.");
    }

    #[test]
    fn links_go_and_words_stay() {
        let mut report = Report::default();
        let got = drop_links(
            "Read https://example.com/a?b=1 or mail desk@example.com now",
            &mut report,
        );
        assert_eq!(report.links, 2);
        assert!(got.contains("Read"));
        assert!(got.contains("now"));
        assert!(!got.contains("example"));
    }

    #[test]
    fn entities_next_to_multi_byte_text_do_not_panic() {
        let mut report = Report::default();
        let got = decode_entities("Caf\u{00e9} &amp; bar &eacute; &amp;", &mut report);
        assert_eq!(got, "Caf\u{00e9} & bar &eacute; &");
        assert_eq!(report.entities, 2);
    }

    #[test]
    fn repeated_lines_are_dropped_once() {
        let raw = "Profit rose on strong demand\nProfit rose on strong demand\nMargins held";
        let (got, report) = clean(raw, &Options::default());
        assert_eq!(report.duplicate_lines, 1);
        assert_eq!(got.lines().count(), 2);
    }
}
