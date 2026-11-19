//! What a message says about itself, before anyone reads the body.
//!
//! A router that only reads prose is blind to the cheapest evidence in the
//! message: the header block says whether a human wrote it, whether it went to a
//! list, and whether another robot already answered it. Those three facts decide
//! more cases than any classifier, and they are exact strings rather than
//! scores, so they are read first and cost nothing.
//!
//! The parser is deliberately small. It unfolds folded headers (RFC 5322 puts a
//! continuation on any line beginning with whitespace), keeps repeated headers
//! such as `Received` in order, and extracts addresses from the forms that occur
//! in practice: a bare address, a display name in angle brackets, and a comment
//! in parentheses. It does not decode RFC 2047 encoded words or walk MIME parts,
//! so a subject written in another charset stays as its transport form and a
//! body sent as `multipart/*` is filed as the raw text. The router treats the
//! body as evidence, not as the record, so a rough reading costs a folder choice
//! and never a lost message.

/// One message as the router sees it.
#[derive(Clone, Debug, Default)]
pub struct Envelope {
    /// Header names in the order they appear, with their unfolded values.
    pub headers: Vec<(String, String)>,
    pub subject: String,
    pub from: String,
    pub from_address: String,
    pub reply_to: Vec<String>,
    pub to: Vec<String>,
    pub message_id: String,
    pub return_path: String,
    pub references: String,
    pub date: String,
    pub body: String,
    /// False when the input had no header block at all, which is the case for
    /// a plain document arriving on the server.
    pub has_headers: bool,
}

impl Envelope {
    /// Read one message. A text without a header block is body only.
    pub fn parse(text: &str) -> Self {
        let normalised = text.replace("\r\n", "\n");
        let (mut headers, body, has_headers) = split_headers(&normalised);
        // `split_headers` cannot tell a document from a message whose first line
        // happens to contain a colon, so the caller's subject fallback matters.
        headers.retain(|(name, _)| plausible_name(name));
        let get = |name: &str| -> Vec<String> {
            headers
                .iter()
                .filter(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.clone())
                .collect()
        };
        let single = |name: &str| get(name).into_iter().next().unwrap_or_default();
        let from = single("From");
        let from_address = addresses(&from).into_iter().next().unwrap_or_default();
        let reply_to = get("Reply-To")
            .iter()
            .flat_map(|value| addresses(value))
            .collect();
        let to = get("To")
            .iter()
            .flat_map(|value| addresses(value))
            .collect();
        Self {
            subject: single("Subject"),
            from,
            from_address,
            reply_to,
            to,
            message_id: single("Message-Id").trim().to_string(),
            return_path: single("Return-Path"),
            references: single("References"),
            date: single("Date"),
            headers,
            body,
            has_headers,
        }
    }

    /// Every value of one header, in order.
    pub fn all(&self, name: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
            .collect()
    }

    /// The class-free signals a guard may veto on.
    pub fn bulk_marks(&self, names: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for name in names {
            for value in self.all(name) {
                out.push(format!("{name}: {}", clip(value, 60)));
            }
        }
        out
    }

    /// `Precedence: bulk` and its relatives, lowercased.
    pub fn precedence(&self) -> Option<String> {
        self.all("Precedence")
            .first()
            .map(|value| value.trim().to_ascii_lowercase())
    }

    /// The `Auto-Submitted` value, lowercased. RFC 3834 makes `no` the only
    /// value that means a person wrote this.
    pub fn auto_submitted(&self) -> Option<String> {
        self.all("Auto-Submitted")
            .first()
            .map(|value| value.trim().to_ascii_lowercase())
    }

    /// The local part of the sender, lowercased, for the no-reply list.
    pub fn sender_local_part(&self) -> String {
        local_part(&self.from_address)
    }

    /// A null reverse-path, written `<>`, marks a message that must never be
    /// answered because it has no valid sender to answer (RFC 3834, section 2).
    pub fn null_return_path(&self) -> bool {
        self.return_path.trim() == "<>"
    }

    /// Where a reply would go. After delivery the envelope sender is the
    /// address a responder is told to use (RFC 3834, section 4), so
    /// `Return-Path` wins when it is present and not null. A file saved without
    /// its envelope falls back to `Reply-To` and then to `From`, which is a
    /// deviation from that rule and is reported as one.
    pub fn reply_target(&self) -> Option<String> {
        if self.null_return_path() {
            return None;
        }
        let envelope = addresses(&self.return_path);
        if let Some(address) = envelope.first() {
            return Some(address.clone());
        }
        self.reply_to
            .iter()
            .find(|address| !local_part(address).starts_with("no-reply"))
            .cloned()
            .or_else(|| Some(self.from_address.clone()).filter(|address| !address.is_empty()))
    }

    /// A stable identity used for duplicate suppression. A message without a
    /// `Message-Id` falls back to the sender and subject, which is weaker but
    /// still catches a file delivered twice under two names.
    pub fn identity(&self) -> String {
        if !self.message_id.is_empty() {
            return self
                .message_id
                .trim_matches(|c| c == '<' || c == '>')
                .to_string();
        }
        format!("{}|{}", self.from_address, self.subject)
    }

    /// The text a classifier should read: subject and body, subject first
    /// because a short subject carries more per word than a long body.
    pub fn reading_text(&self) -> String {
        if self.subject.is_empty() {
            self.body.clone()
        } else {
            format!("{}\n{}", self.subject, self.body)
        }
    }
}

fn split_headers(text: &str) -> (Vec<(String, String)>, String, bool) {
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut body_start = None;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            body_start = Some(index + 1);
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            if let Some((_, value)) = headers.last_mut() {
                value.push(' ');
                value.push_str(line.trim());
            }
            continue;
        }
        match line.split_once(':') {
            Some((name, value)) if plausible_name(name) => {
                headers.push((name.trim().to_string(), value.trim().to_string()));
            }
            _ => {
                body_start = Some(index);
                break;
            }
        }
    }
    let body_lines: Vec<&str> = match body_start {
        Some(at) => text.lines().skip(at).collect(),
        None => Vec::new(),
    };
    let has_headers = body_start.is_some() && !headers.is_empty();
    if !has_headers {
        return (Vec::new(), text.to_string(), false);
    }
    (headers, body_lines.join("\n"), true)
}

/// A header name is printable ASCII with no spaces. This is what stops a line
/// like `12:30 standup` at the top of a note from becoming a header.
fn plausible_name(name: &str) -> bool {
    let name = name.trim();
    let mut chars = name.chars();
    let starts_with_letter = chars.next().is_some_and(|c| c.is_ascii_alphabetic());
    starts_with_letter
        && name.len() <= 78
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Pull the addresses out of a header value.
pub fn addresses(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut quoted = false;
    for ch in value.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                continue;
            }
            '(' if !quoted => {
                depth += 1;
                continue;
            }
            ')' if !quoted => {
                depth -= 1;
                continue;
            }
            ',' | ';' if depth == 0 && !quoted => {
                push_address(&mut out, &current);
                current.clear();
                continue;
            }
            _ => {}
        }
        if depth == 0 {
            current.push(ch);
        }
    }
    push_address(&mut out, &current);
    out
}

fn push_address(out: &mut Vec<String>, raw: &str) {
    let candidate = match (raw.find('<'), raw.rfind('>')) {
        (Some(start), Some(end)) if end > start => &raw[start + 1..end],
        _ => raw,
    };
    let trimmed = candidate.trim().trim_matches('"');
    let cleaned = trimmed
        .strip_prefix("mailto:")
        .unwrap_or(trimmed)
        .trim()
        .to_ascii_lowercase();
    if cleaned.contains('@') && !cleaned.contains(' ') {
        out.push(cleaned);
    }
}

/// The part before the `@`, lowercased.
pub fn local_part(address: &str) -> String {
    address
        .split('@')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

/// The part after the `@`, lowercased.
pub fn domain(address: &str) -> String {
    address
        .split_once('@')
        .map(|(_, domain)| domain.trim().to_ascii_lowercase())
        .unwrap_or_default()
}

pub(crate) fn clip(text: &str, width: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= width {
        return text.to_string();
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MESSAGE: &str = "From: Ada Lovelace <Ada@Example.COM>\r\nTo: support@example.org\r\nSubject: cannot log in\r\n  after the reset\r\nMessage-Id: <abc@example.com>\r\nList-Unsubscribe: <mailto:leave@example.com>\r\n\r\nThe form says the password reset\r\nfailed.\r\n";

    #[test]
    fn folded_headers_are_joined_and_addresses_are_lowered() {
        let envelope = Envelope::parse(MESSAGE);
        assert_eq!(envelope.subject, "cannot log in after the reset");
        assert_eq!(envelope.from_address, "ada@example.com");
        assert_eq!(envelope.to, vec!["support@example.org".to_string()]);
        assert_eq!(envelope.message_id, "<abc@example.com>");
        assert!(envelope.body.contains("password reset"));
    }

    #[test]
    fn repeated_headers_are_kept_in_order() {
        let text = "Received: from a\nReceived: from b\nFrom: x@y.z\n\nbody";
        let envelope = Envelope::parse(text);
        assert_eq!(envelope.all("Received"), vec!["from a", "from b"]);
        assert_eq!(envelope.identity(), "x@y.z|");
    }

    #[test]
    fn a_document_without_headers_is_body_only() {
        let envelope = Envelope::parse("Invoice 4471\n\nAmount due: 900.00\n");
        assert!(!envelope.has_headers);
        assert!(envelope.headers.is_empty());
        assert!(envelope.body.starts_with("Invoice 4471"));
    }

    #[test]
    fn a_clock_line_is_not_a_header() {
        let envelope = Envelope::parse("12:30 standup\n\nnotes");
        assert!(!envelope.has_headers);
    }

    #[test]
    fn display_names_comments_and_lists_parse() {
        let found =
            addresses("\"Lovelace, Ada\" <ada@example.com>, bob@example.com (Bob), carol@x");
        assert_eq!(
            found,
            vec![
                "ada@example.com".to_string(),
                "bob@example.com".to_string(),
                "carol@x".to_string()
            ]
        );
    }

    #[test]
    fn auto_submitted_and_precedence_are_lowered() {
        let envelope = Envelope::parse("Auto-Submitted: Auto-Replied\nPrecedence: Bulk\n\nhi");
        assert_eq!(envelope.auto_submitted().as_deref(), Some("auto-replied"));
        assert_eq!(envelope.precedence().as_deref(), Some("bulk"));
    }

    #[test]
    fn reply_target_skips_a_no_reply_mailbox() {
        let envelope =
            Envelope::parse("From: no-reply@example.com\nReply-To: team@example.com\n\nbody");
        assert_eq!(envelope.reply_target().as_deref(), Some("team@example.com"));
        assert_eq!(envelope.sender_local_part(), "no-reply");
    }

    #[test]
    fn bulk_marks_name_the_header_that_fired() {
        let envelope = Envelope::parse(MESSAGE);
        let marks = envelope.bulk_marks(&["List-Id".to_string(), "List-Unsubscribe".to_string()]);
        assert_eq!(marks.len(), 1);
        assert!(marks[0].starts_with("List-Unsubscribe"));
    }
}
