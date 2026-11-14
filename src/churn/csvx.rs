//! A CSV reader for the four shapes this repository actually meets.
//!
//! The churn tables are comma separated, quoted where a memo contains a comma,
//! with `\r\n` line endings and a byte order mark on the first header. That is
//! the whole grammar this reader implements: quoted fields, doubled quotes,
//! embedded newlines, and a header row. It is not a general CSV library and does
//! not pretend to be one; it refuses a row whose field count disagrees with the
//! header rather than shifting columns silently.
//!
//! Everything comes back as strings. A caller that wants a number decides what
//! an empty cell means, because in these tables empty means different things:
//! no churn reason, no dependents, not measured.

use std::fs;
use std::io;

/// A parsed table: one header row and the rows under it.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    /// The column index of a header, matched case and space insensitively.
    pub fn column(&self, name: &str) -> Option<usize> {
        let wanted = fold(name);
        self.header.iter().position(|cell| fold(cell) == wanted)
    }

    /// The column index, refusing a table that lacks it.
    pub fn require(&self, name: &str) -> io::Result<usize> {
        self.column(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the table has no column named {name:?}"),
            )
        })
    }
}

/// Read a file into a table.
pub fn read(path: &str) -> io::Result<Table> {
    let text =
        fs::read_to_string(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?;
    parse(&text).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))
}

/// Parse CSV text. The first non empty row is the header.
pub fn parse(text: &str) -> Result<Table, String> {
    let mut records = records(text);
    if records.is_empty() {
        return Err("the table is empty".to_string());
    }
    let header: Vec<String> = records.remove(0).into_iter().map(clean).collect();
    let mut rows = Vec::with_capacity(records.len());
    for (number, record) in records.into_iter().enumerate() {
        if record.len() == 1 && record[0].trim().is_empty() {
            continue;
        }
        if record.len() != header.len() {
            return Err(format!(
                "row {} holds {} fields where the header holds {}",
                number + 2,
                record.len(),
                header.len()
            ));
        }
        rows.push(record);
    }
    Ok(Table { header, rows })
}

/// Trim, drop a byte order mark and collapse internal whitespace.
fn clean(cell: String) -> String {
    cell.trim_start_matches('\u{feff}')
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

fn fold(cell: &str) -> String {
    clean(cell.to_string())
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn records(text: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                out.push(std::mem::take(&mut record));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        out.push(record);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_comma_stays_in_its_field() {
        let table = parse("id,reason\n1,\"Too, expensive\"\n2,plain\n").unwrap();
        assert_eq!(table.rows[0][1], "Too, expensive");
        assert_eq!(table.rows[1][1], "plain");
    }

    #[test]
    fn a_doubled_quote_escapes() {
        let table = parse("id,reason\n1,\"He said \"\"no\"\"\"\n").unwrap();
        assert_eq!(table.rows[0][1], "He said \"no\"");
    }

    #[test]
    fn a_header_is_matched_ignoring_case_and_punctuation() {
        let table = parse("Customer Status,Tenure in Months\nChurned,9\n").unwrap();
        assert_eq!(table.column("customer status"), Some(0));
        assert_eq!(table.column("TENURE IN MONTHS"), Some(1));
        assert_eq!(table.column("Monthly Charge"), None);
        assert!(table.require("age").is_err());
    }

    #[test]
    fn a_byte_order_mark_is_not_part_of_the_first_name() {
        let table = parse("\u{feff}id,label\n1,x\n").unwrap();
        assert_eq!(table.header[0], "id");
    }

    #[test]
    fn a_short_row_is_a_refusal() {
        let error = parse("id,label\n1\n").unwrap_err();
        assert!(error.contains("1 fields"), "{error}");
    }

    #[test]
    fn a_trailing_newline_adds_no_empty_row() {
        let table = parse("id,label\n1,x\n\n").unwrap();
        assert_eq!(table.rows.len(), 1);
    }
}
