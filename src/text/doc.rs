//! The record shape and the readers and writers.
//!
//! Gzip is handled by the system `gzip` binary through a pipe, so the crate
//! keeps four dependencies and still reads the `.jsonl.gz` files the market
//! studio writes. If `gzip` is missing the reader returns an error rather than
//! pretending the file was empty.
//!
//! The record is intentionally loose: a corpus row may name its text `text`,
//! `text_a` or `body`, and its source `source`, `publisher` or `a_ticker`. The
//! mapping is explicit in [`TextRow::from_json`] so a new corpus needs one line
//! here and nothing else.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::Value;

use crate::text::hash;

/// One document, reduced to what the cleaning and selection stages need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextRow {
    pub id: String,
    pub text: String,
    /// Where it came from, kept for the report.
    pub source: String,
    /// The unit a cap applies to: a publisher, a ticker, a site.
    pub group: String,
    pub split: String,
    pub day: String,
}

fn string_at(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = value.get(*key).and_then(Value::as_str) {
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

impl TextRow {
    /// Read a record from a JSON value, or return `None` when it has no text.
    pub fn from_json(value: &Value) -> Option<Self> {
        let text = string_at(value, &["text", "text_a", "body", "content"])?;
        let group = string_at(value, &["group", "publisher", "source", "a_ticker"])
            .or_else(|| {
                value
                    .get("tickers")
                    .and_then(Value::as_array)
                    .and_then(|list| list.first())
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string());
        let source = string_at(value, &["source", "publisher", "a_ticker"]).unwrap_or_default();
        let id = string_at(value, &["id", "doc", "a", "a_id"])
            .unwrap_or_else(|| hash::hex64(hash::text_hash(&text)));
        Some(Self {
            id,
            text,
            source,
            group,
            split: string_at(value, &["split"]).unwrap_or_else(|| "train".to_string()),
            day: string_at(value, &["day", "ts"]).unwrap_or_default(),
        })
    }
}

/// A line reader over plain or gzip files.
pub struct LineReader {
    inner: Box<dyn BufRead>,
    child: Option<Child>,
}

impl io::Read for LineReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl BufRead for LineReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.inner.consume(amount)
    }
}

impl Drop for LineReader {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
    }
}

/// Open a path for line reading, decompressing when it ends in `.gz`.
pub fn reader(path: &str) -> io::Result<LineReader> {
    if path.ends_with(".gz") {
        let mut child = Command::new("gzip")
            .arg("-dc")
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("gzip produced no output"))?;
        Ok(LineReader {
            inner: Box::new(BufReader::with_capacity(1 << 20, stdout)),
            child: Some(child),
        })
    } else {
        Ok(LineReader {
            inner: Box::new(BufReader::with_capacity(1 << 20, File::open(path)?)),
            child: None,
        })
    }
}

/// Call `f` for every non-empty line, with a buffer reused across lines.
pub fn for_each_line<F>(path: &str, mut f: F) -> io::Result<usize>
where
    F: FnMut(&str),
{
    let mut reader = reader(path)?;
    let mut buffer = String::with_capacity(1 << 12);
    let mut n = 0;
    loop {
        buffer.clear();
        let read = reader.read_line(&mut buffer)?;
        if read == 0 {
            break;
        }
        let line = buffer.trim_end();
        if line.is_empty() {
            continue;
        }
        f(line);
        n += 1;
    }
    Ok(n)
}

/// Read every row, skipping lines that are not JSON or carry no text.
pub fn read_rows(path: &str) -> io::Result<(Vec<TextRow>, usize)> {
    let mut rows = Vec::new();
    let mut bad = 0usize;
    for_each_line(path, |line| match serde_json::from_str::<Value>(line) {
        Ok(value) => match TextRow::from_json(&value) {
            Some(row) => rows.push(row),
            None => bad += 1,
        },
        Err(_) => bad += 1,
    })?;
    Ok((rows, bad))
}

/// A line writer that compresses on `finish` when the path ends in `.gz`.
pub struct Sink {
    writer: BufWriter<File>,
    target: PathBuf,
    tmp: PathBuf,
    gzip: bool,
}

impl Sink {
    /// Create the file, using a temporary path when compression is requested.
    pub fn create(path: &str) -> io::Result<Self> {
        let target = PathBuf::from(path);
        let gzip = path.ends_with(".gz");
        let tmp = if gzip {
            let mut name = target.clone().into_os_string();
            name.push(".tmp");
            PathBuf::from(name)
        } else {
            target.clone()
        };
        let file = File::create(&tmp)?;
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, file),
            target,
            tmp,
            gzip,
        })
    }

    /// Write one line.
    pub fn line(&mut self, value: &str) -> io::Result<()> {
        self.writer.write_all(value.as_bytes())?;
        self.writer.write_all(b"\n")
    }

    /// Flush, and compress when the path asked for it.
    pub fn finish(mut self) -> io::Result<()> {
        self.writer.flush()?;
        drop(self.writer);
        if self.gzip {
            let status = Command::new("gzip")
                .arg("-c")
                .arg(&self.tmp)
                .stdout(File::create(&self.target)?)
                .status()?;
            let _ = std::fs::remove_file(&self.tmp);
            if !status.success() {
                return Err(io::Error::other(format!("gzip exited with {status}")));
            }
        }
        Ok(())
    }
}

/// Percentile of a slice copied out of order, for reporting.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_row_reads_the_three_shapes() {
        let a: Value =
            serde_json::from_str(r#"{"doc":"d1","text":"hello","publisher":"wire"}"#).unwrap();
        let row = TextRow::from_json(&a).expect("row");
        assert_eq!(row.id, "d1");
        assert_eq!(row.group, "wire");
        let b: Value = serde_json::from_str(r#"{"a":"x1","text_a":"pair text"}"#).unwrap();
        assert_eq!(TextRow::from_json(&b).unwrap().text, "pair text");
        let empty: Value = serde_json::from_str(r#"{"y":1}"#).unwrap();
        assert!(TextRow::from_json(&empty).is_none());
    }

    #[test]
    fn sink_round_trips_plain_lines() {
        let path = std::env::temp_dir().join("xcm_mmm_sink_test.jsonl");
        let path_s = path.to_string_lossy().to_string();
        {
            let mut sink = Sink::create(&path_s).expect("sink");
            sink.line(r#"{"text":"one"}"#).unwrap();
            sink.line(r#"{"text":"two"}"#).unwrap();
            sink.finish().unwrap();
        }
        let (rows, bad) = read_rows(&path_s).expect("read");
        assert_eq!(rows.len(), 2);
        assert_eq!(bad, 0);
        let _ = std::fs::remove_file(&path);
    }
}
