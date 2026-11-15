//! The `read` command surface: one text in, one report out.
//!
//! This is the front end for a person, not for a pipeline, so it takes the text
//! as positional arguments or as stdin and it needs no flags to be useful. The
//! flags only change how much evidence is shown or which models answer; the
//! report itself is [`super::report::build`].

use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Read};

use clap::Args;

use super::context::Loaded;
use super::render;
use super::report::{build, Options};
use super::{DETECTORS, EMOTION, SEGMENTS};

/// Arguments for `read`.
#[derive(Args)]
pub struct ReadArgs {
    /// Copy to read. Several arguments are joined with a space; omit them to
    /// read standard input.
    #[arg(value_name = "TEXT")]
    pub text: Vec<String>,
    /// Read a file as one text.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    pub file: Option<String>,
    /// One advertiser's copy set, for the learned brand map. JSON Lines with a
    /// `text` field, or one ad per line. With no text argument the report reads
    /// the first ad and the map reads the set.
    #[arg(long, value_name = "PATH", conflicts_with = "file")]
    pub ads: Option<String>,
    /// Detector artifact to read.
    #[arg(long, default_value = DETECTORS)]
    pub detectors: String,
    /// Emotion artifact to read.
    #[arg(long, default_value = EMOTION)]
    pub emotion: String,
    /// Segment artifact to read. A missing file drops that view rather than
    /// refusing the report.
    #[arg(long, default_value = SEGMENTS)]
    pub segments: String,
    /// Trigger words to show. Zero shows every word the reading kept.
    #[arg(long, default_value_t = 6, value_name = "N")]
    pub triggers: usize,
    /// Targeting terms per ad, when the text is one ad in a known buy.
    #[arg(long, value_name = "N")]
    pub terms: Option<f64>,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Run `read`.
pub fn run(args: ReadArgs) -> Result<(), Box<dyn Error>> {
    let set = match &args.ads {
        Some(path) => Some(copy_set(path)?),
        None => None,
    };
    let text = if args.text.is_empty() && args.file.is_none() {
        match set.as_ref() {
            Some(texts) if !texts.is_empty() => texts[0].clone(),
            _ => input(&args)?,
        }
    } else {
        input(&args)?
    };
    if text.trim().is_empty() {
        return Err("the text is empty: give copy to read".into());
    }
    let loaded = Loaded::open(&args.detectors, &args.emotion, &args.segments)?;
    let report = build(
        &loaded,
        &text,
        &Options {
            triggers: args.triggers,
            terms: args.terms,
            set,
        },
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        render::print(&report);
    }
    Ok(())
}

/// Read one advertiser's copy set: JSON Lines with a `text` field when the
/// lines parse, otherwise one ad per line.
fn copy_set(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let body = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let text = match serde_json::from_str::<serde_json::Value>(line) {
            Ok(value) => value
                .get("text")
                .and_then(|text| text.as_str())
                .unwrap_or("")
                .to_string(),
            Err(_) => line.to_string(),
        };
        if !text.trim().is_empty() {
            out.push(text);
        }
    }
    if out.is_empty() {
        return Err(
            format!("{path} holds no copy: one ad per line, or JSON Lines with `text`").into(),
        );
    }
    Ok(out)
}

fn input(args: &ReadArgs) -> Result<String, Box<dyn Error>> {
    if let Some(path) = &args.file {
        return Ok(fs::read_to_string(path)?);
    }
    if !args.text.is_empty() {
        return Ok(args.text.join(" "));
    }
    if io::stdin().is_terminal() {
        return Err("give the text as arguments, with --file, or on stdin".into());
    }
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_join_with_a_single_space() {
        let args = ReadArgs {
            text: vec!["act".into(), "now".into()],
            file: None,
            ads: None,
            detectors: DETECTORS.into(),
            emotion: EMOTION.into(),
            segments: SEGMENTS.into(),
            triggers: 6,
            terms: None,
            json: false,
        };
        assert_eq!(input(&args).unwrap(), "act now");
    }
}
