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
    let text = input(&args)?;
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
        },
    );
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        render::print(&report);
    }
    Ok(())
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
