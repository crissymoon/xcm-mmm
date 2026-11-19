//! The `topics` command surface, beside the module it reads.
//!
//! Every verb here is read-only. The panel is pulled and built on the Python
//! side, where the ad registers answer; this surface opens what it published
//! and prints a ranking. `check` is the one that can fail: it compares each
//! file against the digest in the folder's manifest, so a file edited after
//! publication is named instead of ranked.

use std::error::Error;

use clap::{Args, Subcommand};

use super::{Panel, Topic, FOLDER, MANIFEST};

use crate::num::commas;

#[derive(Args)]
pub struct TopicsArgs {
    #[command(subcommand)]
    pub command: TopicsCommand,
}

#[derive(Subcommand)]
pub enum TopicsCommand {
    /// Print the panel headline, its languages and its reach.
    Info(TopicsDir),
    /// Rank the topics by the reach their ads carried.
    Topics(TopicsTop),
    /// Rank single terms across the panel, or inside one ticker or area.
    Keywords(TopicsWithin),
    /// Rank the ads by the reach the register measured.
    Ads(TopicsTop),
    /// Compare each published file against the manifest digest.
    Check(TopicsDir),
}

#[derive(Args)]
pub struct TopicsDir {
    /// Folder holding hot-topics.json, keywords.json and manifest.json.
    #[arg(long, default_value = FOLDER)]
    pub dir: String,
}

#[derive(Args)]
pub struct TopicsTop {
    /// Folder holding the published files.
    #[arg(long, default_value = FOLDER)]
    pub dir: String,
    /// How many rows to print.
    #[arg(long, default_value_t = 15)]
    pub top: usize,
}

#[derive(Args)]
pub struct TopicsWithin {
    /// Folder holding the published files.
    #[arg(long, default_value = FOLDER)]
    pub dir: String,
    /// Which axis to break the terms down by: ticker or area.
    #[arg(long, default_value = "ticker")]
    pub axis: String,
    /// The ticker or area to read, or empty for the panel.
    #[arg(long, default_value = "")]
    pub name: String,
    /// How many rows to print.
    #[arg(long, default_value_t = 20)]
    pub top: usize,
}

pub fn run(args: TopicsArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        TopicsCommand::Info(a) => info(&a.dir),
        TopicsCommand::Topics(a) => topics(&a),
        TopicsCommand::Keywords(a) => keywords(&a),
        TopicsCommand::Ads(a) => ads(&a),
        TopicsCommand::Check(a) => check(&a.dir),
    }
}

fn info(dir: &str) -> Result<(), Box<dyn Error>> {
    let panel = Panel::open(dir)?;
    let meta = &panel.hot.meta;
    println!(
        "panel     {} creatives  {} with reach  {} people",
        commas(meta.ads),
        commas(meta.with_reach),
        commas(meta.reach_total)
    );
    println!(
        "shape     {} topics  {} terms  {} tickers  {} areas",
        meta.topics,
        commas(meta.terms),
        meta.tickers,
        meta.areas.len()
    );
    if !meta.read_at.is_empty() {
        println!("read at   {}", meta.read_at);
    }
    let languages: Vec<String> = meta
        .languages
        .iter()
        .map(|(code, count)| format!("{code} {}", commas(*count)))
        .collect();
    if !languages.is_empty() {
        println!("languages {}", languages.join("  "));
    }
    let areas: Vec<String> = meta
        .areas
        .iter()
        .map(|(code, count)| format!("{code} {}", commas(*count)))
        .collect();
    if !areas.is_empty() {
        println!("areas     {}", areas.join("  "));
    }
    if !meta.note.is_empty() {
        println!("note      {}", meta.note);
    }
    Ok(())
}

fn topics(args: &TopicsTop) -> Result<(), Box<dyn Error>> {
    let panel = Panel::open(&args.dir)?;
    let ranked = panel.ranked();
    if ranked.is_empty() {
        println!(
            "no topics at {}: the panel found no cluster above its floor",
            args.dir
        );
        return Ok(());
    }
    println!("{:>10}  {:>5}  {:<38}  tickers", "reach", "ads", "topic");
    for topic in ranked.iter().take(args.top) {
        println!(
            "{:>10}  {:>5}  {:<38}  {}",
            commas(topic.reach),
            topic.ads,
            cut(&topic.name, 38),
            topic
                .tickers
                .iter()
                .take(3)
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(top) = ranked.first() {
        print_example(top);
    }
    Ok(())
}

fn print_example(topic: &Topic) {
    if topic.example.copy.is_empty() {
        return;
    }
    println!();
    println!("heaviest ad under {:?}", cut(&topic.name, 38));
    println!(
        "  {} {} ({})",
        topic.example.ticker, topic.example.page, topic.example.started
    );
    for line in wrap(&topic.example.copy, 88).iter().take(4) {
        println!("  {line}");
    }
}

fn keywords(args: &TopicsWithin) -> Result<(), Box<dyn Error>> {
    let panel = Panel::open(&args.dir)?;
    if args.name.is_empty() {
        println!(
            "{:>10}  {:>5}  {:<22}  {:>5}  leaning",
            "reach", "ads", "term", "lift"
        );
        for term in panel.keywords.keywords.iter().take(args.top) {
            println!(
                "{:>10}  {:>5}  {:<22}  {:>5.2}  {}",
                commas(term.reach),
                term.ads,
                cut(&term.term, 22),
                term.lift,
                term.leaning
            );
        }
        return Ok(());
    }
    let rows = panel.within(&args.axis, &args.name).ok_or_else(|| {
        format!(
            "no {} named {}, expected one of the published groups",
            args.axis, args.name
        )
    })?;
    println!("{} {}: the terms it leans on", args.axis, args.name);
    for term in rows.iter().take(args.top) {
        println!(
            "  {:<22} {:>5} ads  x{:.2}",
            cut(&term.term, 22),
            term.ads,
            term.lift
        );
    }
    Ok(())
}

fn ads(args: &TopicsTop) -> Result<(), Box<dyn Error>> {
    let panel = Panel::open(&args.dir)?;
    let ranked = panel.hottest();
    if ranked.is_empty() {
        println!("no ads at {}", args.dir);
        return Ok(());
    }
    println!("{:>10}  {:<6}  {:<24}  copy", "reach", "ticker", "page");
    for ad in ranked.iter().take(args.top) {
        println!(
            "{:>10}  {:<6}  {:<24}  {}",
            commas(ad.reach),
            ad.ticker,
            cut(&ad.page, 24),
            cut(&one_line(&ad.copy), 56)
        );
    }
    Ok(())
}

fn check(dir: &str) -> Result<(), Box<dyn Error>> {
    let Some(manifest) = super::Manifest::open(dir)? else {
        return Err(format!("no {MANIFEST} at {dir}: run the panel build first").into());
    };
    let found = manifest.disagreements(dir);
    if found.is_empty() {
        println!(
            "{} ok  {} files  {}",
            dir,
            manifest.files.len(),
            manifest.name
        );
        return Ok(());
    }
    for line in &found {
        println!("drift     {line}");
    }
    Err(format!(
        "{} of {} files disagree with the manifest",
        found.len(),
        manifest.files.len()
    )
    .into())
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cut(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}.")
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in one_line(text).split(' ') {
        if !line.is_empty() && line.chars().count() + word.chars().count() + 1 > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
