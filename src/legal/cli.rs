//! The `legal` command surface: list, scan, translate, check.
//!
//! Text comes from the arguments or from standard input, and a document comes
//! from `--file`, because the two jobs are different: a sentence is typed, a
//! notice is opened. Every command prints a report for a person unless `--json`
//! is given, in which case the same structure is written out.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Read};

use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

use crate::text::hash::{fnv1a, hex64};

use super::bench;
use super::brief::{Brief, Kind};
use super::registers;
use super::term::Settings;
use super::translate::Legal;

/// The folder the instrument ships in.
pub const DIR: &str = "data/legal";

/// Arguments for `legal`.
#[derive(Args)]
pub struct LegalArgs {
    #[command(subcommand)]
    pub command: LegalCommand,
    /// Folder holding the glossary, the rules and the settings.
    #[arg(long, default_value = DIR, global = true, value_name = "PATH")]
    pub dir: String,
}

#[derive(Subcommand)]
pub enum LegalCommand {
    /// List the glossary, by kind or in full.
    Terms(TermsArgs),
    /// Read a text and name every legal term in it.
    Scan(ScanArgs),
    /// Rewrite a text in plain words and gloss the terms of art.
    Translate(TranslateArgs),
    /// Explain a text: what it requires, forbids and allows, and what its terms
    /// of art mean.
    Explain(ExplainArgs),
    /// Recompute the folder's digests and coverage.
    Check(CheckArgs),
    /// Score the rewrite on the shipped legal and plain pairs.
    Bench(BenchArgs),
    /// Split the bank by register and report where the glossary is blind.
    Coverage(CoverageArgs),
}

#[derive(Args)]
pub struct BenchArgs {
    /// The pairs to score. Defaults to `pairs.jsonl` in the instrument folder.
    #[arg(long, value_name = "PATH")]
    pairs: Option<String>,
    /// Seed for the bootstrap, so two runs report the same interval.
    #[arg(long, default_value_t = 7, value_name = "N")]
    seed: u64,
    /// Bootstrap resamples for the mean delta. Zero reports point estimates.
    #[arg(long, default_value_t = 2000, value_name = "N")]
    resamples: usize,
    /// Write the report here as JSON, for a chart or a diff between runs.
    #[arg(long, value_name = "PATH")]
    out: Option<String>,
    /// Exit non-zero when the share of pairs that improve is below this.
    #[arg(long, value_name = "RATE")]
    fail_under: Option<f64>,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub struct CoverageArgs {
    /// The sentence bank to split. Defaults to `bank.jsonl` in the folder.
    #[arg(long, value_name = "PATH")]
    bank: Option<String>,
    /// Print the audit as JSON.
    #[arg(long)]
    json: bool,
}

/// Where the text comes from.
#[derive(Args)]
pub struct Source {
    /// Text to read. Several arguments are joined with a space.
    #[arg(value_name = "TEXT")]
    text: Vec<String>,
    /// Read a file as one text.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    file: Option<String>,
}

impl Source {
    fn read(&self) -> Result<String, Box<dyn Error>> {
        if let Some(path) = &self.file {
            return Ok(fs::read_to_string(path)?);
        }
        if !self.text.is_empty() {
            return Ok(self.text.join(" "));
        }
        let terminal = io::stdin().is_terminal();
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        if buffer.trim().is_empty() && terminal {
            return Err("no text given, and standard input is a terminal".into());
        }
        Ok(buffer)
    }
}

#[derive(Args)]
pub struct TermsArgs {
    /// Only terms of this kind, such as deictic, latin or doublet.
    #[arg(long, value_name = "KIND")]
    kind: Option<String>,
    /// Only terms that a rewrite can replace.
    #[arg(long)]
    avoidable: bool,
    /// Only avoidable terms that no applying rule reaches.
    #[arg(long)]
    gaps: bool,
    /// Print the rows as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub struct ScanArgs {
    #[command(flatten)]
    source: Source,
    /// How many terms to show. Zero shows all of them.
    #[arg(long, default_value_t = 20, value_name = "N")]
    top: usize,
    /// Print the reading as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub struct TranslateArgs {
    #[command(flatten)]
    source: Source,
    /// Gloss each term of art on first use.
    #[arg(long)]
    explain: bool,
    /// Also print the advisory rules that were reported but not applied.
    #[arg(long)]
    advisory: bool,
    /// Write the plain text here instead of printing it alone.
    #[arg(long, value_name = "PATH")]
    out: Option<String>,
    /// Print the reading as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub struct ExplainArgs {
    #[command(flatten)]
    source: Source,
    /// How many definitions to show. Zero shows all of them.
    #[arg(long, default_value_t = 0, value_name = "N")]
    top: usize,
    /// Write the brief here as well as printing it.
    #[arg(long, value_name = "PATH")]
    out: Option<String>,
    /// Print the brief as JSON.
    #[arg(long)]
    json: bool,
    /// Exit non-zero when the text defines a term the glossary has no row for.
    #[arg(long)]
    check: bool,
}

#[derive(Args)]
pub struct CheckArgs {
    /// Print every file and count, not only the faults.
    #[arg(long)]
    verbose: bool,
    /// Recompute every row of the manifest, then check it.
    #[arg(long)]
    fix: bool,
}

/// One file as the manifest declares it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileRow {
    pub name: String,
    pub rows: usize,
    pub bytes: usize,
    pub fnv64: String,
}

/// The manifest beside the instrument.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: u32,
    pub built_at: String,
    pub files: Vec<FileRow>,
    #[serde(default)]
    pub counts: BTreeMap<String, u64>,
}

/// Run `legal`.
pub fn run(args: LegalArgs) -> Result<(), Box<dyn Error>> {
    let dir = args.dir.clone();
    match args.command {
        LegalCommand::Terms(list) => terms(&dir, &list),
        LegalCommand::Scan(scan) => read_scan(&dir, &scan),
        LegalCommand::Translate(job) => translate(&dir, &job),
        LegalCommand::Explain(job) => explain(&dir, &job),
        LegalCommand::Check(check) => verify(&dir, &check),
        LegalCommand::Bench(bench) => score_pairs(&dir, &bench),
        LegalCommand::Coverage(coverage) => registers(&dir, &coverage),
    }
}

fn score_pairs(dir: &str, args: &BenchArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let path = args
        .pairs
        .clone()
        .unwrap_or_else(|| format!("{dir}/pairs.jsonl"));
    let pairs = bench::read_pairs(&path)?;
    let report = bench::score(&legal, &pairs, args.seed, args.resamples);

    if let Some(out) = &args.out {
        let text = serde_json::to_string_pretty(&report)?;
        fs::write(out, format!("{text}\n"))?;
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", report.render());
    }
    if let Some(floor) = args.fail_under {
        if report.all.rate < floor {
            return Err(format!(
                "{} of {} pairs improved, below the {floor} floor",
                report.all.won, report.all.cases
            )
            .into());
        }
    }
    Ok(())
}

fn registers(dir: &str, args: &CoverageArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let path = args
        .bank
        .clone()
        .unwrap_or_else(|| format!("{dir}/bank.jsonl"));
    let coverage = registers::Coverage::audit(&legal, &path)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&coverage)?);
    } else {
        println!("{}", coverage.render());
    }
    Ok(())
}

fn terms(dir: &str, args: &TermsArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let rewritten: BTreeSet<&str> = legal
        .rules
        .applying()
        .map(|rule| rule.id.as_str())
        .collect();
    let (_, gaps) = legal.coverage();
    let gaps: BTreeSet<String> = gaps.into_iter().collect();

    let rows: Vec<&super::term::Term> = legal
        .terms()
        .iter()
        .filter(|term| args.kind.as_ref().map(|k| &term.kind == k).unwrap_or(true))
        .filter(|term| !args.avoidable || term.avoidable)
        .filter(|term| !args.gaps || gaps.contains(&term.id))
        .collect();

    if args.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    println!(
        "{} of {} terms{}",
        rows.len(),
        legal.terms().len(),
        args.kind
            .as_ref()
            .map(|k| format!(" of kind {k}"))
            .unwrap_or_default()
    );
    for term in rows {
        let mark = if term.avoidable { " " } else { "*" };
        let rule = if rewritten.contains(term.id.as_str()) {
            ""
        } else if term.avoidable {
            "  [no rule]"
        } else {
            ""
        };
        println!(
            "{mark} {:<22} {:<11} {:<26} {}{rule}",
            term.id, term.kind, term.term, term.plain
        );
    }
    println!("\n* a term of art: counted apart from the density, glossed not replaced");
    Ok(())
}

fn read_scan(dir: &str, args: &ScanArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let text = args.source.read()?;
    let scan = legal.scan(&text);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&scan)?);
        return Ok(());
    }

    let stats = &scan.stats;
    println!(
        "{} words, {} terms, {:.2} avoidable per 100 words, band {}",
        stats.words, stats.terms, stats.avoidable_per_100, stats.band
    );
    println!(
        "{} avoidable, {} necessary, {} inside a citation",
        stats.avoidable, stats.necessary, stats.in_citation
    );
    if !stats.kinds.is_empty() {
        let kinds: Vec<String> = stats
            .kinds
            .iter()
            .map(|(kind, count)| format!("{kind} {count}"))
            .collect();
        println!("by kind: {}", kinds.join(", "));
    }
    if !stats.band_note.is_empty() {
        println!("{}", stats.band_note);
    }

    let show = if args.top == 0 {
        scan.terms.len()
    } else {
        args.top.min(scan.terms.len())
    };
    if show > 0 {
        println!();
    }
    for term in scan.terms.iter().take(show) {
        let mark = if term.in_citation {
            "cite"
        } else if term.avoidable {
            "edit"
        } else {
            "keep"
        };
        println!(
            "  {:<5} {:<20} {:<11} {:<28} {}..{}",
            mark, term.id, term.kind, term.plain, term.span.start, term.span.end
        );
    }
    if show < scan.terms.len() {
        println!("  {} more, raise --top", scan.terms.len() - show);
    }
    Ok(())
}

fn translate(dir: &str, args: &TranslateArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let text = args.source.read()?;
    let reading = legal.translate(&text, args.explain);

    if let Some(path) = &args.out {
        fs::write(path, &reading.text)?;
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&reading)?);
        return Ok(());
    }

    println!("{}", reading.text);
    println!();
    println!(
        "{} words to {}, {} edits, {} glosses, {:.2} to {:.2} avoidable per 100",
        reading.words_in,
        reading.words_out,
        reading.changes.len(),
        reading.glosses.len(),
        reading.stats.avoidable_per_100,
        reading.after.avoidable_per_100
    );
    for change in &reading.changes {
        println!(
            "  {:<22} {:?} -> {:?}  ({})",
            change.rule, change.from, change.to, change.why
        );
    }
    for gloss in &reading.glosses {
        println!("  gloss {:<18} {}", gloss.term, gloss.plain);
    }
    if args.advisory && !reading.suggestions.is_empty() {
        println!("\nadvisory:");
        for advice in &reading.suggestions {
            println!(
                "  {:<22} {:?} -> {:?}  ({})",
                advice.rule, advice.from, advice.to, advice.why
            );
        }
    }
    if !reading.remaining.is_empty() {
        println!("\nleft in place: {}", reading.remaining.join(", "));
    }
    Ok(())
}

fn explain(dir: &str, args: &ExplainArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let text = args.source.read()?;
    let brief = Brief::build(&legal, &text)?;
    let rendered = if args.json {
        serde_json::to_string_pretty(&brief)?
    } else {
        render(&brief, args.top)
    };
    if let Some(path) = &args.out {
        fs::write(path, format!("{rendered}\n"))?;
    }
    println!("{rendered}");
    if args.check {
        return coverage_check(&brief);
    }
    Ok(())
}

/// Fail the run when the glossary cannot answer for a term the text defines.
///
/// This is the check a research loop wants: an exit status, not a sentence to
/// read. An empty list is the pass, and the pass is still said out loud so a
/// pipeline can tell "nothing missing" from "the check never ran".
fn coverage_check(brief: &Brief) -> Result<(), Box<dyn Error>> {
    if brief.gaps.is_empty() {
        eprintln!("check: every defined term in the text has a glossary row");
        return Ok(());
    }
    for gap in &brief.gaps {
        eprintln!(
            "gap: {} at byte {} has no glossary row",
            gap.term, gap.offset
        );
    }
    Err(format!(
        "{} defined term{} the glossary cannot gloss",
        brief.gaps.len(),
        if brief.gaps.len() == 1 { "" } else { "s" }
    )
    .into())
}

/// A brief for a person: the summary, then one section per kind of clause, then
/// the definitions the text depends on.
fn render(brief: &Brief, top: usize) -> String {
    let stats = &brief.stats;
    let mut out = String::new();
    out.push_str(&format!(
        "BRIEF  {} words, {} {}, {:.2} avoidable per 100, band {}\n\n",
        brief.words,
        brief.counts.definitions,
        if brief.counts.definitions == 1 {
            "glossary term"
        } else {
            "glossary terms"
        },
        stats.avoidable_per_100,
        stats.band
    ));
    out.push_str(&format!("{}\n", brief.summary));
    for warning in &brief.warnings {
        out.push_str(&format!("note: {warning}\n"));
    }

    for (kind, title) in [
        (Kind::Duty, "WHAT IT REQUIRES"),
        (Kind::Prohibition, "WHAT IT FORBIDS"),
        (Kind::Permission, "WHAT IT ALLOWS"),
        (Kind::Condition, "CONDITIONS"),
        (Kind::Statement, "ALSO STATES"),
    ] {
        let mut shown = 0;
        let mut section = String::new();
        for clause in brief.of(kind) {
            if clause.heading {
                continue;
            }
            shown += 1;
            section.push_str(&format!("  {shown}. {}\n", one_line(&clause.text)));
            let mut detail: Vec<String> = Vec::new();
            if !clause.actor.is_empty() {
                detail.push(format!("who {}", clause.actor));
            }
            if let Some(condition) = &clause.condition {
                if clause.kind != Kind::Condition {
                    detail.push(format!("condition: {condition}"));
                }
            }
            for deadline in &clause.deadlines {
                detail.push(format!("when {}", deadline));
            }
            if !clause.terms.is_empty() {
                detail.push(format!("see {}", clause.terms.join(", ")));
            }
            if !detail.is_empty() {
                section.push_str(&format!("     {}\n", detail.join(" | ")));
            }
        }
        if !section.is_empty() {
            out.push_str(&format!("\n{title}\n{section}"));
        }
    }

    let definitions: Vec<&super::brief::Definition> = if top == 0 {
        brief.definitions.iter().collect()
    } else {
        brief.definitions.iter().take(top).collect()
    };
    if definitions.is_empty() {
        out.push_str("\nDEFINITIONS\n  none: no glossary term appears outside a citation.\n");
    } else {
        out.push_str("\nDEFINITIONS\n");
        for (n, definition) in definitions.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} ({}): {}\n",
                n + 1,
                definition.term,
                definition.kind,
                definition.plain
            ));
            if !definition.note.is_empty() {
                out.push_str(&format!("     why it matters: {}\n", definition.note));
            }
            if !definition.law.is_empty() {
                out.push_str(&format!("     authority: {}\n", definition.law));
            }
            if !definition.sources.is_empty() {
                let shown = definition
                    .sources
                    .iter()
                    .take(2)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                let more = definition.sources.len().saturating_sub(2);
                out.push_str(&format!(
                    "     sources: {shown}{}\n",
                    if more > 0 {
                        format!(" (+{more} more)")
                    } else {
                        String::new()
                    }
                ));
            }
            if definition.uses > 1 {
                out.push_str(&format!("     used {} times\n", definition.uses));
            }
        }
        if top > 0 && top < brief.definitions.len() {
            out.push_str(&format!(
                "  {} more, raise --top\n",
                brief.definitions.len() - top
            ));
        }
    }

    if !brief.gaps.is_empty() {
        out.push_str("\nUNGLOSSED TERMS\n");
        for (n, gap) in brief.gaps.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} (byte {}): defined here, no glossary row, so no plain reading\n",
                n + 1,
                gap.term,
                gap.offset
            ));
        }
    }

    out.push_str(&format!("\n{}\n", brief.disclaimer));
    out
}

fn one_line(text: &str) -> String {
    let mut line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line
        .chars()
        .last()
        .is_some_and(|c| c.is_alphanumeric() || c == ')' || c == '\'' || c == '"')
    {
        line.push('.');
    }
    line
}

/// Run the reader over the bank and report what it does to both registers.
///
/// A check that only compares digests says the folder did not move, not that
/// the instrument works. This measures the two things a reader would ask: how
/// much drafting a rewrite removes from the legal side, and how often it edits
/// prose that was already plain. The second number is the one to watch: an edit
/// on a self-help page is either a rule reaching a phrase that page really uses
/// or a rule firing where it should not.
fn audit(legal: &Legal, bank: &str) -> Result<Vec<String>, Box<dyn Error>> {
    if fs::metadata(bank).is_err() {
        return Ok(Vec::new());
    }
    let read = crate::detect::bank::read_bank(bank)?;
    if !read.bad.is_empty() {
        return Err(format!("{bank}: {} unreadable rows", read.bad.len()).into());
    }
    let mut out = Vec::new();
    let clause: std::collections::BTreeSet<&str> = legal
        .rules
        .rules()
        .iter()
        .filter(|rule| rule.kind == "clause")
        .map(|rule| rule.id.as_str())
        .collect();
    for (label, name) in [(1u8, "legal"), (0u8, "plain")] {
        let rows: Vec<_> = read.rows.iter().filter(|row| row.label == label).collect();
        if rows.is_empty() {
            continue;
        }
        let (mut words, mut hits, mut after_words, mut after_hits, mut changed) = (0, 0, 0, 0, 0);
        let (mut advice, mut cited, mut rewrites) = (0, 0, 0);
        for row in &rows {
            let reading = legal.translate(&row.text, false);
            words += reading.words_in;
            hits += reading.stats.avoidable;
            after_words += reading.words_out;
            after_hits += reading.after.avoidable;
            advice += reading.after.advisory;
            cited += reading.after.in_citation;
            rewrites += reading
                .changes
                .iter()
                .filter(|change| clause.contains(change.rule.as_str()))
                .count();
            if !reading.changes.is_empty() {
                changed += 1;
            }
        }
        let before = rate(hits, words);
        let after = rate(after_hits, after_words);
        out.push(format!(
            "{name}: {} rows, {before:.2} avoidable per 100 words before, {after:.2} after \
             ({advice} advisory, {cited} in citations left in place), {changed} changed ({:.1}%), \
             {rewrites} clause rewrites",
            rows.len(),
            changed as f64 * 100.0 / rows.len() as f64
        ));
    }
    Ok(out)
}

fn rate(hits: usize, words: usize) -> f64 {
    if words == 0 {
        0.0
    } else {
        (hits as f64 * 100.0 / words as f64 * 100.0).round() / 100.0
    }
}

/// Recompute every declared row from the bytes on disk. Rows keep their order
/// and their names; only the counts, the byte lengths and the digests move.
fn rebuild_manifest(dir: &str) -> Result<Manifest, Box<dyn Error>> {
    let path = format!("{dir}/manifest.json");
    let mut manifest: Manifest = serde_json::from_str(&fs::read_to_string(&path)?)?;
    for row in &mut manifest.files {
        let bytes =
            fs::read(format!("{dir}/{}", row.name)).map_err(|e| format!("{}: {e}", row.name))?;
        row.bytes = bytes.len();
        row.rows = if row.name.ends_with(".jsonl") {
            String::from_utf8_lossy(&bytes)
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count()
        } else {
            1
        };
        row.fnv64 = hex64(fnv1a(&bytes));
    }
    manifest.built_at = crate::dataset::built_at();
    Ok(manifest)
}

fn verify(dir: &str, args: &CheckArgs) -> Result<(), Box<dyn Error>> {
    let legal = Legal::load(dir)?;
    let path = format!("{dir}/manifest.json");
    if args.fix {
        let repaired = rebuild_manifest(dir)?;
        fs::write(
            &path,
            format!("{}\n", serde_json::to_string_pretty(&repaired)?),
        )?;
        println!(
            "manifest: {} files recomputed, built_at {}",
            repaired.files.len(),
            repaired.built_at
        );
    }
    let text = fs::read_to_string(&path)?;
    let manifest: Manifest = serde_json::from_str(&text)?;

    let mut faults = Vec::new();
    let mut lines = Vec::new();
    for row in &manifest.files {
        let file = format!("{dir}/{}", row.name);
        match fs::read(&file) {
            Ok(bytes) => {
                let rows = if row.name.ends_with(".jsonl") {
                    // Counting rows the way the readers do: every non-empty line.
                    String::from_utf8_lossy(&bytes)
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .count()
                } else {
                    1
                };
                let digest = hex64(fnv1a(&bytes));
                lines.push(format!(
                    "  {:<16} {rows:>6} rows {:>9} bytes  {}",
                    row.name,
                    bytes.len(),
                    digest
                ));
                if bytes.len() != row.bytes {
                    faults.push(format!(
                        "{}: manifest says {} bytes, file has {}",
                        row.name,
                        row.bytes,
                        bytes.len()
                    ));
                }
                if rows != row.rows {
                    faults.push(format!(
                        "{}: manifest says {} rows, file has {rows}",
                        row.name, row.rows
                    ));
                }
                if digest != row.fnv64 {
                    faults.push(format!(
                        "{}: digest {} does not match the manifest's {}",
                        row.name, digest, row.fnv64
                    ));
                }
            }
            Err(e) => faults.push(format!("{}: {e}", row.name)),
        }
    }

    let declared: BTreeSet<&str> = manifest.files.iter().map(|f| f.name.as_str()).collect();
    for name in [
        "terms.jsonl",
        "rewrites.jsonl",
        "settings.json",
        "bank.jsonl",
        "pairs.jsonl",
    ] {
        if !declared.contains(name) && fs::metadata(format!("{dir}/{name}")).is_ok() {
            faults.push(format!("{name} is present but not in the manifest"));
        }
    }

    let (orphans, gaps) = legal.coverage();
    for id in &orphans {
        faults.push(format!(
            "rule {id} has no glossary row and is not a phrase rule"
        ));
    }

    let settings = Settings::load(&format!("{dir}/settings.json"))?;
    let mut warnings = Vec::new();
    if settings.protect.is_empty() {
        warnings.push("no citation patterns: a rewrite would edit a statute name".to_string());
    }
    if !gaps.is_empty() {
        warnings.push(format!(
            "{} avoidable terms have no applying rule: {}",
            gaps.len(),
            gaps.join(", ")
        ));
    }

    let audit = audit(&legal, &format!("{dir}/bank.jsonl"))?;
    let bank = format!("{dir}/bank.jsonl");
    let split = if fs::metadata(&bank).is_ok() {
        Some(registers::Coverage::audit(&legal, &bank)?)
    } else {
        None
    };
    let gaps = split.as_ref().map(|split| split.gaps()).unwrap_or_default();
    if !gaps.is_empty() {
        warnings.push(format!(
            "{} target registers are absent, thin or blind, so nothing is claimed there: {}",
            gaps.len(),
            gaps.join(", ")
        ));
    }

    if args.verbose || !faults.is_empty() {
        println!(
            "{} v{} built {}",
            manifest.name, manifest.version, manifest.built_at
        );
        for line in &lines {
            println!("{line}");
        }
        let mut counts: Vec<String> = manifest
            .counts
            .iter()
            .map(|(name, value)| format!("{name} {value}"))
            .collect();
        counts.sort();
        if !counts.is_empty() {
            println!("  {}", counts.join(", "));
        }
    }
    for line in &audit {
        println!("audit: {line}");
    }
    for warning in &warnings {
        println!("warning: {warning}");
    }
    if faults.is_empty() {
        println!(
            "ok: {} files, {} terms, {} rules, {} of them applying",
            manifest.files.len(),
            legal.terms().len(),
            legal.rules.len(),
            legal.rules.applying().count()
        );
        return Ok(());
    }
    for fault in &faults {
        println!("fault: {fault}");
    }
    Err(format!("{} faults", faults.len()).into())
}
