//! The `route` command surface: decide, organize, policy, check.
//!
//! `decide` answers one message and prints every number behind the answer.
//! `organize` runs a folder of arrivals, dry by default, and writes the ledger
//! that makes the run auditable. Neither sends mail: a reply or a forward is
//! written as an `eml` draft for a person or a queue to release, because a tool
//! that decides from a score has no business also being the thing that
//! transmits.

use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Read};
use std::path::Path;

use crate::detect::Detector;
use clap::{Args, Subcommand};

use super::decide::{decide, forward_draft, reply_draft, Decision, Readings};
use super::envelope::Envelope;
use super::organize::{organize, read_arrival, Summary};
use super::policy::{Policy, POLICY};

/// Arguments for `route`.
#[derive(Args)]
pub struct RouteArgs {
    #[command(subcommand)]
    pub command: RouteCommand,
    /// Folder holding the policy.
    #[arg(long, default_value = super::policy::DIR, global = true, value_name = "PATH")]
    pub dir: String,
}

#[derive(Subcommand)]
pub enum RouteCommand {
    /// Read one message and name the action, the class and every veto.
    Decide(DecideArgs),
    /// Read a folder of arrivals and file them, or show what filing would do.
    Organize(OrganizeArgs),
    /// Print the rulebook: classes, folders, costs and guards.
    Policy(PolicyArgs),
    /// Validate the rulebook and print its stamp.
    Check(CheckArgs),
}

#[derive(Args)]
pub struct DecideArgs {
    /// Text to read. Several arguments are joined with a space.
    #[arg(value_name = "TEXT")]
    text: Vec<String>,
    /// Read a message or a document from a file.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    file: Option<String>,
    /// Add the legal and manipulation readings from a detector artifact.
    #[arg(long, value_name = "PATH")]
    detectors: Option<String>,
    /// Print the reply or forward that the action would produce.
    #[arg(long)]
    draft: bool,
    /// One JSON object instead of a report.
    #[arg(long)]
    json: bool,
    /// Identity written into a draft's From line. Nothing is sent.
    #[arg(long, default_value = "xcm-router@localhost")]
    from: String,
}

#[derive(Args)]
pub struct OrganizeArgs {
    /// Folder of arrivals to read.
    #[arg(long, value_name = "PATH")]
    pub inbox: String,
    /// Folder to write the tree, the drafts and the ledger into.
    #[arg(long, value_name = "PATH")]
    pub out: String,
    /// Write the copies. Without it the run only reports.
    #[arg(long)]
    pub apply: bool,
    /// Also copy what was ignored, such as bulk mail.
    #[arg(long)]
    pub keep_ignored: bool,
    /// Add detector readings, at the cost of one model load per file.
    #[arg(long, value_name = "PATH")]
    pub detectors: Option<String>,
    #[arg(long)]
    pub json: bool,
    #[arg(long, default_value = "xcm-router@localhost")]
    pub from: String,
}

#[derive(Args)]
pub struct PolicyArgs {
    /// The rulebook as JSON, which is the form the file is in.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct CheckArgs {}

pub fn run(args: RouteArgs) -> Result<(), Box<dyn Error>> {
    let dir = args.dir.clone();
    match args.command {
        RouteCommand::Decide(job) => decide_one(&dir, &job),
        RouteCommand::Organize(job) => organize_many(&dir, &job),
        RouteCommand::Policy(show) => show_policy(&dir, &show),
        RouteCommand::Check(_) => check(&dir),
    }
}

fn decide_one(dir: &str, args: &DecideArgs) -> Result<(), Box<dyn Error>> {
    let policy = load(dir)?;
    let text = match (&args.file, args.text.is_empty()) {
        (Some(path), _) => {
            let envelope = read_arrival(Path::new(path))?;
            envelope.reading_text()
        }
        (None, false) => args.text.join(" "),
        (None, true) => {
            if io::stdin().is_terminal() {
                return Err("a text, --file, or standard input is required".into());
            }
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            buffer
        }
    };
    let envelope = match &args.file {
        Some(path) => read_arrival(Path::new(path))?,
        None => {
            let mut envelope = Envelope::parse(&text);
            if !envelope.has_headers && envelope.subject.is_empty() {
                envelope.subject = first_line(&text);
            }
            envelope
        }
    };
    let readings = readings(args.detectors.as_deref(), &envelope.reading_text())?;
    let decision = decide(&envelope, &policy, &readings, 0);

    if args.json {
        let row = serde_json::json!({
            "decision": decision,
            "envelope": {
                "from": envelope.from_address,
                "to": envelope.to,
                "subject": envelope.subject,
                "message_id": envelope.message_id,
                "auto_submitted": envelope.auto_submitted(),
                "precedence": envelope.precedence(),
            },
            "drafts": {
                "reply": reply_draft(&envelope, &decision, &policy, &args.from),
                "forward": forward_draft(&envelope, &decision, &args.from),
            },
        });
        println!("{}", serde_json::to_string_pretty(&row)?);
        return Ok(());
    }

    print_report(&decision);
    if args.draft {
        let draft = match decision.action.as_str() {
            "reply" => reply_draft(&envelope, &decision, &policy, &args.from),
            "forward" => forward_draft(&envelope, &decision, &args.from),
            _ => None,
        };
        match draft {
            Some(body) => println!("\n--- draft, not sent ---\n{body}"),
            None => println!("\nno draft: the action {} sends nothing", decision.action),
        }
    }
    Ok(())
}

fn organize_many(dir: &str, args: &OrganizeArgs) -> Result<(), Box<dyn Error>> {
    if args.detectors.is_some() {
        // The folder run holds one decision per file; a detector adds a load per
        // file, so it is accepted here only to be refused rather than silently
        // ignored. `route decide` is the place for a detector reading.
        return Err("organize reads headers and cues only; pass --detectors to decide".into());
    }
    let policy = load(dir)?;
    let summary = organize(
        Path::new(&args.inbox),
        Path::new(&args.out),
        &policy,
        args.apply,
        args.keep_ignored,
        &args.from,
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        print_summary(&summary);
    }
    Ok(())
}

fn show_policy(dir: &str, args: &PolicyArgs) -> Result<(), Box<dyn Error>> {
    let policy = load(dir)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&policy)?);
        return Ok(());
    }
    println!("{}", policy.disclaimer);
    println!("\nactions");
    for action in &policy.actions {
        println!("  {:<8} {}", action.id, action.note);
    }
    println!("\nclasses and the folder each one files into");
    for class in &policy.classes {
        let folder = policy.folder(&class.id).unwrap_or("unrouted");
        let queue = policy
            .forward_to
            .get(&class.id)
            .map(|to| format!(", forward {to}"))
            .unwrap_or_default();
        println!("  {:<16} -> {folder}{queue}", class.id);
        println!("  {:<16}    {}", "", class.question);
    }
    println!("\nexpected cost, class down the side and action across");
    let actions = policy.action_ids();
    print!("  {:<16}", "class");
    for action in &actions {
        print!("{:>8}", action);
    }
    println!();
    for class in &policy.classes {
        print!("  {:<16}", class.id);
        for action in &actions {
            print!(
                "{:>8.1}",
                policy.cost(&class.id, action).unwrap_or(f64::NAN)
            );
        }
        println!();
    }
    println!("\nguards");
    println!(
        "  top share at least {:.2}, actions within {:.2} cost units go to a person",
        policy.guards.min_top_share, policy.guards.min_margin
    );
    println!(
        "  at most {} reply per sender per run",
        policy.guards.max_replies_per_sender
    );
    println!(
        "  never reply to: {}",
        policy.guards.blocked_local_parts.join(", ")
    );
    println!(
        "  always read by a person: {}",
        policy.guards.always_hold_classes.join(", ")
    );
    println!(
        "  never forwarded: {}",
        policy.guards.never_forward_classes.join(", ")
    );
    Ok(())
}

fn check(dir: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(dir).join(POLICY);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let policy: Policy =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let problems = policy.validate();
    let stamp = policy.stamp();
    if problems.is_empty() {
        println!(
            "ok: {} classes, {} actions, {} bytes, policy {}",
            stamp.classes, stamp.actions, stamp.bytes, stamp.fnv64
        );
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("problem: {problem}");
        }
        Err(format!("{} problems in {}", problems.len(), path.display()).into())
    }
}

fn load(dir: &str) -> Result<Policy, Box<dyn Error>> {
    Ok(Policy::load(
        &Path::new(dir).join(POLICY).display().to_string(),
    )?)
}

fn readings(model: Option<&str>, text: &str) -> Result<Readings, Box<dyn Error>> {
    let Some(path) = model else {
        return Ok(Readings::new());
    };
    let detector = Detector::load(path)?;
    let mut out = Readings::new();
    for reading in detector.score(text) {
        out.insert(reading.axis, reading.score);
    }
    Ok(out)
}

fn first_line(text: &str) -> String {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn print_report(decision: &Decision) {
    println!("action: {}", decision.action);
    println!(
        "class:  {} ({:.2} of the evidence)",
        decision.class, decision.shares[0].1
    );
    println!("folder: {}", decision.folder);
    if let Some(queue) = &decision.forward_to {
        println!("queue:  {queue}");
    }
    print!("cost:  ");
    for (action, cost) in &decision.expected {
        print!(" {action} {cost:.2}");
    }
    println!();
    if !decision.reasons.is_empty() {
        println!("why:");
        for reason in &decision.reasons {
            println!("  + {reason}");
        }
    }
    if !decision.vetoes.is_empty() {
        println!("against:");
        for veto in &decision.vetoes {
            println!("  - {veto}");
        }
    }
    if let Some(gate) = &decision.gate {
        println!("gate:   {gate}");
    }
    println!("digest: {}", decision.digest);
}

fn print_summary(summary: &Summary) {
    println!(
        "{} files from {}, {} duplicate{}, policy {}",
        summary.files,
        summary.inbox,
        summary.duplicates,
        if summary.duplicates == 1 { "" } else { "s" },
        summary.policy.fnv64
    );
    let width = summary
        .rows
        .iter()
        .map(|row| row.file.chars().count())
        .max()
        .unwrap_or(10)
        .min(44);
    for row in &summary.rows {
        let mark = if row.duplicate_of.is_some() {
            "duplicate"
        } else {
            row.action.as_str()
        };
        println!(
            "  {:<width$}  {:<10} {:<16} {}",
            row.file,
            mark,
            row.class,
            row.dest.as_deref().unwrap_or("-"),
            width = width
        );
        for draft in &row.drafts {
            println!("  {:<width$}  draft {draft}", "", width = width);
        }
    }
    print!(
        "{}",
        if summary.applied {
            ""
        } else {
            "\ndry run: nothing was written\n"
        }
    );
    print!("actions:");
    for (action, count) in &summary.by_action {
        print!(" {action} {count}");
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_first_line_becomes_a_subject() {
        assert_eq!(first_line("\n\nInvoice 4471\nAmount due"), "Invoice 4471");
        assert_eq!(first_line("   "), "");
    }
}
