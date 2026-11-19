//! Applying the decision to a folder of arriving files.
//!
//! The unit is the arrival, not the message: a server hands over whatever
//! landed, which may be an `eml` with headers, a plain text notice, a saved
//! page, or a bill exported from another system. Headers are used when they are
//! there, the file name is used when they are not, and the text is read either
//! way.
//!
//! Four properties make the run safe to repeat:
//!
//! * it is a dry run unless `apply` is set, and the dry run prints exactly the
//!   copy it would make;
//! * the original is never moved or rewritten, so a wrong reading costs a
//!   duplicate under the wrong folder rather than a lost file;
//! * every decision is appended to `ledger.jsonl` beside the tree, with the
//!   digest of the policy that produced it and the second the decision was
//!   made, so the filing can be audited and replayed after the rulebook
//!   changes;
//! * the ledger is read back before the run, so a sender answered less than
//!   `reply_window_days` ago is not answered again. That is the repeat window
//!   of RFC 3834, section 7, and it is the one thing a run cannot learn from
//!   the arrivals alone.
//!
//! Duplicate suppression uses the `Message-Id`, falling back to sender and
//! subject. Two files with one identity are recorded once and the second is
//! reported as a duplicate, which is the same defence an auto-responder needs
//! against a message redelivered by a queue.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::decide::{decide, forward_draft, reply_draft, Budget, Decision, Readings, WINDOW_VETO};
use super::envelope::Envelope;
use super::history::{Facts, History, LEDGER};
use super::policy::{Policy, PolicyStamp};

/// Everything one run needs, gathered so a call site cannot swap two booleans
/// and silently file bulk mail into the review folder.
#[derive(Clone, Copy, Debug)]
pub struct Run<'a> {
    pub inbox: &'a Path,
    pub out: &'a Path,
    pub policy: &'a Policy,
    pub apply: bool,
    pub keep_ignored: bool,
    pub from: &'a str,
    /// Seconds since the Unix epoch: the ledger stamp and the clock the repeat
    /// window is measured against.
    pub now: u64,
}

/// What one arrival became.
#[derive(Clone, Debug, Serialize)]
pub struct Row {
    pub file: String,
    pub subject: String,
    pub sender: String,
    pub action: String,
    pub class: String,
    pub folder: String,
    pub bytes: u64,
    pub duplicate_of: Option<String>,
    pub dest: Option<String>,
    pub drafts: Vec<String>,
    /// When the decision was made, so a later run can measure the repeat window
    /// against it.
    pub at: u64,
    pub decision: Decision,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub inbox: String,
    pub out: String,
    pub applied: bool,
    pub now: u64,
    pub policy: PolicyStamp,
    /// The repeat window this run measured against, from the policy.
    pub window_days: u64,
    pub files: usize,
    pub duplicates: usize,
    /// Arrivals whose answer was refused because the same sender was answered
    /// inside the window.
    pub window_blocked: usize,
    /// What the ledger held before this run.
    pub history: Facts,
    pub by_action: BTreeMap<String, usize>,
    pub rows: Vec<Row>,
}

/// Read one arrival: headers if present, else the file stem as the subject.
pub fn read_arrival(path: &Path) -> Result<Envelope, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    let mut envelope = Envelope::parse(&text);
    if !envelope.has_headers {
        envelope.subject = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().replace(['_', '-'], " "))
            .unwrap_or_default();
    }
    Ok(envelope)
}

/// Decide and, when `run.apply` is set, file everything under `run.out`.
pub fn organize(run: &Run<'_>) -> Result<Summary, String> {
    let (out, policy) = (run.out, run.policy);
    let mut paths: Vec<PathBuf> = fs::read_dir(run.inbox)
        .map_err(|e| format!("{}: {e}", run.inbox.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .filter(|path| {
            !path
                .file_name()
                .map(|name| name.to_string_lossy().starts_with('.'))
                .unwrap_or(true)
        })
        .collect();
    paths.sort();

    if run.apply {
        fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    }

    // The ledger is the only memory a run has. Reading it is cheap and a
    // missing file is an empty history, so a first run needs no special case.
    let mut history = History::load(out)?;
    let before = history.facts();

    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    let mut sent: BTreeMap<String, usize> = BTreeMap::new();
    let mut rows = Vec::new();
    let mut duplicates = 0usize;
    let mut window_blocked = 0usize;

    for path in paths {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());
        let bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        let envelope = read_arrival(&path)?;
        let replies = *sent.get(&envelope.from_address).unwrap_or(&0);
        let budget = Budget {
            in_run: replies,
            since_reply_days: history.reply_age_days(&envelope.from_address, run.now),
        };

        let identity = envelope.identity();
        if let Some(first) = seen.get(&identity) {
            duplicates += 1;
            let decision = decide(&envelope, policy, &Readings::new(), budget);
            rows.push(Row {
                file: name,
                subject: envelope.subject,
                sender: envelope.from_address,
                action: "ignore".to_string(),
                class: decision.class.clone(),
                folder: decision.folder.clone(),
                bytes,
                duplicate_of: Some(first.clone()),
                dest: None,
                drafts: Vec::new(),
                at: run.now,
                decision,
            });
            continue;
        }
        seen.insert(identity, name.clone());

        let decision = decide(&envelope, policy, &Readings::new(), budget);
        if decision
            .vetoes
            .iter()
            .any(|line| line.starts_with(WINDOW_VETO))
        {
            window_blocked += 1;
        }

        let mut drafts = Vec::new();
        if decision.action == "reply" {
            // The counter moves in a dry run too, so the report names the
            // second message a real run would hold rather than promising an
            // answer it would not write. Only an applied run leaves a record.
            *sent.entry(envelope.from_address.clone()).or_insert(0) += 1;
            if run.apply {
                if let Some(body) = reply_draft(&envelope, &decision, policy, run.from) {
                    let rel = format!("drafts/{}-reply.eml", &decision.digest[..8]);
                    write_file(&out.join(&rel), &body)?;
                    drafts.push(rel);
                    history.record(&envelope.from_address, run.now);
                }
            }
        }
        if run.apply && decision.action == "forward" {
            if let Some(body) = forward_draft(&envelope, &decision, run.from) {
                let rel = format!("drafts/{}-forward.eml", &decision.digest[..8]);
                write_file(&out.join(&rel), &body)?;
                drafts.push(rel);
            }
        }

        let keep = decision.action != "ignore" || run.keep_ignored;
        let dest = if run.apply && keep {
            let folder = if decision.action == "hold" {
                "review".to_string()
            } else {
                decision.folder.clone()
            };
            let rel = unique_path(out, &folder, &name, &decision.digest);
            let dest_path = out.join(&rel);
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            fs::copy(&path, &dest_path).map_err(|e| format!("{} -> {rel}: {e}", path.display()))?;
            Some(rel)
        } else {
            None
        };

        rows.push(Row {
            file: name,
            subject: envelope.subject,
            sender: envelope.from_address,
            action: decision.action.clone(),
            class: decision.class.clone(),
            folder: decision.folder.clone(),
            bytes,
            duplicate_of: None,
            dest,
            drafts,
            at: run.now,
            decision,
        });
    }

    let mut by_action: BTreeMap<String, usize> = BTreeMap::new();
    for row in &rows {
        *by_action.entry(row.action.clone()).or_insert(0) += 1;
    }

    let summary = Summary {
        inbox: run.inbox.display().to_string(),
        out: out.display().to_string(),
        applied: run.apply,
        now: run.now,
        policy: policy.stamp(),
        window_days: policy.guards.reply_window_days,
        files: rows.len(),
        duplicates,
        window_blocked,
        history: before,
        by_action,
        rows,
    };
    if run.apply {
        append_ledger(out, &summary)?;
    }
    Ok(summary)
}

fn unique_path(out: &Path, folder: &str, name: &str, digest: &str) -> String {
    let first = format!("{folder}/{name}");
    if !out.join(&first).exists() {
        return first;
    }
    format!("{folder}/{}-{name}", &digest[..8])
}

fn write_file(path: &Path, body: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))
}

fn append_ledger(out: &Path, summary: &Summary) -> Result<(), String> {
    let path = out.join(LEDGER);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    for row in &summary.rows {
        let line = serde_json::to_string(row).map_err(|e| e.to_string())?;
        writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// The set of folder names a run may create, used by the report.
pub fn folders(policy: &Policy) -> BTreeSet<String> {
    policy.folders.values().cloned().collect()
}
