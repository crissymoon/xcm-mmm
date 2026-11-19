//! Behaviour the router claims, written as messages it has to get right.
//!
//! The samples in `data/route/samples/` are the fixtures. Each test names the
//! decision a competent human would make for that file and the rule that
//! forces it, so a change to a cue weight that moves a demand letter into a
//! folder shows up as a failure rather than as a quiet re-tune.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::decide::{decide, forward_draft, reply_draft};
use super::envelope::Envelope;
use super::organize::{organize, read_arrival};
use super::policy::Policy;

const SAMPLES: &str = "data/route/samples";

fn policy() -> Policy {
    Policy::load("data/route/policy.json").unwrap()
}

fn sample(name: &str) -> Envelope {
    read_arrival(&Path::new(SAMPLES).join(name)).unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xcm-route-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn a_question_from_a_person_is_answered() {
    let envelope = sample("support-login.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "support_request");
    assert_eq!(decision.action, "reply");
    assert!(
        !decision
            .vetoes
            .iter()
            .any(|v| v.starts_with("reply vetoed")),
        "{:?}",
        decision.vetoes
    );
    let draft = reply_draft(&envelope, &decision, &policy(), "router@example.org").unwrap();
    assert!(draft.contains("Auto-Submitted: auto-replied"));
    assert!(draft.contains("In-Reply-To: <support-1@example.com>"));
    assert!(draft.contains("To: ada@example.com"));
    assert!(draft.contains("not read by a person"));
}

#[test]
fn an_invoice_goes_to_finance_with_a_forward_draft() {
    let envelope = sample("invoice-4471.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "invoice");
    assert_eq!(decision.action, "forward");
    assert_eq!(decision.folder, "finance/invoices");
    let draft = forward_draft(&envelope, &decision, "router@example.org").unwrap();
    assert!(draft.contains("To: billing@example.invalid"));
    assert!(draft.contains("----- Original message -----"));
}

#[test]
fn a_demand_letter_is_held_and_never_forwarded() {
    let envelope = sample("legal-demand.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "legal_notice");
    assert_eq!(decision.action, "hold");
    assert!(decision.vetoes.iter().any(|v| v.contains("forward vetoed")));
    assert!(decision.gate.is_some());
}

#[test]
fn a_list_message_is_ignored_and_never_answered() {
    let envelope = sample("newsletter.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "bulk_list");
    assert_eq!(decision.action, "ignore");
    assert!(decision
        .vetoes
        .iter()
        .any(|v| v.contains("bulk or list headers")));
    assert!(reply_draft(&envelope, &decision, &policy(), "a@b.c").is_none());
}

#[test]
fn an_application_is_filed_and_not_forwarded() {
    let envelope = sample("application.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "application");
    assert_eq!(decision.action, "file");
    assert!(decision.vetoes.iter().any(|v| v.contains("must not leave")));
}

#[test]
fn a_document_without_headers_still_files() {
    let envelope = sample("purchase-order.txt");
    assert!(!envelope.has_headers);
    assert_eq!(envelope.subject, "purchase order");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.class, "invoice");
    assert_eq!(decision.action, "forward");
}

#[test]
fn an_unremarkable_document_goes_to_a_person() {
    let envelope = sample("minutes.txt");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(decision.action, "hold");
    assert!(decision.gate.unwrap().contains("top share"));
}

#[test]
fn auto_submitted_suppresses_the_reply() {
    let text = "From: robot@example.com\nAuto-Submitted: auto-generated\nSubject: ticket 12\n\nYour ticket was created.\n";
    let decision = decide(&Envelope::parse(text), &policy(), &BTreeMap::new(), 0);
    assert!(decision.vetoes.iter().any(|v| v.contains("Auto-Submitted")));
    assert_ne!(decision.action, "reply");
}

#[test]
fn a_machine_mailbox_is_never_answered() {
    let text = "From: MAILER-DAEMON@example.com\nSubject: failure notice\n\nUndelivered mail.\n";
    let decision = decide(&Envelope::parse(text), &policy(), &BTreeMap::new(), 0);
    assert!(decision
        .vetoes
        .iter()
        .any(|v| v.contains("machine mailbox")));
}

#[test]
fn an_empty_reverse_path_blocks_a_reply() {
    let text = "From: list@example.com\nReturn-Path: <>\nSubject: cannot log in\n\nThe password reset failed.\n";
    let envelope = Envelope::parse(text);
    assert!(envelope.null_return_path());
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert!(decision
        .vetoes
        .iter()
        .any(|v| v.contains("reverse-path is empty")));
}

#[test]
fn the_reply_budget_holds_the_second_message() {
    let envelope = sample("support-login.eml");
    let decision = decide(&envelope, &policy(), &BTreeMap::new(), 1);
    assert_eq!(decision.action, "hold");
    assert!(decision
        .vetoes
        .iter()
        .any(|v| v.contains("already has 1 replies")));
}

#[test]
fn a_detector_reading_can_move_a_close_call() {
    let text = "Subject: Important notice\n\nPlease read the enclosed document and reply.\n";
    let mut readings = BTreeMap::new();
    readings.insert("legal".to_string(), 0.9);
    let decision = decide(&Envelope::parse(text), &policy(), &readings, 0);
    assert_eq!(decision.class, "legal_notice");
    assert_eq!(decision.action, "hold");
}

#[test]
fn the_same_message_decides_the_same_way_twice() {
    let envelope = sample("invoice-4471.eml");
    let first = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    let second = decide(&envelope, &policy(), &BTreeMap::new(), 0);
    assert_eq!(first.digest, second.digest);
    assert_eq!(first.action, second.action);
}

#[test]
fn a_dry_run_writes_nothing() {
    let out = scratch("dry");
    let summary = organize(
        Path::new(SAMPLES),
        &out,
        &policy(),
        false,
        false,
        "router@example.org",
    )
    .unwrap();
    assert_eq!(summary.files, 7);
    assert!(summary.rows.iter().all(|row| row.dest.is_none()));
    assert_eq!(fs::read_dir(&out).unwrap().count(), 0);
}

#[test]
fn an_applied_run_files_every_message_and_writes_the_ledger() {
    let out = scratch("apply");
    let summary = organize(
        Path::new(SAMPLES),
        &out,
        &policy(),
        true,
        false,
        "router@example.org",
    )
    .unwrap();
    assert!(summary.applied);
    assert!(out.join("finance/invoices/invoice-4471.eml").exists());
    assert!(out.join("support/inbox/support-login.eml").exists());
    assert!(out.join("review/legal-demand.eml").exists());
    assert!(out.join("people/applications/application.eml").exists());
    assert!(!out.join("bulk/lists/newsletter.eml").exists());
    let ledger = fs::read_to_string(out.join("ledger.jsonl")).unwrap();
    assert_eq!(ledger.lines().count(), 7);
    assert!(ledger.contains("\"policy\""));
    // A support reply, and a forward for each of the two invoices: the eml and
    // the headerless purchase order are the same queue.
    let drafts: Vec<_> = summary
        .rows
        .iter()
        .filter(|row| !row.drafts.is_empty())
        .collect();
    assert_eq!(drafts.len(), 3);
    for row in drafts {
        for draft in &row.drafts {
            assert!(out.join(draft).exists(), "{draft}");
        }
    }
}

#[test]
fn keep_ignored_copies_bulk_mail_too() {
    let out = scratch("keep");
    organize(
        Path::new(SAMPLES),
        &out,
        &policy(),
        true,
        true,
        "router@example.org",
    )
    .unwrap();
    assert!(out.join("bulk/lists/newsletter.eml").exists());
}

#[test]
fn a_redelivered_message_is_counted_once() {
    let inbox = scratch("inbox-dupe");
    let out = scratch("out-dupe");
    fs::copy(
        Path::new(SAMPLES).join("invoice-4471.eml"),
        inbox.join("invoice-4471.eml"),
    )
    .unwrap();
    fs::copy(
        Path::new(SAMPLES).join("invoice-4471.eml"),
        inbox.join("invoice-4471-copy.eml"),
    )
    .unwrap();
    let summary = organize(&inbox, &out, &policy(), true, false, "a@b.c").unwrap();
    assert_eq!(summary.files, 2);
    assert_eq!(summary.duplicates, 1);
    assert!(summary.rows[1].duplicate_of.is_some());
}

#[test]
fn the_shipped_policy_and_its_samples_agree() {
    // The samples are part of the instrument: if a cue changes, this fails and
    // names the file whose expected action moved.
    let policy = policy();
    let expected = [
        ("support-login.eml", "reply"),
        ("invoice-4471.eml", "forward"),
        ("legal-demand.eml", "hold"),
        ("newsletter.eml", "ignore"),
        ("application.eml", "file"),
        ("purchase-order.txt", "forward"),
        ("minutes.txt", "hold"),
    ];
    for (name, action) in expected {
        let decision = decide(&sample(name), &policy, &BTreeMap::new(), 0);
        assert_eq!(decision.action, action, "{name}");
    }
}
