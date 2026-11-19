//! From evidence to one action, with the reason each veto fired.
//!
//! The procedure is a decision rule, not a classifier. Classes get a share of
//! the available evidence, each action gets the cost it would incur averaged
//! over those shares, and the cheapest action wins. That is the expected-cost
//! rule for an asymmetric loss matrix (Elkan, IJCAI, 2001): the matrix is what
//! makes "ignore a newsletter" cheap and "ignore a demand letter" expensive,
//! without the router ever needing a probability it cannot calibrate.
//!
//! Two features keep the rule honest about what it does not know.
//!
//! * Vetoes. Some actions are forbidden regardless of cost: RFC 3834 says a
//!   robot must not answer a message that arrives with `Auto-Submitted` other
//!   than `no`, a list message carries `List-Id` and often `Precedence: bulk`,
//!   and a class marked as privileged may not leave the organisation at any
//!   price. A veto removes the action from the argmin and is printed.
//! * The reject option. When the top share is below `min_top_share`, or the two
//!   cheapest actions sit within `min_margin` cost units, the router holds the
//!   message for a person. Chow (1970) is the origin of the error-reject
//!   tradeoff, and the practical reason is automation bias: a plausible action
//!   with no gate is accepted more readily than it deserves (Parasuraman and
//!   Riley, *Human Factors*, 1997).
//!
//! The shares are shares of evidence, not probabilities. Nothing here measures
//! whether a share of 0.7 is right seven times in ten, which is exactly the
//! mistake Guo and others document for uncalibrated scores (ICML, 2017). The
//! cost matrix and the gate are what carry the behaviour until a labelled
//! routing log exists to fit against.

use std::collections::BTreeMap;

use serde::Serialize;

use super::envelope::{domain, local_part, Envelope};
use super::policy::Policy;

/// One reading, with the action it supports and everything that argued with it.
#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    pub action: String,
    pub class: String,
    pub shares: Vec<(String, f64)>,
    pub expected: Vec<(String, f64)>,
    pub margin: f64,
    pub folder: String,
    pub forward_to: Option<String>,
    pub reply_to: Option<String>,
    pub reasons: Vec<String>,
    pub vetoes: Vec<String>,
    pub gate: Option<String>,
    pub identity: String,
    pub subject: String,
    pub sender: String,
    pub digest: String,
    pub policy: String,
}

/// Detector readings the caller already computed. Absent axes change nothing.
pub type Readings = BTreeMap<String, f64>;

/// Decide one message. `sent` is how many replies this sender already received
/// in the current run, which is the only loop budget a stateless call can hold.
pub fn decide(envelope: &Envelope, policy: &Policy, readings: &Readings, sent: usize) -> Decision {
    let text = envelope.reading_text().to_lowercase();
    let mut weights: BTreeMap<String, f64> = policy
        .classes
        .iter()
        .map(|class| (class.id.clone(), class.prior))
        .collect();
    let mut reasons: Vec<String> = Vec::new();

    for class in &policy.classes {
        for (phrase, weight) in &class.cues {
            if text.contains(&phrase.to_lowercase()) {
                *weights.entry(class.id.clone()).or_insert(0.0) += weight;
                reasons.push(format!("cue {:?} adds {weight:.1} to {}", phrase, class.id));
            }
        }
    }

    let marks = envelope.bulk_marks(&policy.guards.bulk_headers);
    let precedence = envelope.precedence();
    let precedence_bulk = precedence
        .as_deref()
        .is_some_and(|value| policy.guards.bulk_precedence.iter().any(|p| p == value));
    let bulk_by_header = !marks.is_empty() || precedence_bulk;
    if !marks.is_empty() {
        *weights.entry("bulk_list".to_string()).or_insert(0.0) += 6.0;
        if let Some(mark) = marks.first() {
            reasons.push(format!("bulk header {mark} adds 6.0 to bulk_list"));
        }
    }
    if precedence_bulk {
        // RFC 3834, section 3.1.8 and RFC 2076, section 3.9: Precedence is not
        // standard and is discouraged, so it may stop a reply but it may not by
        // itself move a message into the bulk class.
        reasons.push(format!(
            "precedence {} is a discouragement hint only (RFC 2076, 3.9)",
            precedence.clone().unwrap_or_default()
        ));
    }

    if let Some(score) = readings.get("legal") {
        if *score >= 0.5 {
            *weights.entry("legal_notice".to_string()).or_insert(0.0) += 3.0 * score;
            reasons.push(format!(
                "detector legal reads {score:.2}, adds {:.1} to legal_notice",
                3.0 * score
            ));
        }
    }
    if let Some(score) = readings.get("manipulation") {
        if *score >= 0.6 {
            *weights.entry("bulk_list".to_string()).or_insert(0.0) += 2.0 * score;
            reasons.push(format!(
                "detector manipulation reads {score:.2}, adds {:.1} to bulk_list",
                2.0 * score
            ));
        }
    }

    let total: f64 = weights.values().sum();
    let shares: Vec<(String, f64)> = if total > 0.0 {
        let mut rows: Vec<(String, f64)> = weights
            .iter()
            .map(|(id, weight)| (id.clone(), round3(weight / total)))
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rows
    } else {
        vec![("general".to_string(), 1.0)]
    };
    let class = shares
        .first()
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| "general".to_string());
    let top = shares.first().map(|(_, share)| *share).unwrap_or(0.0);

    let mut expected: Vec<(String, f64)> = policy
        .action_ids()
        .into_iter()
        .map(|action| {
            let cost = weights
                .iter()
                .map(|(id, weight)| {
                    let share = if total > 0.0 { weight / total } else { 0.0 };
                    share * policy.cost(id, &action).unwrap_or(0.0)
                })
                .sum::<f64>();
            (action, round3(cost))
        })
        .collect();
    expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut vetoes: Vec<String> = Vec::new();
    let blocked = policy.guards.blocked_local_parts.iter().any(|part| {
        *part == envelope.sender_local_part() || *part == local_part(&envelope.from_address)
    });
    if blocked {
        vetoes.push(format!(
            "reply vetoed: {} is a machine mailbox (RFC 3834 loop safety)",
            envelope.from_address
        ));
    }
    let auto = envelope
        .all(&policy.guards.auto_header)
        .first()
        .map(|value| value.trim().to_ascii_lowercase());
    if let Some(auto) = auto {
        if auto != "no" {
            vetoes.push(format!(
                "reply vetoed: Auto-Submitted is {auto}, so this message is already automatic"
            ));
        }
    }
    if envelope.reply_target().is_none() {
        vetoes.push("reply vetoed: no address survives the envelope checks".to_string());
    }
    if envelope.null_return_path() {
        vetoes.push(
            "reply vetoed: the reverse-path is empty, so there is no sender to answer (RFC 3834, 2)"
                .to_string(),
        );
    }
    if bulk_by_header {
        vetoes.push("reply vetoed: bulk or list headers make a robot answer wrong".to_string());
    }
    let never_reply = policy
        .guards
        .never_reply_classes
        .iter()
        .any(|id| id == &class);
    if never_reply {
        vetoes.push(format!("reply vetoed: policy never answers a {class}"));
    } else if policy.reply_template(&class).is_none() {
        vetoes.push(format!("reply vetoed: no template for {class}"));
    }
    if sent >= policy.guards.max_replies_per_sender {
        vetoes.push(format!(
            "reply vetoed: this sender already has {sent} replies in this run"
        ));
    }
    let never_forward = policy
        .guards
        .never_forward_classes
        .iter()
        .any(|id| id == &class);
    if never_forward {
        vetoes.push(format!(
            "forward vetoed: a {class} must not leave the organisation"
        ));
    }
    match policy.forward_to.get(&class) {
        None if !never_forward => vetoes.push(format!(
            "forward vetoed: no queue is configured for {class}"
        )),
        None => {}
        Some(queue) => {
            if envelope.body.len() > policy.guards.max_forward_bytes {
                vetoes.push(format!(
                    "forward vetoed: {} bytes exceeds the transfer limit",
                    envelope.body.len()
                ));
            }
            if queue.is_empty() || local_part(queue).is_empty() {
                vetoes.push("forward vetoed: the queue address is malformed".to_string());
            }
        }
    }

    let is_vetoed = |action: &str| -> bool {
        vetoes.iter().any(|line| {
            (action == "reply" && line.starts_with("reply vetoed"))
                || (action == "forward" && line.starts_with("forward vetoed"))
        })
    };

    let mut gate: Option<String> = None;
    let mut action = expected
        .iter()
        .find(|(id, _)| !is_vetoed(id))
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| {
            gate = Some("every action was vetoed, so the message is held".to_string());
            "hold".to_string()
        });

    if policy
        .guards
        .always_hold_classes
        .iter()
        .any(|id| id == &class)
    {
        gate = Some(format!("a {class} is always read by a person first"));
        action = "hold".to_string();
    }

    // A veto that removes the answer a class expects must not fall through to a
    // silent filing. When reply is the class's own cheapest action and it was
    // vetoed, the message goes to a person instead of a folder.
    let reply_expected = policy.cost(&class, "reply").unwrap_or(f64::MAX)
        < policy.cost(&class, "ignore").unwrap_or(f64::MAX);
    let reply_blocked_temporarily = vetoes.iter().any(|line| {
        line.starts_with("reply vetoed") && {
            line.contains("already has") || line.contains("no address survives")
        }
    });
    if gate.is_none()
        && action == "file"
        && reply_expected
        && reply_blocked_temporarily
        && !envelope.from_address.is_empty()
    {
        gate = Some(format!(
            "a {class} expects an answer and reply is vetoed, so a person must answer"
        ));
        action = "hold".to_string();
    }

    let margin = match (expected.first(), expected.get(1)) {
        (Some((_, best)), Some((_, second))) => round3(second - best),
        _ => 0.0,
    };
    if gate.is_none() && top < policy.guards.min_top_share {
        gate = Some(format!(
            "top share {top:.2} is below {:.2}, so no class owns the message",
            policy.guards.min_top_share
        ));
    }
    if gate.is_none() && margin < policy.guards.min_margin {
        gate = Some(format!(
            "the two cheapest actions differ by {margin:.2} cost units, below {:.2}",
            policy.guards.min_margin
        ));
    }
    if gate.is_some() && action != "hold" {
        vetoes.push(format!("hold forced: {}", gate.clone().unwrap_or_default()));
        action = "hold".to_string();
    }

    let folder = policy.folder(&class).unwrap_or("unrouted").to_string();
    let forward_to = policy.forward_to.get(&class).cloned();
    let reply_to = match action.as_str() {
        "reply" => envelope.reply_target(),
        _ => None,
    };
    let identity = envelope.identity();
    let digest = {
        let seed = format!("{}|{}|{}|{}", action, class, identity, policy.stamp().fnv64);
        crate::text::hash::hex64(crate::text::hash::fnv1a(seed.as_bytes()))
    };

    Decision {
        action,
        class,
        shares,
        expected,
        margin,
        folder,
        forward_to,
        reply_to,
        reasons,
        vetoes,
        gate,
        identity,
        subject: envelope.subject.clone(),
        sender: envelope.from_address.clone(),
        digest,
        policy: policy.stamp().fnv64,
    }
}

/// A reply body a person can read and a robot can ignore, carrying the header
/// RFC 3834 requires so a second robot does not answer it.
pub fn reply_draft(
    envelope: &Envelope,
    decision: &Decision,
    policy: &Policy,
    from: &str,
) -> Option<String> {
    let template = policy.reply_template(&decision.class)?;
    let to = decision.reply_to.clone()?;
    let subject = if envelope.subject.to_lowercase().starts_with("re:") {
        envelope.subject.clone()
    } else {
        format!("Re: {}", envelope.subject)
    };
    let mut out = String::new();
    out.push_str(&format!("To: {to}\n"));
    out.push_str(&format!("From: {from}\n"));
    out.push_str(&format!("Subject: {subject}\n"));
    if !envelope.message_id.is_empty() {
        out.push_str(&format!("In-Reply-To: {}\n", envelope.message_id));
        let references = if envelope.references.is_empty() {
            envelope.message_id.clone()
        } else {
            format!("{} {}", envelope.references, envelope.message_id)
        };
        out.push_str(&format!("References: {references}\n"));
    }
    out.push_str("Auto-Submitted: auto-replied\n");
    out.push_str(&format!(
        "X-Xcm-Route: {} {} {}\n",
        decision.action, decision.class, decision.digest
    ));
    out.push('\n');
    out.push_str(template);
    out.push_str("\n\n--\nThis acknowledgement was written by a router. It was not read by a person before it was sent.\n");
    Some(out)
}

/// A forward body for the queue that owns the class. The original headers are
/// kept so the queue can see where it came from.
pub fn forward_draft(envelope: &Envelope, decision: &Decision, from: &str) -> Option<String> {
    let queue = decision.forward_to.clone()?;
    let mut out = String::new();
    out.push_str(&format!("To: {queue}\n"));
    out.push_str(&format!("From: {from}\n"));
    out.push_str(&format!("Subject: Fwd: {}\n", envelope.subject));
    out.push_str(&format!(
        "X-Xcm-Route: {} {} {}\n",
        decision.action, decision.class, decision.digest
    ));
    out.push_str("\nForwarded by the router as a ");
    out.push_str(&decision.class);
    out.push_str(".\n\n");
    out.push_str("----- Original message -----\n");
    out.push_str(&format!("From: {}\n", envelope.from));
    out.push_str(&format!("Date: {}\n", envelope.date));
    out.push_str(&format!("Subject: {}\n", envelope.subject));
    out.push_str(&format!("Message-Id: {}\n\n", envelope.message_id));
    out.push_str(&envelope.body);
    Some(out)
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// The sender domain, exposed for the report rather than the decision.
pub fn sender_domain(envelope: &Envelope) -> String {
    domain(&envelope.from_address)
}
