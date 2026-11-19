//! The rulebook, in one file a person can read and change.
//!
//! Every number the router uses lives in `data/route/policy.json`: the cost of
//! each action under each class, the cues that raise a class, the folder tree,
//! the forward queues and the guard thresholds. Code holds the procedure and the
//! file holds the policy, which is the only arrangement in which an operator can
//! answer "why did it file my invoice under legal" without reading Rust.
//!
//! The file is validated as a whole. A class with no cost for one action would
//! quietly make that action free, and a folder that two classes share would make
//! a filing look successful while it landed somewhere else, so both are errors
//! rather than warnings.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use serde::{Deserialize, Serialize};

use crate::text::hash::{fnv1a, hex64};

/// The folder the instrument ships in.
pub const DIR: &str = "data/route";
/// The policy file inside it.
pub const POLICY: &str = "policy.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub version: u32,
    pub disclaimer: String,
    pub actions: Vec<ActionSpec>,
    pub classes: Vec<ClassSpec>,
    pub costs: BTreeMap<String, BTreeMap<String, f64>>,
    pub folders: BTreeMap<String, String>,
    pub forward_to: BTreeMap<String, String>,
    pub replies: BTreeMap<String, String>,
    pub guards: Guards,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionSpec {
    pub id: String,
    pub note: String,
    pub sends_mail: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassSpec {
    pub id: String,
    pub question: String,
    pub prior: f64,
    /// Phrase and weight. A phrase matches case insensitively as a substring,
    /// because the alternative is a tokeniser arguing about hyphens.
    pub cues: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Guards {
    pub min_top_share: f64,
    pub min_margin: f64,
    pub max_replies_per_sender: usize,
    pub blocked_local_parts: Vec<String>,
    pub never_reply_classes: Vec<String>,
    pub never_forward_classes: Vec<String>,
    pub always_hold_classes: Vec<String>,
    pub bulk_headers: Vec<String>,
    pub bulk_precedence: Vec<String>,
    pub auto_header: String,
    pub max_forward_bytes: usize,
}

impl Policy {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let policy: Policy = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let problems = policy.validate();
        if problems.is_empty() {
            Ok(policy)
        } else {
            Err(format!("{path}: {}", problems.join("; ")))
        }
    }

    /// Every way the file could mislead a reader, named.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let action_ids: Vec<&str> = self.actions.iter().map(|a| a.id.as_str()).collect();
        if action_ids.is_empty() {
            problems.push("no actions declared".to_string());
        }
        let unique: BTreeSet<&str> = action_ids.iter().copied().collect();
        if unique.len() != action_ids.len() {
            problems.push("action ids repeat".to_string());
        }
        let mut folders = BTreeSet::new();
        for class in &self.classes {
            match self.costs.get(&class.id) {
                Some(row) => {
                    for action in &action_ids {
                        match row.get(*action) {
                            Some(cost) if cost.is_finite() && *cost >= 0.0 => {}
                            Some(_) => problems.push(format!(
                                "class {} has a non-positive or non-finite cost for {action}",
                                class.id
                            )),
                            None => problems
                                .push(format!("class {} has no cost for {action}", class.id)),
                        }
                    }
                }
                None => problems.push(format!("class {} has no cost row", class.id)),
            }
            match self.folders.get(&class.id) {
                Some(folder) if !folder.trim().is_empty() => {
                    if !folders.insert(folder.clone()) {
                        problems.push(format!("two classes share the folder {folder}"));
                    }
                }
                _ => problems.push(format!("class {} has no folder", class.id)),
            }
            if !class.prior.is_finite() || class.prior < 0.0 {
                problems.push(format!("class {} has a negative prior", class.id));
            }
            for (phrase, weight) in &class.cues {
                if phrase.trim().is_empty() || !weight.is_finite() || *weight <= 0.0 {
                    problems.push(format!("class {} has an empty or weightless cue", class.id));
                }
            }
        }
        for id in self.costs.keys() {
            if !self.classes.iter().any(|class| &class.id == id) {
                problems.push(format!("cost row for unknown class {id}"));
            }
        }
        for id in self.forward_to.keys() {
            if !self.classes.iter().any(|class| &class.id == id) {
                problems.push(format!("forward queue for unknown class {id}"));
            }
        }
        for id in self.replies.keys() {
            if !self.classes.iter().any(|class| &class.id == id) {
                problems.push(format!("reply template for unknown class {id}"));
            }
        }
        let guards = &self.guards;
        for (name, value) in [
            ("min_top_share", guards.min_top_share),
            ("min_margin", guards.min_margin),
        ] {
            if !(0.0..=1.0).contains(&value) {
                problems.push(format!("{name} is outside 0..1"));
            }
        }
        if guards.max_replies_per_sender == 0 {
            problems.push("max_replies_per_sender is zero, so reply is never allowed".to_string());
        }
        for class in guards
            .never_reply_classes
            .iter()
            .chain(&guards.never_forward_classes)
            .chain(&guards.always_hold_classes)
        {
            if !self.classes.iter().any(|known| &known.id == class) {
                problems.push(format!("guard names unknown class {class}"));
            }
        }
        problems
    }

    pub fn class(&self, id: &str) -> Option<&ClassSpec> {
        self.classes.iter().find(|class| class.id == id)
    }

    pub fn action_ids(&self) -> Vec<String> {
        self.actions
            .iter()
            .map(|action| action.id.clone())
            .collect()
    }

    /// The cost of doing `action` when the message really is `class`.
    pub fn cost(&self, class: &str, action: &str) -> Option<f64> {
        self.costs
            .get(class)
            .and_then(|row| row.get(action))
            .copied()
    }

    pub fn folder(&self, class: &str) -> Option<&str> {
        self.folders.get(class).map(String::as_str)
    }

    pub fn reply_template(&self, class: &str) -> Option<&str> {
        self.replies.get(class).map(String::as_str)
    }

    /// A receipt for the file the run used, so a decision can be traced to the
    /// policy that produced it even after the file changes.
    pub fn stamp(&self) -> PolicyStamp {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        PolicyStamp {
            bytes: bytes.len(),
            classes: self.classes.len(),
            actions: self.actions.len(),
            fnv64: hex64(fnv1a(&bytes)),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PolicyStamp {
    pub bytes: usize,
    pub classes: usize,
    pub actions: usize,
    pub fnv64: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> Policy {
        Policy::load(&format!("{DIR}/{POLICY}")).unwrap()
    }

    #[test]
    fn the_shipped_policy_validates() {
        assert!(shipped().validate().is_empty());
    }

    #[test]
    fn every_class_can_name_every_action_cost() {
        let policy = shipped();
        for class in &policy.classes {
            for action in policy.action_ids() {
                assert!(
                    policy.cost(&class.id, &action).is_some(),
                    "{} {}",
                    class.id,
                    action
                );
            }
        }
    }

    #[test]
    fn a_missing_cost_is_an_error_not_a_zero() {
        let mut policy = shipped();
        policy.costs.get_mut("invoice").unwrap().remove("reply");
        let problems = policy.validate();
        assert!(problems.iter().any(|p| p.contains("no cost for reply")));
    }

    #[test]
    fn two_classes_may_not_share_a_folder() {
        let mut policy = shipped();
        policy
            .folders
            .insert("invoice".to_string(), "correspondence".to_string());
        assert!(policy.validate().iter().any(|p| p.contains("share")));
    }

    #[test]
    fn the_stamp_changes_when_the_rulebook_changes() {
        let mut policy = shipped();
        let before = policy.stamp().fnv64;
        policy.guards.min_margin += 0.01;
        assert_ne!(before, policy.stamp().fnv64);
    }
}
