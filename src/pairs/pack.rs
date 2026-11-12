//! Merging pair files into one corpus with a chosen task mix.
//!
//! Four filters run before the mix, in order of cost:
//!
//! * a row without both a task and a text is malformed and is dropped;
//! * two rows with the same task and the same document ids are one row;
//! * two sides made of the same words, or of nearly the same words, teach
//!   nothing and are dropped;
//! * a task over its quota is sampled down, not truncated from the front, so a
//!   file sorted by date does not become a file of one month.
//!
//! The mix itself is proportional to the weight per task. When a task cannot
//! fill its quota the shortfall is spread over the tasks that still have rows,
//! and the report says the realised share next to the requested one, because a
//! requested 2:1:1 that arrives as 2:1:0.2 is worth knowing before training.

use std::collections::{HashMap, HashSet};

use crate::pairs::{self, Pair, Task};
use crate::rng::Rng;
use crate::text::hash;
use crate::text::tokenize;

/// What to keep.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Total rows. Zero keeps every row that survives the filters.
    pub total: usize,
    /// Weight per task, in any scale.
    pub weights: Vec<(Task, f64)>,
    /// Pairs whose two sides are more similar than this are dropped.
    pub max_similarity: f64,
    pub seed: u64,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            total: 0,
            weights: vec![
                (Task::Rank, 3.0),
                (Task::Preference, 2.0),
                (Task::Contrast, 2.0),
                (Task::Label, 1.0),
                (Task::Correction, 1.0),
                (Task::Retrieval, 1.0),
            ],
            max_similarity: 0.98,
            seed: 5,
        }
    }
}

impl Plan {
    /// Weight of one task, zero when it is not listed.
    pub fn weight_of(&self, task: Task) -> f64 {
        self.weights
            .iter()
            .find(|(t, _)| *t == task)
            .map(|(_, w)| *w)
            .unwrap_or(0.0)
            .max(0.0)
    }
}

/// What a merge did.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub files: usize,
    pub read: usize,
    pub malformed: usize,
    pub duplicate_pairs: usize,
    pub trivial: usize,
    pub kept: usize,
    pub by_task: Vec<(Task, usize)>,
    pub by_split: Vec<(String, usize)>,
    /// Realised share per task, in [0, 1].
    pub realised: Vec<(Task, f64)>,
}

fn sides(text: &str) -> Vec<u64> {
    let words: Vec<u64> = tokenize::words(text).map(hash::word_hash).collect();
    hash::shingles(&words, 1)
}

fn trivial(pair: &Pair, max_similarity: f64) -> bool {
    let Some(b) = &pair.text_b else {
        return false;
    };
    if pair.text_a.eq_ignore_ascii_case(b) {
        return true;
    }
    let a = sides(&pair.text_a);
    let b = sides(b);
    hash::jaccard(&a, &b) >= max_similarity
}

/// Read and merge pair files.
pub fn merge(inputs: &[String], plan: &Plan) -> std::io::Result<(Vec<Pair>, Report)> {
    let mut report = Report {
        files: inputs.len(),
        ..Report::default()
    };
    let mut rows: Vec<Pair> = Vec::new();
    for path in inputs {
        let mut bad = 0usize;
        crate::text::doc::for_each_line(path, |line| match pairs::from_line(line) {
            Some(pair) if !pair.text_a.trim().is_empty() => rows.push(pair),
            _ => bad += 1,
        })?;
        report.malformed += bad;
    }
    report.read = rows.len();

    let mut seen: HashSet<String> = HashSet::with_capacity(rows.len());
    let mut kept: Vec<Pair> = Vec::with_capacity(rows.len());
    for mut pair in rows {
        if pair.a.is_empty() {
            pair.a = hash::hex64(hash::text_hash(&pair.text_a));
        }
        pair.stamp();
        if !seen.insert(pair.id.clone()) {
            report.duplicate_pairs += 1;
            continue;
        }
        if trivial(&pair, plan.max_similarity) {
            report.trivial += 1;
            continue;
        }
        kept.push(pair);
    }

    let mut by_task: HashMap<Task, Vec<Pair>> = HashMap::new();
    for pair in kept {
        by_task.entry(pair.task).or_default().push(pair);
    }
    let total_available: usize = by_task.values().map(Vec::len).sum();
    let weight_sum: f64 = Task::ALL.iter().map(|t| plan.weight_of(*t)).sum();
    let mut rng = Rng::new(plan.seed);
    let mut out: Vec<Pair> = Vec::new();

    if plan.total == 0 || plan.total >= total_available || weight_sum <= 0.0 {
        for list in by_task.into_values() {
            out.extend(list);
        }
    } else {
        let mut quota: HashMap<Task, usize> = HashMap::new();
        for task in Task::ALL {
            let available = by_task.get(&task).map(Vec::len).unwrap_or(0);
            let share = plan.weight_of(task) / weight_sum;
            let want = ((plan.total as f64 * share).round() as usize).min(available);
            quota.insert(task, want);
        }
        // Spread a shortfall over the tasks that still have rows, largest
        // remainder first, until the budget is used or nothing is left.
        let mut assigned: usize = quota.values().sum();
        let mut guard = 0;
        while assigned < plan.total && guard < 1000 {
            guard += 1;
            let mut progressed = false;
            for task in Task::ALL {
                if assigned >= plan.total {
                    break;
                }
                let available = by_task.get(&task).map(Vec::len).unwrap_or(0);
                let have = quota.get(&task).copied().unwrap_or(0);
                if have < available {
                    quota.insert(task, have + 1);
                    assigned += 1;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        for (task, mut list) in by_task {
            let want = quota.get(&task).copied().unwrap_or(0);
            if want == 0 {
                continue;
            }
            if list.len() > want {
                rng.shuffle(&mut list);
                list.truncate(want);
            }
            out.extend(list);
        }
    }

    out.sort_by(|a, b| {
        a.task
            .cmp(&b.task)
            .then(a.split.cmp(&b.split))
            .then(a.id.cmp(&b.id))
    });
    report.kept = out.len();
    let mut task_counts: HashMap<Task, usize> = HashMap::new();
    let mut split_counts: HashMap<String, usize> = HashMap::new();
    for pair in &out {
        *task_counts.entry(pair.task).or_insert(0) += 1;
        *split_counts.entry(pair.split.clone()).or_insert(0) += 1;
    }
    report.by_task = Task::ALL
        .iter()
        .map(|t| (*t, task_counts.get(t).copied().unwrap_or(0)))
        .filter(|(_, n)| *n > 0)
        .collect();
    for task in Task::ALL {
        let count = task_counts.get(&task).copied().unwrap_or(0);
        report
            .realised
            .push((task, count as f64 / out.len().max(1) as f64));
    }
    let mut splits: Vec<(String, usize)> = split_counts.into_iter().collect();
    splits.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    report.by_split = splits;
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(task: Task, a: &str, text: &str, split: &str) -> Pair {
        let mut p = Pair::new(task, split);
        p.a = a.into();
        p.text_a = text.into();
        p.stamp();
        p
    }

    #[test]
    fn trivial_pairs_are_recognised() {
        let mut p = pair(
            Task::Rank,
            "a",
            "Fed holds rates steady as inflation cools",
            "train",
        );
        p.text_b = Some("Fed holds rates steady as inflation cools".into());
        assert!(trivial(&p, 0.98));
        p.text_b = Some("Airline cancels flights as a storm closes airports".into());
        assert!(!trivial(&p, 0.98));
    }

    #[test]
    fn quota_spreads_a_shortfall() {
        let plan = Plan {
            total: 4,
            weights: vec![(Task::Rank, 1.0), (Task::Contrast, 1.0)],
            ..Plan::default()
        };
        assert_eq!(plan.weight_of(Task::Rank), 1.0);
        assert_eq!(plan.weight_of(Task::Label), 0.0);
    }
}
