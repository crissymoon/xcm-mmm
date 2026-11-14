//! k-means over standardised behaviour, with the audit numbers beside it.
//!
//! Seeding is k-means++: the first centre is drawn from the entities, every
//! later centre from the squared distance still unexplained. Lloyd's loop then
//! alternates assignment and the mean of each cluster. An empty cluster is
//! repaired rather than dropped, by moving the centre to the entity the current
//! partition represents worst, so a fit never returns fewer groups than asked.
//!
//! Every draw comes from the crate's own generator and every tie breaks on the
//! lower index, so a fit is reproducible bit for bit and a shuffled control runs
//! under exactly the same protocol as the fit it is compared with.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// One converged partition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub k: usize,
    pub centroids: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub inertia: f64,
}

/// What one value of k scored, for the selection table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sweep {
    pub k: usize,
    pub silhouette: f64,
    pub inertia: f64,
    pub sizes: Vec<usize>,
    pub min_share: f64,
}

/// Squared euclidean distance.
pub fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The nearest centroid and its squared distance. Ties go to the lower index.
pub fn nearest(point: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (i, centre) in centroids.iter().enumerate() {
        let d = dist2(point, centre);
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

/// k-means++ seeding. Falls back to a uniform draw when every point already
/// sits on a centre, which happens only when k exceeds the distinct rows.
fn seed_centres(rows: &[Vec<f64>], k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut centres = Vec::with_capacity(k);
    centres.push(rows[rng.below(n)].clone());
    let mut d2: Vec<f64> = rows.iter().map(|r| dist2(r, &centres[0])).collect();
    while centres.len() < k {
        let total: f64 = d2.iter().sum();
        let pick = if total <= 0.0 {
            rng.below(n)
        } else {
            let target = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut chosen = n - 1;
            for (i, value) in d2.iter().enumerate() {
                acc += value;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let centre = rows[pick].clone();
        for (i, row) in rows.iter().enumerate() {
            d2[i] = d2[i].min(dist2(row, &centre));
        }
        centres.push(centre);
    }
    centres
}

/// One run of Lloyd's algorithm from one seeding.
fn lloyd(rows: &[Vec<f64>], mut centres: Vec<Vec<f64>>, iterations: usize) -> (Vec<usize>, f64) {
    let n = rows.len();
    let k = centres.len();
    let dim = rows[0].len();
    let mut labels = vec![usize::MAX; n];
    for _ in 0..iterations.max(1) {
        let mut changed = false;
        for i in 0..n {
            let (label, _) = nearest(&rows[i], &centres);
            if labels[i] != label {
                labels[i] = label;
                changed = true;
            }
        }
        let mut counts = vec![0usize; k];
        for &label in &labels {
            counts[label] += 1;
        }
        for cluster in 0..k {
            if counts[cluster] > 0 {
                continue;
            }
            // Move an entity from the largest cluster into the empty one. The
            // donor keeps at least one member, so the repair cannot empty a
            // second group, and a fit never reports a group it did not find.
            let donor = (0..k)
                .filter(|&c| counts[c] > 1)
                .max_by_key(|&c| counts[c])
                .unwrap_or(0);
            let mut worst = 0usize;
            let mut worst_d = -1.0;
            for i in 0..n {
                if labels[i] != donor {
                    continue;
                }
                let d = dist2(&rows[i], &centres[donor]);
                if d > worst_d {
                    worst_d = d;
                    worst = i;
                }
            }
            counts[donor] -= 1;
            labels[worst] = cluster;
            counts[cluster] += 1;
            changed = true;
        }
        let mut sums = vec![vec![0.0; dim]; k];
        for i in 0..n {
            let label = labels[i];
            for (d, value) in rows[i].iter().enumerate() {
                sums[label][d] += value;
            }
        }
        for cluster in 0..k {
            if counts[cluster] == 0 {
                continue;
            }
            for d in 0..dim {
                centres[cluster][d] = sums[cluster][d] / counts[cluster] as f64;
            }
        }
        if !changed {
            break;
        }
    }
    let inertia = (0..n).map(|i| dist2(&rows[i], &centres[labels[i]])).sum();
    (labels, inertia)
}

/// Fit k groups from `restarts` seedings and keep the lowest inertia.
pub fn fit(rows: &[Vec<f64>], k: usize, restarts: usize, iterations: usize, seed: u64) -> Fit {
    assert!(!rows.is_empty(), "fit on an empty matrix");
    assert!(k >= 1 && k <= rows.len(), "k out of range");
    let mut best: Option<Fit> = None;
    for run in 0..restarts.max(1) {
        let mut rng = Rng::stream(seed, run as u64);
        let centres = seed_centres(rows, k, &mut rng);
        let (labels, inertia) = lloyd(rows, centres.clone(), iterations);
        let better = match &best {
            None => true,
            Some(current) => inertia < current.inertia - 1e-12,
        };
        if better {
            // The centroid of a cluster is the mean of its members, which is
            // what Lloyd left in `centres`. Recover it exactly for the artifact.
            let dim = rows[0].len();
            let mut sums = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, label) in labels.iter().enumerate() {
                counts[*label] += 1;
                for (d, value) in rows[i].iter().enumerate() {
                    sums[*label][d] += value;
                }
            }
            let mut means = vec![vec![0.0; dim]; k];
            for cluster in 0..k {
                if counts[cluster] == 0 {
                    means[cluster] = centres[cluster].clone();
                    continue;
                }
                means[cluster] = sums[cluster]
                    .iter()
                    .map(|sum| sum / counts[cluster] as f64)
                    .collect();
            }
            best = Some(Fit {
                k,
                centroids: means,
                labels,
                inertia,
            });
        }
    }
    best.expect("at least one restart")
}

/// Mean silhouette over every entity, in [-1, 1].
///
/// A point alone in its cluster scores zero, and so does a point whose two
/// nearest clusters are the same distance away. That convention is stated here
/// because a single entity cluster would otherwise inflate the mean.
pub fn silhouette(rows: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
    let n = rows.len();
    if n < 2 || k < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let own = labels[i];
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            sums[labels[j]] += dist2(&rows[i], &rows[j]).sqrt();
            counts[labels[j]] += 1;
        }
        if counts[own] == 0 {
            continue;
        }
        let a = sums[own] / counts[own] as f64;
        let mut b = f64::INFINITY;
        for cluster in 0..k {
            if cluster == own || counts[cluster] == 0 {
                continue;
            }
            let mean = sums[cluster] / counts[cluster] as f64;
            if mean < b {
                b = mean;
            }
        }
        if !b.is_finite() {
            continue;
        }
        let denom = a.max(b);
        if denom > 0.0 {
            total += (b - a) / denom;
        }
    }
    total / n as f64
}

/// Assignment of every row to its nearest centroid.
pub fn assign(rows: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    rows.iter().map(|row| nearest(row, centroids).0).collect()
}

/// Cluster sizes, in cluster id order.
pub fn sizes(labels: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &label in labels {
        if label < k {
            counts[label] += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(cx: f64, cy: f64, n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.37;
                vec![cx + 0.05 * t.sin(), cy + 0.05 * t.cos()]
            })
            .collect()
    }

    fn rows() -> Vec<Vec<f64>> {
        let mut all = blob(0.0, 0.0, 20);
        all.extend(blob(6.0, 0.0, 20));
        all.extend(blob(0.0, 6.0, 20));
        all
    }

    #[test]
    fn three_separated_blobs_are_found_from_every_seed() {
        let data = rows();
        for seed in 1..8 {
            let got = fit(&data, 3, 8, 100, seed);
            assert_eq!(sizes(&got.labels, 3), vec![20, 20, 20]);
            assert!(silhouette(&data, &got.labels, 3) > 0.8);
        }
    }

    #[test]
    fn a_fit_is_reproducible_and_returns_the_number_of_groups_asked_for() {
        let data = rows();
        let a = fit(&data, 3, 8, 100, 42);
        let b = fit(&data, 3, 8, 100, 42);
        assert_eq!(a, b);
        assert_eq!(a.centroids.len(), 3);
        assert_eq!(a.labels.len(), data.len());
    }

    #[test]
    fn one_cluster_scores_zero_silhouette_and_assigns_to_zero() {
        let data = rows();
        let flat = fit(&data, 1, 2, 50, 7);
        assert_eq!(silhouette(&data, &flat.labels, 1), 0.0);
        assert!(assign(&data, &flat.centroids).iter().all(|&l| l == 0));
    }

    #[test]
    fn the_worst_represented_entity_repairs_an_empty_cluster() {
        // Two identical rows and k of three would collapse a centre; the repair
        // has to keep three groups and leave every row assigned.
        let data = vec![vec![0.0], vec![1.0], vec![1.0]];
        let got = fit(&data, 3, 2, 20, 3);
        assert_eq!(got.labels.len(), 3);
        assert!(got.labels.iter().all(|&l| l < 3));
        assert_eq!(sizes(&got.labels, 3).iter().sum::<usize>(), 3);
    }

    #[test]
    fn silhouette_falls_when_the_partition_is_wrong() {
        let data = rows();
        let good = fit(&data, 3, 8, 100, 1);
        let wrong: Vec<usize> = (0..data.len()).map(|i| i % 3).collect();
        assert!(silhouette(&data, &good.labels, 3) > silhouette(&data, &wrong, 3));
    }

    #[test]
    fn nearest_breaks_ties_on_the_lower_index() {
        let centres = vec![vec![1.0], vec![-1.0]];
        assert_eq!(nearest(&[0.0], &centres).0, 0);
    }
}
