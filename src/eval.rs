//! Pareto tools: dominance, non-dominated sorting, crowding distance and a
//! bounded non-dominated archive.

use rayon::prelude::*;

use crate::problem::{Problem, Score};

/// Tolerance used when comparing objectives, so that floating point noise does
/// not turn one point into two.
pub const EPS: f64 = 1e-12;

/// True when `a` Pareto dominates `b` for minimisation.
///
/// Dominance needs `a` to be no worse on every objective and strictly better on
/// at least one.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut strictly_better = false;
    for (x, y) in a.iter().zip(b) {
        if x > &(y + EPS) {
            return false;
        }
        if x < &(y - EPS) {
            strictly_better = true;
        }
    }
    strictly_better
}

/// True when two objective vectors are equal within [`EPS`].
pub fn same_point(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= EPS)
}

/// Evaluate a whole population in parallel.
///
/// Only evaluation is parallel, never the evolution step, so the result is
/// identical at any thread count.
pub fn evaluate_population<P: Problem + ?Sized>(
    problem: &P,
    population: &[Vec<f64>],
) -> Vec<Score> {
    population
        .par_iter()
        .map(|x| problem.score(x))
        .collect::<Vec<_>>()
}

/// Fast non-dominated sorting, the Deb et al. 2002 procedure.
///
/// Returns front indices, front 0 first. Each front is ordered by the input
/// order, which the caller can sort further.
pub fn non_dominated_sort(points: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut count = vec![0usize; n];
    let mut first = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&points[i], &points[j]) {
                dominated_by[i].push(j);
                count[j] += 1;
            } else if dominates(&points[j], &points[i]) {
                dominated_by[j].push(i);
                count[i] += 1;
            }
        }
    }
    for (i, &c) in count.iter().enumerate() {
        if c == 0 {
            first.push(i);
        }
    }

    let mut fronts = Vec::new();
    let mut current = first;
    while !current.is_empty() {
        let mut next = Vec::new();
        for &p in &current {
            for &q in &dominated_by[p] {
                count[q] -= 1;
                if count[q] == 0 {
                    next.push(q);
                }
            }
        }
        fronts.push(current);
        current = next;
    }
    fronts
}

/// Crowding distance of each point inside one front, the NSGA-II measure.
///
/// Boundary points receive infinity so that they survive truncation.
// The column index is the sort key and also indexes every point, so the plain
// range is clearer than an iterator here.
#[allow(clippy::needless_range_loop)]
pub fn crowding_distance(front: &[Vec<f64>]) -> Vec<f64> {
    let n = front.len();
    let mut distance = vec![0.0; n];
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }
    let m = front[0].len();
    for k in 0..m {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| front[a][k].partial_cmp(&front[b][k]).unwrap());
        let lo = front[order[0]][k];
        let hi = front[order[n - 1]][k];
        distance[order[0]] = f64::INFINITY;
        distance[order[n - 1]] = f64::INFINITY;
        let span = hi - lo;
        if span <= 0.0 {
            continue;
        }
        for w in 1..(n - 1) {
            let prev = front[order[w - 1]][k];
            let next = front[order[w + 1]][k];
            distance[order[w]] += (next - prev) / span;
        }
    }
    distance
}

/// A bounded set of mutually non-dominated points.
///
/// New points are rejected when any member dominates them, and members that the
/// new point dominates are dropped. When the capacity is reached the densest
/// members go first, measured by crowding distance.
#[derive(Clone, Debug)]
pub struct Archive {
    points: Vec<Vec<f64>>,
    capacity: usize,
}

impl Archive {
    /// An archive holding at most `capacity` points.
    pub fn new(capacity: usize) -> Self {
        Self {
            points: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// Current members.
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Consume the archive.
    pub fn into_points(self) -> Vec<Vec<f64>> {
        self.points
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Offer one objective vector.
    ///
    /// Returns true when the archive changed.
    pub fn offer(&mut self, candidate: Vec<f64>) -> bool {
        if self.points.iter().any(|p| same_point(p, &candidate)) {
            return false;
        }
        if self.points.iter().any(|p| dominates(p, &candidate)) {
            return false;
        }
        self.points.retain(|p| !dominates(&candidate, p));
        self.points.push(candidate);
        if self.points.len() > self.capacity {
            self.prune();
        }
        // Reaching this line always means a new member was inserted, even when
        // pruning brought the length back to where it started.
        true
    }

    /// Add every non-dominated member of a population.
    pub fn extend<I: IntoIterator<Item = Vec<f64>>>(&mut self, points: I) {
        for p in points {
            self.offer(p);
        }
    }

    fn prune(&mut self) {
        let distances = crowding_distance(&self.points);
        let mut order: Vec<usize> = (0..self.points.len()).collect();
        order.sort_by(|&a, &b| {
            distances[b]
                .partial_cmp(&distances[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(self.capacity);
        order.sort_unstable();
        let kept: Vec<Vec<f64>> = order.into_iter().map(|i| self.points[i].clone()).collect();
        self.points = kept;
    }

    /// Sort the members by the first objective, useful for reports.
    pub fn sorted(mut self) -> Vec<Vec<f64>> {
        self.points
            .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        self.points
    }
}

/// Remove dominated points from a set, keeping input order.
pub fn pareto_filter(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut keep = vec![true; points.len()];
    for i in 0..points.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..points.len() {
            if i == j || !keep[j] {
                continue;
            }
            if dominates(&points[j], &points[i]) || (same_point(&points[i], &points[j]) && j < i) {
                keep[i] = false;
                break;
            }
        }
    }
    points
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| p.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_is_strict_somewhere() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 1.0], &[1.0, 2.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0]));
    }

    #[test]
    fn sorting_splits_a_known_front() {
        let points = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
            vec![3.0, 3.0],
        ];
        let fronts = non_dominated_sort(&points);
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0].len(), 4);
        assert_eq!(fronts[1], vec![4]);
    }

    #[test]
    fn archive_rejects_dominated_and_limits_size() {
        let mut a = Archive::new(3);
        assert!(a.offer(vec![1.0, 3.0]));
        assert!(a.offer(vec![2.0, 2.0]));
        assert!(!a.offer(vec![3.0, 3.0]));
        assert!(a.offer(vec![0.5, 4.0]));
        assert!(a.offer(vec![3.0, 1.0]));
        assert!(a.len() <= 3);
        assert!(a.points().iter().all(|p| p.len() == 2));
    }

    #[test]
    fn crowding_marks_the_edges() {
        let front = vec![
            vec![1.0, 4.0],
            vec![2.0, 2.5],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let d = crowding_distance(&front);
        assert!(d[0].is_infinite() && d[3].is_infinite());
        assert!(d[1] > 0.0 && d[2] > 0.0);
    }
}
