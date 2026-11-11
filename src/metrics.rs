//! Quality indicators for a Pareto front, and the reference fronts used with
//! them.
//!
//! Hypervolume is exact in two and three dimensions and Monte Carlo above that,
//! because the exact methods grow exponentially with the number of objectives.
//! The Monte Carlo estimate is seeded, so a report reproduces exactly; its
//! standard error is reported by [`hypervolume_report`].

use crate::benchmarks::Kind;
use crate::rng::Rng;

/// Number of samples used by the default Monte Carlo hypervolume.
pub const MONTE_CARLO_SAMPLES: usize = 200_000;

/// The five arcs of the ZDT3 Pareto front, from the roots of its defining
/// equation, as used by the standard reference sets.
pub const ZDT3_ARCS: [(f64, f64); 5] = [
    (0.0, 0.083_001_534_9),
    (0.182_228_780, 0.257_762_363_4),
    (0.409_313_674_8, 0.453_882_104_1),
    (0.618_396_794_4, 0.652_511_703_8),
    (0.823_331_798_3, 0.851_832_865_4),
];

/// Hypervolume dominated by `points` above `reference`, for minimisation.
///
/// Points are assumed to be in the box below the reference; any coordinate at
/// or above the reference makes that point's contribution empty.
pub fn hypervolume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    match reference.len() {
        2 => hypervolume_2d(points, reference),
        3 => hypervolume_3d(points, reference),
        _ => hypervolume_monte_carlo(points, reference, MONTE_CARLO_SAMPLES),
    }
}

/// Exact hypervolume for two objectives in `O(n log n)`.
pub fn hypervolume_2d(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let mut usable: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1])
        .map(|p| (p[0], p[1]))
        .collect();
    if usable.is_empty() {
        return 0.0;
    }
    usable.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut area = 0.0;
    let mut best_y = reference[1];
    for (x, y) in usable {
        if y < best_y {
            area += (reference[0] - x) * (best_y - y);
            best_y = y;
        }
    }
    area
}

/// Exact hypervolume for three objectives by slicing along the third axis.
///
/// The slice from `z_i` to the next larger `z` is covered exactly by the points
/// with `z <= z_i`, so a two dimensional sweep per slice gives the exact volume.
pub fn hypervolume_3d(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let mut usable: Vec<&Vec<f64>> = points
        .iter()
        .filter(|p| (0..3).all(|i| p[i] < reference[i]))
        .collect();
    if usable.is_empty() {
        return 0.0;
    }
    usable.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let flat: Vec<(f64, f64)> = usable.iter().map(|p| (p[0], p[1])).collect();
    let mut volume = 0.0;
    for i in 0..usable.len() {
        let z = usable[i][2];
        let z_next = if i + 1 < usable.len() {
            usable[i + 1][2]
        } else {
            reference[2]
        };
        if z_next <= z {
            continue;
        }
        let slice = &flat[..=i];
        let area = slice_area(slice, reference[0], reference[1]);
        volume += (z_next - z) * area;
    }
    volume
}

/// Sweep a 2D slice without allocating.
fn slice_area(points: &[(f64, f64)], rx: f64, ry: f64) -> f64 {
    let mut usable: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|p| p.0 < rx && p.1 < ry)
        .collect();
    if usable.is_empty() {
        return 0.0;
    }
    usable.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut area = 0.0;
    let mut best_y = ry;
    for (x, y) in usable {
        if y < best_y {
            area += (rx - x) * (best_y - y);
            best_y = y;
        }
    }
    area
}

/// Seeded Monte Carlo hypervolume for four or more objectives.
///
/// Samples are drawn uniformly in the box between the points' lower corner and
/// the reference, and the fraction dominated by any point is multiplied by the
/// box volume. Deterministic for a given seed.
pub fn hypervolume_monte_carlo(points: &[Vec<f64>], reference: &[f64], samples: usize) -> f64 {
    let dim = reference.len();
    if points.is_empty() || samples == 0 {
        return 0.0;
    }
    let mut lower = vec![f64::INFINITY; dim];
    for p in points {
        for (low, v) in lower.iter_mut().zip(p) {
            *low = low.min(*v);
        }
    }
    for low in lower.iter_mut() {
        *low = low.min(0.0);
    }
    let volume: f64 = (0..dim)
        .map(|i| (reference[i] - lower[i]).max(0.0))
        .product();
    if volume <= 0.0 {
        return 0.0;
    }
    let mut rng = Rng::new(0x5EED_1234);
    let mut hits = 0usize;
    let mut sample = vec![0.0; dim];
    for _ in 0..samples {
        for i in 0..dim {
            sample[i] = rng.uniform(lower[i], reference[i]);
        }
        if points
            .iter()
            .any(|p| p.iter().zip(&sample).all(|(a, b)| a <= b))
        {
            hits += 1;
        }
    }
    volume * hits as f64 / samples as f64
}

/// A hypervolume figure with its sampling error, for reports.
#[derive(Clone, Debug)]
pub struct Hypervolume {
    /// Estimated volume.
    pub value: f64,
    /// Half width of the 95 percent interval, zero for the exact methods.
    pub error: f64,
    /// True when the value is exact.
    pub exact: bool,
}

/// Hypervolume plus an error estimate.
pub fn hypervolume_report(points: &[Vec<f64>], reference: &[f64]) -> Hypervolume {
    if reference.len() <= 3 {
        return Hypervolume {
            value: hypervolume(points, reference),
            error: 0.0,
            exact: true,
        };
    }
    let n = MONTE_CARLO_SAMPLES as f64;
    let volume = box_volume(points, reference);
    let value = hypervolume_monte_carlo(points, reference, MONTE_CARLO_SAMPLES);
    let p = if volume > 0.0 {
        (value / volume).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Standard error of a proportion, widened to a 95 percent interval.
    let error = 1.96 * (p * (1.0 - p) / n).sqrt() * volume;
    Hypervolume {
        value,
        error,
        exact: false,
    }
}

fn box_volume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let dim = reference.len();
    let mut lower = vec![f64::INFINITY; dim];
    for p in points {
        for (low, v) in lower.iter_mut().zip(p) {
            *low = low.min(*v);
        }
    }
    (0..dim)
        .map(|i| (reference[i] - lower[i].min(0.0)).max(0.0))
        .product()
}

/// Euclidean distance between two objective vectors.
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Inverted generational distance: mean distance from each reference point to
/// the nearest point of the approximation.
///
/// Lower is better. The reference front should be dense and uniformly spread,
/// because the measure charges for every gap.
pub fn igd(approximation: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    if approximation.is_empty() || reference.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = reference
        .iter()
        .map(|r| {
            approximation
                .iter()
                .map(|a| distance(a, r))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    total / reference.len() as f64
}

/// Generational distance: mean distance from each approximation point to the
/// nearest reference point. Lower is better.
pub fn gd(approximation: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    if approximation.is_empty() || reference.is_empty() {
        return f64::INFINITY;
    }
    let total: f64 = approximation
        .iter()
        .map(|a| {
            reference
                .iter()
                .map(|r| distance(a, r))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    total / approximation.len() as f64
}

/// Spacing: standard deviation of the nearest neighbour distances inside the
/// approximation. Lower means a more even spread.
pub fn spacing(approximation: &[Vec<f64>]) -> f64 {
    if approximation.len() < 2 {
        return 0.0;
    }
    let nearest: Vec<f64> = approximation
        .iter()
        .enumerate()
        .map(|(i, a)| {
            approximation
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, b)| distance(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let mean = nearest.iter().sum::<f64>() / nearest.len() as f64;
    (nearest.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / nearest.len() as f64).sqrt()
}

/// A dense, evenly spread sample of the true Pareto front of a benchmark.
///
/// ZDT fronts are parameterised by the first variable, so sampling that
/// variable gives an exact front. ZDT3 is filtered to its non-dominated
/// segments. DTLZ fronts come from a simplex lattice, which is exact for DTLZ1
/// and lies on the unit sphere for DTLZ2.
pub fn reference_front(kind: Kind, samples: usize) -> Vec<Vec<f64>> {
    let n = samples.max(2);
    match kind {
        Kind::Zdt1 | Kind::Zdt2 | Kind::Zdt4 | Kind::Zdt6 => (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                let f1 = if kind == Kind::Zdt6 {
                    1.0 - (-4.0 * t).exp() * (6.0 * std::f64::consts::PI * t).sin().powi(6)
                } else {
                    t
                };
                let f2 = match kind {
                    Kind::Zdt2 | Kind::Zdt6 => 1.0 - f1 * f1,
                    _ => 1.0 - f1.sqrt(),
                };
                vec![f1, f2]
            })
            .collect(),
        Kind::Zdt3 => {
            // The ZDT3 front is five disconnected arcs, not one curve. Sampling
            // the parameter uniformly over [0, 1] would put points in the gaps,
            // so each arc is sampled where it actually exists.
            let per_arc = (n / ZDT3_ARCS.len()).max(2);
            let mut front = Vec::with_capacity(per_arc * ZDT3_ARCS.len());
            for &(lo, hi) in ZDT3_ARCS.iter() {
                for i in 0..per_arc {
                    let t = i as f64 / (per_arc - 1) as f64;
                    let f1 = lo + (hi - lo) * t;
                    let f2 = 1.0 - f1.sqrt() - f1 * (10.0 * std::f64::consts::PI * f1).sin();
                    front.push(vec![f1, f2]);
                }
            }
            front
        }
        Kind::Dtlz1 => das_dennis(samples, 3)
            .into_iter()
            .map(|w| w.iter().map(|v| 0.5 * v).collect())
            .collect(),
        Kind::Dtlz2 => das_dennis(samples, 3)
            .into_iter()
            .map(|w| {
                let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
                w.iter().map(|v| v / norm).collect()
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The Das and Dennis simplex lattice, the standard weight generator.
///
/// With `divisions` steps the count is `C(divisions + m - 1, m - 1)`, the
/// number of points of a regular lattice on the unit simplex.
pub fn das_dennis(divisions: usize, objectives: usize) -> Vec<Vec<f64>> {
    assert!(divisions >= 1, "need at least one division");
    assert!(objectives >= 2, "a simplex needs at least two objectives");
    let mut out = Vec::new();
    let mut current = vec![0.0; objectives];
    fill_lattice(objectives, divisions, divisions, 0, &mut current, &mut out);
    out
}

fn fill_lattice(
    objectives: usize,
    remaining: usize,
    total: usize,
    index: usize,
    current: &mut Vec<f64>,
    out: &mut Vec<Vec<f64>>,
) {
    if index == objectives - 1 {
        current[index] = remaining as f64 / total as f64;
        out.push(current.clone());
        return;
    }
    for i in 0..=remaining {
        current[index] = i as f64 / total as f64;
        fill_lattice(objectives, remaining - i, total, index + 1, current, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn two_dimensional_hypervolume_matches_the_boxes() {
        let hv = hypervolume_2d(&[vec![1.0, 1.0]], &[2.0, 2.0]);
        assert_relative_eq!(hv, 1.0);
        let hv = hypervolume_2d(&[vec![0.5, 1.0], vec![1.0, 0.5]], &[2.0, 2.0]);
        assert_relative_eq!(hv, 2.0);
        assert_eq!(hypervolume_2d(&[vec![3.0, 3.0]], &[2.0, 2.0]), 0.0);
    }

    #[test]
    fn three_dimensional_hypervolume_matches_the_union() {
        let hv = hypervolume_3d(&[vec![1.0, 1.0, 1.0]], &[2.0, 2.0, 2.0]);
        assert_relative_eq!(hv, 1.0);
        let hv = hypervolume_3d(
            &[vec![0.5, 0.5, 1.0], vec![1.0, 1.0, 0.5]],
            &[2.0, 2.0, 2.0],
        );
        // 0.5 * 1 * 1 + 1 * 2.25 = 2.75
        assert_relative_eq!(hv, 2.75);
    }

    #[test]
    fn monte_carlo_agrees_with_the_exact_method() {
        let points = vec![
            vec![0.2, 0.6, 0.3],
            vec![0.5, 0.2, 0.4],
            vec![0.3, 0.3, 0.8],
        ];
        let reference = [1.0, 1.0, 1.0];
        let exact = hypervolume_3d(&points, &reference);
        let mc = hypervolume_monte_carlo(&points, &reference, 400_000);
        assert!((exact - mc).abs() < 0.01, "exact {exact} mc {mc}");
    }

    #[test]
    fn igd_is_zero_on_an_identical_front() {
        let front = vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![1.0, 0.0]];
        assert_relative_eq!(igd(&front, &front), 0.0);
        let far = vec![vec![2.0, 2.0]];
        assert!(igd(&far, &front) > 1.0);
    }

    #[test]
    fn spacing_is_zero_on_a_uniform_line() {
        let front: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 4.0 - i as f64]).collect();
        assert_relative_eq!(spacing(&front), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn das_dennis_count_is_the_simplex_lattice() {
        assert_eq!(das_dennis(4, 2).len(), 5);
        assert_eq!(das_dennis(4, 3).len(), 15);
        assert_eq!(das_dennis(12, 3).len(), 91);
        for w in das_dennis(6, 3) {
            assert_relative_eq!(w.iter().sum::<f64>(), 1.0);
        }
    }

    #[test]
    fn zdt1_reference_lies_on_its_curve() {
        for p in reference_front(Kind::Zdt1, 50) {
            assert_relative_eq!(p[1], 1.0 - p[0].sqrt(), epsilon = 1e-12);
        }
    }

    #[test]
    fn dtlz2_reference_lies_on_the_unit_sphere() {
        for p in reference_front(Kind::Dtlz2, 91) {
            let r = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_relative_eq!(r, 1.0, epsilon = 1e-12);
        }
    }
}
