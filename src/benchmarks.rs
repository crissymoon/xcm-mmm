//! Standard benchmark problems.
//!
//! The single objective set (sphere, Rosenbrock, Rastrigin, Ackley) checks that
//! the differential evolution engine converges and gives the timing figures. The
//! multi-objective set (ZDT1 to ZDT4, ZDT6, DTLZ1, DTLZ2) is the standard one
//! from the evolutionary literature, so the numbers in
//! `README/DOCS/BENCHMARKS.md` can be compared with published results.
//!
//! Every problem is deterministic and side effect free, and the bounds are
//! stored with the instance so that no global state is involved.

use crate::problem::{Bounds, Problem};

/// Which benchmark formula to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Convex, separable, minimum zero at the origin.
    Sphere,
    /// Narrow curved valley, minimum zero at all ones.
    Rosenbrock,
    /// Many local minima on a regular lattice, minimum zero at the origin.
    Rastrigin,
    /// Flat outer region with one deep well, minimum zero at the origin.
    Ackley,
    /// Convex Pareto front, the easiest ZDT instance.
    Zdt1,
    /// Concave Pareto front.
    Zdt2,
    /// Disconnected Pareto front.
    Zdt3,
    /// Many local fronts, a hard multimodal two objective problem.
    Zdt4,
    /// Non-uniform density along the front.
    Zdt6,
    /// Linear front on the simplex, with a deceptive barrier.
    Dtlz1,
    /// Spherical front on the positive orthant.
    Dtlz2,
}

impl Kind {
    /// The name the CLI and the reports use.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Sphere => "sphere",
            Kind::Rosenbrock => "rosenbrock",
            Kind::Rastrigin => "rastrigin",
            Kind::Ackley => "ackley",
            Kind::Zdt1 => "zdt1",
            Kind::Zdt2 => "zdt2",
            Kind::Zdt3 => "zdt3",
            Kind::Zdt4 => "zdt4",
            Kind::Zdt6 => "zdt6",
            Kind::Dtlz1 => "dtlz1",
            Kind::Dtlz2 => "dtlz2",
        }
    }

    /// Number of objectives.
    pub fn objectives(self) -> usize {
        match self {
            Kind::Zdt1 | Kind::Zdt2 | Kind::Zdt3 | Kind::Zdt4 | Kind::Zdt6 => 2,
            Kind::Dtlz1 | Kind::Dtlz2 => 3,
            _ => 1,
        }
    }

    /// Lowest dimension at which the problem is defined.
    pub fn minimum_dim(self) -> usize {
        match self {
            Kind::Dtlz1 | Kind::Dtlz2 => 3,
            _ => 2,
        }
    }
}

/// One benchmark instance.
#[derive(Clone, Debug)]
pub struct Benchmark {
    kind: Kind,
    dim: usize,
    bounds: Bounds,
}

/// Catalogue entries: name, default dimension, description.
pub const CATALOGUE: &[(&str, usize, &str)] = &[
    ("sphere", 30, "convex bowl, minimum 0 at the origin"),
    ("rosenbrock", 30, "curved valley, minimum 0 at all ones"),
    ("rastrigin", 30, "regular lattice of local minima"),
    ("ackley", 30, "flat rim with one deep well"),
    ("zdt1", 30, "convex Pareto front"),
    ("zdt2", 30, "concave Pareto front"),
    ("zdt3", 30, "disconnected Pareto front"),
    ("zdt4", 10, "multimodal, many local fronts"),
    ("zdt6", 10, "non-uniform density along the front"),
    ("dtlz1", 7, "linear front, deceptive barrier"),
    ("dtlz2", 12, "spherical front on the positive orthant"),
];

impl Benchmark {
    /// Build a benchmark by name at the given dimension.
    ///
    /// Returns `None` for an unknown name and for a dimension the problem
    /// cannot take.
    pub fn named(name: &str, dim: usize) -> Option<Self> {
        let kind = match name.to_ascii_lowercase().as_str() {
            "sphere" => Kind::Sphere,
            "rosenbrock" => Kind::Rosenbrock,
            "rastrigin" => Kind::Rastrigin,
            "ackley" => Kind::Ackley,
            "zdt1" => Kind::Zdt1,
            "zdt2" => Kind::Zdt2,
            "zdt3" => Kind::Zdt3,
            "zdt4" => Kind::Zdt4,
            "zdt6" => Kind::Zdt6,
            "dtlz1" => Kind::Dtlz1,
            "dtlz2" => Kind::Dtlz2,
            _ => return None,
        };
        Self::build(kind, dim)
    }

    /// Build a known kind for the CLI.
    pub fn build(kind: Kind, dim: usize) -> Option<Self> {
        if dim < kind.minimum_dim() {
            return None;
        }
        let bounds = if kind == Kind::Zdt4 {
            // The first variable stays in [0, 1] and the rest in [-5, 5],
            // which is where the multimodality comes from.
            let mut lower = vec![-5.0; dim];
            let mut upper = vec![5.0; dim];
            lower[0] = 0.0;
            upper[0] = 1.0;
            Bounds::new(lower, upper)
        } else {
            Bounds::unit(dim)
        };
        Some(Self { kind, dim, bounds })
    }

    /// Which formula this instance uses.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// True for a problem with more than one objective.
    pub fn is_multi_objective(&self) -> bool {
        self.kind.objectives() > 1
    }
}

impl Problem for Benchmark {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn objectives(&self) -> usize {
        self.kind.objectives()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], _g: &mut [f64]) {
        match self.kind {
            Kind::Sphere => {
                f[0] = x.iter().map(|v| v * v).sum();
            }
            Kind::Rosenbrock => {
                let mut sum = 0.0;
                for i in 0..(x.len() - 1) {
                    let a = 1.0 - x[i];
                    let b = x[i + 1] - x[i] * x[i];
                    sum += 100.0 * b * b + a * a;
                }
                f[0] = sum;
            }
            Kind::Rastrigin => {
                let n = x.len() as f64;
                f[0] = 10.0 * n
                    + x.iter()
                        .map(|v| v * v - 10.0 * (std::f64::consts::TAU * v).cos())
                        .sum::<f64>();
            }
            Kind::Ackley => {
                let n = x.len() as f64;
                let sum_sq = x.iter().map(|v| v * v).sum::<f64>() / n;
                let sum_cos = x
                    .iter()
                    .map(|v| (std::f64::consts::TAU * v).cos())
                    .sum::<f64>()
                    / n;
                f[0] = -20.0 * (-0.2 * sum_sq.sqrt()).exp() - sum_cos.exp()
                    + std::f64::consts::E
                    + 20.0;
            }
            Kind::Zdt1 => {
                let (f1, g) = zdt_head(x);
                f[0] = f1;
                f[1] = g * (1.0 - (f1 / g).sqrt());
            }
            Kind::Zdt2 => {
                let (f1, g) = zdt_head(x);
                f[0] = f1;
                f[1] = g * (1.0 - (f1 / g).powi(2));
            }
            Kind::Zdt3 => {
                let (f1, g) = zdt_head(x);
                let s = (10.0 * std::f64::consts::PI * f1).sin();
                f[0] = f1;
                f[1] = g * (1.0 - (f1 / g).sqrt() - f1 / g * s);
            }
            Kind::Zdt4 => {
                let f1 = x[0];
                let g = 1.0
                    + 10.0 * (x.len() - 1) as f64
                    + x[1..]
                        .iter()
                        .map(|v| v * v - 10.0 * (4.0 * std::f64::consts::PI * v).cos())
                        .sum::<f64>();
                f[0] = f1;
                f[1] = g * (1.0 - (f1 / g).sqrt());
            }
            Kind::Zdt6 => {
                let f1 =
                    1.0 - (-4.0 * x[0]).exp() * (6.0 * std::f64::consts::PI * x[0]).sin().powi(6);
                let g = 1.0 + 9.0 * (x[1..].iter().sum::<f64>() / (x.len() - 1) as f64).powf(0.25);
                f[0] = f1;
                f[1] = g * (1.0 - (f1 / g).powi(2));
            }
            Kind::Dtlz1 => dtlz1(x, f, self.objectives()),
            Kind::Dtlz2 => dtlz2(x, f, self.objectives()),
        }
    }
}

fn zdt_head(x: &[f64]) -> (f64, f64) {
    let f1 = x[0];
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    (f1, g)
}

fn dtlz1(x: &[f64], f: &mut [f64], m: usize) {
    let k = x.len() - m + 1;
    let tail = &x[x.len() - k..];
    // The cosine term is minus one at the midpoint, so the leading k cancels
    // there and g reaches zero exactly on the front.
    let g = 100.0
        * (k as f64
            + tail
                .iter()
                .map(|v| {
                    let d = v - 0.5;
                    d * d - (20.0 * std::f64::consts::PI * d).cos()
                })
                .sum::<f64>());
    for i in 0..m {
        let mut value = 0.5 * (1.0 + g);
        for &v in &x[..(m - 1 - i)] {
            value *= v;
        }
        if i > 0 {
            value *= 1.0 - x[m - 1 - i];
        }
        f[i] = value;
    }
}

fn dtlz2(x: &[f64], f: &mut [f64], m: usize) {
    let k = x.len() - m + 1;
    let g: f64 = x[x.len() - k..].iter().map(|v| (v - 0.5) * (v - 0.5)).sum();
    for i in 0..m {
        let mut value = 1.0 + g;
        for &v in &x[..(m - 1 - i)] {
            value *= (v * std::f64::consts::FRAC_PI_2).cos();
        }
        if i > 0 {
            value *= (x[m - 1 - i] * std::f64::consts::FRAC_PI_2).sin();
        }
        f[i] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn eval(b: &Benchmark, x: &[f64]) -> Vec<f64> {
        let mut f = vec![0.0; b.objectives()];
        let mut g = vec![0.0; b.constraints()];
        b.evaluate(x, &mut f, &mut g);
        f
    }

    #[test]
    fn sphere_is_zero_at_origin() {
        let b = Benchmark::named("sphere", 5).unwrap();
        assert_relative_eq!(eval(&b, &[0.0; 5])[0], 0.0);
        assert_relative_eq!(eval(&b, &[1.0; 5])[0], 5.0);
    }

    #[test]
    fn rosenbrock_is_zero_at_ones() {
        let b = Benchmark::named("rosenbrock", 4).unwrap();
        assert_relative_eq!(eval(&b, &[1.0; 4])[0], 0.0, epsilon = 1e-12);
        assert!(eval(&b, &[0.0; 4])[0] > 0.0);
    }

    #[test]
    fn zdt1_front_matches_the_formula() {
        let b = Benchmark::named("zdt1", 5).unwrap();
        let f = eval(&b, &[0.25, 0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(f[0], 0.25);
        assert_relative_eq!(f[1], 1.0 - 0.5);
    }

    #[test]
    fn zdt4_bounds_are_mixed() {
        let b = Benchmark::named("zdt4", 6).unwrap();
        let bounds = b.bounds();
        assert_eq!(bounds.lower(0), 0.0);
        assert_eq!(bounds.upper(0), 1.0);
        assert_eq!(bounds.lower(1), -5.0);
        assert_eq!(bounds.upper(5), 5.0);
    }

    #[test]
    fn dtlz1_is_zero_on_its_front() {
        // With every tail variable at the midpoint g is zero, and the sum of
        // the objectives is then exactly one half.
        let b = Benchmark::named("dtlz1", 7).unwrap();
        let mut x = vec![0.5; 7];
        x[0] = 0.2;
        x[1] = 0.3;
        x[2] = 0.5;
        let f = eval(&b, &x);
        assert_relative_eq!(f.iter().sum::<f64>(), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn dtlz2_points_land_on_the_unit_sphere() {
        let b = Benchmark::named("dtlz2", 12).unwrap();
        let mut x = vec![0.5; 12];
        x[0] = 0.3;
        x[1] = 0.4;
        let f = eval(&b, &x);
        let r = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_relative_eq!(r, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn every_catalogue_entry_builds() {
        for (name, dim, _) in CATALOGUE {
            let b = Benchmark::named(name, *dim).unwrap_or_else(|| panic!("{name} failed"));
            assert_eq!(b.dim(), *dim);
            let p = b.bounds().sample(&mut crate::rng::Rng::new(1));
            assert!(b.bounds().contains(&p));
            let f = eval(&b, &p);
            assert!(f.iter().all(|v| v.is_finite()), "{name} produced {f:?}");
        }
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert!(Benchmark::named("nope", 10).is_none());
        assert!(Benchmark::named("zdt1", 1).is_none());
    }
}
