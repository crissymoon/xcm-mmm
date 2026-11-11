# Usage

[Back to the project](../../README.md)

The crate builds one binary, `xcm-mmm`, and one library, `xcm_mmm`. The binary is a thin wrapper over the library, so anything the CLI does can be done in code.

## Build and install

```bash
cargo build --release
./target/release/xcm-mmm --help
```

To put it on PATH:

```bash
cargo install --path .
xcm-mmm list
```

Requirements and platform notes are in [Install](INSTALL.md).

## Commands

### list

Prints the benchmark catalogue with the default dimension and a one line description.

```bash
xcm-mmm list
```

### de

Solves a single objective problem with differential evolution.

| Flag | Default | Meaning |
|---|---|---|
| `--problem` | sphere | catalogue name |
| `--dim` | catalogue value | number of variables |
| `--budget` | `10 000 * dim` | objective evaluations |
| `--population` | `clamp(18 * dim, 40, 1000)` | individuals |
| `--seed` | 1 | random seed |
| `--adaptation` | shade | `shade`, `jde` or `static` |
| `--strategy` | current-to-p-best | `current-to-p-best`, `rand1` or `best1` |
| `--crossover` | binomial | `binomial` or `exponential` |
| `--f` | 0.5 | fixed scale factor |
| `--cr` | 0.9 | fixed crossover rate |
| `--stall` | 0 | restart the worse half after N stalled generations, 0 is off |
| `--no-reduce` | off | keep a fixed population instead of the L-SHADE schedule |
| `--json` | off | print a JSON report |

```bash
xcm-mmm de --problem rosenbrock --dim 30 --budget 300000 --seed 4
xcm-mmm de --problem sphere --dim 10 --adaptation jde --json
xcm-mmm de --problem rastrigin --dim 20 --stall 40 --budget 400000
```

The table output reports the best objective, whether it is feasible, the violation, evaluations, generations, restarts, final population, diversity and wall clock. The last line shows the first six coordinates.

### moead

Solves a multi-objective problem with MOEA/D.

| Flag | Default | Meaning |
|---|---|---|
| `--problem` | zdt1 | catalogue name |
| `--dim` | catalogue value | number of variables |
| `--population` | 100 | target subproblems |
| `--divisions` | derived | simplex lattice divisions |
| `--neighbourhood` | 20 | `T` |
| `--delta` | 0.9 | probability of mating in the neighbourhood |
| `--replacements` | 2 | neighbours replaced per child |
| `--update-probability` | 0.2 | share of subproblems that produce a child per generation |
| `--update-delay` | 0.1 | warm-up share before the partial schedule starts |
| `--ri-window` | 50 | stagnation window for relative-improvement allocation; 0 draws the share uniformly at random |
| `--scalar` | tchebycheff | `tchebycheff`, `pbi` or `weighted-sum` |
| `--theta` | 5.0 | PBI penalty |
| `--f` | 0.5 | differential scale factor |
| `--cr` | 1.0 | crossover rate |
| `--budget` | 50 000 | objective evaluations |
| `--seed` | 1 | random seed |
| `--out` | none | write the front as CSV |
| `--json` | off | print a JSON report |

```bash
xcm-mmm moead --problem zdt1 --dim 30
xcm-mmm moead --problem dtlz2 --scalar pbi --budget 100000 --out tmp/front.csv
xcm-mmm moead --problem zdt3 --seed 9 --json
```

Against a benchmark with a known front, the report includes IGD and hypervolume. Against any problem it reports the number of points, generations, evaluations, the ideal point and wall clock. The CSV has one row per point, objective values only.

### synth

Writes a synthetic panel so the forecasting path can be tried without data. The first `--informative` features drive the target; the rest are noise. The target carries an autoregressive term, which makes a walk-forward split meaningful.

| Flag | Default | Meaning |
|---|---|---|
| `--rows` | 600 | rows |
| `--features` | 12 | feature columns |
| `--informative` | 4 | columns that carry signal |
| `--seed` | 7 | random seed |
| `--out` | synthetic.csv | destination |

```bash
xcm-mmm synth --rows 800 --features 12 --informative 4 --out tmp/panel.csv
```

### forecast

Selects features, fits ridge regressions on walk-forward folds, and reports out-of-sample quality.

| Flag | Default | Meaning |
|---|---|---|
| `data` | required | CSV or JSON Lines path |
| `--target` | target | target column name |
| `--folds` | 4 | walk-forward folds |
| `--embargo` | 1 | rows dropped before each test block |
| `--lambda` | 1.0 | ridge penalty |
| `--method` | both | `de`, `moead` or `both` |
| `--sparsity` | 0.01 | error per unit of feature fraction, single objective run |
| `--budget` | 6 000 | evaluations for each search |
| `--seed` | 1 | random seed |
| `--out` | none | write the Pareto front as CSV |
| `--holdout` | 0.0 | fraction of rows held back from the search and scored once at the end |
| `--json` | off | print a JSON report |

```bash
xcm-mmm forecast tmp/panel.csv --method both --folds 4
xcm-mmm forecast prices.csv --target forward_return --folds 6 --embargo 5 --json
xcm-mmm forecast prices.csv --target forward_return --embargo 2500 --holdout 0.25
```

The table has a row per model: all features, the single objective selection, the MOEA/D knee and the best point on the MOEA/D front. Each row shows the number of features, pooled out-of-sample mean squared error, R2, mean information coefficient and skill against the training mean. `1 - mse / baseline_mse` is the skill, so 71 percent means the model removed 71 percent of the error of predicting the training mean.

With `--holdout` the run prints a second table underneath, scored on rows the search never saw. Use it whenever you intend to quote a number: without it the search chooses the subset on the same folds the report scores, so the selected rows always look better than `all`.

The JSON report carries the same numbers plus the per model column indices and names, the full front, fold counts and elapsed seconds. See [Forecasting](FORECASTING.md) for what the numbers mean and how to improve them.

### bench

Runs the benchmark suite, five seeds unless told otherwise, and prints two tables.

```bash
xcm-mmm bench --runs 5
xcm-mmm bench --runs 10 --budget 100000 --json
```

The single objective table reports the best, median and mean objective over the runs. The multi-objective table reports mean IGD, mean hypervolume, the point count and the time per run.

## Error behaviour

A wrong engine for a problem is refused with an explanation rather than run. `de --problem zdt1` says that the problem has two objectives and points at `moead`; `moead --problem sphere` does the reverse. Unknown problem names, a dimension below the problem minimum, an unreadable file, a missing target column and a panel too short for the requested folds all exit with status 1 and a single line on standard error.

## Library

The trait surface is small. Implement `Problem` and pass it to an engine.

```rust
use xcm_mmm::de::{minimize, DeConfig};
use xcm_mmm::problem::{Bounds, Problem};

struct Bowl {
    bounds: Bounds,
}

impl Problem for Bowl {
    fn name(&self) -> &str {
        "bowl"
    }

    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], _g: &mut [f64]) {
        f[0] = x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum();
    }
}

let problem = Bowl {
    bounds: Bounds::symmetric(20, -1.0, 1.0),
};
let out = minimize(
    &problem,
    &DeConfig {
        budget: 50_000,
        seed: 7,
        ..DeConfig::default()
    },
);
assert!(out.best_objective() < 1e-10);
```

Constraints work the same way: report their number from `constraints()` and write each value into the constraint slice. The convention is `g <= 0`.

```rust
struct ConstrainedBowl {
    bounds: Bounds,
}

impl Problem for ConstrainedBowl {
    fn name(&self) -> &str {
        "constrained-bowl"
    }

    fn dim(&self) -> usize {
        self.bounds.dim()
    }

    fn constraints(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn evaluate(&self, x: &[f64], f: &mut [f64], g: &mut [f64]) {
        f[0] = x.iter().map(|v| v * v).sum();
        g[0] = 1.0 - x[0]; // satisfied when x[0] >= 1
    }
}
```

A multi-objective problem reports its length from `objectives()` and fills that many slots.

```rust
use xcm_mmm::moead::{minimize, MoeadConfig};

let out = minimize(&problem, &MoeadConfig::seeded(3));
for solution in &out.solutions {
    println!("{:?} at {:?}", solution.objectives, solution.weight);
}
```

A forecast can be driven directly, which is useful when the panel is already in memory.

```rust
use std::sync::Arc;
use xcm_mmm::data::Dataset;
use xcm_mmm::forecast::{SelectionProblem, SparseAccuracy, DEFAULT_LAMBDA};

let mut data = Dataset::synthetic(600, 10, 3, 1);
data.standardize();
let problem = SelectionProblem::new(Arc::new(data), 4, 1, DEFAULT_LAMBDA);
let sparse = SparseAccuracy::new(problem.clone(), 0.01);
let all = problem.decode(&vec![1.0; problem.dim()]);
let report = problem.report(&all);
println!("{}", report.summary());
```

`SelectionProblem` returns the front; `SparseAccuracy` is the single objective wrapper for `de::minimize`. Both expose `report`, which computes the pooled metrics and the per fold detail.
