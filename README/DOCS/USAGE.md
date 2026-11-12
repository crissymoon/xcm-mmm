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

### clean

Cleans a corpus, drops near duplicates, and keeps a relevant budget. Each stage prints what it removed, so a run that cuts too much is visible in the first twenty lines.

```bash
xcm-mmm clean docs.jsonl.gz clean.jsonl.gz --budget 200000 --dedupe 0.80
xcm-mmm clean docs.jsonl.gz clean.jsonl.gz --target target.jsonl --budget 50000
```

| Flag | Default | Meaning |
|---|---|---|
| `--dedupe` | 0.80 | Jaccard gate for near duplicates, 0 disables the pass |
| `--budget` | 0 | rows to keep, 0 keeps every survivor |
| `--min-words` | 5 | floor of the length rule |
| `--max-words` | 3000 | ceiling of the length rule |
| `--floor` | 0 | composite quality floor for the coverage pass |
| `--theta` | 0.85 | cosine similarity at which a row is redundant |
| `--group-cap` | 0.10 | largest share of the budget one source may take |
| `--target` | none | curated target set, switches selection to importance resampling |

Input and output may be `.jsonl.gz`; compression uses the system `gzip`, so no crate is added. Without `--target` the pass orders candidates by the composite quality score, refuses a source that has spent its cap, and refuses a row whose cosine similarity to an already accepted row is at or above `--theta`. With `--target` it fits a target density and a proposal density over hashed n-grams and draws rows with probability proportional to `p/q`, which is the importance resampling rule.

### pairs

Three subcommands: `mine` builds banks from labelled rows, `pack` merges pair files into one corpus with a task mix, `audit` checks a corpus and exits 1 when it finds leakage, trivial pairs or contamination.

```bash
xcm-mmm pairs mine rows.jsonl mined.jsonl --min-margin 0.02
xcm-mmm pairs pack --input mined.jsonl,ads.jsonl --out pairs.jsonl --total 500000
xcm-mmm pairs audit pairs.jsonl --eval eval.jsonl
```

| Command | Flag | Default | Meaning |
|---|---|---|---|
| mine | `--min-margin` | 0.01 | smallest outcome gap a pair may carry |
| mine | `--max-similarity` | 0.95 | largest cosine the two sides may share |
| mine | `--neighbours` | 5 | how many hardest negatives to choose among |
| mine | `--max-per-task` | 0 | cap per bank, 0 keeps every pair |
| mine | `--max-chars` | 700 | longest text copied into a pair |
| pack | `--total` | 0 | rows in the corpus, 0 keeps every row |
| pack | `--weights` | rank=3,contrast=2,preference=2,label=1,correction=1,retrieval=1 | task mix |
| pack | `--max-similarity` | 0.98 | similarity at which two sides are one text |
| pack | `--manifest` | prints | where to write the run manifest |
| audit | `--eval` | none | evaluation rows to screen for contamination |
| audit | `--contamination` | 0.80 | share of an evaluation text that counts as a hit |

`mine` writes the three banks the market corpus supports: `rank` pairs two rows from the same session whose outcomes landed in different quartiles, `preference` pairs two rows on the same group with a clear gap in outcome, and `contrast` pairs an anchor with a positive from an earlier session and a negative from the same session in another group. Each row records `margin`, `difficulty`, `quality_a`, `quality_b` and `neg_source`, so an audit can filter without re-reading the texts.

### transformer

One shared encoder, one head per task, one file that holds the tokenizer, the config and the weights.

```bash
xcm-mmm transformer train --pairs pairs.jsonl --out model.json --epochs 3
xcm-mmm transformer eval model.json --pairs pairs.jsonl
xcm-mmm transformer score model.json --rows rows.jsonl --out scores.jsonl
xcm-mmm transformer encode model.json "the company beat estimates as demand rose"
```

| Command | Flag | Default | Meaning |
|---|---|---|---|
| train | `--vocab` | 2048 | tokenizer size, control tokens included |
| train | `--max-len` | 64 | longest token sequence |
| train | `--dim` | 64 | residual width |
| train | `--heads` | 4 | attention heads, must divide `--dim` |
| train | `--ff` | 128 | feed forward width |
| train | `--layers` | 2 | transformer blocks |
| train | `--epochs` | 3, or the checkpoint's plan on `--resume` | passes over the corpus |
| train | `--batch` | 8 | rows per gradient step |
| train | `--lr` | 0.003 | peak learning rate |
| train | `--warmup` | 0.1 | share of the schedule spent warming up |
| train | `--clip` | 1.0 | gradient norm limit |
| train | `--dropout` | 0.1 | dropout on the two residual branches |
| train | `--weight-decay` | 0.01 | applied to weight matrices only |
| train | `--task-weights` | rank=1,contrast=1,preference=1,label=0.5,correction=0.5,retrieval=1 | sampling weight per task |
| train | `--limit` | 0 | read at most this many pairs, 0 reads everything |
| train | `--patience` | 2 | epochs without improvement before stopping, 0 disables it |
| train | `--min-count` | 2 | smallest pair count a merge may come from |
| train | `--resume` | none | continue from a checkpoint written by an earlier leg |
| train | `--stop-after` | the plan | end this leg after this many epochs, schedule unchanged |
| train | `--checkpoint` | none | write weights, best epoch, Adam moments and tokenizer after each group |
| train | `--checkpoint-every` | 1 | epochs between checkpoint writes |
| eval | `--pairs` | required | pair files, comma separated |
| eval | `--limit` | 0 | read at most this many pairs |
| score | `--rows` | required | text rows, one JSON object per line |
| score | `--out` | required | where the scored rows are written |
| score | `--task` | every head | one task head per line instead of all six |

`train` reads the split field each pair carries: `train` is trained on, `val`, `valid`, `dev` and `validation` are used for early stopping, and `test` or `holdout` is never trained on. When no validation split exists the last tenth of the rows is held out by position. `score` reads the same fields the cleaning stages read (`text`, `text_a` or `body`) and writes one line per row with every head unless `--task` picks one.

The rows of a batch are filled in parallel: each example writes its own gradient buffer and the buffers are added in batch order, so the update is the same on one thread and on eight. Measured on the development machine over 4,191 pairs for three epochs at width 64, that is 9.8 s of training on one thread against 3.0 s on eight, with byte identical weights. Dropout draws from a per-example seed derived from the run seed, the step and the row, and the order of an epoch is drawn from the epoch number alone, which is why a batch and an epoch no longer depend on what ran before them.

A long run can be split. `--checkpoint` writes the current weights, the best weights, the two Adam moments, the step counter, the tokenizer and the plan after every epoch group, and `--resume` continues from that file. With `--resume` the plan is the checkpoint's, so `--epochs` may be left out; `--stop-after` ends a leg early while the learning rate schedule still uses the plan:

```bash
xcm-mmm transformer train --pairs pairs.jsonl --out model.json --epochs 8 \
    --stop-after 4 --checkpoint run.cp.json
xcm-mmm transformer train --pairs pairs.jsonl --out model.json --resume run.cp.json
```

The second command runs epochs 5 to 8 and lands on the model the uninterrupted eight epoch run produces, byte for byte, which `tests/transformer_pipeline.rs` asserts. The flags that shape the model (`--dim`, `--heads`, `--ff`, `--layers`, `--max-len`) must match the checkpoint and are refused with a sentence when they do not; `--dropout` may change. A checkpoint is transient and roughly four times the size of a model file.

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
