# xcm-mmm

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/crissymoon/xcm-mmm/blob/main/LICENSE)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://github.com/crissymoon/xcm-mmm)
[![Website](https://img.shields.io/badge/website-xcaliburmoon.net-black.svg)](https://xcaliburmoon.net/)

Differential evolution and MOEA/D in Rust, with walk-forward forecasting.

xcm-mmm solves single and multi-objective optimisation problems with two research grade engines. SHADE carries a success history of scale factors and crossover rates, an external archive and the L-SHADE population schedule; jDE and the classic Storn and Price algorithm are included for comparison. MOEA/D-DE decomposes a front into weighted subproblems, mates inside a neighbourhood and updates a bounded number of neighbours per child. Both engines evaluate on a rayon pool and both are bit for bit reproducible at any thread count.

The forecasting module turns feature selection into a search problem over blocked walk-forward folds. A ridge model is fitted on a training block, scored on the block that follows, and the search sees only the cross-validated error. Run it with two objectives and it returns the whole accuracy against model size curve, so a smaller model can be chosen on evidence rather than hope.

**Version:** 0.1.0

## Why this exists

Most of the prediction accuracy in a panel is lost before the model is fitted: features are chosen on the same rows they are scored on, the split ignores time, and a handful of noisy columns are kept because they helped in sample. This crate makes the honest version the easy one.

- Ridge is closed form, deterministic and fast enough to be evaluated tens of thousands of times inside an evolutionary loop.
- Every fold trains on the past and tests on the future, with an embargo between them, so a label can never look at its own future.
- The single objective search minimises error plus a sparsity penalty; the two objective search returns the trade-off between error and model size.
- The report is out of sample: mean squared error, R2, mean absolute error, rank information coefficient, and skill against predicting the training mean.

On the synthetic panel the search lowers its own walk-forward error: all 12 features score R2 0.661, and the 4 true features score 0.695. That comparison flatters the search, because it scores every fold it also uses to choose the subset. Holding the last quarter of the panel back with `--holdout 0.25` removes the advantage: 0.7026 for the selected four against 0.7058 for all twelve. On a wide noisy panel selection pays; on a panel whose columns are already few and informative it does not, and the tool is built to show you which case you are in.

## What is inside

- **SHADE and L-SHADE.** Current-to-pbest mutation, success history memory with a weighted Lehmer mean for F, Cauchy sampling for F and normal sampling for CR, an external archive, and the linear population reduction schedule.
- **jDE and classic DE.** Per individual self adaptation, and the Storn and Price rand/1/bin baseline, with binomial or exponential crossover.
- **Constraints.** Deb's feasibility rules everywhere: a feasible point always beats an infeasible one, and two infeasible points compare by total violation.
- **MOEA/D-DE.** Das and Dennis simplex weights, Euclidean neighbourhoods, Tchebycheff, PBI and weighted sum scalarisation, neighbourhood mating with delta, two neighbour replacements per child, polynomial mutation, and partial updates that vary a fifth of the subproblems per generation, weighted by relative improvement with a stagnation reset.
- **Quality indicators.** Exact hypervolume in two and three dimensions, seeded Monte Carlo above that with a reported error, IGD, generational distance, spacing, and the standard ZDT and DTLZ reference fronts.
- **Parallel and deterministic.** Evaluation runs on rayon. Mutation, selection and all random draws stay on one thread, so the result does not depend on the thread count.
- **Forecasting.** CSV and JSON Lines reading, blocked walk-forward folds with an embargo, ridge regression by Cholesky, and a report with per fold detail.
- **Corpus cleaning and pairs.** Character folding, markup and boilerplate screening, quality rules with a called-out reason per rejection, MinHash near duplicate clustering, two selection rules (thresholded marginal relevance and target-aware importance resampling), and a pair miner that writes rank, preference and contrast banks in one schema with an audit that gates leakage, triviality and contamination.
- **One transformer for every task.** A byte level byte pair tokenizer with no unknown token, a pre norm encoder whose backward pass is checked against finite differences, six task heads on one shared trunk, AdamW with a warmup and cosine schedule, early stopping on the validation split the pairs carry, and one model file holding the tokenizer, the config and the weights.
- **Two front ends.** A library with a small trait surface, and a CLI that can run a benchmark suite, solve a problem or score a panel.

## Training data

`data/training/` is the set this tool ships: 20,000 rank, preference and contrast pairs mined from the public marketlab news corpus, in `train.jsonl`, `valid.jsonl` and `test.jsonl` beside a generated `card.md` and a `manifest.json` that names the digest and row count of every file.

The set is a stratified sample of a retention bank that keeps growing, so it is fixed rather than current. The same source and the same seed rebuild the same bytes:

```bash
./xcm-mmm dataset build --input bank.jsonl --out data/training --budget 20000 --seed 20261114
./xcm-mmm dataset check --dir data/training
```

`dataset check` reads the folder back, parses every row and compares each file with its manifest, so a copy that lost a line or a byte is caught rather than trained on. The full path from labelled rows is one script:

```bash
scripts/train-data.sh --rows rows/ --out data/training
```

It mines each month, folds the pairs into a bank that only improves, samples the bank, and checks the result. The bank lives under `tmp/` and stays out of the repository because it grows every week; what enters version control is the sample and the card that describes it.

## Quick start

```bash
cargo build --release
./target/release/xcm-mmm list
./target/release/xcm-mmm de --problem sphere --dim 30
./target/release/xcm-mmm moead --problem zdt1
```

A forecast needs a CSV with a header row and a numeric target column. To try the whole path without data of your own:

```bash
./target/release/xcm-mmm synth --rows 800 --features 12 --informative 4 --out tmp/panel.csv
./target/release/xcm-mmm forecast tmp/panel.csv --method both --folds 4
```

That prints one line per model:

```text
rows        800   features 12   folds 4
model        feat          mse       r2      ic   skill
all            12     0.371781   0.6616   0.796   67.8%
de              4     0.335217   0.6949   0.820   71.0%
  keeps: x1, x2, x3, x4
moead-knee      2     0.524956   0.5222   0.718   54.6%
  keeps: x1, x3
moead-best      4     0.335217   0.6949   0.820   71.0%
  keeps: x1, x2, x3, x4
  front: 4 points, knee 2, best 0
```

From labelled rows to one trained model:

```bash
./target/release/xcm-mmm pairs mine tmp/rows.jsonl tmp/pairs.jsonl
./target/release/xcm-mmm transformer train --pairs tmp/pairs.jsonl --out tmp/model.json
./target/release/xcm-mmm transformer eval tmp/model.json --pairs tmp/pairs.jsonl
./target/release/xcm-mmm transformer score tmp/model.json --rows tmp/rows.jsonl --out tmp/scores.jsonl
```

Training fills a batch in parallel and can be paused: `--checkpoint tmp/run.cp.json` writes the optimizer state after each epoch group, and `transformer train --pairs tmp/pairs.jsonl --out tmp/model.json --resume tmp/run.cp.json` continues the same plan and lands on the model the uninterrupted run produces.

## Benchmarks

Both tables come from `xcm-mmm bench --runs 5` on one development machine, release build. Times are wall clock per run and only show scale; compare the objective values, not the clocks.

Differential evolution, SHADE with the L-SHADE reduction, 300 000 evaluations at 30 variables:

| problem | best | median | mean | seconds |
|---|---:|---:|---:|---:|
| sphere | 1.510e-13 | 1.629e-13 | 1.639e-13 | 0.21 |
| rosenbrock | 1.371e-13 | 1.502e-13 | 1.509e-13 | 0.20 |
| rastrigin | 5.684e-14 | 1.137e-13 | 1.023e-13 | 0.21 |
| ackley | 4.405e-13 | 5.649e-13 | 5.585e-13 | 0.22 |

MOEA/D with Tchebycheff scalarisation, 100 subproblems, 50 000 evaluations, five runs, partial updates at a 0.2 share weighted by relative improvement:

| problem | variables | IGD | hypervolume | points | seconds |
|---|---:|---:|---:|---:|---:|
| zdt1 | 30 | 0.00550 | 0.86646 | 100 | 0.119 |
| zdt2 | 30 | 0.00561 | 0.53220 | 98 | 0.119 |
| zdt3 | 30 | 0.01230 | 1.30270 | 84 | 0.122 |
| zdt4 | 10 | 0.00710 | 0.86345 | 100 | 0.109 |
| zdt6 | 10 | 0.00190 | 0.50436 | 100 | 0.105 |
| dtlz1 | 7 | 0.03186 | 0.18272 | 63 | 0.110 |
| dtlz2 | 12 | 0.07700 | 0.70066 | 60 | 0.112 |

Hypervolume uses the reference point 1.1 in every objective for the ZDT set, 0.6 for DTLZ1 and 1.1 for DTLZ2. IGD is measured against a 500 point sample of the true front, which the DTLZ rows serve with a 496 point lattice; the three objective rows are resolution limited, and the scaling law behind that is in [Benchmarks](README/DOCS/BENCHMARKS.md). Reproduce with `xcm-mmm bench --runs 5`.

## Documentation

- [Algorithms](README/DOCS/ALGORITHMS.md) - the operators, the success history formulas, decomposition, constraints and every default.
- [Usage](README/DOCS/USAGE.md) - the full command line, JSON reports and library examples.
- [Forecasting](README/DOCS/FORECASTING.md) - data formats, folds, metrics, how to read the front and where accuracy comes from.
- [Benchmarks](README/DOCS/BENCHMARKS.md) - methodology, the tables above and how they compare with published results.
- [Install](README/DOCS/INSTALL.md) - requirements, build, install and using the crate as a dependency.
- [Development](README/DOCS/DEVELOPMENT.md) - layout, design rules, tests, lints and how to add a problem.

## Testing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

The suite has 97 tests. It covers convergence on sphere, Rosenbrock, Rastrigin, Ackley, ZDT1 to ZDT4, ZDT6, DTLZ1 and DTLZ2, the exact hypervolume identities, the weight lattice counts, constraint handling, reproducibility at one and four threads, the forecasting path on a synthetic panel, and every CLI subcommand.

## License

MIT License, copyright 2026 Crissy Moon. See [LICENSE](LICENSE) for the full text.

Crissy Moon - [https://xcaliburmoon.net/](https://xcaliburmoon.net/)

Repository: [https://github.com/crissymoon/xcm-mmm](https://github.com/crissymoon/xcm-mmm)
