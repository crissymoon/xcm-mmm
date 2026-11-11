# Benchmarks

[Back to the project](../../README.md)

The numbers below come from the binary in this repository, not from a paper. They exist to show that the engines work and to catch a regression, not to claim a record. Published tables differ in budget, population, reference front density and the machine, so compare the order of magnitude, not the fourth digit.

## Method

- Machine: Apple M5, 10 cores, macOS 26.6, rustc 1.99.1, release build with thin LTO.
- Every run starts from a fresh seed. The reported figure is the mean over the stated number of seeds, with 1, 2, ... as seeds.
- Single objective runs use 300 000 evaluations at 30 variables, which is `10 000 * dim`, the CEC convention.
- Multi-objective runs use 100 subproblems and 50 000 evaluations, which is 500 generations. Tchebycheff scalarisation, neighbourhood 20, delta 0.9, two replacements per child, F 0.5, CR 1.0, polynomial mutation with index 20.
- IGD is measured against 500 points of the generated true front.
- Hypervolume uses the reference point 1.1 in every objective for ZDT, 0.6 for DTLZ1 and 1.1 for DTLZ2.

Reproduce everything with:

```bash
cargo build --release
./target/release/xcm-mmm bench --runs 5
```

The adaptation comparison uses one command per cell:

```bash
./target/release/xcm-mmm de --problem rosenbrock --adaptation static --budget 300000 --seed 1
```

## Single objective

Mean best objective over three seeds, 30 variables, 300 000 evaluations. Lower is better.

| problem | SHADE | jDE | classic DE |
|---|---:|---:|---:|
| sphere | 1.61e-13 | 2.48e-13 | 2.98e-03 |
| rosenbrock | 1.49e-13 | 1.84e+01 | 1.99e+01 |
| rastrigin | 1.14e-13 | 1.38e+00 | 1.56e+01 |
| ackley | 5.33e-13 | 7.91e-13 | 2.56e-02 |

SHADE is the default for a reason. On sphere and Ackley the three variants are separated by orders of magnitude that all count as solved, but on Rosenbrock and Rastrigin the success history and the population reduction are the difference between a solved problem and a stalled one. The classic algorithm is included as the baseline, not as a recommendation.

Per seed detail for the default configuration, five runs:

| problem | best | median | mean | seconds per run |
|---|---:|---:|---:|---:|
| sphere | 1.510e-13 | 1.629e-13 | 1.639e-13 | 0.21 |
| rosenbrock | 1.371e-13 | 1.502e-13 | 1.509e-13 | 0.20 |
| rastrigin | 5.684e-14 | 1.137e-13 | 1.023e-13 | 0.21 |
| ackley | 4.405e-13 | 5.649e-13 | 5.585e-13 | 0.22 |

The spread between seeds is within a factor of two at the precision floor, which is what a converged run looks like.

## Multi-objective

Five seeds, 50 000 evaluations, 100 subproblems. IGD lower is better, hypervolume higher is better.

| problem | variables | IGD | hypervolume | points | seconds per run |
|---|---:|---:|---:|---:|---:|
| zdt1 | 30 | 0.01066 | 0.85703 | 100 | 0.056 |
| zdt2 | 30 | 0.00719 | 0.52725 | 99 | 0.055 |
| zdt3 | 30 | 0.01792 | 1.28145 | 80 | 0.055 |
| zdt4 | 10 | 0.00801 | 0.86152 | 95 | 0.046 |
| zdt6 | 10 | 0.00189 | 0.50452 | 100 | 0.045 |
| dtlz1 | 7 | 0.03017 | 0.18294 | 61 | 0.056 |
| dtlz2 | 12 | 0.07501 | 0.69888 | 60 | 0.055 |

Two observations:

- The ZDT fronts are dense and close. ZDT3 is the hardest of the two objective set because its front is disconnected, and it still lands within 0.02.
- The DTLZ rows carry fewer points and a larger IGD. Three objectives give the lattice 105 weights and the front is a surface rather than a curve, so the same budget buys less coverage per point. DTLZ1 is also deceptive by construction: its barrier rewards a population that keeps exploring.

At 30 variables, ZDT1 needs about 500 generations to converge. At 25 000 evaluations, which is 250 generations, IGD is around 0.12; at 50 000 it is 0.011; at 100 000 it is 0.005. The engine does not cheat by using a smaller search space, and the budget is the honest way to compare it.

## Scalarisation choice

Three objectives are where PBI earns its place. Same budget and population, one seed, 50 000 evaluations:

| problem | Tchebycheff IGD | PBI IGD | Tchebycheff points | PBI points |
|---|---:|---:|---:|---:|
| zdt1 | 0.00932 | 0.01981 | 100 | 96 |
| dtlz1 | 0.03032 | 0.02398 | 61 | 91 |
| dtlz2 | 0.07466 | 0.07032 | 58 | 83 |

PBI improves the three objective front and its density, and loses on ZDT1. The default stays Tchebycheff because two objective problems are the common case; pass `--scalar pbi` when the front is a surface.

## What the numbers do not say

- **Time is not a benchmark.** The problems are cheap, so a large part of the wall clock is Rayon start up and the allocation of scores. The forecasting path, where a single evaluation is four ridge fits, is the realistic measure of throughput.
- **The reference fronts are generated.** ZDT1, ZDT2, ZDT4, ZDT6 and the DTLZ rows are exact; ZDT3 is sampled on its five published arcs. A different reference set changes IGD in the third digit.
- **Monte Carlo hypervolume is not used above three objectives.** Every row in the table is exact. For four or more objectives the crate reports a seeded estimate and its 95 percent error, and the two should be quoted together.
- **The engine is synchronous.** It evaluates a whole generation at once so the expensive part can use every core, and a child does not see its neighbours' updates until the next generation. Sequential MOEA/D converges slightly faster per generation. This is visible at short budgets, not at 500 generations.
- **These are synthetic problems.** They test the optimiser, not the forecasting claim. The forecasting evidence is the worked example in [Forecasting](FORECASTING.md).
