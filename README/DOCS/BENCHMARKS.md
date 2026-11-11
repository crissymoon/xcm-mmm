# Benchmarks

[Back to the project](../../README.md)

The numbers below come from the binary in this repository, not from a paper. They exist to show that the engines work and to catch a regression, not to claim a record. Published tables differ in budget, population, reference front density and the machine, so compare the order of magnitude, not the fourth digit.

## Method

- Machine: Apple M5, 10 cores, macOS 26.6, rustc 1.99.1, release build with thin LTO.
- Every run starts from a fresh seed. The reported figure is the mean over the stated number of seeds, with 1, 2, ... as seeds.
- Single objective runs use 300 000 evaluations at 30 variables, which is `10 000 * dim`, the CEC convention.
- Multi-objective runs use 100 subproblems and 50 000 evaluations, which is 500 generations. Tchebycheff scalarisation, neighbourhood 20, delta 0.9, two replacements per child, F 0.5, CR 1.0, polynomial mutation with index 20, partial updates at a 0.2 share weighted by relative improvement with a 50 generation stagnation window.
- IGD is measured against a 500 point sample of the generated true front. For the DTLZ rows a 500 point request is met by the 30 division lattice, which holds 496 points.
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
| zdt1 | 30 | 0.00550 | 0.86646 | 100 | 0.119 |
| zdt2 | 30 | 0.00561 | 0.53220 | 98 | 0.119 |
| zdt3 | 30 | 0.01230 | 1.30270 | 84 | 0.122 |
| zdt4 | 10 | 0.00710 | 0.86345 | 100 | 0.109 |
| zdt6 | 10 | 0.00190 | 0.50436 | 100 | 0.105 |
| dtlz1 | 7 | 0.03186 | 0.18272 | 63 | 0.110 |
| dtlz2 | 12 | 0.07700 | 0.70066 | 60 | 0.112 |

Two observations:

- The ZDT fronts are dense and close. ZDT3 is the hardest of the two objective set because its front is disconnected, and it still lands at 0.012.
- The DTLZ rows carry fewer points and a larger IGD, and that gap is the resolution of a surface rather than a convergence failure. Three objectives spread 91 subproblems over a two dimensional front instead of a curve, so the nearest neighbour distance grows like `1 / sqrt(N)`. The section on front resolution below measures it: IGD times the square root of the subproblem count stays between 0.65 and 0.74 for DTLZ2 and between 0.27 and 0.31 for DTLZ1 over a tenfold range of populations.

At 30 variables, ZDT1 needs about 50 000 evaluations to converge. At 25 000 evaluations the mean IGD over five seeds is 0.021 with a spread of 0.014 to 0.032; at 50 000 it is 0.0055; at 100 000 it is 0.0044. The engine does not cheat by using a smaller search space, and the budget is the honest way to compare it.

## Resource allocation

Only a fifth of the subproblems produce a child in a generation by default, and which fifth is drawn in proportion to recent relative improvement rather than uniformly at random. Thirty seeds per setting at 50 000 evaluations, mean IGD and mean hypervolume over all seven problems:

| update probability | allocation | mean IGD | mean hypervolume |
|---:|---|---:|---:|
| 1.0 | plain MOEA/D-DE | 0.02534 | 0.69987 |
| 0.4 | uniform random | 0.02086 | 0.70527 |
| 0.4 | relative improvement | 0.02027 | 0.70719 |
| 0.2 | uniform random | 0.02066 | 0.70610 |
| 0.2 | relative improvement, default | 0.02003 | 0.70785 |
| 0.1 | uniform random | 0.02062 | 0.70611 |
| 0.1 | relative improvement | 0.02032 | 0.70732 |

Per problem at the default, the improvement weighting takes IGD from 0.00790 to 0.00612 on ZDT1, 0.00559 to 0.00526 on ZDT2, 0.01281 to 0.01170 on ZDT3 and 0.00751 to 0.00628 on ZDT4. ZDT6 is flat at 0.0019 and both DTLZ rows are flat to the fourth digit. The random subset alone is worth most of the move against plain MOEA/D-DE: ZDT3 falls from 0.0184 to 0.0128, ZDT4 from 0.0131 to 0.0075 and DTLZ1 from 0.0496 to 0.0318. The improvement weighting then adds 3 percent on the mean.

The cost is wall clock. The same budget is spread over five times as many generations, each generation recomputes one scalar value per subproblem for the priorities, and a row takes about 1.6 times as long as plain MOEA/D-DE at the same evaluation count. Objective evaluations do not change.

## Front resolution

The DTLZ rows look weak next to the ZDT rows for a geometric reason: their fronts are surfaces, not curves. Holding the generation count at 500 and raising the population spreads more points over the same surface:

| DTLZ1 subproblems | IGD | IGD x sqrt(N) | DTLZ2 subproblems | IGD | IGD x sqrt(N) |
|---:|---:|---:|---:|---:|---:|
| 91 | 0.03186 | 0.304 | 91 | 0.07700 | 0.735 |
| 190 | 0.02088 | 0.288 | 190 | 0.05085 | 0.701 |
| 300 | 0.01615 | 0.280 | 300 | 0.03865 | 0.670 |
| 780 | 0.01031 | 0.288 | 780 | 0.02502 | 0.699 |

The product is close to constant, so the DTLZ IGD at 91 subproblems is the spacing of a two dimensional sampling, not a failure to converge. It also means an IGD figure for a three objective problem is only comparable to another with the same population, and that spending the budget on more subproblems buys coverage in a way that more generations do not.

One caution: a population whose lattice coincides with the reference lattice scores better than its neighbours in this table. At 496 subproblems DTLZ2 reads 0.0294, against 0.0387 at 300 and 0.0250 at 780; the 496 point reference and the 496 point population overlap exactly, so distances collapse. The effect is invisible for the random or very dense reference sets used elsewhere.

## Scalarisation choice

Three objectives are where PBI earns its place. Same budget and population, one seed, 50 000 evaluations:

| problem | Tchebycheff IGD | PBI IGD | Tchebycheff points | PBI points |
|---|---:|---:|---:|---:|
| zdt1 | 0.00577 | 0.02321 | 100 | 97 |
| dtlz1 | 0.03193 | 0.02491 | 61 | 91 |
| dtlz2 | 0.07706 | 0.06851 | 59 | 81 |

PBI improves the three objective front and its density, and loses on ZDT1. The default stays Tchebycheff because two objective problems are the common case; pass `--scalar pbi` when the front is a surface.

## What the numbers do not say

- **Time is not a benchmark.** The problems are cheap, so a large part of the wall clock is Rayon start up and the allocation of scores. The forecasting path, where a single evaluation is four ridge fits, is the realistic measure of throughput.
- **The reference fronts are generated.** ZDT1, ZDT2, ZDT4, ZDT6 and the DTLZ rows are exact; ZDT3 is sampled on its five published arcs. A different reference set changes IGD in the second digit for the three objective rows, because a denser lattice measures coverage more harshly, and a set that aliases with the population lattice measures it more kindly.
- **Monte Carlo hypervolume is not used above three objectives.** Every row in the table is exact. For four or more objectives the crate reports a seeded estimate and its 95 percent error, and the two should be quoted together.
- **The engine is synchronous.** It evaluates a whole generation at once so the expensive part can use every core, and a child does not see its neighbours' updates until the next generation. Sequential MOEA/D converges slightly faster per generation. This is visible at short budgets, not at the budgets in this table.
- **Partial updates change the generation count, not the evaluation count.** The budget is still 50 000 evaluations, but at an update probability of 0.2 each generation builds about twenty children, so a row runs around 2 500 generations instead of 500. Any figure per generation, including the wall clock, has to be read with that in mind.
- **The resource allocation comparison is on seven problems and thirty seeds.** The improvement weighting wins on the mean at three shares and on hypervolume at three shares, but its per problem edge is small outside ZDT1 and ZDT3, and the standard deviation of a single run is larger than the difference between the schedules.
- **These are synthetic problems.** They test the optimiser, not the forecasting claim. The forecasting evidence is the worked example in [Forecasting](FORECASTING.md).
