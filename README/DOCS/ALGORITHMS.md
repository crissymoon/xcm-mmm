# Algorithms

[Back to the project](../../README.md)

This page describes what the two engines actually do, with the formulas and the defaults. Every constant named here exists in the source, and every default can be changed through `DeConfig` or `MoeadConfig`.

A problem is minimised, never maximised. Decision variables live in a box `lo <= x <= hi`. A constraint is satisfied when `g(x) <= 0`; the total violation is the sum of the positive constraint values.

## Differential evolution

The single objective engine is in `src/de`. It offers three adaptations and three mutation operators, and it always uses greedy selection under Deb's feasibility rules.

### SHADE

The default is SHADE, from Tanabe and Fukunaga (2013). The population of `NP` points starts uniformly in the box, and every point is evaluated once before the loop.

Two ring buffers of size `H = 6` hold the memory of successful parameters, `M_F` and `M_CR`. Both start at 0.5. At every generation, for individual `i`:

1. Draw a slot `r` uniformly from `0..H`.
2. Draw the scale factor `F_i` from a Cauchy distribution centred on `M_F[r]` with scale 0.1. A value at or below zero is redrawn, and a value above one is capped at one.
3. Draw the crossover rate `CR_i` from a normal distribution centred on `M_CR[r]` with standard deviation 0.1, clipped to `[0, 1]`.
4. Choose `x_pbest` from the best `max(2, round(p NP))` points, with `p = 0.11`.
5. Build the mutant with current-to-pbest/1:

   ```text
   v = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - x_r2)
   ```

   The index `r1` comes from the population. The index `r2` comes from the union of the population and the external archive, which holds parents that were replaced and is capped at `NP` entries.
6. Apply binomial crossover against `x_i`, with one coordinate forced to come from the mutant.
7. Repair a coordinate that left the box with the midpoint rule: an offending value is replaced by the midpoint between the parent and the violated face, rather than clipped. Reflection and clipping are available and are chosen with `Boundary`.
8. Evaluate the trial and replace the parent when the trial wins.

The memory is updated once per generation from the successful trials:

```text
w_k    = |f(x_i) - f(u_i)|, or the violation reduction when a constraint binds
M_F    = sum(w F^2) / sum(w F)        (weighted Lehmer mean)
M_CR   = sum(w CR) / sum(w)           (weighted arithmetic mean)
```

The Lehmer mean for `F` is deliberate: it leans towards larger successful values, which keeps progress on flat and narrow ridges where small steps stall. `CR` stays arithmetic, which is steadier when most successes carry a small weight.

### L-SHADE population reduction

When `reduce_population` is on, and only for SHADE, the population shrinks linearly with the evaluation count:

```text
NP_g = round(NP_max - (NP_max - NP_min) * FE / FE_max)
```

`NP_min` is four and `NP_max` is `18 * dim`, clamped to 40 and 1000. Shrinking removes the worst ranks first, so the best point survives. jDE and the classic algorithm keep a fixed population, because the schedule is a SHADE result and does not belong to them.

### jDE and classic DE

`Adaptation::Jde` keeps one `(F, CR)` pair per individual. Before mutation, with probability 0.1, the pair is replaced by `F = 0.1 + 0.9 u` and `CR = u` with `u` uniform. A child that wins inherits the new pair, a child that loses restores the old one. The mutation is rand/1/bin.

`Adaptation::Static` is the classic Storn and Price algorithm: rand/1/bin with `F = 0.5` and `CR = 0.9`. It is the baseline the adaptive variants are measured against.

`Strategy::Best1Bin` replaces the base vector with the best point and converges quickly, at the cost of diversity. `Strategy::CurrentToPBest1` is the SHADE operator and can be used with fixed parameters as well.

### Constraints

All comparisons use Deb's rules:

- A feasible point beats an infeasible one, whatever the objective values.
- Two infeasible points compare by total violation; the smaller violation wins.
- Two feasible points compare by Pareto dominance, which for one objective is the ordinary comparison.

The rules apply to selection in DE, to replacement in MOEA/D, and to ranking inside the population reduction. A run therefore reports a feasible answer whenever one was found, and the violation is always part of the score.

### Restarts

`stall_generations` turns on a partial restart. When the best score has not improved for that many generations, the worse half of the population is drawn again, the parameter memory returns to 0.5, and the archive is cleared. Zero, the default, turns this off, because a restart changes the budget semantics. It is useful on multimodal problems such as Rastrigin at high dimension.

### Determinism and cost

All random draws and all decisions happen on the calling thread. Only `Score` evaluation is sent to the rayon pool, and results come back in input order. A run with one thread and a run with eight threads therefore produce the same population, the same history and the same best point, and the tests assert it.

One generation costs `NP` evaluations, each of `O(dim)` work plus whatever the problem costs. Memory is `O(NP * dim)` plus the archive. The evaluation budget is the only stop condition.

### Defaults

| Setting | Default | Meaning |
|---|---|---|
| population | `clamp(18 * dim, 40, 1000)` | L-SHADE recommendation |
| budget | `10 000 * dim` | CEC convention |
| adaptation | SHADE | success history |
| strategy | current-to-pbest/1 | SHADE operator |
| crossover | binomial | one forced coordinate |
| F, CR | 0.5, 0.9 | used by the static adaptation |
| p_best | 0.11 | top fraction for `x_pbest` |
| archive scale | 1.0 | archive size as a multiple of the population |
| history | 6 | memory slots |
| boundary | midpoint | jSO and L-SHADE choice |
| reduce population | on | L-SHADE schedule |
| stall generations | 0 | restarts off |

## MOEA/D

The multi-objective engine is in `src/moead` and follows MOEA/D-DE from Li and Zhang (2009).

### Weights and neighbourhoods

The weight vectors are the Das and Dennis simplex lattice with `H` divisions and `m` objectives:

```text
W = { (i_1/H, ..., i_m/H) : sum(i_k) = H, i_k >= 0 }
|W| = C(H + m - 1, m - 1)
```

`H = 12` gives 91 weights for three objectives and `H = 99` gives 100 for two. When the caller gives a target population rather than a resolution, the code picks the largest `H` whose count fits.

The neighbourhood `B(i)` is the `T` weights closest to `w_i` by Euclidean distance, and always contains `i`. Ties break by index, so the layout is deterministic.

### Scalarisation

For an ideal point `z`, where `z_k` is the best value seen for objective `k`:

```text
Tchebycheff   g = max_k  w_k |f_k - z_k|
Weighted sum  g = sum_k  w_k f_k
PBI           d1 = ((f - z) . w) / ||w||
              d2 = ||(f - z) - d1 w / ||w|| ||
              g  = d1 + theta d2
```

Tchebycheff is the default because it reaches points on a concave front, where weighted sum cannot. PBI with `theta = 5` is the usual choice for three or more objectives. In this crate PBI improved DTLZ1 and DTLZ2 at 50 000 evaluations, and Tchebycheff was clearly better on ZDT1; the tables are in [Benchmarks](BENCHMARKS.md).

### One generation

For each subproblem `i`:

1. With probability `delta`, draw parents from `B(i)`; otherwise draw them from the whole population.
2. Build the mutant with rand/1 from three distinct parents:

   ```text
   y = x_r1 + F (x_r2 - x_r3)
   ```

3. Cross `y` with the incumbent `x_i` using binomial crossover. The default `CR = 1.0` makes the child equal to the mutant, which is the MOEA/D-DE setting.
4. Apply polynomial mutation with distribution index 20 and rate `1 / dim`, then clip to the box.
5. Evaluate, then update the ideal point `z` from the child.
6. Walk `B(i)` in a seeded random order. Replace a neighbour when the child's scalar value is at least as good, and stop after `nr = 2` replacements.

The algorithm is synchronous: every child in a generation is produced from the population at the start of the round, then the whole round is evaluated in parallel, then the replacements are applied in index order. Sequential MOEA/D lets a child see its neighbours' updates immediately and converges a little faster per generation. The synchronous form is chosen because the expensive part is the evaluation, and batching it is what makes the engine usable on a forecasting problem where one evaluation fits four regressions. The cost is visible in the benchmarks: the engines reach published quality at 50 000 evaluations rather than at 25 000, and the difference is a constant factor rather than a wrong answer.

### Constraints

Replacement is feasibility first: a feasible child replaces an infeasible neighbour, two infeasible points compare by violation, and two feasible points by the scalar value. The ideal point is updated from feasible children; if the population has no feasible point yet, the best available objectives are used so the scalariser stays finite.

### Returned front

The front is read from the final population, not from a history of every non-dominated point ever seen. A history keeps early points that were never dominated but sit far behind the converged front, which inflates the point count and worsens IGD without telling the caller what the run achieved. Points with identical objective vectors are merged. If no feasible point exists, the least infeasible points are returned as a diagnostic.

### Complexity

One generation costs `N` evaluations of `O(dim)` plus `O(N * T * m)` scalar work, with `N` subproblems, neighbourhood `T` and `m` objectives. Memory is `O(N * dim + N * m)`. The weight lattice and the neighbourhoods are built once.

### Defaults

| Setting | Default | Meaning |
|---|---|---|
| population | 100 | target subproblems |
| divisions | derived | largest lattice that fits |
| neighbourhood | 20 | `T` |
| delta | 0.9 | local mating probability |
| replacements | 2 | `nr` |
| scalar | Tchebycheff | decomposition |
| F, CR | 0.5, 1.0 | MOEA/D-DE operator |
| mutation eta | 20 | polynomial mutation index |
| budget | 50 000 | 500 generations at 100 subproblems |
| boundary | clip | standard for MOEA/D |

Tuning notes, in the order that usually matters:

- `replacements`: 2 is the paper setting and keeps diversity. Raising it to 5 improved ZDT1 at 30 variables from IGD 0.0093 to 0.0066 at the same budget, and narrows the front.
- `neighbourhood`: larger values slow convergence and improve spread. At 30 variables, `T = 10` beat `T = 20` on ZDT1 at a short budget, and `T = 40` was worse than both.
- `delta`: 0.9 keeps most mating local, which is the point of the decomposition. Raising it to 1.0 is reasonable for a front with many objectives.
- `scalar`: Tchebycheff for two objectives, PBI for three or more.

## Quality indicators

`src/metrics` implements the indicators the reports use.

- **Hypervolume**: the volume of the region dominated by the front and bounded by a reference point. Exact in two dimensions by a sweep over sorted points, exact in three dimensions by slicing along the third axis and sweeping each slice, and seeded Monte Carlo above three with a reported 95 percent error. Hypervolume is Pareto compliant: a front that dominates another always scores higher.
- **IGD**: mean distance from each reference point to the nearest point of the approximation. It rewards convergence and spread at once, and it is the number in the benchmark table. Lower is better.
- **Generational distance**: mean distance from each approximation point to the nearest reference point. It measures convergence only. Lower is better.
- **Spacing**: standard deviation of the nearest neighbour distances inside the front. Lower means a more even spread.

Reference fronts are generated, not downloaded. ZDT1, ZDT2, ZDT4 and ZDT6 are parameterised by the first variable, so sampling it gives the exact front. ZDT3 is five disconnected arcs, sampled per arc using the published interval roots. DTLZ1 and DTLZ2 use a simplex lattice; DTLZ1 maps it to the plane `sum(f) = 0.5`, DTLZ2 normalises it onto the unit sphere.

## Benchmark problems

The suite includes four single objective problems and seven multi-objective ones. Definitions follow the original papers, with `x_1` the first variable and `n` the dimension.

| Problem | Objectives | Front or optimum |
|---|---|---|
| sphere | 1 | `f = sum(x^2)`, zero at the origin |
| rosenbrock | 1 | `f = sum(100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)`, zero at all ones |
| rastrigin | 1 | lattice of local minima, zero at the origin |
| ackley | 1 | flat rim, one well, zero at the origin |
| zdt1 | 2 | convex, `f2 = 1 - sqrt(f1)` |
| zdt2 | 2 | concave, `f2 = 1 - f1^2` |
| zdt3 | 2 | five disconnected arcs |
| zdt4 | 2 | multimodal, many local fronts |
| zdt6 | 2 | non-uniform density, `f1` in about `[0.28, 1]` |
| dtlz1 | 3 | linear, `sum(f) = 0.5` |
| dtlz2 | 3 | spherical, `sum(f^2) = 1` |

ZDT4 keeps `x_1` in `[0, 1]` and every other variable in `[-5, 5]`. DTLZ1 and DTLZ2 use `k = n - m + 1` tail variables and the standard `g` functions.

## Reproducibility

Every engine draws from a xoshiro256++ generator seeded by `seed`. The generator is implemented in the crate, so results do not drift with a dependency update. Parallel evaluation does not touch the generator. Two runs with the same configuration and any thread count produce identical objectives, decision vectors and history, which the test suite checks against a one thread and a four thread pool.

## References

- Storn, R. and Price, K. (1997). Differential evolution, a simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization 11, 341-359. DOI 10.1023/A:1008202821328
- Tanabe, R. and Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. IEEE Congress on Evolutionary Computation. DOI 10.1109/CEC.2013.6557555
- Tanabe, R. and Fukunaga, A. (2014). Improving the search performance of SHADE using linear population size reduction. IEEE Congress on Evolutionary Computation. DOI 10.1109/CEC.2014.6900380
- Brest, J., Greiner, S., Boskovic, B., Mernik, M. and Zumer, V. (2006). Self-adapting control parameters in differential evolution. IEEE Transactions on Evolutionary Computation 10(6), 646-657. DOI 10.1109/TEVC.2006.872133
- Zhang, Q. and Li, H. (2007). MOEA/D, a multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation 11(6), 712-731. DOI 10.1109/TEVC.2007.892759
- Li, H. and Zhang, Q. (2009). Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II. IEEE Transactions on Evolutionary Computation 13(2), 284-302. DOI 10.1109/TEVC.2008.925798
- Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm, NSGA-II. IEEE Transactions on Evolutionary Computation 6(2), 182-197. DOI 10.1109/4235.996017
- Das, I. and Dennis, J. E. (1998). Normal-boundary intersection, a new method for generating the Pareto surface in nonlinear multicriteria optimization problems. SIAM Journal on Optimization 8(3), 631-657. DOI 10.1137/S1052623496307510
- Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms, empirical results. Evolutionary Computation 8(2), 173-195. DOI 10.1109/4235.843494
- Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2005). Scalable test problems for evolutionary multiobjective optimization. Evolutionary Multiobjective Optimization, 105-145. DOI 10.1109/TEVC.2004.826067
- While, L., Bradstreet, L. and Barone, L. (2012). A fast way of calculating exact hypervolumes. IEEE Transactions on Evolutionary Computation 16(1), 86-95. DOI 10.1109/TEVC.2012.2206644
