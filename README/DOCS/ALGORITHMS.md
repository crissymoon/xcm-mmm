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
4. Choose `x_pbest` from the best `max(2, round(p NP))` points, with `p = 0.11`. Setting `p_best_final` moves `p` linearly towards that value over the run, which is the jSO schedule of 0.25 down to 0.125. It is off by default; the measured reason is in the next section but one.
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

### The x_pbest schedule, measured and left off

jSO narrows `p` from 0.25 to 0.125 as the budget is spent, and the crate can do the same with `p_best_final`. On this suite it does not pay. Median of 30 seeds at 30 000 evaluations in 30 variables, constant `p = 0.11` against the jSO schedule:

| problem | p = 0.11 | 0.25 to 0.125 | ratio |
|---|---:|---:|---:|
| sphere | 2.690e-13 | 2.654e-13 | 0.99 |
| rosenbrock | 2.268e+01 | 2.296e+01 | 1.01 |
| rastrigin | 5.140e+00 | 5.932e+00 | 1.15 |
| ackley | 7.923e-13 | 7.141e-13 | 0.90 |

Rastrigin loses 15 percent, the other three do not move beyond seed noise. The narrower pool late in the run is meant to sharpen a search that is already close, and L-SHADE with its own population reduction is already sharp; the wider pool early costs more than the late narrowing returns. The knob stays for callers who want the published schedule, and the default stays constant.

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

`stall_generations` turns on a partial restart. When the best score has not improved for that many generations, the worse half of the population is drawn again, the parameter memory returns to 0.5, and the archive is cleared. Zero, the default, turns this off, because a restart changes the budget semantics.

The trigger is deliberately strict: any strict improvement in the best score resets the counter, so with the default tolerance of `1e-14` a value of 20 or 50 almost never fires on this suite. At 5 and 10 it fires often, and on Rastrigin it hurts, from a median of 5.14 to 16.0 and 15.0 over 30 seeds at 30 000 evaluations. Restarting the worse half discards the parameter memory that L-SHADE had spent the run learning, which costs more than the jump is worth here. The option is therefore left for landscapes where the run truly stalls, and it is not a default.

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
| p_best_final | 0 | end of the schedule, 0 holds `p_best` |
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

### Partial updates and resource allocation

Plain MOEA/D-DE produces one child per subproblem per generation. That is not what this engine does by default. Each generation first selects a subset of subproblems, and only those produce a child. Two schedules are available.

The uniform random subset, the strategy measured by Lavinas, Aranha, Ladeira and Campelo (2020), draws each subproblem independently:

```text
active = { i : U_i < p_s }                    p_s = update_probability
```

The relative improvement schedule, after the dynamic resource allocation of Zhang, Liu and Li (2009), keeps a priority per subproblem. Each generation it measures how much the subproblem's scalar value `g_i` improved and accumulates that with a window of `W = ri_window` generations:

```text
delta_i = max(0, (g_i_old - g_i_new) / |g_i_old|)
pi_i   <- (1 - 1/W) pi_i + delta_i
```

A subproblem that has not improved for `W` generations is reset to the mean priority, so a subproblem that will only pay off late is retried instead of being abandoned. Selection is independent per subproblem, proportional to priority, and never returns an empty set:

```text
P(i in active) = min(1, p_s pi_i / mean(pi))   p_s = update_probability
```

For the first `update_delay` share of the planned generations every subproblem is active, which is the two phase warm-up of the relative improvement rule. A subproblem that is not active can still be replaced by a neighbour's child, because resource allocation limits variation, not replacement.

The defaults are `p_s = 0.2` and `ri_window = 50`, against 1.0 and no allocation for plain MOEA/D-DE. Lavinas et al. found that a random subset beats both no allocation and the relative improvement rule, and that the gain grows as the subset shrinks, because the population gains inertia: most incumbents survive a round, so the front moves by selection rather than by churn. Their 2021 follow up explains the mechanism: partial update converges as fast as a small population and explores as widely as a large one, so it removes most of the sensitivity to the population size choice.

On this suite the random subset reproduces that result against no allocation, and the improvement weighting then adds to it. Thirty seeds, 50 000 evaluations, mean IGD over the seven problems falls from 0.0253 at no allocation to 0.0207 with a random fifth, and then to 0.0200 when that fifth is weighted by relative improvement. The weighting wins at shares of 0.1, 0.2 and 0.4, on IGD and on hypervolume, with the gain concentrated in ZDT1 (0.0079 to 0.0061) and ZDT3 (0.0128 to 0.0117); ZDT6 and both DTLZ rows do not move. The tables are in [Benchmarks](BENCHMARKS.md).

Resource allocation of this shape has one failure mode worth naming: without the stagnation reset the priorities drift onto the subproblems that improved early, and the tail of the front stops being sampled. The reset is the defence, and this implementation gives a stale subproblem the current mean priority rather than a fixed one, which keeps the expected active set equal to the requested share. The measured gain over the random subset is consistent with that argument, but the two rules were not compared in isolation, so the attribution is open.

### One generation

For each active subproblem `i`:

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

Under partial update a generation costs about `p_s N` evaluations, so the same budget is spent over roughly `1 / p_s` more generations. The evaluation budget is still the stop condition and the number of objective evaluations does not change; what grows is the per generation bookkeeping and, for a cheap problem, the wall clock. On the benchmark suite the default costs about 1.6 times the plain run at the same 50 000 evaluations, of which the relative improvement bookkeeping is around 8 percent: one scalar value per subproblem per generation, `O(N m)`, which is small next to `O(N T m)` in the replacement loop.

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
| update probability | 0.2 | share of subproblems active per generation |
| relative improvement window | 50 | stagnation window; 0 draws the share uniformly at random |
| update delay | 0.1 | warm-up share before the partial schedule |
| budget | 50 000 | 500 generations at 100 subproblems |
| boundary | clip | standard for MOEA/D |

Tuning notes, in the order that usually matters:

- `update_probability`: the largest single lever. 30 seeds at 50 000 evaluations, mean over all seven problems: mean IGD falls from 0.0253 at 1.0 to 0.0209 at 0.4, 0.0207 at 0.2 and 0.0206 at 0.1 with the uniform random subset, and to 0.0203, 0.0200 and 0.0203 with relative improvement; mean hypervolume rises from 0.6999 to about 0.707. ZDT3 and DTLZ1 improve by about 30 percent, DTLZ2 is flat. The default is 0.2, which keeps the gain without the very small active sets that cost ZDT4 at 0.1.
- `ri_window`: 50 is the measured default. Setting it to 0 returns to the uniform random subset and costs about 3 percent of mean IGD, mostly on ZDT1. At 10 the mean was indistinguishable from the random subset, and at 200 the ZDT1 gain was paid for by a loss on ZDT2 and ZDT3, so 50 is the flat part of the curve rather than a sharp optimum.
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
- Brest, J., Maucec, M. S. and Boskovic, B. (2017). Single objective real-parameter optimization, algorithm jSO. IEEE Congress on Evolutionary Computation. DOI 10.1109/CEC.2017.7969456
- Lavinas, Y., Aranha, C., Ladeira, M. and Campelo, F. (2020). MOEA/D with random partial update strategy. IEEE Congress on Evolutionary Computation. DOI 10.1109/CEC48606.2020.9185527, arXiv 2001.06980
- Lavinas, Y., Ladeira, M. and Aranha, C. (2021). Faster convergence in multi-objective optimization algorithms based on decomposition. arXiv 2112.11939
- Brest, J., Greiner, S., Boskovic, B., Mernik, M. and Zumer, V. (2006). Self-adapting control parameters in differential evolution. IEEE Transactions on Evolutionary Computation 10(6), 646-657. DOI 10.1109/TEVC.2006.872133
- Zhang, Q., Liu, W. and Li, H. (2009). The performance of a new version of MOEA/D on CEC09 unconstrained MOP test instances. IEEE Congress on Evolutionary Computation, 203-208. DOI 10.1109/CEC.2009.4982949
- Zhang, Q. and Li, H. (2007). MOEA/D, a multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation 11(6), 712-731. DOI 10.1109/TEVC.2007.892759
- Li, H. and Zhang, Q. (2009). Multiobjective optimization problems with complicated Pareto sets, MOEA/D and NSGA-II. IEEE Transactions on Evolutionary Computation 13(2), 284-302. DOI 10.1109/TEVC.2008.925798
- Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm, NSGA-II. IEEE Transactions on Evolutionary Computation 6(2), 182-197. DOI 10.1109/4235.996017
- Das, I. and Dennis, J. E. (1998). Normal-boundary intersection, a new method for generating the Pareto surface in nonlinear multicriteria optimization problems. SIAM Journal on Optimization 8(3), 631-657. DOI 10.1137/S1052623496307510
- Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms, empirical results. Evolutionary Computation 8(2), 173-195. DOI 10.1109/4235.843494
- Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2005). Scalable test problems for evolutionary multiobjective optimization. Evolutionary Multiobjective Optimization, 105-145. DOI 10.1109/TEVC.2004.826067
- While, L., Bradstreet, L. and Barone, L. (2012). A fast way of calculating exact hypervolumes. IEEE Transactions on Evolutionary Computation 16(1), 86-95. DOI 10.1109/TEVC.2012.2206644

The literature behind these choices is pulled, not remembered. `scripts/research.py` queries arXiv, Crossref and OpenAlex through the house scrape tool, caches every response under `research/raw`, and merges the records into `research/papers.json`, 146 papers across 28 mechanism specific queries at the time of writing. The digest groups them by mechanism and prints the abstract, so a claim in this document can be traced to a retrieved abstract rather than to a recollection of one.
