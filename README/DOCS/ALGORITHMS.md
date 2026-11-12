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

## Corpus cleaning and pair mining

The optimiser answers questions about a model. This layer answers the question in front of it: which text is worth keeping, and what pairs should one model be trained on. Every rule below is a published one, and where the default differs from the paper the reason is stated.

### Normalisation and screening

Characters are folded rather than transliterated: curly quotes to straight, dash variants to hyphen, non breaking spaces to a space, soft hyphens and zero width marks removed, full width forms mapped to ASCII, and other control characters dropped. Text is then stripped of HTML tags, script, style and comment blocks, and the named entities that appear in wire copy are decoded. Links, emails and phone numbers are removed as tokens, so the words around them survive.

Boilerplate is screened by line. A line is dropped when it carries a known consent, subscription or navigation phrase and is at most 22 words long, or when it is pure navigation (two or more pipes with few words, or a title case run with no terminator), or when it repeats an earlier line of the same document. Every removal is counted in a report, because a cleaner that silently deletes a fifth of a corpus is indistinguishable from one that corrupts it.

### Quality statistics and rules

Nine statistics are measured per document: word count, mean word length, share of letters, share of digits, share of function words, type to token ratio, repeated line share, most repeated content word share, and words per sentence. The rule set is in the shape of the Gopher quality filters (Rae et al 2021, arXiv:2112.11446) and the C4 line rules (Raffel et al 2020), with the length floor lowered from 50 words to 5 and the symbol ceiling raised, because this crate is pointed at headlines and ad copy rather than web pages.

The defaults: 5 to 3000 words, mean word length 2.8 to 12, at least 0.04 function words, at most 0.45 uppercase, at most 0.22 punctuation plus digits, at most 0.30 repeated lines, at most 0.20 for one content word, at most 0.60 list lines. Each failure has a name, so a rejected document can be explained in one line.

The composite score is a weighted sum of six bounded components: length (peaking at 40 words), letters, variety, repetition, sentence structure and function words. It is a fallback, not a truth. When labelled keep and drop sets exist, `QualityModel` fits a logistic regression over the same nine statistics, balanced by class weight, with L2 shrinkage and full batch gradient descent from zero, so two runs produce the same weights and the weights can be printed and argued with.

### Near duplicate clustering

MinHash over word shingles, with double hashing for the permutation family: `a_i * x + b_i mod 2^64` with `a_i` odd, coefficients from one SplitMix64 stream. The signature is cut into bands and two documents that agree on a whole band become candidates, with the standard collision law `P(s) = 1 - (1 - s^rows)^bands`.

The default is 14 bands of 8 rows over 112 permutations, the configuration datatrove released with FineWeb (Penedo et al 2024, arXiv:2406.17557). Its crossing point is 0.72, so a pair at Jaccard 0.8 is found with probability 0.92. The Lee et al 2021 setting (arXiv:2107.06499) of 450 bands of 20 rows, 9000 permutations, is available by setting `perms` and `bands`; its crossing point is 0.74 with a much steeper curve. Candidates are verified with the signature estimate, which is unbiased for Jaccard and free.

The shingle is 3 words by default rather than the 5 the web pipelines use, because a six word headline produces two 5-grams and a near duplicate headline is exactly the case this pass exists to catch. A body corpus should raise it to 5.

Exact duplicates are merged first on a hash of the cleaned text. One survivor per cluster remains: highest quality score, then lowest index, so two runs on the same input keep the same rows.

SimHash and Hamming distance are provided for callers that need a single 64 bit fingerprint; the deduplication pass does not use them, because at a Hamming threshold of 3 a 64 bit SimHash collides often enough that banded MinHash is both cheaper to reason about and better calibrated.

### Selection

Two rules, and they answer different questions.

The coverage rule is thresholded maximum marginal relevance: order by the composite score, accept a document only when its cosine similarity to every accepted document is below `theta` (default 0.85), and refuse a source that has spent its group cap. Features are hashed unigrams and bigrams with an inverse document frequency weight, stopwords removed, vectors normalised, and the similarity test runs through an inverted index over accepted documents with an early exit, because every value is positive and a partial dot product is a lower bound.

The target rule is importance resampling (DSIR, Xie et al 2023, arXiv:2302.03169): unigram and bigram counts hashed into `m` buckets (10,000 in the paper), a target density `p` from a small curated set and a proposal density `q` from the corpus, log weight `sum count * (ln p - ln q)` with a 1e-8 floor, and Gumbel top-k to draw without replacement. The paper reports +1.18 average F1 over random selection on the Pile at 25M selected documents, and a 100M document pass in 4.5 hours on one 96 core node. No weight clipping is applied, because the paper applies none.

### Pair mining

Three banks, one schema. `rank` pairs the top quartile of a session with the bottom quartile, from different groups. `preference` pairs two documents on the same group with a clear gap in outcome, inside a 60 row and 30 day window. `contrast` pairs an anchor with a positive from the same group in the previous five sessions and a negative from the same session in another group.

Three guards come straight from the retrieval literature. The negative is drawn by similarity, never at random, so it is hard. It is sampled from the hardest five rather than the single hardest, because the hardest neighbour is often a document that agrees with the anchor and carries a different label; that false negative is the failure ANCE (arXiv:2007.00808), RocketQA (arXiv:2010.08191) and Robinson et al (arXiv:2010.04592) all report. And a pair whose two sides are more similar than 0.95 is dropped, so the model cannot win on one word.

A pair never crosses a split, and a pair whose outcome gap is below `min_margin` is dropped rather than clipped. The contrast bank is the exception on margins: its label is temporal and cross group, so it records both outcomes and no margin, and the audit reports margins only where a task claims one.

### Measured

On this machine, a release build of `clean` over 100,000 rows of short market copy (45 MB, 12.9 million characters, 61 words per row) took 2.6 s wall clock with 240 MB peak memory: 11,256 rows dropped by the rules, 1,059 merged as near duplicates, and 20,000 kept after the cap and the second pass. Signature building is the parallel part; bucketing and selection are serial. The selection pass keeps an inverted index over the accepted rows in memory, so it is comfortable to about a million rows and should be run per shard above that.

### The audit

Six checks, and the command exits 1 when any of the first five finds something: a document id under two splits, a text under two splits, the same text on both sides, two sides above the triviality threshold, contamination against an evaluation set, and thin margins. Contamination has two rules: at least 80 percent of an evaluation document's 3-grams, or a verbatim run of 13 words, which is the overlap rule GPT-3 used (Brown et al 2020, arXiv:2005.14165). Length bias and the margin distribution are reported, not enforced, because a corpus can legitimately be skewed and the reader should see it.

### What is not implemented

Semantic deduplication (SemDeDup, arXiv:2303.09540) needs one embedding per document and removes 37 percent of LAION with no measured loss; it is the right second pass after MinHash. The crate now carries a small encoder (see The transformer), so the blocker is no longer a dependency: a pass that encodes each survivor, clusters in the pooled space and keeps one member per cluster would fit. It is still not implemented, and neither is QuRating (arXiv:2402.09739), which needs a 1.3B parameter rater rather than a small encoder. Both are named so a reader knows what the gap is rather than assuming the pass is complete.

## The transformer

The pair corpus feeds one model rather than six. A single pre norm encoder is shared, and the task selects a linear head over the pooled representation.

### Tokenizer

A byte level byte pair encoding over lowercased word types, weighted by how often each type appears. The vocabulary starts with 260 tokens (four control tokens and the 256 byte values) and grows by merging the most frequent adjacent pair until the requested size or until no pair reaches `--min-count`. Pairs are ranked by count and then by the pair itself, and word type indices are sorted before a merge rewrites them, so two runs on the same text produce the same merges. Encoding never emits an unknown token: a word outside the vocabulary is still a sequence of bytes. Training merges over word types rather than raw text keeps the pair counting linear in the number of distinct words.

### Encoder

For a sequence of length `n`, width `d`, `h` heads and `f` feed forward width:

```text
x0 = token embedding + position embedding
xk+1 = xk + attention(layer_norm(xk))
xk+2 = xk+1 + feed_forward(layer_norm(xk+1))
pooled = mean(xL)
h = layer_norm(pooled)
score = w_task . h + b_task
```

Attention is bidirectional, which is what an encoder wants, and a sequence is one document, so there is no padding mask. The feed forward block is linear, the tanh approximation of GELU, then linear. The parameter count is `vocab d + max_len d + L (4 d^2 + 2 d f + 9 d + f) + 2 d + tasks (d + 1)`, where the per block terms are four attention matrices, two feed forward matrices, four biases, four layer norm parameters, and the two feed forward biases.

### Losses

A pairwise task pushes the difference of the two scores through a logistic, so the loss is the Bradley-Terry negative log likelihood `softplus(-(s_a - s_b))` and the gradient to the first score is `sigmoid(s_a - s_b) - 1`. A pointwise task uses binary cross entropy on the direction of the outcome, `softplus(s) - y s`, with gradient `sigmoid(s) - y`. Both are written with `softplus` rather than a logarithm of a sigmoid, so a confident wrong pair does not overflow. The tests check both gradients against a central finite difference.

### Training

AdamW with `beta1 = 0.9`, `beta2 = 0.999`, epsilon `1e-8`, and decoupled weight decay on matrices only: a layer norm gain and a bias are a single row, so decaying them shrinks the function rather than the weights. The learning rate warms up linearly over the first tenth of the schedule and then follows a cosine to a tenth of the peak. The global gradient norm is clipped after the batch average and before the step.

An epoch samples the training rows without replacement, weighted by task. Each row draws an exponential key scaled by its task weight and the smallest keys come first, which is weighted sampling without replacement and keeps a row from appearing twice in one epoch. Gradients accumulate over a batch of sequences and are applied once. Dropout is applied to the attention projection and the feed forward projection, scaled by `1/(1-p)`.

### Parallel steps and resuming

A batch is filled with one gradient buffer per example, each computed on whichever rayon thread takes it, and the buffers are added into the batch gradient in batch order. Nothing is shared while a buffer is being written and nothing is reduced out of order, so the update is the serial one to the bit: thread count changes the wall clock and nothing else. On the development machine, three epochs over 4,191 pairs at width 64 and length 64 took 9.8 s on one thread, 5.2 s on two, 3.6 s on four and 3.0 s on eight, and every thread count ended on the same weights and the same validation accuracy.

Two counters drive the randomness, and neither is a stream that advances with history. A dropout mask comes from a splitmix64 mix of the run seed, the optimizer step and the row position; an epoch permutation comes from the same mix of the run seed and the epoch number. A row therefore sees the same mask whether it is the first or the last of a batch, and epoch five is the same permutation in a resumed run as in an uninterrupted one. This is what makes a checkpoint a real pause rather than an approximation: the state to carry is the weights, the two Adam moments, the step counter, the epoch cursor and the tokenizer, and nothing else. The file stores floats as sixteen hex digits of their bit pattern, because the decimal JSON round trip moves some `f64` values by one unit in the last place, which a few hundred Adam steps amplify into visible drift; with the bit pattern a run resumed at an epoch boundary produces a model file identical to the uninterrupted run's, which the integration test asserts.

The plan and the leg are separate. `--epochs` sets the schedule's length and `--stop-after` ends a leg early, so splitting a run at an epoch boundary does not move the cosine, and a resume without `--epochs` takes the plan from the checkpoint.

Validation uses the split field the pair schema carries. Early stopping keeps the epoch with the best mean accuracy across the tasks that are present, and the file that is written is that epoch.

### Correctness and cost

The backward pass is hand written. The model test perturbs forty randomly chosen scalars across every parameter block, compares the analytic gradient with a central finite difference of `0.5 score^2`, and requires a relative agreement of `1e-4`. Layer normalisation, GELU and both losses have their own finite difference tests. A separate test round trips the model file and compares scores.

Forward and backward cost `O(n^2 d + n d f)` per block, so a sequence of 64 tokens at width 64 with two blocks is about 3 million multiply accumulates forward. On one core in release a pair of short documents costs roughly half a millisecond at that size, which is what the measured run on 19,108 pairs reflects: 12.5 seconds for one epoch at width 48. The cost of a step is dominated by that per pair work, which is why the next speed step was threads rather than a wider model: at width 64 the 4,191 pair corpus spends 9.8 s single threaded and 3.0 s on eight threads, a factor of 3.3 on a machine with four performance cores.

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
