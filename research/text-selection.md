# Keeping the most relevant text: what the literature actually specifies

One page per finding, with the identifier checked against the arXiv abstract, because a parameter that is quoted from a blog is worth less than one quoted from the released pipeline. Every entry ends with the line that says what it costs in Rust. The companion note on pairs is `pairs.md`.

## 1. Gopher quality filters

**Title** Scaling Language Models: Methods, Analysis and Insights from Training Gopher
**Authors** Jack W. Rae et al, 2021
**Identifier** arXiv:2112.11446
**URL** https://arxiv.org/abs/2112.11446

**Rule** A document is dropped when it fails any of: word count outside 50 to 100,000, mean word length outside 3 to 10 characters, more than 10 percent of words without an alphabetic character, more than 10 percent of words carrying a symbol, fewer than 80 percent of words containing a stop word, more than 90 percent of lines starting with a bullet, more than 30 percent of lines ending with an ellipsis, or a repeated line share above 30 percent. A separate repetition filter drops documents where a single 2-gram occupies more than 20 percent of the text, or the most common 3-gram more than 18 percent, or the top 4-gram more than 16 percent.

**Effect** Reported as a data quality ablation rather than a per-rule count; the paper's point is that the rules are cheap and explainable.

**Rust** Nine counters and eight comparisons. Implemented in `src/text/quality.rs` as `Rules::flags`, with the length floor lowered to 5 and the symbol ceiling raised because the corpus here is headlines and ad copy. Every failure returns a named `Flag`, so a rejection can be printed.

## 2. C4 line rules

**Title** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
**Authors** Colin Raffel et al, 2020
**Identifier** JMLR 21(140)
**URL** https://www.jmlr.org/papers/v21/20-074.html

**Rule** Keep only lines that end in a terminal punctuation mark, drop a page with fewer than three sentences, drop a sentence with fewer than five words, and drop pages containing any word on a bad-word list.

**Effect** The cleaned C4 release is the reference point every later web pipeline measures against.

**Rust** The terminator and sentence-length rules are in `Rules`; the bad word list is deliberately not implemented, because a profanity list is corpus policy rather than an algorithm and belongs in the caller.

## 3. FineWeb and the datatrove MinHash step

**Title** The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
**Authors** Guilherme Penedo, Hynek Kydlicek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, Thomas Wolf, 2024
**Identifier** arXiv:2406.17557
**URL** https://arxiv.org/abs/2406.17557

**Rule** Before deduplication text is simplified: lowercased, whitespace collapsed, punctuation removed, diacritics and Unicode normalised, numbers normalised, then word tokenised and shingled at `n_grams = 5`. The released config is `num_buckets = 14`, `hashes_per_bucket = 8`, `seed = 1`, `HashConfig(precision = 64, hash_fc = "xxhash")`, so 112 permutations. Permutations are the universal family `h_i(x) = (a_i x + b_i) mod 2^61 - 1`; a shared packed bucket is a candidate. The file's own docstring states the crossing point as `(1/14)^(1/8) = 0.72` with `P(0.8) = 0.924`, `P(0.7) = 0.565`, `P(0.6) = 0.211`. The paper describes the same configuration as targeting documents at least 75 percent similar, which is the recall intent, not the computed crossing point.

**Effect** FineWeb reports the corpus with an ablation of each filter, and publishes the deduplicated release that the 14 by 8 setting comes from.

**Rust** This is the default in `src/text/dedupe.rs`: 14 bands of 8 rows over 112 permutations, double hashing instead of 61 bit modular arithmetic because the modulus buys nothing at 112 permutations, and a 3 word shingle because a six word headline would otherwise have two shingles. Use 5 for body text.

## 4. NearDup, the strict setting

**Title** Deduplicating Training Data Makes Language Models Better
**Authors** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, Nicholas Carlini, 2021
**Identifier** arXiv:2107.06499
**URL** https://arxiv.org/abs/2107.06499

**Rule** Space tokenise, hash every 5-gram, keep the minimum under each of `k = 9000` permutations, split the 9000 minima into `r = 450` buckets of `b = 20`, and treat a shared bucket as a candidate. Verify with the true Jaccard, and require Jaccard above 0.8 **and** edit similarity above 0.8. Collision law `P = 1 - (1 - s^b)^r`; crossing point 0.737, `P(0.8) = 0.995`.

**Effect** Removed 3.04 percent of C4, 0.39 percent of Wiki-40B and 13.63 percent of RealNews training examples. 1.59 percent of C4 validation examples had a near duplicate in train, 4.60 percent inside validation. Memorised generations fell by about 10x and perplexity did not rise.

**Rust** 9000 permutations is 80x the default work per document; expose it, do not default to it.

## 5. Semantic deduplication

**Title** SemDeDup: Data-efficient learning at web-scale through semantic deduplication
**Authors** Amro Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, Ari S. Morcos, 2023
**Identifier** arXiv:2303.09540
**URL** https://arxiv.org/abs/2303.09540

**Rule** Embed every document, run k-means, compare only inside a cluster, mark a pair as a duplicate when `cos >= 1 - epsilon`, take connected components, and keep the member with the **lowest** cosine to its centroid. Cluster counts: 11,000 for C4, 50,000 for LAION-440M. The threshold is tuned per dataset to a retention target by running 10 percent of clusters and interpolating; at epsilon 0.03 half of LAION has a duplicate.

**Effect** 37 percent of LAION removed with no measured ImageNet loss, 50 percent with under 0.5 percent loss, and training twice as fast at 50 percent removal. On C4 it matches MinHash at 80 percent retention while MinHash keeps 96.1 percent, which is the whole argument: MinHash needs shared n-grams, so paraphrases and templated boilerplate pass it.

**Rust** Needs an embedding model or an imported table of vectors. Not implemented; `ALGORITHMS.md` names the gap rather than pretending the pass is complete.

## 6. Target-aware importance resampling

**Title** Data Selection for Language Models via Importance Resampling
**Authors** Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang, 2023
**Identifier** arXiv:2302.03169
**URL** https://arxiv.org/abs/2302.03169

**Rule** Hash unigram and bigram counts into `m = 10,000` buckets. Estimate a target density `p` from a small curated set and a proposal density `q` from the raw corpus as normalised counts. Weight each document by `log w = sum_j count_j (log p_j - log q_j)`, with a `1e-8` floor inside the logs and no clipping. Resample `k` documents without replacement with probability proportional to `exp(log w)`, implemented as `topk(log w + Gumbel(0,1))`.

**Effect** Continued pretraining RoBERTa on 25M selected examples from the Pile: 83.76 average F1 against 82.58 random and 82.88 heuristic classification, so +1.18 over random; from-scratch pretraining gains 2.0 GLUE points over random. Top-k without the resampling step is 0.9 percent worse. 100M Pile documents in 4.5 hours on one 96 core node.

**Rust** Two count arrays of 10,000 `f64`, one pass per document, one sort. Implemented in `src/text/select.rs` as `Dsir`, wired to `clean --target`.

## 7. Model-based quality ratings

**Title** QuRating: Selecting High-Quality Data for Training Language Models
**Authors** Alexander Wettig, Aatmik Gupta, Saumya Malik, Danqi Chen, 2024
**Identifier** arXiv:2402.09739
**URL** https://arxiv.org/abs/2402.09739

**Rule** Rate each document on four axes (writing style, required expertise, facts and trivia, educational value) with a fine-tuned 1.3B rater trained on 250K GPT-3.5 pairwise judgements per axis. Select with `p_i = exp(s_i / tau) / sum exp(s_j / tau)`, `tau = 2.0`, sampled without replacement.

**Effect** Educational value gives +1.8 percent average in-context-learning accuracy across ten tasks over uniform selection, about the same as training 50 percent longer. Writing style improves perplexity but barely moves downstream accuracy.

**Rust** Needs a 1.3B forward pass or imported ratings. Out of scope for the same reason as SemDeDup.

## 8. Diversity as a selection objective

**Title** The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
**Authors** Jaime Carbonell, Jade Goldstein, 1998
**Identifier** DOI 10.1145/290941.291025
**URL** https://doi.org/10.1145/290941.291025

**Rule** Score a candidate as `lambda * relevance - (1 - lambda) * max similarity to anything already chosen`, and take the maximum each round.

**Effect** The ancestor of every diversity pass. The formal warrant for the greedy form comes later, in Nemhauser, Wolsey and Fisher 1978, which proves a `1 - 1/e` guarantee for monotone submodular objectives; what MMR adds is the redundancy term that stops a corpus from becoming one story.

**Rust** Implemented in `src/text/select.rs` in thresholded form: accept a document only when its cosine to every accepted document is below `theta`, checked through an inverted index with an early exit, since every tf-idf value is positive and a partial dot product is a lower bound. The greedy optimum is not claimed.

## 9. Implementation notes that changed the code

- **Parsing.** `serde_json`'s own documentation states that `from_reader` is usually slower than reading into memory and calling `from_slice`, because `IoRead` cannot seek; measured 55 ms for a 23 MiB file either way, and a derived struct parses 1.7 to 1.8 times faster than `Value` on the serde benchmark corpus (canada 320 vs 580 MB/s, citm 420 vs 710 MB/s, twitter 300 vs 550 MB/s). Source: https://github.com/serde-rs/json-benchmark. The pipeline keeps one `String` buffer and calls `from_str` per line, and reads gzip through the system binary so no crate is added.
- **MinHash cost.** datasketch on 113,333 synthetic 300 word articles with 128 permutations: 0.8 ms per article for fingerprinting on one core, 21 s to build the 16 band index, 3.5 s to query all. Source: https://www.sevydevy.com/learn/near-duplicate-detection. At 112 permutations and 100k documents the whole local pass, cleaning included, runs in 2.7 s on this machine.
- **Existing crates.** `simhash` (crate, 1.7M downloads) is fixed at 64 bits and has no band index; `minhash-rs` reports about 4.8x for parallel signature building at 1M signatures with 128 permutations and 16 bands; `lsh-rs` has no documentation and no published numbers. Source: https://docs.rs/minhash-rs/0.6.0/minhash_rs/. The house implementation stays in the crate because the corpus shape (short documents, 3 word shingles) is not what any of them default to.
- **Scale reference.** deduplicating 39M peS2o documents with MinHash LSH takes more than 35 hours on a 32 core node, against 3 hours for a bloom filter LSH. Source: arXiv:2411.04257. That is the honest cost of the pairwise candidate check, and the reason the pass here is parallel over signatures and serial over buckets.

## Recommended defaults, in the order they are applied

1. Fold characters, strip markup and decode entities before counting anything.
2. Screen boilerplate by line, with a 22 word ceiling, and report every removal.
3. Enforce a 5 word floor for short form corpora, 50 for body text.
4. Reject documents with almost no function words, which is the cheapest list detector.
5. Deduplicate exactly on a text hash first, then approximately.
6. Use 14 bands of 8 rows over 112 permutations for a 0.72 crossing point; raise to 450 by 20 with 9000 permutations when a 0.8 gate is required.
7. Shingle at 3 words for headlines, 5 for bodies.
8. Keep one survivor per cluster by quality, with an index tie break, so runs are reproducible.
9. Select with a redundancy threshold and a per-source cap, and fill any shortfall in a second pass that says how many rows it relaxed.
10. When a curated target exists, use importance resampling instead, because it is the only rule that knows what relevant means.
11. Mine negatives by similarity, sample from the hardest few, and refuse a pair whose sides are more similar than 0.95.
12. Audit leakage, triviality, contamination (80 percent of 3-grams or a 13 word run) and thin margins before training, and let the audit fail the build.
