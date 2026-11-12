# Fast Rust text dedup and corpus filtering: parameters, costs, sources

Scope: xcm-mmm targets 10 million short documents (80 to 900 characters), JSONL input,
dependencies limited to rayon, serde, serde_json, clap. No crate may be added, so every
sketch, hash and tokenizer below is written in std.

Measurement method. All local numbers were measured on 2026-11-11, Apple M5 (10 rayon
threads), rustc 1.99.1, release with thin LTO and codegen-units=1, serde_json 1.0.154,
rayon 1.12. Corpus: 200,000 synthetic JSONL lines, 97.6 MB, 488 bytes per line, 400-char
text field averaging 67 words. Published numbers are quoted with their source URL.

## 1. MinHash configuration and memory for 10M short documents

Standard configuration: 5-word shingles, k = 128 permutations, signature stored as u32,
LSH index of b = 16 bands of r = 8 rows. 128 is the datasketch default, and 16 x 8 is the
split used in the datasketch-based near-duplicate benchmark (0.8 ms per 300-word article).

Band math. Each band matches with probability s^r, at least one band matches with

    P(s) = 1 - (1 - s^r)^b          (S-curve)
    t    = (1/b)^(1/r)              (the similarity where the curve is 0.5)
    P(t) = 1 - (1 - 1/b)^b ~ 0.632  (1 - 1/e, so a threshold always loses 37% recall)

At k = 128 the usable splits (b x r) and their measured behaviour:

| b x r | t | P(0.80) | P(0.75) | P(0.60) | memory signature |
|-------|---|---------|---------|---------|------------------|
| 4 x 32 | 0.958 | 0.003 | 0.000 | 0.000 | 512 B/doc |
| 8 x 16 | 0.878 | 0.204 | 0.077 | 0.002 | 512 B/doc |
| 16 x 8 | 0.707 | 0.947 | 0.815 | 0.237 | 512 B/doc |
| 32 x 4 | 0.420 | 1.000 | 1.000 | 0.988 | 512 B/doc |
| 64 x 2 | 0.125 | 1.000 | 1.000 | 1.000 | 512 B/doc |

Consequences for a 0.80 target:
- Set t about 0.10 below the target (0.707, 16 bands x 8 rows). Recall at J = 0.80 is 94.7%.
- Do not try to put t on 0.80. At 20 x 13 (k=256, t=0.794) recall at 0.80 falls to 67.7%
  because the S-curve is 0.632 at t by construction. More permutations do not fix this;
  moving t does.
- 32 x 4 and 64 x 2 are useless for filtering: they return 98.8% and 100% of pairs at J=0.6.
- False positives cost one signature comparison (~1 us), false negatives lose documents.
  Choose recall and let the verifier clean up.

Memory for 10M documents:

| item | bytes/doc | 10M docs |
|------|-----------|----------|
| k = 128, u32 signature | 512 | 5.12 GB |
| k = 128, u64 signature | 1024 | 10.24 GB |
| k = 64, u32 signature (8 x 8, t=0.771, recall at 0.8 = 0.770) | 256 | 2.56 GB |
| LSH table, b=16, u64 band key + u32 doc id, load 0.7 | 192 | 2.74 GB |
| LSH table, b=25 | 300 | 4.29 GB |
| total, k=128 + b=16 | ~704 | 7.86 GB |

u32 signatures are mandatory here: they halve 10.24 GB to 5.12 GB, and 10M doc ids fit in
u32 too (limit 4.29B). datasketch 2.0.0 made the same move; its default "affine32" scheme
stores uint32 and "halves memory use compared to legacy". Hash each r=8 band (256 bits) down
to a u64 key; expected collisions over 160M entries are (1.6e8)^2 / 2^65 = 0.0007, about one
spurious band match in the corpus, which the verifier rejects.

Estimation error. Standard error of the Jaccard estimate is sqrt(J(1-J)/k):
k=64 gives 0.050 at J=0.8, k=128 gives 0.0354, k=256 gives 0.0250. A 0.70 cut on the
signature is 2.8 sigma below a true 0.80 at k=128, so signature-only verification is
safe at a 0.70 cut.

Recommended (10M docs, 0.80 target): 5-word shingles, k=128 u32, b=16, r=8, candidate
verification at signature J >= 0.70, 7.9 GB total. Low-memory variant if 8 GB is too
much: k=64, b=8, r=8, 3.9 GB total, 77% recall at 0.80.

Sources: https://ekzhu.com/datasketch/lsh.html (P(s), (b,r) choices at num_perm=128)
and https://en.wikipedia.org/wiki/MinHash (O(1/sqrt(k)) error, 400 hashes for 0.05).

## 2. Is 64-bit SimHash enough at Hamming distance 3?

No. At 64 bits and k=3, SimHash only detects documents that are almost literally identical
(J = 0.995 and above). For feature sets A and B, a random-hyperplane bit agrees with
probability

    p = 1 - arccos(cos) / pi,   cos = J / (2 - J) for equal-size binary sets
    E[hamming] = 64 * arccos(cos) / pi
    P(hamming <= 3) = sum_{i=0..3} C(64,i) p^i (1-p)^(64-i)

| J | cos | E[hamming] | P(h <= 3) | P(h <= 6) | P(h <= 10) |
|---|-----|------------|-----------|-----------|------------|
| 0.60 | 0.4286 | 23.0 | 3.5e-09 | 1.2e-06 | 2.9e-04 |
| 0.70 | 0.5385 | 20.4 | 1.0e-07 | 2.1e-05 | 2.5e-03 |
| 0.80 | 0.6667 | 17.1 | 5.1e-06 | 5.3e-04 | 2.6e-02 |
| 0.90 | 0.8182 | 12.5 | 6.8e-04 | 2.3e-02 | 2.7e-01 |
| 0.95 | 0.9048 | 9.0 | 1.6e-02 | 1.9e-01 | 7.2e-01 |
| 0.99 | 0.9802 | 4.1 | 4.2e-01 | 8.9e-01 | 1.00 |

Read the answer directly: at J = 0.80, P(hamming <= 3) = 5.1e-06, so 5 duplicate pairs per
million are found and 99.9995% are missed. At J = 0.60 it is 3.5e-09. The false positive
rate on an unrelated pair (J = 0, p = 0.5) is 2.4e-15, so k=3 is precise and blind, which
is exactly why Google could use 64-bit SimHash with k=3: Manku et al validated it on 8B
web pages whose duplicates differ only by ad blocks (J ~ 0.99).

Equivalence table, 64 bits, inverted: a threshold of 3 bits means J = 0.9946; 6 bits means
J = 0.978; 10 bits means J = 0.937; 17 bits means J = 0.80. A threshold of 17 bits is not
usable, since at J = 0.50 the mean Hamming distance is 25.1 and the distributions overlap.

128-bit SimHash is the smallest width that separates 0.8 from 0.6, but weakly: at J = 0.80
E[h] = 34.3, and a cut at d = 42 gives recall 0.947 at J=0.80, 0.628 at J=0.70, 0.264 at
J=0.60 and 0.082 at J=0.50, so 16 bytes/doc and a 26% false positive rate at J=0.6.
MinHash banding at 128 permutations gives the same 0.947 recall at 0.80 with a 0.237
candidate rate at 0.60 from the same 512 bytes, and its threshold is a Jaccard estimate
instead of a cosine surrogate. Prefer MinHash for a Jaccard target; keep SimHash only if
8 bytes/doc matters (80 MB for 10M) and only to catch exact and near-exact copies.

Sources: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33026.pdf
(64-bit fingerprints and k=3 validated for 8B pages; bit-agreement model),
https://en.wikipedia.org/wiki/SimHash (Charikar 2002; hamming distance to Jaccard).

## 3. Cheapest pure-std hashing, and k MinHashes from one 64-bit hash

Shingle hashing, measured on 40-byte 5-word shingles and on 400-byte documents, same
binary, black_boxed, 10M iterations:

| function | 40 B shingle | 400 B doc | note |
|----------|--------------|-----------|------|
| FNV-1a 64 | 2.90 GB/s, 13.8 ns | 1.22 GB/s, 329 ns | one dependent multiply per byte, 3-cycle latency chain |
| wyhash-style, 8 B rounds | 17.75 GB/s, 2.25 ns | 7.00 GB/s, 57 ns | 8 bytes per multiply |
| 4-lane xxhash-style | 17.82 GB/s, 2.24 ns | 22.13 GB/s, 18 ns | 2 multiplies per 8 bytes, 4 independent lanes |

FNV-1a is 6.1x more expensive per 40-byte shingle (13.8 vs 2.25 ns). For a 400-char doc
with 63 shingles that is 0.87 us versus 0.14 us, a saving of 0.7 us/doc or 7 s over 10M
documents. The 4-lane version wins on long text because four independent multiply chains
hide latency; on 40-byte shingles there is not enough work to fill them, so the simple
8-byte wyhash-style loop is as good and much shorter to write. Use the 8-byte loop for
shingles, and switch to 4 lanes only if you hash whole documents. All three are
non-cryptographic and fine for sketching; FNV-1a is the one to avoid in the hot loop, not
because it is insecure but because it is latency-bound.

k independent hashes from one 64-bit hash. Two standard tricks:

1. Affine universal hashing: h_i = (a_i * h + b_i) mod 2^32, with odd a_i (a bijection on
   2^32), b_i random. This is what datasketch 2.0.0 does by default ("affine32", input
   pre-mixed with the MurmurHash3 finalizer, values stored as uint32). It is collision-free
   within a permutation and gives independent-looking values.
2. Double hashing: h_i = h1 + i * h2 (Kirsch and Mitzenmacher, "Less Hashing, Same
   Performance"), with h1 and h2 from two independent mixes of the same 64-bit shingle.
   Requires h2 odd, or take h2 | 1. This is equivalent in quality to k independent hashes
   for Bloom filters and for MinHash in practice, at one multiply per value.

Cost measured for 128 affine updates from one mix: 1.66 ns per document. In the full
pipeline, 63 shingles x 128 updates take 1.5 us of the 2.25 us per doc, so the permutation
step is the cost, not the shingle hash. Do not use the mod-(2^61-1) form (datasketch's
legacy scheme): it needs u128 multiply and reduce, roughly 4x the arithmetic, for no gain.
Wrap in u64 and take the high 32 bits (low bits of an affine map over a wrap are poorly
mixed), or work in 2^32 with odd multipliers.

Sources: https://raw.githubusercontent.com/ekzhu/datasketch/master/datasketch/minhash.py
(affine32, odd a, uint32 storage), https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf
(double hashing), https://crates.io/crates/fnv and https://crates.io/crates/xxhash-rust
(954M and 139M downloads; both off-limits here, the implementations above are 20 lines).

## 4. Unicode-aware words and sentences without a crate

What std gives you. char::is_alphabetic is the Unicode Alphabetic property,
char::is_numeric is Nd+Nl+No, and char::is_alphanumeric is exactly the union of the two.
char::to_lowercase is the full unconditional Unicode mapping (may expand to 3 chars, for
example U+0130 to "i" + U+0307), with no locale tailoring. is_digit(radix) is ASCII only,
so never use it for text. Rust 1.75 ships Unicode 15.0.0, 1.99 ships 17.0.0. There is no
General_Category, grapheme cluster or normalisation API in std, so handle the cases below.

Word tokenizer that is correct enough:
- Scan char_indices and split on runs of is_alphanumeric. Measured on a 400-char doc: 398 ns
  with char::is_alphanumeric (including lowercase push) versus 414 ns with a byte and
  is_ascii_alphanumeric scan, so Unicode-awareness is free on ASCII. One code path.
- Keep U+0300 to U+036F (combining marks), U+200C (ZWNJ), U+200D (ZWJ) and the apostrophes
  U+0027 and U+2019 attached to the current token. UAX #29 rule WB4 says Extend, Format and
  ZWJ never start a boundary; a plain predicate splits "e" + U+0301 into two tokens.
- Do not fold U+2010 or U+2011 into a word; UAX #29 leaves them unlisted and breaks there.
  Treat hyphen as a separator, which is what shingle-based dedup wants.
- Han, Hiragana, Katakana and Thai letters are unlisted in UAX #29, so a run-based scanner
  turns a whole Chinese or Thai sentence into one token. Emit one token per such character
  (CJK bigrams for the shingle set) and accept that Thai needs a dictionary you do not have.
- Lowercase with char::to_lowercase().next() only on the ASCII fast path; for full folding
  push the whole iterator output, since one char can become three.

Sentence splitter. Three rules cover almost everything:
1. Break after . ! or ? (and U+3002, U+FF01, U+FF1F) when whitespace is followed by an
   uppercase letter, a digit, a quote or an opening bracket.
2. Never break after a digit (3.14, version 1.75) or after a single capital letter
   (initials, "J. Smith") unless the next token is not a capital.
3. Never break when the token before the period is in the abbreviation set below.

Abbreviation guards, minimum useful set (English):
Mr, Mrs, Ms, Dr, Prof, Sr, Jr, St, vs, etc, e.g, i.e, cf, Fig, No, approx, Dept, Univ,
Inc, Ltd, Co, U.S, a.m, p.m, and the months Jan, Feb, Mar, Apr, Jun, Jul, Aug, Sep, Sept,
Oct, Nov, Dec. Why: a period after a capital looks like a sentence end ("Mr. Jones"), a
truncation before a number or lowercase splits wrongly ("Fig. 3", "No. 5", "cf. the"),
corporate suffixes split before a preposition ("Acme Inc. in New York"), and multi-period
tokens must match whole or "e.g." becomes "e." + "g.". NLTK's Punkt ships a 156-type
abbreviation list, the right size to copy. UAX #29 states SB6 to SB8 cannot detect
"Mr. Jones" without such a tailoring.

Cost: a full punctuation scan of a 400-char doc is 191 ns. Tokenization plus sentence
splitting is under 0.5 us/doc, about 20% of the dedup pipeline.

Sources: https://doc.rust-lang.org/std/primitive.char.html,
https://www.unicode.org/reports/tr29/ (WB4, SB6-SB8, hyphens, apostrophes),
https://www.nltk.org/api/nltk.tokenize.punkt.html.

## 5. JSONL streaming: buffer reuse versus serde_json::from_reader

Measured on the 97.6 MB, 200,000-line corpus (488 B/line), warm page cache, single thread:

| method | MB/s | us/doc | note |
|--------|------|--------|------|
| from_reader over File, Vec<T> | 3.48 | 140.4 | one read syscall per byte, 680x slower |
| StreamDeserializer over File | 3.44 | 141.8 | same trap, same fix |
| BufReader(1 MB) + from_reader, Vec<T> | 473.7 | 1.03 | buffering fixes the syscall |
| BufReader(1 MB) + StreamDeserializer, T | 485.1 | 1.01 | still byte-at-a-time parsing |
| read_line reused + from_str, Value | 1325 | 0.37 | 1.8x slower than typed |
| read_line reused + from_str, typed struct | 2387 | 0.20 | best streaming option |
| read_until Vec + from_slice, typed | 2481 | 0.20 | fastest, avoids UTF-8 check |
| read_line reused, default 8 KB BufReader | 2156 | 0.23 | 12% slower than 1 MB |
| read_to_end + lines(), typed | 2217 | 0.22 | 98 MB resident, not streaming |

Findings:
1. from_reader without a BufReader is a disaster at 3.5 MB/s: serde_json's IoRead has no
   lookahead and pulls one byte per Read call, and File does a syscall each time. Its own
   docs say "counter to intuition, this function is usually slower than reading a file
   completely into memory and then applying from_str or from_slice on it".
2. Read a line into one String with s.clear() then read_line. read_line appends and
   String::clear keeps capacity, so after the first long line there is no per-line
   allocation. Borrow &str fields via #[serde(borrow)]; that works only with
   from_str/from_slice, never with from_reader.
3. read_until into a Vec<u8> plus from_slice skips UTF-8 validation of the whole line and
   was fastest at 2.48 GB/s; use it when invalid UTF-8 must be filtered, otherwise
   read_line is within 4%.
4. Never parse to serde_json::Value in the hot loop. It is 1.8x slower (1325 vs 2387 MB/s)
   and allocates per object: Map<String, Value> is a BTreeMap by default with one String
   per key and 32 bytes per Value, nothing shared. serde_json issue 635 measures a 10.8 MB
   document growing to about 188 MB as Value, over 10x the input. Use a typed struct with
   &str fields, or &RawValue for payloads you only forward.

Sources: https://docs.rs/serde_json/latest/serde_json/fn.from_reader.html,
https://github.com/serde-rs/json/issues/635,
https://doc.rust-lang.org/std/io/trait.BufRead.html (read_line appends).

## 6. Parallel dedup with rayon: the split, and why one global map dies

The split that works, three stages:
1. Map: read the file once to line byte offsets (or read chunks and split on newlines), then
   par_chunks over those offsets. Each worker parses its own lines, tokenizes, shingles,
   builds a k=128 u32 signature and inserts (band key, doc id) into its own private
   HashMap<u64, ...>. No shared state in this stage.
2. Shuffle: reduce each worker map by key high bits into S shards (S = thread count), so
   every key lands in exactly one shard and a shard is built by one thread.
3. Reduce: per shard, emit candidate pairs per band bucket; union-find or a sort of
   (min_id, max_id) pairs collapses duplicates, then verify by comparing signatures.

Measured, 200k documents, 16 bands, 3.2M inserts, 10 threads:

| approach | time | scale to 10M docs |
|----------|------|-------------------|
| global Mutex<HashMap> | 624 ms | about 31 s |
| per-thread maps | 22.9 ms | about 1.1 s |
| per-thread maps + single-threaded merge into one map | 22.9 + 177 ms | about 10 s |
| per-thread maps + 10-way partitioned merge | 22.9 + 31.6 ms | about 2.7 s |

Why a single global hash map is a bottleneck: every insert takes the mutex, so all workers
serialize on one lock and one cache line; the line ping-pongs between cores at roughly
100 ns per transfer, longer than the insert itself, and the structure degrades into a lock
convoy. Measured 27x slower than private maps and 20x slower end to end. The same applies
to Mutex<Vec>, to a global allocator under per-line String allocation, and to
Arc<Mutex<HashSet>> "seen" checks. Amdahl also caps the gain: a 10% serial merge limits 10
threads to 5.3x; partitioning the merge removes that serial part.

Only the signature build is embarrassingly parallel; the index build must be partitioned,
never locked. If memory allows, skip the merge and query the per-thread maps in parallel.

Parallel speedup in a comparable implementation: minhash-rs reports about 4.8x for
from_signatures_par at 1M signatures with P=128, BANDS=16 on a 32-core Threadripper
5975WX, 15% efficiency at 32 cores, so expect 5x to 8x on 10 cores and measure it.

Sources: https://docs.rs/minhash-rs/0.6.0/minhash_rs/ (4.8x parallel LSH build),
https://ekzhu.com/datasketch/lsh.html (band/row index and merge semantics).

## 7. Published implementations and measured baselines

Crates (crates.io API, 2026-11-11):

| crate | version | downloads | what it is |
|-------|---------|-----------|------------|
| simhash | 0.3.0 | 1,718,343 total, 264,182 recent | functions only, simhash(&str) -> u64, fixed 64 bits, whitespace split, optional fnv or siphasher |
| minhash-rs | 0.6.0 | 5,381 | MinHash<W, const PERMUTATIONS>, LshIndex<..., const BANDS>, sparse hashes, rayon path |
| probabilistic-collections | 0.7.0 | 535,073 | MinHash similarity, pairwise only, no bands or index |
| lsh-rs | 0.4.0 (2020) | 10,118 | generic LSH, minhash projection, undocumented |

Published throughput:
- Python datasketch 2.0.0: 113,333 articles of about 300 words, 128 permutations, 16 bands
  x 8 rows = 90 s fingerprinting, 0.8 ms per article on one core, plus 21 s index build.
  That is 1,250 docs/s, 355x slower per document than the Rust pipeline below.
- HuggingFace text-dedup: MinHash 11.09 s versus SimHash 626.11 s on the same corpus,
  MinHash is 56x faster because SimHash needs a 64-bit pass and an O(n) scan per query.
  Its defaults are num_perm=240, threshold 0.7, ngram 5, hash_bits 64.
- peS2o, 39M documents, 32-core Zen 3: MinHashLSH dedup "more than 35 hours", LSHBloom
  3 hours (12x), which is the cost of exact candidate verification at that scale.
- Rust minhash-rs: rayon build about 4.8x at 1M signatures, P=128, BANDS=16.

Local baseline for this exact pipeline (pure std, no crates beyond serde_json): tokenize,
5-word shingles, 128 affine MinHashes, u32 signature, 67 words and 63 shingles per doc.
Result: 444k docs/s, 2.25 us/doc, 217 MB/s single core, 22.5 s for 10M documents, and
about 2.5 s to 3 s at 10 threads if the pipeline is memory-light, bounded by the 7.9 GB
signature and index memory. That is 3.6 G hash updates/s, and it makes MinHash the
dominant cost (1.5 us of 2.25 us), not JSON parsing (0.2 us) or tokenizing (0.4 us).

No published Rust project states a MB/s or docs/s figure for text dedup specifically.
The numbers above are what exists, plus this measurement; treat any Rust "GB/s dedup"
claim without a corpus and a CPU as unverified.

Sources: https://crates.io/crates/simhash, https://docs.rs/minhash-rs/0.6.0/minhash_rs/,
https://github.com/ChenghaoMou/text-dedup#benchmarks,
https://arxiv.org/html/2411.04257v3, https://www.sevydevy.com/learn/near-duplicate-detection.

## Design decisions

1. Shingle on 5 consecutive lowercased words, hashing each word once with an 8-byte-round
   wyhash-style loop, not FNV-1a (2.25 ns versus 13.8 ns per 40-byte shingle).
2. Build 128 MinHash values per document with the affine trick h_i = (a_i * h + b_i) mod
   2^32, odd a_i, and store them as u32 (512 B/doc, 5.12 GB for 10M documents).
3. Split the 128 values into 16 bands of 8 rows, target threshold 0.707, and expect 94.7%
   recall at J=0.80 and a 23.7% candidate rate at J=0.60.
4. Hash each band to a u64 key, never store the raw 256-bit band, and expect about one
   spurious band collision per 160M index entries.
5. Verify every candidate by comparing the two u32 signatures and keep pairs at estimated
   J >= 0.70; do not re-read or re-tokenize the text.
6. Drop SimHash from the primary path. If 8 bytes/doc is needed, use 64-bit SimHash only
   with Hamming <= 3 as an exact-plus copy filter, and know that it means J >= 0.995.
7. Never call serde_json::from_reader on a File; use BufReader::with_capacity(1<<20) plus
   read_line into one reused String, then from_str into a struct with borrowed &str fields.
8. Prefer read_until into a reused Vec<u8> plus from_slice when invalid UTF-8 must be
   filtered, and skip String allocation entirely.
9. Reject serde_json::Value in the hot loop: it is 1.8x slower and over 10x the memory.
10. Split dedup into offset map, per-thread band maps, key-partitioned shuffle and shard
    reduce; never put a Mutex<HashMap> in the insertion path (27x slower here).
11. Cap the design at 4.29B documents so u32 ids stay valid, and pre-size every HashMap to
    b * docs / threads to avoid rehash spikes.
12. Guard sentence splitting with the 156-type abbreviation list class (titles, months,
    e.g, i.e, Fig, No, Inc, Ltd, U.S), a digit rule and an initials rule; tokenize with
    char::is_alphanumeric and keep combining marks and U+2019 attached.
