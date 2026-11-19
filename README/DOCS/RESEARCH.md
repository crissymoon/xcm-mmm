# Research and provenance

Two questions sit behind every table in this repository: why the evaluation is
shaped the way it is, and which binary and which artefacts produced a given
number. The first is answered by the literature below, the second by a chain of
records that starts in `build.rs` and ends in an assertion inside the Kaggle
notebook.

## The chain in one view

| step | artefact | what it holds | who checks it |
| --- | --- | --- | --- |
| 1 | `build.rs` | commit, dirty flag, target triple, `rustc`, `SOURCE_DATE_EPOCH` | compiled in, so it cannot drift after the fact |
| 2 | `xcm-mmm provenance --json` | the same six fields, read back out of the ELF | `scripts/build-linux.sh`, any reader |
| 3 | `dist/kaggle/provenance.json` | sha256, byte count, full commit, flags, epoch | written by the packaging fetch, next to the binary |
| 4 | the dataset card | version, bytes, sha256, target | Kaggle, at upload |
| 5 | `kernel-metadata.json` | the dataset slug with an optional version segment | Kaggle, at kernel push |
| 6 | the notebook | recomputes the sha256 and asserts it equals step 3 | the reader, on every run |

Step 6 is the one that matters. A hash printed in a card proves only that a file
did not change after it was hashed. Recomputed inside the notebook and compared
against a record that travelled with the file, it proves the notebook is running
the artefact the card describes.

## Selection is not an estimate

The forecasting path picks one configuration out of a candidate set and reports
how well it did. Those are two different statistical problems, and the tool keeps
them apart on purpose:

- `forecast` scores every candidate on the outer folds and reports the best.
  That number is a maximum of many correlated estimates, so it is optimistic.
- `forecast --nested` runs the selection inside each outer training window and
  scores only the winner's out-of-sample performance. That number can be
  reported.

Cawley and Talbot (2010) show the bias is driven by the number of candidates
tested against the noise of the criterion, not by the fold count alone. Varma
and Simon (2006) quantify the same bias for cross-validated selection and
prescribe the outer loop. With four outer folds the estimate itself has three
degrees of freedom, so the notebook reports per fold rather than a single mean,
and `--nested` is the default route in the documentation for anything a reader
would quote.

Three further constraints shaped the defaults:

- Bergmeir, Hyndman and Koo (2018) show that unblocked cross-validation is valid
  for autoregressive models with uncorrelated errors, so the fold geometry should
  follow the residual autocorrelation the tool measures rather than assume
  blocking is always required. `horizon_embargo` exists for the case where it is.
- Tashman (2000) fixes the vocabulary of rolling-origin evaluation and is the
  reason the folds are successive origins with a declared window policy rather
  than shuffled rows, and also the reason four origins is described as a small
  sample rather than a comfortable one.
- Cerqueira, Torgo and Mozetič (2020) find that repeating a cheap scheme buys
  more reliability than enlarging a single one, which is why a nested run is
  expected to be repeated rather than widened when a budget allows.

The gate that prices the search, described in [Forecasting](FORECASTING.md),
is a diagnostics layer over this: it reports how many candidates the data can
support before a difference is indistinguishable from noise. It does not repair
a selection made without nesting, and the documentation says so.

## The sources behind the defaults

| mechanism | source | code |
| --- | --- | --- |
| success-history adaptation | Tanabe and Fukunaga (2013) | `src/de` |
| linear population size reduction | Tanabe and Fukunaga (2014) | `src/de/memory.rs` |
| MOEA/D-DE | Li and Zhang (2009) | `src/moead` |
| Tchebycheff and PBI scalarisation | Zhang and Li (2007) | `src/moead/scalar.rs` |
| random partial update | Lavinas et al. (2020) | `src/moead/allocation.rs` |
| relative improvement schedule | Zhang, Liu and Li (2009) | `src/moead/allocation.rs` |

Full DOIs are in the references at the end of [Algorithms](ALGORITHMS.md), which
is the page that describes what each one does to the search.

## Reproducing the binary

The release artefact is built by one script and is byte identical across
rebuilds of one commit. That property is what makes step 3 above worth
publishing, and it is checked rather than claimed:

```bash
scripts/build-linux.sh          # writes target/x86_64-unknown-linux-musl/release/
OUT=/tmp/a scripts/build-linux.sh
OUT=/tmp/b scripts/build-linux.sh
shasum -a 256 /tmp/a/xcm-mmm /tmp/b/xcm-mmm    # the two digests agree
```

Three settings do the work, and each one removes a specific source of drift:

- `SOURCE_DATE_EPOCH`, set from the commit timestamp unless the environment
  already names one, pins the date `build.rs` bakes in. Without it a rebuild
  would embed the wall clock and no two builds would agree.
- `--remap-path-prefix` keeps the build machine's home directory out of the
  artefact. This is the flag the Rust reference documents for reproducible
  paths, and Debian applies the same trick to packaged Rust crates.
- `strip = "symbols"` in the release profile, rather than only in the shell
  script, so a plain `cargo install` produces the same shape of artefact that
  the dataset describes.

`--build-id=sha1` is added for identification rather than reproducibility: the
ELF gets a note naming the link inputs, which is what matches a crash report to
one artefact when several versions are in circulation. `file` prints it, and the
notebook does not need to.

What the build does not do is claim more than it can. There is no signature and
no attestation: a single maintainer building on one host cannot attest to a
process a reader did not watch. The chain above gives a reader something
checkable instead, a hash and a commit they can rebuild and compare.

## Kaggle specifics

The notebook is written around four documented properties of the platform,
because each one breaks a naive port:

- A CPU session runs for 12 hours and an interactive one ends after 20 minutes
  idle. The published run has to finish well inside both, which is why the
  workload is four shards and not a full sweep.
- `/kaggle/working` is capped at 20 GB and is the only directory saved with the
  version. The binary, the `data/` tree, `charts/` and `provenance.json` all
  live there.
- With `enable_internet` false there is no outbound network, so `pip install`
  fails. The notebook uses the image's own pandas and matplotlib, and the
  research below is about not needing a download at run time.
- `dataset_sources` accepts `owner/slug/version`, and the third segment is the
  exact version. `kernel-metadata.json` is generated with the pin when
  `XCM_BINARY_VERSION` and `XCM_MODELS_VERSION` are set. What was observed:
  the push accepted the pinned refs and the run used the version named, but
  `kernels pull` returns the sources as bare slugs, so the number is resolved at
  push time rather than recorded on the notebook. The hash the notebook
  recomputes is the check that survives, which is why the provenance cell comes
  before the workload rather than after it.

## Routing a reading to an action

`route` asks a different question from the rest of the crate. The detectors say
what a text looks like and the legal reading says what it does; the router says
what to do about it, and the answer is one of five actions: reply, forward, file,
hold for a person, or ignore. The interesting part is not the classification. It
is that a good action depends on consequences, protocol rules and the cost of
being wrong, and none of those live in a score.

| question | what the source says | what it forced in the code |
| --- | --- | --- |
| may a robot answer this at all? | RFC 3834, sections 2 and 5.1: `Auto-Submitted` other than `no`, a null reverse-path and machine mailboxes must not be answered | three separate vetoes, each printed with its reason |
| where may the answer go? | RFC 3834, section 4: after delivery the reply destination is the envelope sender, not `From` or `Reply-To` | `Return-Path` first, and the fallback to `Reply-To` is reported as a deviation |
| is this list mail? | RFC 2919 says only list software generates `List-Id`; RFC 2369 defines `List-Unsubscribe`; RFC 2076 calls `Precedence` non-standard and discouraged | `List-*` headers raise the bulk class; `Precedence` may veto a reply but may never move a message into the class on its own |
| what should filing look like? | RFC 5228: `fileinto` is a SHOULD, a missing mailbox may be an error, and an error must leave the message kept; RFC 5232 flags are per-message | folders are created explicitly, the original is copied and never moved, and no state is written into flags |
| which action, given the costs? | Elkan (2001): choose the action that minimises expected cost under an asymmetric loss matrix | `costs` in `policy.json`, with every action's expected cost printed |
| when should it refuse to choose? | Chow (1970) reject option; El-Yaniv and Wiener (2010) risk-coverage | a top-share gate and a cost-margin gate both force `hold` |
| may the score be read as a probability? | Guo et al. (2017): modern scores are poorly calibrated | the numbers are called shares of evidence, never probabilities, and the gate carries the safety |
| what keeps a person in the loop? | Parasuraman and Riley (1997): the failure mode is over-reliance, not under-use | `hold` is a real queue, and the draft says it was not read before it was sent |
| what must the record keep? | ISO 15489-1:2016: records stay authentic, reliable, complete and usable | copies are additive and the ledger stores the policy digest beside each decision |

The consequence is a policy file an operator can argue with rather than a model
they cannot. A demand letter costs nine to answer unreviewed and one to hold; a
newsletter costs ten to answer and one to ignore. Reverse those numbers and the
router does the opposite, which is the point of putting them in a file.

Two things the command deliberately cannot do, because each would be a stronger
claim than the implementation supports:

- **Send.** A reply or a forward is written as an `eml` body with the headers
  RFC 3834 requires (`Auto-Submitted: auto-replied`, `In-Reply-To`,
  `References`), and a person or a queue releases it. Nothing in this crate
  opens a socket.
- **Delete or move.** Every filing is a copy beside a ledger. A wrong reading
  costs a duplicate under the wrong folder, not a message nobody can find, which
  is the implicit-keep rule of RFC 5228 applied one level up.

The repeat window is enforced, from memory it can name. RFC 3834, section 7
illustrates a vacation responder that refuses a message when the sender "has
been sent a response within the previous 7 days"; the section is illustration
rather than a rule, and the window is still the only thing between an automatic
responder and writing to the same person on every follow-up. `decide` takes the
run's own count and the age of the last answer the ledger holds for that sender,
and a sender answered inside `reply_window_days` is refused, then held for a
person, because they did write again. The limit is the record: a ledger line
without a timestamp is counted as undated and reported rather than guessed at,
and an answer a person sent by hand is invisible to it. The code says that
instead of implying a guarantee it does not make.

The seven files in `data/route/samples/` are fixtures, not decoration: the test
suite asserts the action each one must receive, so a cue weight that moves a
demand letter into a folder fails the build.

## What was read and not adopted

- **A user-space allocator.** The case for one is real: musl's malloc is slower
  than glibc's, most visibly once several threads free each other's objects, and
  mimalloc and jemalloc both beat glibc in rust-analyzer's benchmark. Neither is
  a pure Rust crate, so adding one means cross compiling C for the musl target,
  and `scripts/build-linux.sh` exists to need nothing beyond rustup. The shard
  timing cell is where the effect would show, since it is the only place the
  binary allocates from several threads; that measurement has not been made, so
  the default allocator stays and this note records why.
- **`-C target-cpu=native`.** It buys a few percent on vectorisable loops and
  costs the property that one artefact runs on every reader's machine, and it
  makes the build non-reproducible besides. A named level is the portable
  version for anyone compiling for their own fleet: `x86-64-v2` as the floor,
  `x86-64-v3` where AVX2 is safe.
- **Profile-guided optimisation.** The gains for parsing code are real but
  input-dependent, measured between 36 percent faster and 24 percent slower on
  different corpora, so it would need a training workload checked into the
  repository to stay honest. The binary's time is dominated by the optimiser
  rather than by dispatch, so it is recorded as an option and not taken.
- **A signature or an attestation service.** The options are real: `cargo-dist`
  can emit GitHub artifact attestations, and `cosign sign-blob` writes a bundle a
  reader verifies against an OIDC identity. Both need an identity to attest to,
  which means a hosted CI job building the artefact. This release is built on one
  laptop and uploaded by hand, so either would prove only that the same person
  signed the file they had just made. The checksum, the commit, the build id and
  a rebuild that a reader can run themselves are the honest version of the same
  guarantee at this scale.
- **`trim-paths`, the Cargo replacement for `--remap-path-prefix`.** It is a
  nightly profile setting in the toolchain used here, so the stable flag is what
  the build script passes. Worth switching when it stabilises.
- **One-click unsubscribe.** RFC 8058 makes the receiver POST to the list's
  unsubscribe URI, and the same document requires user consent before that
  request is made. A router that unsubscribed on its own would be spending a
  person's consent it never had, so `List-Unsubscribe` only marks the message as
  list mail; it never triggers a request.
- **A folder per list.** `List-Id` is stable and would make a tidy tree, but the
  value is chosen by the sender, so the folder names would be an untrusted
  string turned into a path. The ledger records the `List-Id` instead and the
  tree stays bounded by the policy.
- **A learned router.** With a labelled log of past decisions a classifier would
  beat the cue weights. There is no such log here, and fitting one on the seven
  samples would be selection on the test set, which is the error the rest of this
  document is about. The cost matrix and the gate are the honest version until
  the log exists.

## References

1. Cawley, G. C. and Talbot, N. L. C. (2010). On over-fitting in model selection
   and subsequent selection bias in performance evaluation. Journal of Machine
   Learning Research 11, 2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html
2. Varma, S. and Simon, R. (2006). Bias in error estimation when using
   cross-validation for model selection. BMC Bioinformatics 7, 91.
   https://doi.org/10.1186/1471-2105-7-91
3. Bergmeir, C., Hyndman, R. J. and Koo, B. (2018). A note on the validity of
   cross-validation for evaluating autoregressive time series prediction.
   Computational Statistics and Data Analysis 120, 70-83.
   https://doi.org/10.1016/j.csda.2017.11.003
4. Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an
   analysis and review. International Journal of Forecasting 16(4), 437-450.
   https://doi.org/10.1016/S0169-2070(00)00065-0
5. Bergmeir, C. and Benítez, J. M. (2012). On the use of cross-validation for
   time series predictor evaluation. Information Sciences 191, 192-213.
   https://doi.org/10.1016/j.ins.2011.12.028
6. Cerqueira, V., Torgo, L. and Mozetič, I. (2020). Evaluating time series
   forecasting models: an empirical study on performance estimation methods.
   Machine Learning 109(11), 1997-2028.
   https://doi.org/10.1007/s10994-020-05910-7
7. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
   sets. Journal of Machine Learning Research 7, 1-30.
   https://www.jmlr.org/papers/v7/demsar06a.html
8. Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of
   Statistical Learning, 2nd edition, section 7.10. Springer.
   https://doi.org/10.1007/978-0-387-84858-7
9. Tanabe, R. and Fukunaga, A. (2013). Success-history based parameter
   adaptation for differential evolution. IEEE Congress on Evolutionary
   Computation. DOI 10.1109/CEC.2013.6557555
10. Tanabe, R. and Fukunaga, A. (2014). Improving the search performance of
    SHADE using linear population size reduction. IEEE Congress on Evolutionary
    Computation. DOI 10.1109/CEC.2014.6900380
11. Zhang, Q. and Li, H. (2007). MOEA/D: a multiobjective evolutionary algorithm
    based on decomposition. IEEE Transactions on Evolutionary Computation 11(6),
    712-731. DOI 10.1109/TEVC.2007.892759
12. Li, H. and Zhang, Q. (2009). Multiobjective optimization problems with
    complicated Pareto sets, MOEA/D and NSGA-II. IEEE Transactions on
    Evolutionary Computation 13(2), 284-302. DOI 10.1109/TEVC.2008.925798
13. Lavinas, Y., Aranha, C., Ladeira, M. and Campelo, F. (2020). MOEA/D with
    random partial update strategy. IEEE Congress on Evolutionary Computation.
    DOI 10.1109/CEC48606.2020.9185527
14. Zhang, Q., Liu, W. and Li, H. (2009). The performance of a new version of
    MOEA/D on CEC09 unconstrained MOP test instances. IEEE Congress on
    Evolutionary Computation, 203-208. DOI 10.1109/CEC.2009.4982949
15. Cargo Book. Profiles, the `strip` option.
    https://doc.rust-lang.org/cargo/reference/profiles.html
16. The rustc book. `--remap-path-prefix`.
    https://doc.rust-lang.org/rustc/command-line-arguments.html
17. Reproducible Builds. SOURCE_DATE_EPOCH.
    https://reproducible-builds.org/docs/source-date-epoch/
18. Kaggle. Notebooks, technical specifications.
    https://www.kaggle.com/docs/notebooks#technical-specifications
19. Kaggle API. Kernel metadata reference.
    https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
20. Reproducible Builds. The SOURCE_DATE_EPOCH specification.
    https://reproducible-builds.org/specs/source-date-epoch/
21. GNU binutils. `ld` options, `--build-id`.
    https://sourceware.org/binutils/docs/ld/Options.html
22. SLSA. Build track basics and build provenance.
    https://slsa.dev/spec/v1.2/build-provenance
23. sigstore. Signing blobs with cosign.
    https://docs.sigstore.dev/cosign/signing/signing_with_blobs/
24. Sivers, N. Default musl allocator considered harmful to performance.
    https://nickb.dev/blog/default-musl-allocator-considered-harmful-to-performance/
25. rust-analyzer. Allocator benchmark.
    https://github.com/rust-lang/rust-analyzer/issues/1441
26. The rustc book. Profile-guided optimization.
    https://doc.rust-lang.org/rustc/profile-guided-optimization.html
27. Wagner, I. Optimizing Rust builds with target flags.
    https://ianwwagner.com/optimizing-rust-builds-with-target-flags.html
28. Chow, C. K. (1970). On optimum recognition error and reject tradeoff. IEEE
    Transactions on Information Theory 16(1), 41-46.
    https://doi.org/10.1109/TIT.1970.1054406
29. Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI 2001,
    973-978. https://cseweb.ucsd.edu/~elkan/rescale.pdf
30. El-Yaniv, R. and Wiener, Y. (2010). On the foundations of noise-free
    selective classification. Journal of Machine Learning Research 11,
    1605-1641. https://jmlr.org/papers/v11/el-yaniv10a.html
31. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of
    modern neural networks. ICML 2017, PMLR 70, 1321-1330.
    https://proceedings.mlr.press/v70/guo17a.html
32. Parasuraman, R. and Riley, V. (1997). Humans and automation: use, misuse,
    disuse, abuse. Human Factors 39(2), 230-253.
    https://doi.org/10.1518/001872097778543886
33. Moore, K. (2004). Recommendations for automatic responses to electronic
    mail. RFC 3834.
    https://www.rfc-editor.org/rfc/rfc3834.html
34. Resnick, P. (2001). Internet Message Format. RFC 5322.
    https://www.rfc-editor.org/rfc/rfc5322.html
35. Palme, J. (1996). Common Internet message headers. RFC 2076, section 3.9.
    https://www.rfc-editor.org/rfc/rfc2076.html
36. Chandhok, R. and Wenger, K. (2010). The mailing list `List-Id` header.
    RFC 2919. https://www.rfc-editor.org/rfc/rfc2919.html
37. Neufeld, G. and Baer, J. (1998). The use of URLs as meta-syntax for core mail
    list commands. RFC 2369. https://www.rfc-editor.org/rfc/rfc2369.html
38. Levine, J. and Herk, T. (2017). Signaling one-click functionality for list
    email headers. RFC 8058. https://www.rfc-editor.org/rfc/rfc8058.html
39. Guenther, P. and Showalter, T. (2008). Sieve: an email filtering language.
    RFC 5228. https://www.rfc-editor.org/rfc/rfc5228.html
40. Melnikov, A. (2008). Sieve email filtering: `imap4flags` extension.
    RFC 5232. https://www.rfc-editor.org/rfc/rfc5232.html
41. International Organization for Standardization (2016). ISO 15489-1:2016,
    Information and documentation, records management, part 1: concepts and
    principles.
