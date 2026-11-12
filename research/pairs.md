# Building training pairs at scale for one encoder and reward model

Published recipes, exact parameters, reported effects, and the failure each guards against. Figures were read
from the arXiv full text unless marked secondary; where a paper is silent, that is stated.

Reading rule: every recipe is a filter plus a sampler. The filter decides which pairs may exist at all (false
negatives, triviality, leakage); the sampler decides how often each row is seen (batch composition,
temperature, curriculum stage). Most published gains are filters, not losses.

## 1. SimCSE: dropout duplication, NLI positives, one hard negative

- Title: Simple Contrastive Learning of Sentence Embeddings
- Authors/Year: Tianyu Gao, Xingcheng Yao, Danqi Chen, EMNLP 2021
- Identifier: arXiv:2104.08821
- URL: https://arxiv.org/abs/2104.08821
- Recipe, unsupervised: positive is the same sentence encoded twice under two independent dropout masks.
Dropout on FC and attention p=0.1, nothing added. InfoNCE with in-batch negatives, cosine on normalized
vectors, tau=0.05. Encoder [CLS] plus the BERT MLP pooler, no projection head. 1e6 Wikipedia sentences, 1
epoch. Batch grid {64,128,256,512} by lr {1e-5,3e-5,5e-5}; best unsup BERT-base batch 64 at 3e-5, RoBERTa
512. N-1 in-batch negatives (63 at N=64, 511 at N=512).
- Recipe, supervised: SNLI + MNLI, 314k entailment premise-hypothesis pairs as positives; exactly one
sampled contradiction per pair appended as a hard negative with weight alpha=1.0 in the denominator. Batch
512, lr 5e-5, 3 epochs.
- Effect: unsupervised STS-B dev 82.5; dropout 0.0 gives 71.1, 0.05 gives 81.1, 0.1 gives 82.5, 0.2 gives
80.5, a fixed mask gives 43.6, and batch size is not sensitive once lr is retuned. Supervised STS-B dev
84.9 with no hard negative, 86.1 at alpha 0.5, 86.2 at alpha 1.0 and 2.0, 85.3 with contradiction plus
neutral; all 942k easy pairs gives 81.9, so one good hard negative beats a much larger easy set. Test
average 76.25 unsupervised and 81.57 supervised.
- Failure mode: collapse when both views share a mask or dropout is removed; in-batch false negatives when
near-duplicate sentences share a batch; neutral or unrelated pairs used as hard negatives dilute the signal.
- Rust: one encoder called twice with distinct dropout seeds derived from (row_id, view), cosine logits
divided by 0.05, a negatives table with a role column (entail, contradict, neutral), a cap of one mined hard
negative per anchor, and a duplicate-cluster check before batching.

## 2. E5: weakly supervised web pairs with a consistency filter

- Title: Text Embeddings by Weakly-Supervised Contrastive Pre-training
- Authors/Year: Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
Furu Wei, 2022
- Identifier: arXiv:2212.03533
- URL: https://arxiv.org/abs/2212.03533
- Recipe: CCPairs from 7 sources, 1.3B raw pairs to 270M after filtering. Wikipedia (section title, passage)
24M; Reddit (post, upvoted comment) 60M; Common Crawl (title, passage) 69M; Stack Exchange (title, answer)
19M; S2ORC title/abstract/citation pairs 90M; news 3M; other 6M. Filters: Reddit comments over 4096 chars or
score below 1 dropped; high-perplexity passages dropped; language id and dedup; exact-string overlap with eval
sets removed; S2ORC weight 0.3. Consistency filter: score each pair against a fixed 1M random passages and
keep only the top 2. Shared encoder, average pooling, "query:" and "passage:" prefixes, tau 0.01. Batch 32768
(32767 in-batch negatives), 20k steps, lr 3e-4/2e-4/1e-4. Fine-tune 3 epochs, batch 256, lr 3e-5/2e-5/1e-5 on
NLI, MS MARCO, NQ.
- Effect: BEIR zero-shot BM25 41.7, E5 base 42.9, large 44.2; after fine-tune 48.7 and 50.0. Batch 1k 45.8, 8k
50.2, 32k 51.6. Filter at 1M pairs 34.9 to 40.7; at full scale 50.0 to 51.6 with 4x less data.
- Failure mode: noisy positives teach false positives; small batches starve the negative pool; mining over
270M pairs must be resumable.
- Rust: a pair store keyed by source with a dedup hash, a resumable pass that ranks each pair against a fixed
1M passage matrix, and the keep_top_k=2 rule as data.

## 3. DPR: in-batch negatives plus one BM25 negative

- Title: Dense Passage Retrieval for Open-Domain Question Answering
- Authors/Year: Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, Wen-tau Yih, EMNLP 2020
- Identifier: arXiv:2004.04906
- URL: https://arxiv.org/abs/2004.04906
- Recipe: 21M Wikipedia passages of 100 words. One gold positive per question; TriviaQA positives are the
highest-ranked BM25 passage containing the answer, and questions with no answer in the BM25 top 100 are
dropped. Negatives: in-batch at batch 128 plus one BM25 negative that does not contain the answer, so 128
negatives per question. Two BM25 negatives is worse. Two BERT-base encoders, dot product, NLL, 40 epochs, lr
1e-5.
- Effect: NQ test top-20 79.4 versus BM25 59.1; top-100 86.0. Ablation top-20: 127 in-batch gold negatives
73.0, plus one BM25 78.0, plus two 76.4. At 7 negatives random 64.3, BM25 63.3, gold 63.1, so count matters
more than negative type until count is large.
- Failure mode: BM25 passages containing the answer are false negatives and cap accuracy; distant supervision
costs about 1 point.
- Rust: a BM25 pass with an answer-span exclusion predicate, one hard-neg column per row, and batch size
exposed as the negative count.

## 4. ANCE: negatives from a live ANN index

- Title: Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval
- Authors/Year: Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed,
Arnold Overwijk, ICLR 2021
- Identifier: arXiv:2007.00808
- URL: https://arxiv.org/abs/2007.00808
- Recipe: keep a Faiss IndexFlatIP over the corpus. A parallel inferencer encodes with a recent checkpoint and
rebuilds the index every 10k batches (5k converges more smoothly at twice the inference GPUs). Sample one
negative uniformly from the ANN top 200 per positive; no in-batch negatives. Warm up on BM25 negatives, LAMB
5e-6 then 1e-6 after 5000 steps, about 10 epochs.
- Effect: MS MARCO MRR@10 0.330 versus DPR 0.311, BM25-negative 0.299, random 0.256; TREC-DL passage nDCG
0.677; NQ coverage 81.9/87.5 versus 78.4/85.4. Roughly 100x cheaper than BM25 plus BERT reranking at similar
accuracy.
- Failure mode: a stale asynchronous index plus a large lr drifts the negative distribution; the single
hardest neighbour is mostly a false negative, which is why sampling is uniform over top 200, not top 1.
- Rust: an index handle with a generation counter, an inference worker publishing a new generation every N
steps, and uniform sampling over the retrieved window.

## 5. RocketQA: cross-batch negatives and cross-encoder denoising

- Title: RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question
Answering
- Authors/Year: Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Dacheng Tao, Hua Wu,
Ji-Rong Wen, NAACL 2021
- Identifier: arXiv:2010.08191
- URL: https://arxiv.org/abs/2010.08191
- Recipe: cross-batch negatives gather A*B-1 candidates, 4095 at 8 GPUs x batch 512, 1023 on NQ. Denoise with
a cross-encoder: keep score < 0.1 as negative, > 0.9 as positive, discard the 0.1 to 0.9 band. Ratio 1:4
positive to hard negative on MS MARCO, 1:1 on NQ, plus answer-span filtering.
- Effect: MS MARCO MRR@10 in-batch 32.39, cross-batch 33.32, undenoised hard negatives 26.03, denoised 36.38,
with augmentation 37.02. Undenoised hard negatives are 7.3 points worse than in-batch.
- Failure mode: about 70 percent of top-retrieved unlabeled passages were actually relevant, so unlabeled hard
negatives are mostly false negatives; the score band is the guard, not retrieval rank.
- Rust: a three-way verdict enum Drop/KeepNeg/KeepPos on the cross-encoder score, and a sampler that enforces
the per-query ratio.

## 6. Hardness-weighted negative sampling with cosine similarity

- Title: Contrastive Learning with Hard Negative Samples
- Authors/Year: Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, Stefanie Jegelka, ICLR 2021
- Identifier: arXiv:2010.04592
- URL: https://arxiv.org/abs/2010.04592
- Recipe: draw batch negatives with weight proportional to exp(beta * s), s a cosine, and debias the
denominator by estimating E[exp(beta*s)] as mean exp((beta+1)s) / mean exp(beta*s), conditioning on a
different latent class. beta is the hardness knob: 0.5 best on CIFAR100 and STL10, 0.2 for MoCo-v2, 1 to 2 in
the text tables. tau+ 0.01 or 0. Negatives are the batch (N=2(batch-1), up to 510) or a 65536 memory bank; one
positive.
- Effect: +7.3 STL10 and +3.0 CIFAR100 over SimCLR at N=510; tinyImageNet 53.4 to 57.0; MoCo-v2 88.08 to
88.47. At beta 6 the class similarity histograms overlap and training collapses; annealing beta to 0 recovers.
- Failure mode: the hardest negatives are the likeliest false negatives; a high beta concentrates the batch on
them and destroys class structure.
- Rust: f64 weights exp(beta*s) with max subtraction first, the debias ratio from the same logits, and beta
plus an anneal schedule in config.

## 7. DPO: mining pairs, and what beta does not do

- Title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- Authors/Year: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea
Finn, NeurIPS 2023
- Identifier: arXiv:2305.18290
- URL: https://arxiv.org/abs/2305.18290
- Recipe: loss = -log sigmoid(beta * log(pi(yw|x)/pi_ref(yw|x)) - beta * log(pi(yl|x)/pi_ref(yl|x))). beta 0.1
default, 0.5 for TL;DR; batch 64, RMSprop, lr 1e-6, 150-step warmup. Pairs: Anthropic HH, 170k dialogues each
ending in a labelled pair; TL;DR human pairs; IMDb 25k prefixes x 4 sampled completions labelled by a
classifier.
- Effect: TL;DR GPT-4 win 61 percent versus PPO 57; human 58 percent versus 17 for PPO at temperature 1.0; OOD
CNN/DailyMail 0.36 versus 0.26. The gradient weights each pair by sigma(r_hat(yl) - r_hat(yw)), so
already-misranked pairs get the largest update.
- Failure mode: no margin, so near-tie pairs push noise; beta was not tuned, so KL strength is a guess; length
and OOD drift untested; GPT-4 prefers longer summaries than humans.
- Rust: beta as f64 config, one cached reference log-prob pass per pair, and a pair_weight column so the sigma
weight is logged per row.

## 8. ORPO: one stage, odds ratio, lambda 0.1

- Title: ORPO: Monolithic Preference Optimization without Reference Model
- Authors/Year: Jiwoo Hong, Noah Lee, James Thorne, EMNLP 2024
- Identifier: arXiv:2403.07691
- URL: https://arxiv.org/abs/2403.07691
- Recipe: L = L_SFT + lambda * L_OR, with L_OR = -log sigmoid(log(odds(yw|x) / odds(yl|x))) and odds =
P/(1-P). No reference model, no separate SFT stage. lambda 0.1 on Mistral-7B, 0.2 on Llama-2-7B, 0.25 on
Phi-2. Data: binarized UltraFeedback, 61k pairs after dropping yw == yl and empty sides. Max lr 8e-6, 10
epochs, cosine decay.
- Effect: Mistral 87.92 AlpacaEval 1.0 and 11.33 AlpacaEval 2.0 (alpha), 91.41 and 12.20 (beta); MT-Bench 7.23
and 7.32; IFEval instruction-loose 66.19. Lambda ablation: 0.1 keeps the rejected log-prob flat and lifts the
chosen one; 0.5 starts dropping rejected; 1.0 widens the margin while extraction, math and reasoning get
worse.
- Failure mode: SFT on chosen responses alone raises the rejected log-prob too, so the model drifts in style
and miscalibrates the pair; a larger lambda is not a better margin; a 61k only model lacks coding and math.
- Rust: log-odds as logp - log1p(-exp(logp)) to avoid cancellation near p=1, mean token log-prob before the
ratio, then one log-sigmoid and one lambda multiply.

## 9. Margin in the loss: SimPO gamma

- Title: SimPO: Simple Preference Optimization with a Reference-Free Reward
- Authors/Year: Yu Meng, Mengzhou Xia, Danqi Chen, NeurIPS 2024
- Identifier: arXiv:2405.14734
- URL: https://arxiv.org/abs/2405.14734
- Recipe: reward = (beta/|y|) * log pi(y|x), so the training reward equals the generation metric. Loss = -log
sigmoid(r_w - r_l - gamma). No reference model, no KL. beta in {2.0, 2.5}, gamma in {0.3, 0.5, 1.0, 1.2, 1.4,
1.6}; guidance beta 2.0 to 2.5 and gamma 0.5 to 1.5. UltraChat 200k SFT, max length 2048, lr 3e-7 to 1e-6.
- Effect: Gemma-2-9B-it-SimPO 72.4 length-controlled AlpacaEval 2 and 59.1 Arena-Hard, up to +6.4 and +7.5
over DPO. Mistral ablation 21.5/16.6 versus DPO 15.1/10.4; dropping length normalization gives 11.9/9.4, about
half. Length does not grow relative to SFT or DPO.
- Failure mode: drop the /|y| and length becomes the reward, halving win rate; too large a gamma, or a badly
calibrated SFT model, weakens or inverts the margin.
- Rust: per-sequence mean log-prob (sum / max(len,1)), scale by beta, subtract the other side and gamma, then
log_sigmoid; no reference forward pass.

## 10. Deterministic pairs: IPO's finite target

- Title: A General Theoretical Paradigm to Understand Learning from Human Preferences
- Authors/Year: Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal
Valko, Remi Munos, AISTATS 2024
- Identifier: arXiv:2310.12036
- URL: https://arxiv.org/abs/2310.12036
- Recipe: replace the unbounded log-sigmoid with a squared loss on the log-ratio whose target is a finite
margin set by a temperature-like constant, so a pair already ranked correctly by that margin stops producing
gradient.
- Effect: on synthetic bandit and preference experiments the squared objective avoids the overfitting DPO
shows when preferences are near deterministic, where DPO keeps pushing the probability toward 1.
- Failure mode: labels come from a noisy rater, so "always preferred" is usually a small-sample artifact; an
unbounded loss turns that noise into a saturated policy.
- Rust: a loss enum with target_margin, plus a squared-error branch that clamps the log-ratio.

## 11. Trivial surface differences: length as the shortcut

- Title: Disentangling Length from Quality in Direct Preference Optimization
- Authors/Year: Ryan Park, Rafael Rafailov, Stefano Ermon, Chelsea Finn, ACL 2024
- Identifier: arXiv:2403.19159
- URL: https://arxiv.org/abs/2403.19159
- Recipe: add a length penalty r(x,y) - alpha|y| inside the DPO derivation, which becomes a margin of alpha
times (chosen length minus rejected length) on the logit. This is a per-example learning-rate multiplier: it
up-weights shorter chosen answers and down-weights longer ones. Pythia 2.8B full fine-tune, beta in
{0.5,0.1,0.05}, alpha 0.01 on HH and 0.05 on TL;DR.
- Effect: dataset bias is small (HH preferred 79.6 tokens versus rejected 75.7; TL;DR 37.9 versus 35.2), yet
vanilla DPO doubles answer length, almost within the first 10 percent of an epoch, and smaller beta gives
longer answers. Regularized runs land at about half the length and under 40 percent of the KL budget with
higher win rate: about +20 percent on HH and +15 percent on TL;DR at matched length. Phi-2 on UltraFeedback at
alpha 0.05 cuts length and slightly raises MT-Bench.
- Failure mode: pairs whose main difference is length, bullets or a preamble are learned as quality, and the
model optimizes the surface feature.
- Rust: store len_pos and len_neg, apply the alpha margin before log_sigmoid, and report the length delta
distribution per source.

## 12. Curriculum for retrieval pairs: easy lists before hard lists

- Title: Curriculum Learning for Dense Retrieval Distillation
- Authors/Year: Hansi Zeng, Hamed Zamani, Vishwa Vinay, SIGIR 2022
- Identifier: arXiv:2204.13679 (2108.08095 is an unrelated fundus-imaging paper)
- URL: https://arxiv.org/abs/2204.13679
- Recipe: fixed query set; take the student's top 200 documents and rerank with a MiniLM cross-encoder
teacher; group 1 is top K (pseudo-relevant, label 1/rank), group 2 is the next N_h (hard negatives, label 0),
group 3 the next N_s (label -1); 30 documents per query. Three iterations: K = 5, 10, 30 with N_h = 12, 10, 0
and N_s = 13, 10, 0, so fine-grained within-group-1 pairs grow 10, 45, 435. DistilBERT, batch 8, warmup 4000,
lr 7e-6, 3e-6, 3e-6.
- Effect: TAS-B MRR@10 0.344 to 0.382, 11 percent relative; TREC-DL 2019 nDCG 0.717 to 0.725; ColBERTv2 0.384
to 0.394. The reverse curriculum gives 0.378, 0.715, 0.683, below the forward order, so the ordering carries
most of the gain.
- Failure mode: hard negatives too early, when the student has no coarse signal and the hardest negatives are
mostly false negatives.
- Rust: stages: Vec<Stage {k, n_h, n_s}> with per-stage lr, and the negative role stored as a signed i8 label.

## 13. Curriculum for preference pairs: largest rating gap first

- Title: Curry-DPO: Enhancing Alignment using Curriculum Learning & Ranked Preferences
- Authors/Year: Pulkit Pattnaik, Rishabh Maheshwary, Kelechi Ogueji, Vikas Yadav et al., EMNLP 2024 Findings
- Identifier: arXiv:2403.07230
- URL: https://arxiv.org/abs/2403.07230
- Recipe: four UltraFeedback responses per prompt with GPT-4 ratings; always keep the best as chosen and build
(R1,R4), (R1,R3), (R1,R2) ordered by rating gap, largest first, one epoch per iteration, three iterations.
From iteration 2 the reference model is the previous checkpoint, so the KL anchor moves. Batch 32, max lr
5e-7, 10 percent warmup, 5K prompts. A variant orders by the reference model's log-prob gap.
- Effect: Zephyr-7B SFT 6.28 to 7.43 MT-Bench, against 7.08 for the easiest pair alone, 6.84 for the hardest,
and 7.30 for vanilla DPO on 64K pairs. WizardLM 83.4 to 87.1; UltraFeedback 82.9 to 87.9; Mistral-7B 5.11 to
5.71, WizardLM 69.5 to 81.8.
- Failure mode: reverse order falls below vanilla DPO; shuffling gives 6.91, below the easiest pair alone;
ordering by model log-prob gives 7.01 versus 7.43 for the judge gap.
- Rust: a pair list sorted by score gap, a stage index, and reference log-probs reloaded from the previous
checkpoint between stages.

## 14. Mixture sampling: exponent, cap and temperature

- Title: mT5, XLM and T5 task mixing
- Authors/Year: Xue et al 2021; Lample and Conneau 2019; Raffel et al 2020
- Identifier: arXiv:2010.11934, arXiv:1901.07291, arXiv:1910.10683
- URL: https://arxiv.org/abs/2010.11934
- Recipe: p(L) proportional to |L|^alpha. mT5 uses alpha 0.3 over 101 languages, counts from mC4 pages (cld3
at 70 percent confidence, minimum 10k pages), alpha 0.1 in fine-tuning. XLM uses alpha 0.7 for CLM/MLM batches
and 0.5 for BPE learning, 64 streams of 256 tokens. T5 caps counts and applies an exponent: r_m = min(e_m,K) /
sum min(e_n,K), then r_m^(1/T), renormalized; K = 2^21, T = 2 best, T = 4 and 8 tabulated; 15 percent span
corruption is mixed as another task. Equal mixing is the negative control.
- Effect: mT5 alpha 0.7 raises Russian XNLI 81.5 to 82.8 but cuts Swahili 75.4 to 70.6, average 80.7; alpha
0.2 keeps 80.7 average and lifts Urdu 73.5 to 73.9. T5 reports equal mixing degrades badly, proportional
mixing has a per-task sweet spot in K, and T = 2 is reasonable on most tasks.
- Failure mode: uncapped proportional mixing lets the biggest corpus swamp every task; one alpha is zero-sum
between high and low resource, and no alpha is good for all.
- Rust: fn mix(counts: &[u64], k: u64, t: f64) -> Vec<f64> doing min-cap, powf(1/t), normalize; K and T in
config, with per-source pair weights.

## 15. DoReMi: learned domain weights instead of guessed ratios

- Title: DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- Authors/Year: Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V.
Le, Tengyu Ma, Adams Wei Yu, NeurIPS 2023
- Identifier: arXiv:2305.10429
- URL: https://arxiv.org/abs/2305.10429
- Recipe: reference model on raw token proportions, then a proxy trained with Group DRO. Reference and proxy
280M, main model 8B. Per step: uniform minibatch u = 1/k; lambda_t[i] = token-normalized mean of
relu(proxy_nll - reference_nll); alpha'_t = alpha_{t-1} exp(eta * lambda_t); renormalize and smooth with
alpha_t = (1-c) alpha'_t/sum + c*u; eta = 1, c = 1e-3. 200k steps, batch 512, length 1024. Use step-averaged
weights and sample hierarchically, domain then example.
- Effect: +6.5 points average one-shot accuracy; baseline reached at 75k steps instead of 200k, 2.6x fewer
steps, 4x when proxy and main match size; tuning costs 8 percent of large model compute. Pile weights move a
lot: Pile-CC 0.1121 to 0.6057, ArXiv 0.1052 to 0.0036, PubMed Central 0.1071 to 0.0046, Wikipedia 0.0919 to
0.0699.
- Failure mode: naive proportional mixing (the Pile default) gives ArXiv and PubMed Central about 25x their
optimal weight; weights are scale dependent, a 1B proxy picks a different minimum, so retune per model size.
- Rust: struct Doremi { alpha: Vec<f64>, eta: f64, c: f64, k: usize } with update(excess) doing exp,
normalize, lerp to uniform, and a running-mean accumulator.

## 16. Pair count versus quality

- Title: What Makes Good Data for Alignment? (Deita); Less Is More for Alignment (LIMA)
- Authors/Year: Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He, ICLR 2024; Chunting Zhou et al, 2023
- Identifier: arXiv:2312.15685, arXiv:2305.11206
- URL: https://arxiv.org/abs/2312.15685
- Recipe (Deita): score s = c * q, with c Evol Complexity and q Evol Quality from 2K seeds evolved 6 times and
rank-scored by ChatGPT, distilled into two LLaMA-1 7B scorers. Sort the pool by s and accept only if the repr
filter passes: cosine distance to the nearest accepted neighbour d < tau, tau = 0.9. Budgets 6K and 10K from
300K. The DPO stage adds 10K pairs sampled at random from Zephyr's UltraFeedback, so pairs are not filtered.
- Effect: Mistral-7B 6K 7.22 MT-Bench and 80.78 AlpacaEval, 10K 7.32 and 81.67, plus 10K DPO 7.55 and 90.06.
Random 10K at the same budget scores 5.89 and 56.90, a loss of 1.3 MT-Bench and about 24 AlpacaEval points.
Zephyr-beta with 200K SFT plus 60K DPO scores 7.34 and 90.60, so 16K selected matches 260K unfiltered; 3K
selected matches the 300K pool. LIMA: 1,000 curated pairs, 43 percent equal or preferred to GPT-4; on 7B,
doubling the set does not help and quality plateaus across a 16x increase.
- Failure mode: scaling count without quality and diversity buys nothing and can dilute style; repeated
sampling from one prompt inflates count without new signal.
- Rust: quality and complexity scores as f32 columns, a diversity gate over an embedding index with tau=0.9,
and a budget parameter separate from pool size.

## Recommended pair schema

One row per (anchor, candidate set) unit, JSONL on disk, one Rust struct in memory. The enum keeps rank,
contrast, label, preference, correction and retrieval tasks in one table without nullable-field sprawl.

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
enum Task { Rank, Contrast, Label, Preference, Correction, Retrieval }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role { Pos, Neg, HardNeg, Neutral, Gold, Corrected, Rejected }

#[derive(Clone, Copy, PartialEq, Eq)]
enum NegSource { InBatch, Bm25, Ann, CrossEncoder, Random, ModelSample, None }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Split { Train, Dev, Test }

struct Candidate {
    text: String,
    role: Role,
    score: Option<f32>,   // judge or reward score, comparable only within group_id
    rank: Option<u32>,    // 1 = best, for listwise and rank tasks
    cosine: Option<f32>,  // similarity to the anchor at mining time
    token_len: u32,
    truncated: bool,
}

struct PairRow {
    id: u64,
    task: Task,
    group_id: u64,        // prompt, question, claim or query shared by candidates
    anchor: String,       // prompt, question, claim, query or context
    anchor_hash: u64,     // normalized hash, used for leakage and dedup
    anchor_lang: Option<[u8; 2]>,
    candidates: Vec<Candidate>,
    neg_source: NegSource,
    label: Option<u8>,    // Label task class id, None otherwise
    margin: Option<f32>,  // score_pos - score_neg, or the judge gap
    difficulty: Option<f32>, // mining-time cosine, or 1 - rank percentile
    stage: u8,            // curriculum stage, 0 = none
    weight: f32,          // mixture weight read by the sampler
    source: String,       // dataset or generator id
    generator: Option<String>,
    dedup_cluster: Option<u64>,
    eval_overlap: u8,     // count of eval items hit by leakage checks
    split: Split,
    created_unix: i64,
}
```

Field rules:

- `task` selects the loss: Rank uses listwise cross-entropy over candidates, Contrast uses InfoNCE, Label uses
cross-entropy on `label`, Preference uses the pair loss over Pos and Neg, Correction uses Pos = corrected text
and Neg = original, Retrieval is Contrast with `neg_source` recorded.
- Exactly one `Role::Pos` for Contrast, Preference, Correction and Retrieval; `Role::Gold` marks the single
answer passage; Rank may hold any number of scored candidates.
- `margin` and `difficulty` are computed at build time, never at train time, so audits can filter on them.
`cosine` is stored from the miner's encoder so provenance survives encoder changes. `weight` is the only field
the sampler reads.

Minimum audit checks, run on every build and blocking if they fail:

1. Leakage. Normalize case and punctuation, then reject a row if the anchor or any candidate has a 13-gram overlap with an eval item (GPT-3 rule, Brown et al 2020, arXiv:2005.14165), an exact normalized match, or an `anchor_hash` collision with a dev or test row; cosine >= 0.95 to an eval anchor also rejects. Report the hit rate per benchmark as in Dodge et al 2021 (arXiv:2104.08758), which found QNLI sentence overlap at 53.6 percent in C4.

2. Triviality. Reject identical normalized texts, token Jaccard >= 0.9, differences only in punctuation, whitespace or one stopword, or edit distance < 3 tokens. Require a positive `margin` for Preference and Correction. Warn when one source supplies over 20 percent of rows with Jaccard >= 0.7.

3. Length bias. Report mean and median token delta per source and task; fail when the absolute mean delta exceeds 5 percent of mean chosen length, or when the Pearson correlation between `margin` and token delta exceeds 0.3. Keep the sign balance near 50/50 and store lengths so the loss can apply the length margin (alpha 0.01 to 0.05).

4. Margin. Require `margin` >= 0.5 on judge scores normalized to [0,1], drop ties, and cap the share of rows with margin above 3 at 20 percent so the set is not all easy pairs. For retrieval require the hard-negative cosine inside a band (for example 0.7 to 0.95) and reject neighbours above 0.95 as likely false negatives.

5. Duplication. Report the share of rows in dedup clusters larger than one, cap repeated anchors at k rows, and dedup 50-token substrings across the pair corpus (Lee et al 2021, arXiv:2107.06499, which cut memorized tokens from 1.93 to 0.14 percent).

6. Balance. Report task, source, neg_source and stage mix before every run. A single source above 40 percent of the mixture, or a task below its floor, fails the build.
