# Emotion readings

xcm-mmm-emotion v1, built 2026-11-14T23:00:51Z. Eight Plutchik axes over English words: a listed seed, a
learned adjustment from a labelled bank of contact-centre transcripts, and a
character-analogy fallback so a word no list carries still reads.

## Source

| file | rows | digest |
|---|---:|---|
| `seed.json` | 673 words | `11576acbf80e2194` |
| `bank.jsonl` | 20 rows | `4bbd2f0c7b6f636f` |
| `listed.tsv` | 7208 tokens | `5aa92f7c57c4f5cc` |

## Fit

Blend 0.35. Neutral band 0.12. A reading whose valence stays inside the
band, or whose total affect mass is under 0.90, is labelled neutral.

| reading | value |
|---|---:|
| bank rows | 20 |
| distinct labels | 5 |
| fit accuracy | 0.850 |
| leave-one-out accuracy | 0.850 |
| majority baseline | 0.350 |

| label | accuracy |
|---|---:|
| Angry | 1.000 |
| Confused | 0.000 |
| Frustrated | 0.667 |
| Happy | 1.000 |
| Neutral | 0.857 |

## Seed by axis

| emotion | words | valence | arousal |
|---|---:|---:|---:|
| joy | 103 | +1.0 | 0.6 |
| trust | 102 | +0.7 | 0.1 |
| anticipation | 84 | +0.5 | 0.4 |
| surprise | 40 | +0.1 | 0.9 |
| sadness | 85 | -1.0 | 0.2 |
| fear | 89 | -0.8 | 1.0 |
| anger | 133 | -0.9 | 1.0 |
| disgust | 56 | -1.0 | 0.6 |

## Learned terms

Words the bank added to an axis, strongest first, capped at 0.35.

- **joy**: `thank`

## Tiers

A word is read at the first tier that answers.

1. exact: a seeded or learned word;
2. listed: a VADER polarity word mapped onto the axes;
3. lemma: an inflection folded onto a listed stem (`happiness`, `running`);
4. prefix: a negating prefix over a listed base (`unhappy`, `dislike`);
5. suffix: an ending folded onto a listed base (`joyless`, `hopefully`);
6. analogy: character trigrams shared with listed words;
7. prior: no evidence, read as neutral.

`emotion coverage` counts these tiers over a word list. The seed is
673 words and every word in a list lands on a tier, so "every English
word" is a measurement rather than a claim.

## Limits

- The bank is twenty transcripts. Its adjustment is a calibration, not
  a fit a reader should trust on new domains.
- Negation flips the next three affect words; a sentence with an
  ambiguous scope will read the wrong word.
- Analogy is spelling, not meaning. It needs a shared opening three letters
  unless the spelling is very close, so `joyful` reaches `joy` and `plugh`
  reaches nothing, while `carpet` still sits near `carp` rather than `comfort`.
- The seed is a working vocabulary, not a published lexicon; a word the
  seed and the valence list both miss reads at the derivation or analogy tier,
  or as neutral when neither has evidence.
- The valence list is VADER (Hutto and Gilbert, 2014), MIT licence. It
  supplies polarity only; the eight axes remain this model's mapping.
