//! A reading for a word no lexicon lists, by character similarity to words it does.
//!
//! A seed lexicon is finite and English is not. The gap is closed here with the
//! cheapest thing that generalises: the character trigrams of a word, compared
//! with the trigrams of every seed word, cosine weighted. `glorious` shares most
//! of its trigrams with `glory` and `joyful`; `dreadful` shares them with
//! `dread` and `awful`. A word with no overlap at all gets no reading, and the
//! caller falls through to the neutral prior rather than inventing one.
//!
//! The index is inverted, so a query touches only the seeds that share a
//! trigram. That is what makes this affordable over a 236,000 word dictionary:
//! the naive comparison is 150 million pairs and this is a few thousand.
//!
//! The returned vector is scaled by the best cosine, so analogy never reads as
//! loudly as a listed word. A reader can see the tier on every hit.

use std::collections::BTreeMap;

/// How many neighbours an analogy reads, and how weak a match may be.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AnalogyConfig {
    pub take: usize,
    pub min_similarity: f64,
    /// A neighbour below this similarity must share the first three letters.
    /// A two letter overlap alone is noise, not morphology.
    #[serde(default = "default_strong")]
    pub strong_similarity: f64,
    /// Multiplier on the blended vector, after the best similarity is applied.
    pub scale: f64,
}

fn default_strong() -> f64 {
    0.55
}

impl Default for AnalogyConfig {
    fn default() -> Self {
        Self {
            take: 12,
            min_similarity: 0.18,
            strong_similarity: 0.55,
            scale: 0.8,
        }
    }
}

/// One seed word and how alike the query is to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Neighbour {
    pub word: String,
    pub similarity: f64,
}

/// A trigram index over the seed words.
#[derive(Clone, Debug, Default)]
pub struct Analogy {
    words: Vec<String>,
    vectors: Vec<Vec<f64>>,
    norms: Vec<f64>,
    anchors: Vec<u32>,
    starts: Vec<u32>,
    index: BTreeMap<u32, Vec<usize>>,
    config: AnalogyConfig,
}

impl Analogy {
    pub fn build(entries: &BTreeMap<String, Vec<f64>>, config: AnalogyConfig) -> Self {
        let mut words = Vec::with_capacity(entries.len());
        let mut vectors = Vec::with_capacity(entries.len());
        let mut norms = Vec::with_capacity(entries.len());
        let mut anchors = Vec::with_capacity(entries.len());
        let mut starts = Vec::with_capacity(entries.len());
        let mut index: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (word, vector) in entries {
            let grams = grams(word);
            let norm = grams.values().map(|c| c * c).sum::<f64>().sqrt();
            if norm <= 0.0 {
                continue;
            }
            let at = words.len();
            for gram in grams.keys() {
                index.entry(*gram).or_default().push(at);
            }
            words.push(word.clone());
            vectors.push(vector.clone());
            norms.push(norm);
            anchors.push(anchor(word));
            starts.push(start(word));
        }
        Self {
            words,
            vectors,
            norms,
            anchors,
            starts,
            index,
            config,
        }
    }

    pub fn config(&self) -> &AnalogyConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The nearest seed words to a query, strongest first.
    pub fn nearest(&self, word: &str, take: usize) -> Vec<Neighbour> {
        let query = grams(word);
        let norm = query.values().map(|c| c * c).sum::<f64>().sqrt();
        if norm <= 0.0 || self.words.is_empty() {
            return Vec::new();
        }
        let mut dot: Vec<f64> = vec![0.0; self.words.len()];
        for (gram, count) in &query {
            if let Some(postings) = self.index.get(gram) {
                for at in postings {
                    dot[*at] += count;
                }
            }
        }
        let anchor = anchor(word);
        let start = start(word);
        let mut hits: Vec<Neighbour> = dot
            .iter()
            .enumerate()
            .filter(|(at, d)| {
                **d > 0.0
                    && self.anchors[*at] == anchor
                    && (self.starts[*at] == start
                        || *d / (norm * self.norms[*at]) >= self.config.strong_similarity)
            })
            .map(|(at, d)| Neighbour {
                word: self.words[at].clone(),
                similarity: d / (norm * self.norms[at]),
            })
            .filter(|hit| hit.similarity >= self.config.min_similarity)
            .collect();
        hits.sort_by(|a, b| {
            b.similarity
                .total_cmp(&a.similarity)
                .then_with(|| a.word.cmp(&b.word))
        });
        hits.truncate(take.max(1));
        hits
    }

    /// A blended reading for a word the lexicon does not list.
    pub fn read(&self, word: &str) -> Option<AnalogyRead> {
        let neighbours = self.nearest(word, self.config.take);
        let best = neighbours.first()?.similarity;
        let position: BTreeMap<&str, usize> = self
            .words
            .iter()
            .enumerate()
            .map(|(at, word)| (word.as_str(), at))
            .collect();
        let mut weights = vec![0.0; self.vectors.first().map(Vec::len).unwrap_or(0)];
        let mut total = 0.0;
        for neighbour in &neighbours {
            let at = *position.get(neighbour.word.as_str())?;
            let weight = neighbour.similarity * neighbour.similarity;
            for (slot, value) in weights.iter_mut().zip(&self.vectors[at]) {
                *slot += weight * value;
            }
            total += weight;
        }
        if total <= 0.0 {
            return None;
        }
        let confidence = best * self.config.scale;
        for slot in &mut weights {
            *slot = (*slot / total) * confidence;
        }
        Some(AnalogyRead {
            weights,
            confidence,
            neighbours,
        })
    }

    /// The seed vectors the index holds, in order, for a rebuild check.
    pub fn words(&self) -> &[String] {
        &self.words
    }
}

/// What an analogy found: a reading, how sure it is, and the words behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalogyRead {
    pub weights: Vec<f64>,
    pub confidence: f64,
    pub neighbours: Vec<Neighbour>,
}

/// Character trigrams with boundary markers, counted.
pub fn grams(word: &str) -> BTreeMap<u32, f64> {
    let mut marked: Vec<char> = Vec::with_capacity(word.chars().count() + 2);
    marked.push('^');
    marked.extend(word.chars().flat_map(char::to_lowercase));
    marked.push('$');
    let mut out: BTreeMap<u32, f64> = BTreeMap::new();
    if marked.len() < 3 {
        return out;
    }
    for window in marked.windows(3) {
        let key = hash(&window.iter().collect::<String>());
        *out.entry(key).or_insert(0.0) += 1.0;
    }
    out
}

/// The opening characters, so an analogy must share the word's first sounds.
///
/// Without this, `connection` reads as `infection`, `range` as `strange` and
/// `explain` as `complain`: a shared ending is enough to pass a cosine. The
/// anchor is `^` plus the first three letters, so the filter costs one
/// comparison and removes both suffix-only matches and the two letter overlaps
/// that let `plugh` borrow `plan`. A word shorter than three letters anchors on
/// what it has, and still matches its own length.
pub fn anchor(word: &str) -> u32 {
    let mut marked = String::from("^");
    marked.extend(word.chars().flat_map(char::to_lowercase).take(2));
    hash(&marked)
}

/// The opening three letters, so `nearest` can tell morphology from a two
/// letter overlap. A weak neighbour is accepted only when it shares this.
pub fn start(word: &str) -> u32 {
    let mut marked = String::from("^");
    marked.extend(word.chars().flat_map(char::to_lowercase).take(3));
    hash(&marked)
}

/// FNV-1a over the gram, so the index key is a number rather than a string.
fn hash(gram: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in gram.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(pairs: &[(&str, &[f64])]) -> BTreeMap<String, Vec<f64>> {
        pairs
            .iter()
            .map(|(word, weights)| (word.to_string(), weights.to_vec()))
            .collect()
    }

    #[test]
    fn a_near_spelling_finds_the_listed_word() {
        let entries = entry(&[
            ("joy", &[1.0, 0.0, 0.0]),
            ("anger", &[0.0, 1.0, 0.0]),
            ("sadness", &[0.0, 0.0, 1.0]),
        ]);
        let analogy = Analogy::build(&entries, AnalogyConfig::default());
        let neighbours = analogy.nearest("joyful", 3);
        assert_eq!(neighbours[0].word, "joy");
        assert!(neighbours[0].similarity > 0.2);
    }

    #[test]
    fn an_unrelated_string_gets_no_reading() {
        let entries = entry(&[("joy", &[1.0, 0.0]), ("anger", &[0.0, 1.0])]);
        let analogy = Analogy::build(&entries, AnalogyConfig::default());
        assert!(analogy.read("zzqqxx").is_none());
        assert!(analogy.nearest("zzqqxx", 3).is_empty());
    }

    #[test]
    fn the_blend_is_scaled_below_a_listed_reading() {
        let entries = entry(&[("dread", &[0.0, 1.0]), ("dreadful", &[0.0, 1.0])]);
        let analogy = Analogy::build(&entries, AnalogyConfig::default());
        let read = analogy.read("dreadfully").expect("a neighbour");
        assert!(read.weights[1] > 0.0);
        assert!(read.weights[1] < 1.0);
        assert!(read.confidence <= 0.8);
        assert!(!read.neighbours.is_empty());
    }

    #[test]
    fn a_two_letter_overlap_is_not_an_analogy() {
        let entries = entry(&[("plan", &[1.0, 0.0]), ("place", &[0.0, 1.0])]);
        let analogy = Analogy::build(&entries, AnalogyConfig::default());
        assert!(
            analogy.read("plugh").is_none(),
            "a shared opening pair is not morphology: {:?}",
            analogy.nearest("plugh", 4)
        );
    }

    #[test]
    fn a_three_letter_prefix_still_reaches_a_seed() {
        let entries = entry(&[("dread", &[0.0, 1.0])]);
        let analogy = Analogy::build(&entries, AnalogyConfig::default());
        let read = analogy.read("dreadful").expect("a neighbour");
        assert!(read.weights[1] > 0.0);
    }

    #[test]
    fn the_index_is_deterministic() {
        let entries = entry(&[("happy", &[1.0]), ("sad", &[0.0])]);
        let a = Analogy::build(&entries, AnalogyConfig::default());
        let b = Analogy::build(&entries, AnalogyConfig::default());
        assert_eq!(a.nearest("happily", 3), b.nearest("happily", 3));
        assert_eq!(a.words(), b.words());
    }
}
