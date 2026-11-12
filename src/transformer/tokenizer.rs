//! A byte level byte pair tokenizer with no dependency.
//!
//! The vocabulary starts as the 256 byte values plus four control tokens, and
//! grows by merging the most frequent adjacent pair over the word types of the
//! training text. Two properties matter for this crate:
//!
//! * Nothing is ever unknown. A word outside the training vocabulary still
//!   encodes, because its bytes are in the vocabulary from the start, so a
//!   corpus that shifts topic does not turn into a row of `UNK`.
//! * Training is deterministic. Pairs are ordered by count and then by the pair
//!   itself, and word type indices are sorted before a merge touches them, so
//!   two runs on the same text with the same options produce the same merges.
//!
//! Encoding lowercases and splits on the word rule in [`crate::text::tokenize`],
//! which means punctuation is dropped rather than tokenised. For a scorer that
//! reads the words of a document that is the right trade: the vocabulary goes
//! to words instead of to variants of a comma.

use std::collections::{HashMap, HashSet};
use std::io;

use serde::{Deserialize, Serialize};

use crate::text::tokenize;

/// Padding token.
pub const PAD: u32 = 0;
/// Unknown token. Present for shape compatibility; the byte floor means the
/// encoder never emits it.
pub const UNK: u32 = 1;
/// Start of sequence token. Reserved, not emitted by [`Tokenizer::encode`].
pub const BOS: u32 = 2;
/// Separator token. Reserved, not emitted by [`Tokenizer::encode`].
pub const SEP: u32 = 3;
/// The first byte token. Byte `b` has id `FIRST_BYTE + b`.
pub const FIRST_BYTE: u32 = 4;

const CONTROL: usize = FIRST_BYTE as usize;

/// What the tokenizer writes to disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenizerFile {
    /// Format version, so an old file fails loudly instead of silently.
    pub version: u32,
    /// The merges in rank order, each a pair of token ids.
    pub merges: Vec<(u32, u32)>,
}

/// Byte level byte pair encoding.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    /// Byte string of every token, indexed by id. Control tokens are empty.
    bytes: Vec<Vec<u8>>,
}

impl Tokenizer {
    fn build(merges: Vec<(u32, u32)>) -> Self {
        let total = CONTROL + 256 + merges.len();
        let mut bytes: Vec<Vec<u8>> = vec![Vec::new(); total];
        for b in 0..256usize {
            bytes[CONTROL + b] = vec![b as u8];
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, (a, b)) in merges.iter().enumerate() {
            let id = CONTROL + 256 + rank;
            if *a as usize >= bytes.len() || *b as usize >= bytes.len() {
                continue;
            }
            let mut merged = bytes[*a as usize].clone();
            merged.extend_from_slice(&bytes[*b as usize]);
            bytes[id] = merged;
            ranks.insert((*a, *b), rank as u32);
        }
        Self {
            merges,
            ranks,
            bytes,
        }
    }

    /// An untrained tokenizer, bytes only.
    pub fn bytes_only() -> Self {
        Self::build(Vec::new())
    }

    /// Number of tokens in the vocabulary, control tokens included.
    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    /// Number of merges.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// The bytes behind one token id, empty for a control token.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Encode a text to at most `max_len` tokens.
    ///
    /// Words are lowercased and encoded one at a time, so a token never spans a
    /// word boundary. A text whose words all fall outside the merges still
    /// encodes to bytes.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        if max_len == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(max_len.min(text.len()));
        for word in tokenize::words(text) {
            if out.len() >= max_len {
                break;
            }
            let lower = word.to_lowercase();
            let mut ids: Vec<u32> = lower
                .as_bytes()
                .iter()
                .map(|b| FIRST_BYTE + *b as u32)
                .collect();
            self.merge_all(&mut ids);
            for id in ids {
                if out.len() >= max_len {
                    break;
                }
                out.push(id);
            }
        }
        out
    }

    /// Merge every adjacent pair that has a rank, best rank first.
    fn merge_all(&self, ids: &mut Vec<u32>) {
        while ids.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..ids.len() - 1 {
                if let Some(rank) = self.ranks.get(&(ids[i], ids[i + 1])) {
                    if best.map_or(true, |(_, current)| *rank < current) {
                        best = Some((i, *rank));
                    }
                }
            }
            let Some((at, rank)) = best else { break };
            let merged = CONTROL as u32 + 256 + rank;
            ids.splice(at..at + 2, [merged]);
        }
    }

    /// Decode tokens back to text, joining the pieces with a space.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for id in ids {
            if !out.is_empty() {
                out.push(' ');
            }
            let bytes = self.token_bytes(*id);
            out.push_str(&String::from_utf8_lossy(bytes));
        }
        out
    }

    /// The serialisable form.
    pub fn to_file(&self) -> TokenizerFile {
        TokenizerFile {
            version: 1,
            merges: self.merges.clone(),
        }
    }

    /// Rebuild from the serialisable form.
    pub fn from_file(file: &TokenizerFile) -> Result<Self, String> {
        if file.version != 1 {
            return Err(format!("unsupported tokenizer version {}", file.version));
        }
        Ok(Self::build(file.merges.clone()))
    }

    /// Write the merges as JSON.
    pub fn save(&self, path: &str) -> io::Result<()> {
        let body = serde_json::to_string(&self.to_file())
            .map_err(|e| io::Error::other(format!("tokenizer encode: {e}")))?;
        std::fs::write(path, body)
    }

    /// Read merges from JSON.
    pub fn load(path: &str) -> io::Result<Self> {
        let body = std::fs::read_to_string(path)?;
        let file: TokenizerFile = serde_json::from_str(&body)
            .map_err(|e| io::Error::other(format!("tokenizer decode: {e}")))?;
        Self::from_file(&file).map_err(io::Error::other)
    }
}

/// Options for [`train`].
#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    /// Total vocabulary size, control tokens included.
    pub vocab_size: usize,
    /// A pair must appear at least this many times to be merged.
    pub min_count: u64,
    /// Words longer than this many bytes are left out of the pair counts. They
    /// still encode at run time through bytes.
    pub max_word_bytes: usize,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            vocab_size: 2048,
            min_count: 2,
            max_word_bytes: 40,
        }
    }
}

type Pair = (u32, u32);

fn add_word(
    counts: &mut HashMap<Pair, u64>,
    occ: &mut HashMap<Pair, HashSet<u32>>,
    seq: &[u32],
    index: u32,
    freq: u64,
) {
    for pair in seq.windows(2) {
        let key = (pair[0], pair[1]);
        *counts.entry(key).or_insert(0) += freq;
        occ.entry(key).or_default().insert(index);
    }
}

fn remove_word(
    counts: &mut HashMap<Pair, u64>,
    occ: &mut HashMap<Pair, HashSet<u32>>,
    seq: &[u32],
    index: u32,
    freq: u64,
) {
    for pair in seq.windows(2) {
        let key = (pair[0], pair[1]);
        if let Some(count) = counts.get_mut(&key) {
            *count = count.saturating_sub(freq);
        }
        if let Some(set) = occ.get_mut(&key) {
            set.remove(&index);
        }
    }
}

fn rewrite(seq: &[u32], pair: Pair, merged: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == pair.0 && seq[i + 1] == pair.1 {
            out.push(merged);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

/// Learn merges from a set of texts.
///
/// The texts should be the training split only: a tokenizer fitted on the
/// validation text leaks its vocabulary into the model, which is a small leak
/// but an avoidable one.
pub fn train<'a, I>(texts: I, options: TrainOptions) -> Tokenizer
where
    I: IntoIterator<Item = &'a str>,
{
    let mut frequency: HashMap<String, u64> = HashMap::new();
    for text in texts {
        for word in tokenize::words(text) {
            let lower = word.to_lowercase();
            if lower.len() > options.max_word_bytes {
                continue;
            }
            *frequency.entry(lower).or_insert(0) += 1;
        }
    }

    // A stable order for the word types, so the initial indices do not depend
    // on the hash map iteration order.
    let mut words: Vec<(String, u64)> = frequency.into_iter().collect();
    words.sort_by(|a, b| a.0.cmp(&b.0));

    let mut sequences: Vec<Vec<u32>> = Vec::with_capacity(words.len());
    let mut freqs: Vec<u64> = Vec::with_capacity(words.len());
    let mut counts: HashMap<Pair, u64> = HashMap::new();
    let mut occ: HashMap<Pair, HashSet<u32>> = HashMap::new();
    for (index, (word, count)) in words.iter().enumerate() {
        let seq: Vec<u32> = word
            .as_bytes()
            .iter()
            .map(|b| FIRST_BYTE + *b as u32)
            .collect();
        add_word(&mut counts, &mut occ, &seq, index as u32, *count);
        sequences.push(seq);
        freqs.push(*count);
    }

    let target = options.vocab_size.saturating_sub(CONTROL + 256);
    let mut merges: Vec<Pair> = Vec::with_capacity(target);
    while merges.len() < target {
        let best = counts
            .iter()
            .filter(|(_, count)| **count >= options.min_count && **count > 0)
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(pair, count)| (*pair, *count));
        let Some((pair, _)) = best else { break };
        let merged = CONTROL as u32 + 256 + merges.len() as u32;
        merges.push(pair);

        if let Some(indices) = occ.remove(&pair) {
            let mut indices: Vec<u32> = indices.into_iter().collect();
            indices.sort_unstable();
            for index in indices {
                let slot = index as usize;
                let freq = freqs[slot];
                let old = sequences[slot].clone();
                remove_word(&mut counts, &mut occ, &old, index, freq);
                let new = rewrite(&old, pair, merged);
                add_word(&mut counts, &mut occ, &new, index, freq);
                sequences[slot] = new;
            }
        }
        counts.remove(&pair);
        occ.remove(&pair);
    }

    Tokenizer::build(merges)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "The market fell and the market rose. The market fell again, \
        and the market rose again. Interest rates held steady as the market watched \
        inflation and the market watched employment.";

    #[test]
    fn training_learns_merges_and_is_deterministic() {
        let options = TrainOptions {
            vocab_size: 400,
            min_count: 2,
            ..Default::default()
        };
        let a = train([TEXT], options);
        let b = train([TEXT], options);
        assert_eq!(a.to_file().merges, b.to_file().merges);
        assert!(a.merge_count() > 0);
        assert!(a.vocab_size() <= 400);
    }

    #[test]
    fn encoding_shortens_frequent_words_and_never_emits_unknown() {
        let options = TrainOptions {
            vocab_size: 400,
            min_count: 2,
            ..Default::default()
        };
        let tokenizer = train([TEXT], options);
        let ids = tokenizer.encode("the market", 16);
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|id| *id < tokenizer.vocab_size() as u32));
        assert_ne!(ids[0], UNK);
        let bytes = "the market".len();
        assert!(ids.len() < bytes);
    }

    #[test]
    fn an_unseen_word_still_encodes_to_bytes() {
        let tokenizer = Tokenizer::bytes_only();
        let ids = tokenizer.encode("zebra", 32);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[0], FIRST_BYTE + b'z' as u32);
        assert_eq!(tokenizer.decode(&ids), "z e b r a");
    }

    #[test]
    fn save_and_load_round_trip() {
        let options = TrainOptions {
            vocab_size: 320,
            min_count: 2,
            ..Default::default()
        };
        let tokenizer = train([TEXT], options);
        let back = Tokenizer::from_file(&tokenizer.to_file()).expect("reload");
        assert_eq!(tokenizer.vocab_size(), back.vocab_size());
        assert_eq!(tokenizer.encode(TEXT, 64), back.encode(TEXT, 64));
    }

    #[test]
    fn truncation_stops_at_the_limit() {
        let tokenizer = Tokenizer::bytes_only();
        let ids = tokenizer.encode("one two three four", 7);
        assert_eq!(ids.len(), 7);
    }
}
