//! Hashing for deduplication: shingles, MinHash signatures and SimHash bits.
//!
//! No dependency is used and none is needed. FNV-1a is the base hash because it
//! is a few instructions per byte and stable across releases, which matters
//! when a signature written today is compared with one written last month.
//!
//! The MinHash family is built by double hashing: `a_i * x + b_i mod 2^64` with
//! `a_i` odd and both coefficients derived from one SplitMix64 stream. That is
//! the standard cheap substitute for `k` independent permutations and it costs
//! one multiply and one add per permutation per shingle.

/// FNV-1a over bytes.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// SplitMix64 finaliser, used to spread a hash across all 64 bits.
pub fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Hash of one word, case folded on ASCII so `Profit` and `profit` collide.
pub fn word_hash(word: &str) -> u64 {
    fnv1a(word.as_bytes()) ^ (word.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Hash of a cleaned text, for exact duplicate detection and for the row id
/// when a record arrives without one.
pub fn text_hash(text: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        let b = if byte.is_ascii_uppercase() {
            byte + 32
        } else if byte == b' ' || byte == b'\n' || byte == b'\t' {
            b' '
        } else {
            byte
        };
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A short hexadecimal digest, stable and independent of the hash map seed.
pub fn hex64(h: u64) -> String {
    format!("{h:016x}")
}

/// Coefficients for the MinHash family.
#[derive(Clone, Debug)]
pub struct Perms {
    a: Vec<u64>,
    b: Vec<u64>,
}

impl Perms {
    /// Build `k` coefficient pairs from one seed. `a` is forced odd so the map
    /// is a bijection on 64 bit words.
    pub fn new(k: usize, seed: u64) -> Self {
        let mut state = mix(seed ^ 0x51ed_270b_9e37_79b9);
        let mut a = Vec::with_capacity(k);
        let mut b = Vec::with_capacity(k);
        for _ in 0..k {
            a.push(mix(state) | 1);
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            b.push(mix(state));
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        }
        Self { a, b }
    }

    /// Number of permutations.
    pub fn len(&self) -> usize {
        self.a.len()
    }

    /// Whether the family is empty, which would make every signature equal.
    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }
}

/// Shingle hashes of a word sequence, deduplicated and sorted.
///
/// A document shorter than `shingle` words becomes one shingle over all of its
/// words, so a headline still gets a signature instead of an empty one.
///
/// The inner loop is one multiply and one add per word, with a single SplitMix
/// finaliser per shingle. Hashing each word through the finaliser costs three
/// more multiplies per word and buys nothing once the result is sorted into a
/// set, which is the measurement in `research/rust-notes.md`: 2.25 ns against
/// 13.8 ns for a 40 byte shingle.
pub fn shingles(words: &[u64], shingle: usize) -> Vec<u64> {
    const ODD: u64 = 0x9e37_79b9_7f4a_7c15;
    let k = shingle.max(1);
    let mut out = Vec::new();
    if words.len() <= k {
        let mut h = 0u64;
        for &w in words {
            h = h.wrapping_mul(ODD).wrapping_add(w);
        }
        out.push(mix(h) ^ words.len() as u64);
    } else {
        for window in words.windows(k) {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for &w in window {
                h = h.wrapping_mul(ODD).wrapping_add(w);
            }
            out.push(mix(h));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// MinHash signature of a shingle set.
///
/// The minimum is taken in 64 bits and stored in 32, which halves the memory
/// for a corpus of ten million documents. Truncation only matters when two
/// different shuffles collide after truncation, and 112 values at 32 bits make
/// that rare enough to ignore for an equality based signature comparison.
pub fn signature(shingle_hashes: &[u64], perms: &Perms) -> Vec<u32> {
    let mut sig = vec![u64::MAX; perms.len()];
    for &x in shingle_hashes {
        for (i, (&a, &b)) in perms.a.iter().zip(perms.b.iter()).enumerate() {
            let h = a.wrapping_mul(x).wrapping_add(b);
            if h < sig[i] {
                sig[i] = h;
            }
        }
    }
    sig.into_iter()
        .map(|v| if v == u64::MAX { 0 } else { v as u32 })
        .collect()
}

/// Estimated Jaccard similarity from two signatures of equal length.
pub fn signature_similarity(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    same as f64 / a.len() as f64
}

/// Key for one band of a signature, used as the LSH bucket.
pub fn band_key(sig: &[u32], band: usize, rows: usize) -> Option<u64> {
    let start = band * rows;
    let end = start + rows;
    if end > sig.len() {
        return None;
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (band as u64);
    for &v in &sig[start..end] {
        h = mix(h ^ v as u64);
    }
    Some(h)
}

/// SimHash of a weighted word sequence: one bit per word hash bit, set when the
/// count of agreeing words outweighs the disagreeing ones.
pub fn simhash(words: &[u64]) -> u64 {
    let mut counts = [0i64; 64];
    for &w in words {
        let h = mix(w);
        for (bit, count) in counts.iter_mut().enumerate() {
            if (h >> bit) & 1 == 1 {
                *count += 1;
            } else {
                *count -= 1;
            }
        }
    }
    let mut out = 0u64;
    for (bit, &count) in counts.iter().enumerate() {
        if count > 0 {
            out |= 1 << bit;
        }
    }
    out
}

/// Number of differing bits.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Exact Jaccard similarity of two sorted, deduplicated hash sets.
pub fn jaccard(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let (mut i, mut j, mut both) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                both += 1;
                i += 1;
                j += 1;
            }
        }
    }
    both as f64 / (a.len() + b.len() - both) as f64
}

/// Cosine similarity of two sparse hashed vectors, each sorted by feature.
pub fn cosine(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut dot) = (0, 0, 0.0f32);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_recovers_shingle_jaccard() {
        let perms = Perms::new(128, 7);
        let a: Vec<u64> = (0..40).map(|i| mix(i as u64)).collect();
        let mut b = a.clone();
        // Replace a fifth of the words: exact Jaccard is 4/6 of the sets.
        for slot in b.iter_mut().take(8) {
            *slot = mix(999 + *slot);
        }
        let sa = signature(&shingles(&a, 3), &perms);
        let sb = signature(&shingles(&b, 3), &perms);
        let estimate = signature_similarity(&sa, &sb);
        let exact = jaccard(&shingles(&a, 3), &shingles(&b, 3));
        assert!((estimate - exact).abs() < 0.12, "{estimate} vs {exact}");
    }

    #[test]
    fn band_keys_are_stable_and_distinct() {
        let perms = Perms::new(16, 3);
        let sig = signature(&shingles(&[1, 2, 3, 4, 5], 2), &perms);
        assert_eq!(band_key(&sig, 0, 4), band_key(&sig, 0, 4));
        assert_ne!(band_key(&sig, 0, 4), band_key(&sig, 1, 4));
    }

    #[test]
    fn simhash_is_close_for_similar_texts() {
        let words = |s: &str| -> Vec<u64> { s.split(' ').map(word_hash).collect() };
        let a = simhash(&words("fed holds rates steady as inflation cools"));
        let b = simhash(&words("fed holds rates steady as inflation slows"));
        let c = simhash(&words("chip maker raises guidance on data centre demand"));
        assert!(hamming(a, b) < hamming(a, c));
    }
}
