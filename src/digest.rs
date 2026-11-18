//! Content digests, so a shipped folder can state which bytes it was built from.
//!
//! Two functions, because they answer two different questions. FNV-1a is short
//! enough to print beside a file name in a card, and it is what the manifests
//! use to ask whether an input has been edited since they were stamped. SHA-256
//! is what the registry uses: an audit trail a reviewer can check has to survive
//! a deliberate edit, not only an accidental one, and FNV-1a does not.
//!
//! Neither is a signature. A digest says the bytes are the bytes that were
//! recorded; it says nothing about who recorded them.

/// FNV-1a over bytes, as sixteen lowercase hex digits.
pub fn fnv64(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// [`fnv64`] for a string.
pub fn fnv64_str(text: &str) -> String {
    fnv64(text.as_bytes())
}

/// SHA-256 over bytes, as sixty-four lowercase hex digits (FIPS 180-4).
///
/// Written out rather than pulled in as a dependency: the crate has no crypto
/// dependency and this is the only hash it needs. The known-answer tests below
/// are the contract, including the empty message, the one-block `abc` and the
/// two-block vector, because a hash that is subtly wrong still produces a
/// stable-looking digest.
pub fn sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut message = bytes.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(bytes.len() as u64).wrapping_mul(8).to_be_bytes());

    for chunk in message.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (word, bytes) in schedule.iter_mut().take(16).zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for at in 16..64 {
            let s0 = schedule[at - 15].rotate_right(7)
                ^ schedule[at - 15].rotate_right(18)
                ^ (schedule[at - 15] >> 3);
            let s1 = schedule[at - 2].rotate_right(17)
                ^ schedule[at - 2].rotate_right(19)
                ^ (schedule[at - 2] >> 10);
            schedule[at] = schedule[at - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[at - 7])
                .wrapping_add(s1);
        }

        let mut round = state;
        for at in 0..64 {
            let s1 =
                round[4].rotate_right(6) ^ round[4].rotate_right(11) ^ round[4].rotate_right(25);
            let choose = (round[4] & round[5]) ^ ((!round[4]) & round[6]);
            let temp1 = round[7]
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[at])
                .wrapping_add(schedule[at]);
            let s0 =
                round[0].rotate_right(2) ^ round[0].rotate_right(13) ^ round[0].rotate_right(22);
            let majority = (round[0] & round[1]) ^ (round[0] & round[2]) ^ (round[1] & round[2]);
            let temp2 = s0.wrapping_add(majority);
            round[7] = round[6];
            round[6] = round[5];
            round[5] = round[4];
            round[4] = round[3].wrapping_add(temp1);
            round[3] = round[2];
            round[2] = round[1];
            round[1] = round[0];
            round[0] = temp1.wrapping_add(temp2);
        }
        for (word, add) in state.iter_mut().zip(round) {
            *word = word.wrapping_add(add);
        }
    }

    state.iter().map(|word| format!("{word:08x}")).collect()
}

/// [`sha256`] for a string.
pub fn sha256_str(text: &str) -> String {
    sha256(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_empty_string_is_the_offset_basis() {
        assert_eq!(fnv64(b""), "cbf29ce484222325");
    }

    #[test]
    fn one_byte_change_changes_the_digest() {
        assert_ne!(fnv64_str("bank"), fnv64_str("banks"));
        assert_eq!(fnv64_str("bank"), fnv64_str("bank"));
        assert_eq!(fnv64_str("bank").len(), 16);
    }

    #[test]
    fn sha256_matches_the_published_vectors() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_pads_the_length_it_was_given() {
        // 55 bytes pad inside one block, 56 bytes need a second block. Both
        // match the vectors a reference implementation produces.
        let short = "a".repeat(55);
        let long = "a".repeat(56);
        assert_eq!(sha256_str(&short).len(), 64);
        assert_eq!(
            sha256_str(&short),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        assert_eq!(
            sha256_str(&long),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
        );
        assert_ne!(sha256_str(&short), sha256_str(&long));
    }

    #[test]
    fn sha256_of_a_thousand_a_is_the_known_value() {
        assert_eq!(
            sha256_str(&"a".repeat(1000)),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }
}
