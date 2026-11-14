//! An FNV-1a digest, so a shipped folder can state which bytes it was built from.
//!
//! The hash is not a security primitive. It answers one question a rebuild has
//! to ask: are these the same bytes the fit read, or has an input been edited
//! since the manifest was stamped? A second preimage is not a concern here, and
//! FNV-1a is short enough to print beside a file name in a card.

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
}
