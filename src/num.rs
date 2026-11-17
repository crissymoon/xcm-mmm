//! Small numeric helpers shared across the cards and artifacts.
//!
//! Every artifact rounds before it serialises, so a rebuild writes the same
//! bytes on any machine. That rule only holds while there is one definition of
//! the rounding: a module that carried its own copy drifted, and the churn card
//! ended up printing 0.006 in a table and 0.007 in the sentence beside it for
//! the same error. Everything now calls this.

/// Round to four decimals, the precision the artifacts store.
///
/// Four places is enough to keep a metric readable and coarse enough that a
/// last-bit difference in a floating point reduction does not change the file.
pub fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// Group a count in thousands, which is how every card writes a row count.
///
/// One definition, for the same reason as [`round4`]: five modules carried their
/// own copy of this and a card that printed 1000000 beside a sentence that said
/// "a million" was a disagreement waiting to happen.
pub fn commas(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_is_grouped_in_thousands() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1000), "1,000");
        assert_eq!(commas(1_355_000), "1,355,000");
    }

    #[test]
    fn rounds_half_away_from_zero_at_four_places() {
        assert_eq!(round4(0.00005), 0.0001);
        assert_eq!(round4(-0.00005), -0.0001);
        assert_eq!(round4(1.234567), 1.2346);
    }

    #[test]
    fn a_rounded_value_survives_being_rounded_again() {
        for value in [0.0, 0.00654, -1.99995, 12.5] {
            assert_eq!(round4(round4(value)), round4(value));
        }
    }

    #[test]
    fn no_negative_zero_reaches_an_artifact() {
        assert!(round4(-1e-9).to_string() == "0" || round4(-1e-9) == 0.0);
    }
}
