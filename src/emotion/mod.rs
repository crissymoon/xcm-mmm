//! Emotion readings over the whole English vocabulary.
//!
//! [`detect`](crate::detect) asks a narrow question of a text: does it carry a
//! legal register, a threat, a claim about followers. This module asks the
//! wider question the churn work needed: what does the text feel like, word by
//! word, and can every word in the language get an answer?
//!
//! The answer has four layers:
//!
//! * [`seed`](fit::SEED), 661 words with a weight on each of eight axes;
//! * a labelled bank of twenty contact-centre transcripts that adjusts the seed
//!   and, more importantly, gives the adjustment a held-out number;
//! * [`lemma`], which folds `happiness`, `happier` and `happily` onto `happy`;
//! * [`analogy`], which reads `glorious` from the words `glory` and `joyful`
//!   because it shares most of its characters with them.
//!
//! Every word resolves, so a text never fails to score and
//! [`coverage`](coverage::measure) can report which tier carried how much of a
//! dictionary. The folder this module ships is `data/emotion/`: the seed, the
//! bank, the artifact, its manifest and the card, and `emotion check` rebuilds
//! the artifact from the first two and reports every disagreement.

pub mod analogy;
pub mod card;
pub mod cli;
pub mod coverage;
pub mod fit;
pub mod lemma;
pub mod model;

pub use cli::{run, EmotionArgs};
pub use coverage::{measure as measure_coverage, Coverage};
pub use fit::{build, check, read_bank, BankRow, SeedFile};
pub use model::{EmotionModel, Hit, Reading, Scanner, Tier};

/// The folder this module ships, relative to the repository root.
pub const FOLDER: &str = "data/emotion";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_folder_names_its_files_once() {
        assert_eq!(fit::SEED, "seed.json");
        assert_eq!(fit::BANK, "bank.jsonl");
        assert_eq!(fit::MODEL, "model.json");
        assert!(FOLDER.starts_with("data/"));
    }

    #[test]
    fn every_tier_has_a_name_and_the_names_are_unique() {
        let names: Vec<&str> = Tier::ALL.iter().map(|tier| tier.as_str()).collect();
        let unique: std::collections::BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
        assert!(names.contains(&"prior"));
    }
}
