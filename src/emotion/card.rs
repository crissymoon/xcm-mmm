//! The card that ships beside the emotion artifact.
//!
//! A reader should be able to answer four questions from this file alone: what
//! the model reads, which words it was given, how well it did on held-out rows,
//! and where it is weak. The card is generated from the artifact, so it cannot
//! describe a fit that is not the one on disk.

use super::model::{EmotionModel, Tier};
use crate::num::commas;

/// Render the card.
pub fn render(model: &EmotionModel) -> String {
    let mut out = String::new();
    out.push_str("# Emotion readings\n\n");
    out.push_str(&format!(
        "{} v{}, built {}. Eight Plutchik axes over English words: a listed seed, a\nlearned adjustment from a labelled bank of contact-centre transcripts, and a\ncharacter-analogy fallback so a word no list carries still reads.\n\n",
        model.name, model.version, model.built_at
    ));
    out.push_str("## Source\n\n");
    out.push_str("| file | rows | digest |\n|---|---:|---|\n");
    out.push_str(&format!(
        "| `{}` | {} words | `{}` |\n",
        model.source.seed, model.source.seed_words, model.source.seed_fnv64
    ));
    out.push_str(&format!(
        "| `{}` | {} rows | `{}` |\n",
        model.source.bank, model.source.bank_rows, model.source.bank_fnv64
    ));
    out.push_str(&format!(
        "| `{}` | {} tokens | `{}` |\n\n",
        model.source.listed, model.source.listed_rows, model.source.listed_fnv64
    ));
    out.push_str("## Fit\n\n");
    out.push_str(&format!(
        "Blend {:.2}. Neutral band {:.2}. A reading whose valence stays inside the\nband, or whose total affect mass is under {:.2}, is labelled neutral.\n\n",
        model.blend, model.neutral_band, model.min_mass
    ));
    out.push_str("| reading | value |\n|---|---:|\n");
    let m = &model.metrics;
    out.push_str(&format!("| bank rows | {} |\n", m.rows));
    out.push_str(&format!("| distinct labels | {} |\n", m.labels));
    out.push_str(&format!("| fit accuracy | {:.3} |\n", m.fit_accuracy));
    out.push_str(&format!(
        "| leave-one-out accuracy | {:.3} |\n",
        m.leave_one_out
    ));
    out.push_str(&format!("| majority baseline | {:.3} |\n\n", m.majority));
    out.push_str("| label | accuracy |\n|---|---:|\n");
    for (label, value) in &m.per_label {
        out.push_str(&format!("| {label} | {value:.3} |\n"));
    }
    out.push_str("\n## What the bank can resolve\n\n");
    if m.detectable == 0.0 && m.resolution_note.is_empty() {
        out.push_str(
            "This artifact predates the resolution reading. Rebuild with `emotion fit` to populate it.\n\n",
        );
    } else {
        out.push_str("| reading | value |\n|---|---:|\n");
        out.push_str(&format!(
            "| smallest resolvable accuracy gain at 80% power | {:.3} |\n",
            m.detectable
        ));
        out.push_str(&format!(
            "| rows that would resolve a three point gain | {} |\n",
            commas(m.rows_for_floor)
        ));
        out.push_str(&format!(
            "| resolves the three point floor | {} |\n\n",
            if m.resolves_floor { "yes" } else { "no" }
        ));
        out.push_str(&format!("{}\n\n", m.resolution_note));
        out.push_str(
            "The reading is a property of twenty rows, not of the scanner. It says a small gain cannot be demonstrated on this bank; it does not say the scanner cannot make one.\n\n",
        );
    }
    out.push_str(
        "## Seed by axis\n\n| emotion | words | valence | arousal |\n|---|---:|---:|---:|\n",
    );
    for emotion in &model.emotions {
        let at = model.emotion_index(emotion).unwrap_or(0);
        let words = model
            .seed
            .values()
            .filter(|weights| weights.get(at).copied().unwrap_or(0.0) > 0.0)
            .count();
        out.push_str(&format!(
            "| {emotion} | {words} | {:+.1} | {:.1} |\n",
            model.valence_weights.get(emotion).copied().unwrap_or(0.0),
            model.arousal_weights.get(emotion).copied().unwrap_or(0.0)
        ));
    }
    out.push_str("\n## Learned terms\n\n");
    out.push_str("Words the bank added to an axis, strongest first, capped at 0.35.\n\n");
    for (emotion, terms) in &model.learned_top {
        if terms.is_empty() {
            continue;
        }
        let shown: Vec<String> = terms.iter().take(8).map(|t| format!("`{t}`")).collect();
        out.push_str(&format!("- **{emotion}**: {}\n", shown.join(", ")));
    }
    out.push_str("\n## Tiers\n\n");
    out.push_str("A word is read at the first tier that answers.\n\n");
    for (at, tier) in Tier::ALL.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", at + 1, describe(*tier)));
    }
    out.push_str(&format!(
        "\n`emotion coverage` counts these tiers over a word list. The seed is\n\
         {} words and every word in a list lands on a tier, so \"every English\n\
         word\" is a measurement rather than a claim.\n\n",
        model.source.seed_words
    ));
    out.push_str("## Limits\n\n");
    out.push_str("- The bank is twenty transcripts. Its adjustment is a calibration, not\n  a fit a reader should trust on new domains.\n");
    out.push_str("- Negation flips the next three affect words; a sentence with an\n  ambiguous scope will read the wrong word.\n");
    out.push_str(
        "- Analogy is spelling, not meaning. It needs a shared opening three letters\n  unless the spelling is very close, so `joyful` reaches `joy` and `plugh`\n  reaches nothing, while `carpet` still sits near `carp` rather than `comfort`.\n",
    );
    out.push_str("- The seed is a working vocabulary, not a published lexicon; a word the\n  seed and the valence list both miss reads at the derivation or analogy tier,\n  or as neutral when neither has evidence.\n");
    out.push_str("- The valence list is VADER (Hutto and Gilbert, 2014), MIT licence. It\n  supplies polarity only; the eight axes remain this model's mapping.\n");
    out
}

/// One line for a tier, used by the card and the coverage report.
fn describe(tier: Tier) -> &'static str {
    match tier {
        Tier::Exact => "exact: a seeded or learned word;",
        Tier::Listed => "listed: a VADER polarity word mapped onto the axes;",
        Tier::Lemma => "lemma: an inflection folded onto a listed stem (`happiness`, `running`);",
        Tier::Prefix => "prefix: a negating prefix over a listed base (`unhappy`, `dislike`);",
        Tier::Suffix => "suffix: an ending folded onto a listed base (`joyless`, `hopefully`);",
        Tier::Analogy => "analogy: character trigrams shared with listed words;",
        Tier::Prior => "prior: no evidence, read as neutral.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_card_names_the_fit_and_the_tiers() {
        let path = "data/emotion/model.json";
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let model = super::super::model::EmotionModel::load(path).expect("the shipped artifact");
        let card = render(&model);
        assert!(card.contains("# Emotion readings"));
        assert!(card.contains("leave-one-out accuracy"));
        assert!(card.contains("analogy"));
        assert!(card.contains("## What the bank can resolve"));
        assert!(card.contains("smallest resolvable accuracy gain"));
        assert!(card.contains("No label audit runs on this bank"));
        assert!(card.contains(&model.source.seed_fnv64));
        assert!(card.len() > 1200);
    }
}
