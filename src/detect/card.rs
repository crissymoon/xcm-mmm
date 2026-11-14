//! The card: what the folder holds, in the words a reader needs first.

use std::fmt::Write;

use super::model::{Artifact, Axis, Manifest};

/// Render `card.md` for a fitted folder.
pub fn render(artifact: &Artifact, manifest: &Manifest) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}", artifact.name);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Seven text detectors over advertising copy, fitted in Rust. Five are declared cue indices and two are lexicons learned from a labelled bank. Built {} from the files listed below, and checked by rebuilding them.",
        artifact.built_at
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Source");
    let _ = writeln!(out);
    let _ = writeln!(out, "| file | rows | digest |");
    let _ = writeln!(out, "|---|---:|---|");
    let _ = writeln!(
        out,
        "| {} | {} | `{}` |",
        artifact.source.bank, artifact.source.bank_rows, artifact.source.bank_fnv64
    );
    let _ = writeln!(
        out,
        "| {} | {} | `{}` |",
        artifact.source.ads, artifact.source.ads_rows, artifact.source.ads_fnv64
    );
    let _ = writeln!(
        out,
        "| {} | {} axes | `{}` |",
        artifact.source.cues, artifact.source.cues_axes, artifact.source.cues_fnv64
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "The bank is legal register from Federal Register abstracts and abstraction from grants.gov synopses, each set against advertisers' own copy as the negative side. The ads file is the same copy at document grain with its source split kept, so the index references are fitted on training ads and nothing else."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Fitted axes");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Held out numbers come from {} folds assigned by the digest of the text, with the cut chosen inside each training part. `balanced` is the mean of recall on the positives and specificity on the negatives; `majority` is the score of answering no to everything; `shuffled` is the same fit on permuted labels.",
        artifact.config.folds
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| axis | terms | rows | recall | precision | balanced | majority | shuffled |"
    );
    let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|");
    for (name, axis) in &artifact.axes {
        if let Axis::Lexicon(lex) = axis {
            let m = &lex.metrics;
            let _ = writeln!(
                out,
                "| {name} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                lex.lexicon.terms.len(),
                m.rows,
                m.recall,
                m.precision,
                m.balanced,
                m.majority,
                m.shuffled.unwrap_or(f64::NAN)
            );
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Declared axes");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "No labelled corpus exists for these constructs, so nothing here is an accuracy. `prevalence` is the share of ads that fire, `halves_agree` is the share where two disjoint cue halves agree, `both` is the share where both halves fire, and `r` is their correlation over the ads where at least one did."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| axis | tag | prevalence | halves agree | both | r | checks |"
    );
    let _ = writeln!(out, "|---|---|---:|---:|---:|---:|---:|");
    for (name, axis) in &artifact.axes {
        if let Axis::Index(ind) = axis {
            let r = ind
                .audit
                .halves_correlation
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "null".to_string());
            let _ = writeln!(
                out,
                "| {name} | `{}` | {:.3} | {:.2} | {} | {r} | {}/{} |",
                ind.tag,
                ind.audit.prevalence,
                ind.audit.halves_agree,
                ind.audit.halves_both_fired,
                ind.checks.iter().filter(|c| c.passed).count(),
                ind.checks.len()
            );
        }
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Reproduce");
    let _ = writeln!(out);
    let _ = writeln!(out, "```bash");
    let _ = writeln!(
        out,
        "scripts/export-detect-data.py          # rewrite the three inputs"
    );
    let _ = writeln!(
        out,
        "xcm-mmm detect train --bank data/detectors/bank.jsonl \\"
    );
    let _ = writeln!(
        out,
        "  --ads data/detectors/ads.jsonl --cues data/detectors/cues.json \\"
    );
    let _ = writeln!(out, "  --out data/detectors");
    let _ = writeln!(out, "xcm-mmm detect check --dir data/detectors");
    let _ = writeln!(out, "```");
    let _ = writeln!(out);

    let _ = writeln!(out, "## Limits");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- The legal and jargon banks are instruments and grant synopses set against ad copy. A high held out number says the two registers are separable, which is a fact about administrative English, not proof that a detector reads intent in an advertisement."
    );
    let _ = writeln!(
        out,
        "- The five declared axes have no labels. The audit measures whether a cue set agrees with itself; it cannot measure whether it is right, and a prevalence near zero on this corpus is a fact about Los Angeles political and issue advertising rather than about the construct."
    );
    let _ = writeln!(
        out,
        "- Fear ships its declared seed cues only. The Python artifact adds a threat lexicon learned from advertiser copy to the same component, so a fear score here runs lower than the one published beside the corpus."
    );
    let _ = writeln!(
        out,
        "- Levels are bands over the fitting corpus, so `strong` means strong relative to this bank. Comparing a reading across corpora is comparing two different rulers."
    );
    let _ = writeln!(
        out,
        "- The build time is a fact about the run, not the fit. The lexicon, the cuts and the references rebuild byte for byte; `built_at` does not."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{} axes. `model.json` is {} bytes and its digest is {}.",
        manifest.axes.len(),
        manifest
            .files
            .iter()
            .find(|f| f.name == "model.json")
            .map(|f| f.bytes.to_string())
            .unwrap_or_else(|| "?".to_string()),
        manifest
            .files
            .iter()
            .find(|f| f.name == "model.json")
            .map(|f| f.fnv64.clone())
            .unwrap_or_else(|| "?".to_string())
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::model::Source;
    use crate::detect::TrainConfig;
    use std::collections::BTreeMap;

    fn manifest() -> Manifest {
        Manifest {
            name: "marketlab-ad-detectors".into(),
            version: 1,
            built_at: "now".into(),
            files: Vec::new(),
            config: TrainConfig::default(),
            axes: BTreeMap::new(),
        }
    }

    #[test]
    fn the_card_names_the_source_and_the_limits() {
        let artifact = Artifact {
            name: "marketlab-ad-detectors".into(),
            version: 1,
            built_at: "now".into(),
            source: Source {
                bank: "bank.jsonl".into(),
                bank_rows: 10,
                bank_fnv64: "abc".into(),
                ads: "ads.jsonl".into(),
                ads_rows: 2,
                ads_fnv64: "def".into(),
                cues: "cues.json".into(),
                cues_axes: 5,
                cues_fnv64: "ghi".into(),
            },
            config: TrainConfig::default(),
            axes: BTreeMap::new(),
        };
        let text = render(&artifact, &manifest());
        assert!(text.contains("bank.jsonl"));
        assert!(text.contains("## Limits"));
        assert!(text.contains("detect check"));
    }
}
