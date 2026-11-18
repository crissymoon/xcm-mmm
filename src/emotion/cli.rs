//! The `emotion` command surface, beside the module it drives.
//!
//! Every subcommand reads or prints; the two that write (`fit`) do so through
//! [`super::fit::build`], so a command line invocation and a rebuild produce the
//! same bytes. Nothing here scores a text differently from the scanner, which is
//! the point: the numbers in the report are the numbers the artifact gives.

use std::error::Error;
use std::fs;

use clap::{Args, Subcommand};

use super::coverage::{self, Coverage};
use super::fit;
use super::model::{EmotionModel, Tier};

/// Arguments for `emotion`.
#[derive(Args)]
pub struct EmotionArgs {
    #[command(subcommand)]
    pub command: EmotionCommand,
}

#[derive(Subcommand)]
pub enum EmotionCommand {
    /// Read one text and print its emotions, valence and evidence.
    Score(EmotionScoreArgs),
    /// Print the model, its fit and the terms the bank added.
    Info(EmotionInfoArgs),
    /// Count which tier answers for every word in a list.
    Coverage(EmotionCoverageArgs),
    /// Refit the folder from its own seed and bank.
    Fit(EmotionFitArgs),
    /// Rebuild a shipped folder and report every disagreement.
    Check(EmotionCheckArgs),
}

#[derive(Args)]
pub struct EmotionScoreArgs {
    /// Text to read.
    #[arg(long, conflicts_with = "file")]
    pub text: Option<String>,
    /// File to read as one text.
    #[arg(long)]
    pub file: Option<String>,
    #[arg(long, default_value = "data/emotion/model.json")]
    pub model: String,
    /// Print the reading as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct EmotionInfoArgs {
    #[arg(long, default_value = "data/emotion/model.json")]
    pub model: String,
}

#[derive(Args)]
pub struct EmotionCoverageArgs {
    #[arg(long, default_value = "data/emotion/model.json")]
    pub model: String,
    /// Newline separated word list.
    #[arg(long, default_value = "/usr/share/dict/words")]
    pub words: String,
    /// Words printed per tier that came back without evidence.
    #[arg(long, default_value_t = 6)]
    pub examples: usize,
    /// JSON Lines corpus whose tokens are counted as running text.
    #[arg(long, default_value = "data/emotion/bank.jsonl")]
    pub corpus: String,
}

#[derive(Args)]
pub struct EmotionFitArgs {
    #[arg(long, default_value = "data/emotion")]
    pub dir: String,
}

#[derive(Args)]
pub struct EmotionCheckArgs {
    #[arg(long, default_value = "data/emotion")]
    pub dir: String,
}

/// Run one `emotion` subcommand.
pub fn run(args: EmotionArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        EmotionCommand::Score(a) => score(a),
        EmotionCommand::Info(a) => {
            info(&EmotionModel::load(&a.model)?);
            Ok(())
        }
        EmotionCommand::Coverage(a) => coverage(a),
        EmotionCommand::Fit(a) => {
            let model = fit::build(&a.dir)?;
            println!(
                "seed      {} words  {}",
                commas(model.source.seed_words),
                model.source.seed_fnv64
            );
            println!(
                "bank      {} rows  {} labels",
                commas(model.source.bank_rows),
                model.metrics.labels
            );
            println!(
                "metrics   fit {:.3}  leave-one-out {:.3}  majority {:.3}",
                model.metrics.fit_accuracy, model.metrics.leave_one_out, model.metrics.majority
            );
            println!("wrote     {}/model.json, manifest.json, card.md", a.dir);
            Ok(())
        }
        EmotionCommand::Check(a) => {
            let faults = fit::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     {} rebuilds from its own seed and bank", a.dir);
                Ok(())
            } else {
                for fault in &faults {
                    println!("fault     {fault}");
                }
                Err(format!("{} faults in {}", faults.len(), a.dir).into())
            }
        }
    }
}

fn score(args: EmotionScoreArgs) -> Result<(), Box<dyn Error>> {
    let text = match (&args.text, &args.file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) => fs::read_to_string(path)?,
        (None, None) => return Err("give --text or --file".into()),
        (Some(_), Some(_)) => return Err("give --text or --file, not both".into()),
    };
    let model = EmotionModel::load(&args.model)?;
    let reading = model.scanner().read(&text);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&reading)?);
        return Ok(());
    }
    println!(
        "text      {} words  {} characters",
        commas(text.split_whitespace().count()),
        commas(text.chars().count())
    );
    println!("dominant  {}", reading.dominant);
    println!(
        "valence   {:+.3}  arousal {:.3}  intensity {:.3}",
        reading.valence, reading.arousal, reading.intensity
    );
    println!("emotion        share");
    for (emotion, share) in &reading.emotions {
        println!("  {:<12} {:>5.1}%  {}", emotion, share * 100.0, bar(*share));
    }
    if reading.hits.is_empty() {
        println!("hits      none: every word fell to the neutral prior");
        return Ok(());
    }
    println!("word         tier       stem           weight  valence");
    for hit in &reading.hits {
        println!(
            "  {:<10} {:<10} {:<14} {:>6.2}  {:+.2}",
            hit.word, hit.tier, hit.stem, hit.weight, hit.valence
        );
    }
    Ok(())
}

fn info(model: &EmotionModel) {
    let m = &model.metrics;
    println!("name      {} v{}", model.name, model.version);
    println!("built     {}", model.built_at);
    println!(
        "source    {} seed words  {} bank rows",
        commas(model.source.seed_words),
        commas(model.source.bank_rows)
    );
    println!(
        "digest    seed {}  bank {}",
        model.source.seed_fnv64, model.source.bank_fnv64
    );
    println!(
        "fit       blend {:.2}  band {:.2}  min mass {:.2}  prior {:.1}",
        model.blend, model.neutral_band, model.min_mass, model.prior
    );
    println!(
        "metrics   fit {:.3}  leave-one-out {:.3}  majority {:.3}  labels {}",
        m.fit_accuracy, m.leave_one_out, m.majority, m.labels
    );
    println!("label         accuracy");
    for (label, value) in &m.per_label {
        println!("  {:<12} {:>6.3}", label, value);
    }
    println!("emotion        seed  learned  valence  arousal  top learned terms");
    for emotion in &model.emotions {
        let at = model.emotion_index(emotion).unwrap_or(0);
        let seed = model
            .seed
            .values()
            .filter(|weights| weights.get(at).copied().unwrap_or(0.0) > 0.0)
            .count();
        let learned = model
            .learned
            .values()
            .filter(|weights| weights.get(at).copied().unwrap_or(0.0) > 0.0)
            .count();
        let top: Vec<&str> = model
            .learned_top
            .get(emotion)
            .map(|terms| terms.iter().take(4).map(String::as_str).collect())
            .unwrap_or_default();
        println!(
            "  {:<12} {:>4}  {:>7}  {:>+7.1}  {:>7.1}  {}",
            emotion,
            seed,
            learned,
            model.valence_weights.get(emotion).copied().unwrap_or(0.0),
            model.arousal_weights.get(emotion).copied().unwrap_or(0.0),
            top.join(" ")
        );
    }
    println!("tiers     exact, lemma, prefix, analogy, prior");
}

fn coverage(args: EmotionCoverageArgs) -> Result<(), Box<dyn Error>> {
    let model = EmotionModel::load(&args.model)?;
    let scanner = model.scanner();
    let covered = coverage::measure(&scanner, &args.words, args.examples)?;
    print_coverage(&covered, &args.words);
    if std::path::Path::new(&args.corpus).is_file() {
        let corpus = coverage::measure_corpus(&scanner, &args.corpus)?;
        println!(
            "corpus    {} tokens from {}",
            commas(corpus.tokens),
            args.corpus
        );
        for tier in Tier::ALL {
            let count = corpus.tiers.get(tier.as_str()).copied().unwrap_or(0);
            println!(
                "  {:<12} {:>8}  {:>6.1}%",
                tier.as_str(),
                commas(count),
                corpus.share(count) * 100.0
            );
        }
        println!(
            "corpus    {} tokens, {} content  {:.1}% of content reads above the neutral prior",
            commas(corpus.tokens),
            commas(corpus.content),
            corpus.content_share() * 100.0
        );
    }
    Ok(())
}

fn print_coverage(covered: &Coverage, path: &str) {
    println!("words     {} from {}", commas(covered.words), path);
    println!("tier            words    share   informative");
    for tier in Tier::ALL {
        let count = covered.total(tier);
        let informative = covered.informative.get(tier.as_str()).copied().unwrap_or(0);
        println!(
            "  {:<12} {:>8}  {:>6.1}%  {:>10}",
            tier.as_str(),
            commas(count),
            covered.share(count) * 100.0,
            commas(informative)
        );
    }
    println!(
        "handled   {}  {:.2}% of the list reads above the neutral prior",
        commas(covered.handled()),
        covered.share(covered.handled()) * 100.0
    );
    println!(
        "evidence  {}  {:.2}% of the list reads above {:.2}",
        commas(covered.informative_total()),
        covered.share(covered.informative_total()) * 100.0,
        coverage::INFORMATIVE
    );
    for (tier, examples) in &covered.examples {
        if examples.is_empty() {
            continue;
        }
        println!("  {tier:<10} no evidence: {}", examples.join(", "));
    }
}

/// A twenty cell bar, for reading a share at a glance.
fn bar(share: f64) -> String {
    let filled = (share.clamp(0.0, 1.0) * 20.0).round() as usize;
    "=".repeat(filled)
}

pub use crate::num::commas;
