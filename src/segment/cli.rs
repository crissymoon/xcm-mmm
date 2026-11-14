//! The `segment` command surface, beside the module it drives.
//!
//! The arguments and the three reports live here rather than in the binary so
//! the command line stays one argument block per front end. Nothing in this
//! module fits anything: it parses, calls [`super::build`] or one of its
//! neighbours, and prints.

use std::error::Error;

use clap::{Args, Subcommand};

use super::{Artifact, SegmentConfig};

/// Arguments for `segment`.
#[derive(Args)]
pub struct SegmentArgs {
    #[command(subcommand)]
    pub command: SegmentCommand,
}

#[derive(Subcommand)]
pub enum SegmentCommand {
    /// Fold an ad panel into one behavioural row per advertiser.
    Extract(SegmentExtractArgs),
    /// Fit the segments and write the folder.
    Fit(SegmentFitArgs),
    /// Print the segments, the selection and the audits.
    Info(SegmentInfoArgs),
    /// Place entities against a fitted model.
    Assign(SegmentAssignArgs),
    /// Rebuild a shipped folder and report every disagreement.
    Check(SegmentCheckArgs),
}

#[derive(Args)]
pub struct SegmentExtractArgs {
    /// Ad panel in JSON Lines: one ad per row.
    #[arg(long)]
    pub ads: String,
    /// Where the entities are written.
    #[arg(long, default_value = "data/segments/features.jsonl")]
    pub out: String,
}

#[derive(Args)]
pub struct SegmentFitArgs {
    /// Entities written by `segment extract`.
    #[arg(long, default_value = "data/segments/features.jsonl")]
    pub features: String,
    /// Folder the model, manifest and card are written to.
    #[arg(long, default_value = "data/segments")]
    pub out: String,
    /// Smallest number of groups the search will consider.
    #[arg(long, default_value_t = 3)]
    pub k_min: usize,
    /// Largest number of groups the search will consider.
    #[arg(long, default_value_t = 6)]
    pub k_max: usize,
    /// Smallest share of the panel any one group may hold.
    #[arg(long, default_value_t = 0.05)]
    pub min_share: f64,
    /// Seed for every draw in the fit and its audits.
    #[arg(long, default_value_t = 20261114)]
    pub seed: u64,
    /// Name the card gives the set.
    #[arg(long, default_value = "marketlab-behaviour-segments")]
    pub name: String,
}

#[derive(Args)]
pub struct SegmentInfoArgs {
    #[arg(long, default_value = "data/segments/model.json")]
    pub model: String,
}

#[derive(Args)]
pub struct SegmentAssignArgs {
    #[arg(long, default_value = "data/segments/model.json")]
    pub model: String,
    /// Entities written by `segment extract`.
    #[arg(long, default_value = "data/segments/features.jsonl")]
    pub entities: String,
    /// Entities to print. Zero prints every one.
    #[arg(long, default_value_t = 25)]
    pub limit: usize,
}

#[derive(Args)]
pub struct SegmentCheckArgs {
    #[arg(long, default_value = "data/segments")]
    pub dir: String,
}

/// Run one `segment` subcommand.
pub fn run(args: SegmentArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        SegmentCommand::Extract(a) => {
            let entities = super::extract(&a.ads, &a.out)?;
            let ads: usize = entities.iter().map(|e| e.ads).sum();
            println!(
                "panel     {} entities, {} ads",
                commas(entities.len()),
                commas(ads)
            );
            println!("wrote     {}", a.out);
            Ok(())
        }
        SegmentCommand::Fit(a) => {
            let k_min = a.k_min.max(2);
            let cfg = SegmentConfig {
                name: a.name,
                k_min,
                k_max: a.k_max.max(k_min),
                min_share: a.min_share.clamp(0.0, 1.0),
                seed: a.seed,
                ..Default::default()
            };
            let artifact = super::build(&a.features, &a.out, &cfg)?;
            let m = &artifact.metrics;
            println!(
                "source    {} entities, {} ads, k = {}",
                commas(m.entities),
                commas(artifact.source.ads),
                artifact.k
            );
            println!(
                "fit       silhouette {:.3}  control {:.3}  resampling {:.3}",
                m.silhouette, m.control.mean, m.stability.mean
            );
            for segment in &artifact.segments {
                println!(
                    "  {:>2}  {:>4} entities  {:>5.1}%  {}",
                    segment.id,
                    segment.size,
                    segment.share * 100.0,
                    segment.label
                );
            }
            println!("wrote     {}/model.json, manifest.json, card.md", a.out);
            Ok(())
        }
        SegmentCommand::Info(a) => {
            let artifact = Artifact::load(&a.model)?;
            let m = &artifact.metrics;
            println!("name      {} v{}", artifact.name, artifact.version);
            println!("built     {}", artifact.built_at);
            println!(
                "source    {} entities  {} ads  {} to {}",
                commas(m.entities),
                commas(artifact.source.ads),
                artifact.source.first_day,
                artifact.source.last_day
            );
            println!("digest    features {}", artifact.source.fnv64);
            println!(
                "k         {} groups from {} to {} by silhouette",
                m.k, artifact.config.k_min, artifact.config.k_max
            );
            println!("  k  silhouette  smallest  inertia  sizes");
            for point in &m.sweep {
                let sizes: Vec<String> = point.sizes.iter().map(|n| n.to_string()).collect();
                println!(
                    "  {}{}  {:>9.3}  {:>8.3}  {:>7.1}  {}",
                    point.k,
                    if point.k == m.k { "*" } else { " " },
                    point.silhouette,
                    point.min_share,
                    point.inertia,
                    sizes.join(" ")
                );
            }
            println!(
                "audit     control {:.3} (best {:.3})  resampling {:.3} (p10 {:.3})",
                m.control.mean, m.control.max, m.stability.mean, m.stability.p10
            );
            println!("segment                      entities  share  strongest readings");
            for segment in &artifact.segments {
                let readings: Vec<String> = segment
                    .profile
                    .iter()
                    .take(3)
                    .map(|row| format!("{} {:+.2}", row.name, row.z))
                    .collect();
                println!(
                    "  {:<28} {:>8}  {:>4.1}%  {}",
                    segment.label,
                    segment.size,
                    segment.share * 100.0,
                    readings.join(", ")
                );
            }
            Ok(())
        }
        SegmentCommand::Assign(a) => {
            let artifact = Artifact::load(&a.model)?;
            let entities = super::train::read_entities(&a.entities)?;
            let mut counted = vec![0usize; artifact.k];
            let mut placed = Vec::with_capacity(entities.len());
            for entity in &entities {
                let (segment, distance) = artifact.classify(&entity.obs)?;
                counted[segment] += 1;
                placed.push((segment, entity.id.clone(), distance));
            }
            placed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            println!(
                "placed    {} entities into {} groups from {}",
                commas(placed.len()),
                artifact.k,
                a.model
            );
            for segment in &artifact.segments {
                println!(
                    "  {:>2}  {:>5}  {:>5.1}%  {}",
                    segment.id,
                    counted[segment.id],
                    counted[segment.id] as f64 / placed.len() as f64 * 100.0,
                    segment.label
                );
            }
            let limit = if a.limit == 0 {
                placed.len()
            } else {
                a.limit.min(placed.len())
            };
            println!("entity                   segment  distance");
            for (segment, entity, distance) in placed.iter().take(limit) {
                println!("  {:<22} {:>7}  {:>8.3}", entity, segment, distance);
            }
            Ok(())
        }
        SegmentCommand::Check(a) => {
            let faults = super::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     {} rebuilds from its own entities", a.dir);
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

/// Group thousands, so an entity count reads at a glance.
fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
