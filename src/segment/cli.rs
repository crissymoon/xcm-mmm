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
    /// Fit the learned copy map against the segments.
    TextFit(SegmentTextFitArgs),
    /// Rebuild the learned copy map and report every disagreement.
    TextCheck(SegmentTextCheckArgs),
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
    /// Hold the shape of a group fixed: spherical, diagonal or full. Left out,
    /// the panel chooses on held-out likelihood.
    #[arg(long)]
    pub family: Option<String>,
    /// Folds the held-out likelihood is read on.
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    /// Nats per entity a freer family must add before it is preferred.
    #[arg(long, default_value_t = 0.02)]
    pub min_gain: f64,
    /// Starts per expectation maximisation fit, half soft and half k-means.
    #[arg(long, default_value_t = 4)]
    pub em_restarts: usize,
    /// Share of a declared edge a reading needs before it is read as a gate.
    #[arg(long, default_value_t = 0.20)]
    pub gate_share: f64,
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

#[derive(Args)]
pub struct SegmentTextFitArgs {
    /// Ad panel in JSON Lines: one ad per row, copy in `text`.
    #[arg(long)]
    pub ads: String,
    /// Segment model the labels are read from.
    #[arg(long, default_value = "data/segments/model.json")]
    pub model: String,
    /// Emotion model, so a training row reads emotional density the way a
    /// report does.
    #[arg(long, default_value = "data/emotion/model.json")]
    pub emotion: String,
    /// Folder the map and its derived rows are written to.
    #[arg(long, default_value = "data/segments")]
    pub out: String,
    /// Grouped folds, each holding out whole advertisers.
    #[arg(long, default_value_t = 5)]
    pub folds: usize,
    #[arg(long, default_value_t = 900)]
    pub iterations: usize,
    /// L2 penalty on the weights.
    #[arg(long, default_value_t = 0.03)]
    pub l2: f64,
    #[arg(long, default_value_t = 20261114)]
    pub seed: u64,
}

#[derive(Args)]
pub struct SegmentTextCheckArgs {
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
            let family = match a.family.as_deref() {
                None => None,
                Some(text) => Some(crate::segment::Family::parse(text).ok_or_else(|| {
                    format!("unknown family {text:?}; use spherical, diagonal or full")
                })?),
            };
            let cfg = SegmentConfig {
                name: a.name,
                k_min,
                k_max: a.k_max.max(k_min),
                min_share: a.min_share.clamp(0.0, 1.0),
                family,
                folds: a.folds.max(2),
                min_gain: a.min_gain.max(0.0),
                em_restarts: a.em_restarts.max(2),
                gate_share: a.gate_share,
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
                "choice    {} covariance at k = {}, {:.3} nats per entity held out, {:.3} better than k-means",
                m.family.name(),
                m.k,
                m.heldout,
                m.gain
            );
            println!(
                "fit       silhouette {:.3}  control {:.3}  resampling {:.3}  shuffled likelihood {:.3}",
                m.silhouette, m.control.mean, m.stability.mean, m.likelihood_control.mean
            );
            let flagged: Vec<String> = m
                .stability
                .cluster
                .iter()
                .filter(|row| row.flagged)
                .map(|row| format!("{} at {:.3}", row.cluster, row.mean))
                .collect();
            println!(
                "groups    {} of {} hold their members under resampling{}",
                m.stability.cluster.len().saturating_sub(flagged.len()),
                m.stability.cluster.len(),
                if flagged.is_empty() {
                    String::new()
                } else {
                    format!(
                        "; below the {:.1} Jaccard floor: {}",
                        crate::segment::stability::CLUSTER_FLOOR,
                        flagged.join(", ")
                    )
                }
            );
            println!(
                "shape     worst condition number {:.0} (bound {:.0e})",
                crate::segment::mixture::conditioning(&artifact.components),
                crate::segment::mixture::CONDITION_MAX
            );
            println!(
                "sharp     {:.4} mean posterior, {} of {} above {:.1}; cushioned {:.4}, {} moved; held-out keeps {:.1}%",
                m.sharpness.mean_posterior,
                commas(m.sharpness.sharp),
                commas(m.sharpness.rows),
                crate::segment::sharp::SHARP,
                m.sharpness.cushioned_posterior,
                commas(m.sharpness.cushioned_moved),
                m.sharpness.held_out_kept * 100.0
            );
            print_weights(m);
            for gate in &artifact.gates {
                let audit = artifact
                    .metrics
                    .gates
                    .iter()
                    .find(|audit| audit.name == gate.name);
                println!(
                    "gate      {} edge {:.2}  panel {:.1}%  held-out {}  Brier {}  edge right {}",
                    gate.name,
                    gate.boundary,
                    gate.share * 100.0,
                    audit
                        .map(|a| format!("{:+.3} vs {:+.3}", a.log_gate, a.log_gauss))
                        .unwrap_or_else(|| "unmeasured".into()),
                    audit
                        .map(|a| format!("{:.3} vs {:.3}", a.brier_gate, a.brier_gauss))
                        .unwrap_or_else(|| "unmeasured".into()),
                    audit
                        .map(|a| format!("{:.1}% vs {:.1}%", a.accuracy * 100.0, a.rate * 100.0))
                        .unwrap_or_else(|| "unmeasured".into())
                );
            }
            print_pattern(&artifact.metrics);
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
                "k         {} groups from {} to {}, {} covariance",
                m.k,
                artifact.config.k_min,
                artifact.config.k_max,
                m.family.name()
            );
            println!("choice    {}", m.choice);
            println!(
                "shape     worst condition number {:.0} (bound {:.0e})",
                crate::segment::mixture::conditioning(&artifact.components),
                crate::segment::mixture::CONDITION_MAX
            );
            let sharp = &m.sharpness;
            if sharp.rows == 0 {
                println!("sharp     not measured");
            } else {
                println!(
                    "sharp     {:.4} mean posterior (lowest {:.4}), {} sharp, {} contested, {} soft",
                    sharp.mean_posterior,
                    sharp.min_posterior,
                    commas(sharp.sharp),
                    commas(sharp.contested),
                    commas(sharp.soft)
                );
                println!(
                    "cushion   {:.1} sd: {:.4} mean posterior, {} moved, {:.1}% kept; held-out keeps {:.1}%",
                    sharp.cushion,
                    sharp.cushioned_posterior,
                    commas(sharp.cushioned_moved),
                    sharp.cushioned_kept * 100.0,
                    sharp.held_out_kept * 100.0
                );
                for point in sharp.closest.iter().take(3) {
                    println!(
                        "  nearest {:>2} vs {:>2}  posterior {:.4}  held-out {}  cushioned {}  {}",
                        point.assigned,
                        point.rival,
                        point.posterior,
                        point
                            .held_out
                            .map(|group| group.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                        point.cushioned,
                        point.name
                    );
                }
            }
            print_weights(m);
            for gate in &artifact.gates {
                let audit = m.gates.iter().find(|audit| audit.name == gate.name);
                println!(
                    "gate      {} at {:.2}: {:.1}% of the panel on the edge, held-out {}",
                    gate.name,
                    gate.boundary,
                    gate.share * 100.0,
                    audit
                        .map(|a| format!(
                            "{:+.3} vs {:+.3} for the Gaussian, edge right {:.1}% against {:.1}%",
                            a.log_gate,
                            a.log_gauss,
                            a.accuracy * 100.0,
                            a.rate * 100.0
                        ))
                        .unwrap_or_else(|| "unmeasured".into())
                );
            }
            print_pattern(m);
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
            println!(
                "likelihood {:.3} held out (spread {:.3})  shuffled {:.3}  k-means baseline {:.3}",
                m.heldout, m.heldout_sd, m.likelihood_control.mean, m.baseline.heldout
            );
            println!(
                "family    k   held out    spread       BIC       ICL  params  smallest  verdict"
            );
            for candidate in &m.candidates {
                println!(
                    "  {:<8} {:>2}  {:>9.3}  {:>9.3}  {:>8.0}  {:>8.0}  {:>6}  {:>8.3}  {}",
                    candidate.family.name(),
                    candidate.k,
                    candidate.heldout,
                    candidate.heldout_sd,
                    candidate.bic,
                    candidate.icl,
                    candidate.parameters,
                    candidate.min_share,
                    candidate.verdict
                );
            }
            for row in &m.stability.cluster {
                println!(
                    "group {}  holds {:>4}  Jaccard {:.3} (worst {:.3}){}",
                    row.cluster,
                    row.size,
                    row.mean,
                    row.min,
                    if row.flagged {
                        "  below the 0.6 floor"
                    } else {
                        ""
                    }
                );
            }
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
        SegmentCommand::TextFit(a) => {
            let cfg = crate::profile::textmap::TextConfig {
                folds: a.folds.max(2),
                iterations: a.iterations.max(10),
                l2: a.l2.max(0.0),
                seed: a.seed,
                ..Default::default()
            };
            let map = crate::profile::textfit::build(&a.ads, &a.model, &a.emotion, &a.out, &cfg)?;
            println!(
                "source    {} ads from {} advertisers",
                commas(map.source.ads),
                commas(map.source.entities)
            );
            for unit in map.units.values() {
                println!(
                    "  {:<6} {:>6} rows  accuracy {:>5.1}%  balanced {:>5.1}%  chance {:>4.1}%  control {:>5.1}%  {}",
                    unit.unit,
                    unit.rows,
                    unit.accuracy * 100.0,
                    unit.balanced * 100.0,
                    unit.chance * 100.0,
                    unit.control * 100.0,
                    if unit.usable { "in use" } else { "withheld" }
                );
            }
            println!("wrote     {}/textmap.json, text_features.jsonl", a.out);
            println!("note      {}", map.note);
            Ok(())
        }
        SegmentCommand::TextCheck(a) => {
            let faults = crate::profile::textfit::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     the learned copy map rebuilds from its own rows");
                Ok(())
            } else {
                for fault in &faults {
                    println!("fault     {fault}");
                }
                Err(format!("{} faults in {}", faults.len(), a.dir).into())
            }
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
pub use crate::num::commas;

/// The atom modelled inside the likelihood, printed the same way by `fit` and
/// `info`. The full sentence is on the card; this is the number and the first
/// word of the verdict.
fn print_pattern(metrics: &crate::segment::Metrics) {
    for verdict in &metrics.pattern {
        println!(
            "pattern   {}  two-part {:+.3}  loop {:+.3}  pooled {:+.3}  gain {:+.3} ({:+.1} se)  {}",
            verdict.name,
            verdict.two_stage,
            verdict.pattern,
            verdict.pooled,
            verdict.gain,
            verdict.z,
            verdict.verdict.split(':').next().unwrap_or("not measured")
        );
        println!(
            "          rows per group {:.1} gate / {:.1} loop  residual {:.3} / {:.3}  free block posterior {:.3} / {:.3}",
            verdict.gate_rows,
            verdict.loop_rows,
            verdict.gate_sd,
            verdict.loop_sd,
            verdict.gate_sharp,
            verdict.loop_sharp
        );
    }
}

/// The mixing weight audit, printed the same way by `fit` and `info`.
///
/// One function rather than two print statements, because a number that appears
/// in two places is a number that can disagree with itself.
fn print_weights(metrics: &crate::segment::Metrics) {
    for audit in [&metrics.concomitant, &metrics.concomitant_all] {
        if audit.covariates.is_empty() || audit.folds == 0 {
            continue;
        }
        let names: Vec<&str> = audit
            .covariates
            .iter()
            .map(|covariate| covariate.name.as_str())
            .collect();
        println!(
            "weight    {}  gain {:+.3} nats (se {:.3})  means move {:.3} sd  {}",
            names.join(" + "),
            audit.delta,
            audit.delta_se,
            audit.shift_max,
            if audit.honest() {
                "accepted"
            } else {
                "stays out"
            }
        );
    }
}
