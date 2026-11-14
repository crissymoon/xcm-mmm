//! Command line front end for xcm-mmm.

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};

use xcm_mmm::benchmarks::{Benchmark, Kind, CATALOGUE};
use xcm_mmm::data::{Dataset, Fold};
use xcm_mmm::de::{self, Adaptation, Crossover, DeConfig, Strategy};
use xcm_mmm::forecast::{
    knee, score_over, Report, SelectionProblem, SparseAccuracy, DEFAULT_LAMBDA,
};
use xcm_mmm::metrics::{hypervolume_report, igd, reference_front};
use xcm_mmm::moead::scalar::Scalarization;
use xcm_mmm::moead::{self, MoeadConfig, MoeadOutcome};
use xcm_mmm::problem::Problem;

#[derive(Parser)]
#[command(
    name = "xcm-mmm",
    version,
    about = "Differential evolution and MOEA/D, with walk-forward forecasting, corpus cleaning and pair mining",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List the benchmark problems and their defaults.
    List,
    /// Solve a single objective problem with differential evolution.
    De(DeArgs),
    /// Solve a multi-objective problem with MOEA/D.
    Moead(MoeadArgs),
    /// Select features for a regression and score them out of sample.
    Forecast(ForecastArgs),
    /// Write a synthetic panel so the tool can be tried without data.
    Synth(SynthArgs),
    /// Run the benchmark suite.
    Bench(BenchArgs),
    /// Clean a corpus, drop near duplicates and keep a relevant budget.
    Clean(CleanArgs),
    /// Mine, pack, retain or audit the pairs a single model is trained on.
    Pairs(PairsArgs),
    /// Ship a stratified sample of a pair bank as the training set in this repo.
    Dataset(DatasetArgs),
    /// Train, score or check the seven text detectors.
    Detect(DetectArgs),
    /// Train, score or inspect the shared transformer on a pair corpus.
    Transformer(TransformerArgs),
}

#[derive(clap::Args)]
struct TransformerArgs {
    #[command(subcommand)]
    command: TransformerCommand,
}

#[derive(Subcommand)]
enum TransformerCommand {
    /// Train one model on a mixed pair corpus.
    Train(TransformerTrainArgs),
    /// Score a pair corpus with a saved model.
    Eval(TransformerEvalArgs),
    /// Show the shape and metadata of a saved model.
    Info(TransformerInfoArgs),
    /// Encode a text and score it with every task head.
    Encode(TransformerEncodeArgs),
    /// Score every text row in a corpus with a saved model.
    Score(TransformerScoreArgs),
}

#[derive(clap::Args)]
struct TransformerTrainArgs {
    /// Pair files, comma separated.
    #[arg(long, value_delimiter = ',', required = true)]
    pairs: Vec<String>,
    /// Where the model file is written.
    #[arg(long)]
    out: String,
    /// Tokenizer vocabulary size, control tokens included.
    #[arg(long, default_value_t = 2048)]
    vocab: usize,
    /// Longest token sequence.
    #[arg(long, default_value_t = 64)]
    max_len: usize,
    /// Residual width.
    #[arg(long, default_value_t = 64)]
    dim: usize,
    /// Attention heads.
    #[arg(long, default_value_t = 4)]
    heads: usize,
    /// Feed forward width.
    #[arg(long, default_value_t = 128)]
    ff: usize,
    /// Transformer blocks.
    #[arg(long, default_value_t = 2)]
    layers: usize,
    /// Epochs in the plan, counting resumed ones. Defaults to the checkpoint's
    /// plan when resuming, and to three otherwise.
    #[arg(long)]
    epochs: Option<usize>,
    /// Rows per gradient step.
    #[arg(long, default_value_t = 8)]
    batch: usize,
    /// Peak learning rate.
    #[arg(long, default_value_t = 3e-3)]
    lr: f64,
    /// Decoupled weight decay.
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,
    /// Share of the schedule spent warming up.
    #[arg(long, default_value_t = 0.1)]
    warmup: f64,
    /// Gradient norm limit.
    #[arg(long, default_value_t = 1.0)]
    clip: f64,
    /// Dropout on the residual branches.
    #[arg(long, default_value_t = 0.1)]
    dropout: f64,
    #[arg(long, default_value_t = 11)]
    seed: u64,
    /// Read at most this many pairs. Zero reads everything.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Epochs without improvement before stopping. Zero disables it.
    #[arg(long, default_value_t = 2)]
    patience: usize,
    /// Smallest pair count a merge may come from.
    #[arg(long, default_value_t = 2)]
    min_count: u64,
    /// Sampling weight per task, as `rank=3,label=0.5`.
    #[arg(
        long,
        default_value = "rank=1,contrast=1,preference=1,label=0.5,correction=0.5,retrieval=1"
    )]
    task_weights: String,
    /// Continue from a checkpoint written by an earlier run.
    #[arg(long)]
    resume: Option<String>,
    /// End this leg after this many epochs in total, keeping the plan's schedule.
    #[arg(long)]
    stop_after: Option<usize>,
    /// Write mid-run state here after every epoch group.
    #[arg(long)]
    checkpoint: Option<String>,
    /// Epochs between checkpoint writes.
    #[arg(long, default_value_t = 1)]
    checkpoint_every: usize,
}

#[derive(clap::Args)]
struct TransformerEvalArgs {
    /// Model file written by `transformer train`.
    model: String,
    /// Pair files, comma separated.
    #[arg(long, value_delimiter = ',', required = true)]
    pairs: Vec<String>,
    /// Read at most this many pairs. Zero reads everything.
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

#[derive(clap::Args)]
struct TransformerInfoArgs {
    model: String,
}

#[derive(clap::Args)]
struct TransformerEncodeArgs {
    model: String,
    /// Text to encode.
    text: String,
}

#[derive(clap::Args)]
struct TransformerScoreArgs {
    /// Model file written by `transformer train`.
    model: String,
    /// Text rows, one JSON object per line.
    #[arg(long)]
    rows: String,
    /// Where the scored rows are written.
    #[arg(long)]
    out: String,
    /// One task head. Omit to write every head on each line.
    #[arg(long)]
    task: Option<String>,
}

#[derive(clap::Args)]
struct DeArgs {
    /// Problem name, see `xcm-mmm list`.
    #[arg(long, default_value = "sphere")]
    problem: String,
    /// Number of variables. Zero uses the catalogue default.
    #[arg(long, default_value_t = 0)]
    dim: usize,
    /// Evaluation budget. Zero uses the automatic rule.
    #[arg(long, default_value_t = 0)]
    budget: usize,
    /// Population size. Zero uses the automatic rule.
    #[arg(long, default_value_t = 0)]
    population: usize,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Parameter adaptation.
    #[arg(long, value_enum, default_value_t = Adapter::Shade)]
    adaptation: Adapter,
    /// Mutation operator.
    #[arg(long, value_enum, default_value_t = Operator::CurrentToPBest)]
    strategy: Operator,
    /// Crossover operator.
    #[arg(long, value_enum, default_value_t = CrossoverKind::Binomial)]
    crossover: CrossoverKind,
    /// Fixed scale factor, used by the static adaptation.
    #[arg(long, default_value_t = 0.5)]
    f: f64,
    /// Fixed crossover rate, used by the static adaptation.
    #[arg(long, default_value_t = 0.9)]
    cr: f64,
    /// Share of the population eligible as x_pbest at the start.
    #[arg(long, default_value_t = 0.11)]
    p_best: f64,
    /// End of the x_pbest schedule. Zero holds p_best for the whole run, 0.125
    /// with a start of 0.25 is the jSO schedule.
    #[arg(long, default_value_t = 0.0)]
    p_best_final: f64,
    /// Restart the worst half after this many stalled generations. Zero is off.
    #[arg(long, default_value_t = 0)]
    stall: usize,
    /// Keep the population size fixed instead of the L-SHADE schedule.
    #[arg(long)]
    no_reduce: bool,
    /// Print a JSON report instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct MoeadArgs {
    /// Problem name, see `xcm-mmm list`.
    #[arg(long, default_value = "zdt1")]
    problem: String,
    /// Number of variables. Zero uses the catalogue default.
    #[arg(long, default_value_t = 0)]
    dim: usize,
    /// Target number of subproblems.
    #[arg(long, default_value_t = 100)]
    population: usize,
    /// Lattice divisions. Zero derives them from the population.
    #[arg(long, default_value_t = 0)]
    divisions: usize,
    /// Neighbourhood size.
    #[arg(long, default_value_t = 20)]
    neighbourhood: usize,
    /// Probability of mating inside the neighbourhood.
    #[arg(long, default_value_t = 0.9)]
    delta: f64,
    /// Maximum neighbours replaced per child.
    #[arg(long, default_value_t = 2)]
    replacements: usize,
    /// Share of subproblems that produce a child each generation. One is
    /// plain MOEA/D-DE; the default varies a fifth.
    #[arg(long, default_value_t = 0.2)]
    update_probability: f64,
    /// Share of the run that updates every subproblem before the partial
    /// schedule starts.
    #[arg(long, default_value_t = 0.1)]
    update_delay: f64,
    /// Stagnation window in generations for relative-improvement allocation.
    /// Zero draws the subset uniformly at random.
    #[arg(long, default_value_t = 50)]
    ri_window: usize,
    /// Differential scale factor.
    #[arg(long, default_value_t = 0.5)]
    f: f64,
    /// Crossover rate. One point zero is the MOEA/D-DE default.
    #[arg(long, default_value_t = 1.0)]
    cr: f64,
    /// Scalarising function.
    #[arg(long, value_enum, default_value_t = Scalar::Tchebycheff)]
    scalar: Scalar,
    /// PBI penalty, used when the scalariser is PBI.
    #[arg(long, default_value_t = 5.0)]
    theta: f64,
    /// Evaluation budget. Zero uses 50 000.
    #[arg(long, default_value_t = 0)]
    budget: usize,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Write the front to a CSV file.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print a JSON report instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct ForecastArgs {
    /// CSV or JSON Lines file.
    data: PathBuf,
    /// Name of the target column.
    #[arg(long, default_value = "target")]
    target: String,
    /// Number of walk-forward folds.
    #[arg(long, default_value_t = 4)]
    folds: usize,
    /// Rows dropped between train and test.
    #[arg(long, default_value_t = 1)]
    embargo: usize,
    /// Ridge penalty.
    #[arg(long, default_value_t = DEFAULT_LAMBDA)]
    lambda: f64,
    /// Search to run. Both is the default because the two answer different
    /// questions: one best subset, or the whole accuracy against size curve.
    #[arg(long, value_enum, default_value_t = Method::Both)]
    method: Method,
    /// Error added per unit of feature fraction in the single objective search.
    #[arg(long, default_value_t = 0.01)]
    sparsity: f64,
    /// Evaluation budget for the search.
    #[arg(long, default_value_t = 6_000)]
    budget: usize,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Write the Pareto front to a CSV file.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Fraction of rows held back from the search and scored once at the end.
    /// Zero keeps every row in the search, which flatters the result because
    /// the selection then scores itself on rows it already saw.
    #[arg(long, default_value_t = 0.0)]
    holdout: f64,
    /// Print a JSON report instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct SynthArgs {
    /// Rows to create.
    #[arg(long, default_value_t = 600)]
    rows: usize,
    /// Features to create.
    #[arg(long, default_value_t = 12)]
    features: usize,
    /// How many features carry signal.
    #[arg(long, default_value_t = 4)]
    informative: usize,
    /// Random seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Destination file. A `.jsonl` suffix writes JSON Lines, anything else CSV.
    #[arg(long, default_value = "synthetic.csv")]
    out: PathBuf,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Runs per problem, with consecutive seeds.
    #[arg(long, default_value_t = 3)]
    runs: u64,
    /// Evaluation budget. Zero uses each engine's default.
    #[arg(long, default_value_t = 0)]
    budget: usize,
    /// Start of the x_pbest schedule for the DE section.
    #[arg(long, default_value_t = 0.11)]
    p_best: f64,
    /// End of the x_pbest schedule for the DE section. Zero holds p_best.
    #[arg(long, default_value_t = 0.0)]
    p_best_final: f64,
    /// Restart the worst half of the DE population after this many stalled
    /// generations. Zero is off.
    #[arg(long, default_value_t = 0)]
    stall: usize,
    /// Per generation share of subproblems that produce a child, for the
    /// MOEA/D section. Omitted means the engine default; 1 is plain MOEA/D-DE.
    #[arg(long)]
    update_probability: Option<f64>,
    /// Stagnation window for relative-improvement allocation in the MOEA/D
    /// section. Zero keeps the uniform random subset.
    #[arg(long, default_value_t = 50)]
    ri_window: usize,
    /// Print JSON instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct CleanArgs {
    /// JSON Lines file, gzip accepted.
    input: String,
    /// Destination file, gzip accepted.
    out: String,
    /// Jaccard threshold for near duplicate merging. Zero disables the pass.
    #[arg(long, default_value_t = 0.80)]
    dedupe: f64,
    /// Keep at most this many rows. Zero keeps every survivor.
    #[arg(long, default_value_t = 0)]
    budget: usize,
    /// Smallest word count a row may have.
    #[arg(long, default_value_t = 5)]
    min_words: usize,
    /// Largest word count a row may have.
    #[arg(long, default_value_t = 3000)]
    max_words: usize,
    /// Composite quality floor for selection.
    #[arg(long, default_value_t = 0.0)]
    floor: f64,
    /// Cosine similarity at which a selected row counts as redundant.
    #[arg(long, default_value_t = 0.85)]
    theta: f64,
    /// Largest share of the budget one source may take.
    #[arg(long, default_value_t = 0.10)]
    group_cap: f64,
    /// Curated target set, JSON Lines or plain text. Setting it replaces the
    /// coverage pass with target-aware importance resampling.
    #[arg(long)]
    target: Option<String>,
}

#[derive(clap::Args)]
struct PairsArgs {
    #[command(subcommand)]
    command: PairCommand,
}

#[derive(Subcommand)]
enum PairCommand {
    /// Build rank, preference and contrast pairs from labelled rows.
    Mine(MineArgs),
    /// Merge pair files into one corpus with a task mix.
    Pack(PackArgs),
    /// Check a pair corpus for leakage, triviality and contamination.
    Audit(AuditArgs),
    /// Keep the best pairs across runs in one growing bank.
    Retain(RetainArgs),
    /// Print the composition of a retention bank.
    Bank(BankArgs),
}

#[derive(clap::Args)]
struct DatasetArgs {
    #[command(subcommand)]
    command: DatasetCommand,
}

#[derive(Subcommand)]
enum DatasetCommand {
    /// Sample a bank into train, valid and test, with a manifest and a card.
    Build(DatasetBuildArgs),
    /// Check a shipped folder against its own manifest.
    Check(DatasetCheckArgs),
}

#[derive(clap::Args)]
struct DatasetBuildArgs {
    /// The pair bank to sample.
    #[arg(long)]
    input: String,
    /// Folder the split files, manifest and card are written to.
    #[arg(long)]
    out: String,
    /// Rows the sample may hold across all splits. Zero ships the whole bank.
    #[arg(long, default_value_t = 20_000)]
    budget: usize,
    /// Seed of the sample. The same seed rebuilds the same bytes.
    #[arg(long, default_value_t = 20_261_114)]
    seed: u64,
    /// Bytes any one shipped file may reach.
    #[arg(long, default_value_t = 40)]
    cap_mb: u64,
    /// Name the card gives the set.
    #[arg(long, default_value = "marketlab-news-pairs")]
    name: String,
}

#[derive(clap::Args)]
struct DatasetCheckArgs {
    /// Folder holding manifest.json and the split files.
    #[arg(long)]
    dir: String,
}

#[derive(clap::Args)]
struct DetectArgs {
    #[command(subcommand)]
    command: DetectCommand,
}

#[derive(Subcommand)]
enum DetectCommand {
    /// Fit the seven axes from the bank, the ads and the cue table.
    Train(DetectTrainArgs),
    /// Score text or a JSON Lines file with a fitted model.
    Score(DetectScoreArgs),
    /// Print what a model holds and how well it scored.
    Info(DetectInfoArgs),
    /// Rebuild a shipped folder and report every disagreement.
    Check(DetectCheckArgs),
}

#[derive(clap::Args)]
struct DetectTrainArgs {
    /// Labelled sentence bank: axis, label, source and text per row.
    #[arg(long, default_value = "data/detectors/bank.jsonl")]
    bank: String,
    /// Ad copy at document grain, with its source split.
    #[arg(long, default_value = "data/detectors/ads.jsonl")]
    ads: String,
    /// Declared cue table for the index axes.
    #[arg(long, default_value = "data/detectors/cues.json")]
    cues: String,
    /// Folder the model, manifest and card are written to.
    #[arg(long, default_value = "data/detectors")]
    out: String,
    /// Folds of the out of fold evaluation.
    #[arg(long, default_value_t = 5)]
    folds: usize,
    /// Smallest number of rows either side of an axis may hold.
    #[arg(long, default_value_t = 40)]
    min_rows: usize,
    /// Terms an axis may keep.
    #[arg(long, default_value_t = 120)]
    top_terms: usize,
    /// Name the card gives the set.
    #[arg(long, default_value = "marketlab-ad-detectors")]
    name: String,
}

#[derive(clap::Args)]
struct DetectScoreArgs {
    /// Model artifact written by `detect train`.
    #[arg(long, default_value = "data/detectors/model.json")]
    model: String,
    /// Text to score. Repeat the flag for more than one.
    #[arg(long)]
    text: Vec<String>,
    /// JSON Lines in, one `text` field per row, scored to `--out`.
    #[arg(long)]
    input: Option<String>,
    /// Where the scored rows go. Required with --input.
    #[arg(long)]
    out: Option<String>,
    /// Terms or phrases printed beside each reading.
    #[arg(long, default_value_t = 6)]
    hits: usize,
}

#[derive(clap::Args)]
struct DetectInfoArgs {
    #[arg(long, default_value = "data/detectors/model.json")]
    model: String,
}

#[derive(clap::Args)]
struct DetectCheckArgs {
    #[arg(long, default_value = "data/detectors")]
    dir: String,
}

#[derive(clap::Args)]
struct MineArgs {
    /// Labelled rows: a text and a forward outcome per row.
    rows: String,
    out: String,
    /// Smallest outcome gap a pair may carry.
    #[arg(long, default_value_t = 0.01)]
    min_margin: f64,
    /// Largest cosine similarity the two sides may share.
    #[arg(long, default_value_t = 0.95)]
    max_similarity: f64,
    /// How many of the hardest negatives to choose among.
    #[arg(long, default_value_t = 5)]
    neighbours: usize,
    /// Cap per task. Zero keeps every pair.
    #[arg(long, default_value_t = 0)]
    max_per_task: usize,
    /// Longest text copied into a pair.
    #[arg(long, default_value_t = 700)]
    max_chars: usize,
    #[arg(long, default_value_t = 11)]
    seed: u64,
}

#[derive(clap::Args)]
struct PackArgs {
    /// Pair files, comma separated.
    #[arg(long, value_delimiter = ',', required = true)]
    input: Vec<String>,
    #[arg(long)]
    out: String,
    /// Total rows. Zero keeps every row that survives the filters.
    #[arg(long, default_value_t = 0)]
    total: usize,
    /// Weight per task, as `rank=3,contrast=2`.
    #[arg(
        long,
        default_value = "rank=3,contrast=2,preference=2,label=1,correction=1,retrieval=1"
    )]
    weights: String,
    /// Similarity at which two sides of one pair count as the same text.
    #[arg(long, default_value_t = 0.98)]
    max_similarity: f64,
    #[arg(long, default_value_t = 5)]
    seed: u64,
    /// Write the manifest here instead of printing it.
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(clap::Args)]
struct AuditArgs {
    input: String,
    /// JSON Lines or plain text evaluation rows.
    #[arg(long)]
    eval: Option<String>,
    #[arg(long, default_value_t = 0.98)]
    max_similarity: f64,
    #[arg(long, default_value_t = 0.80)]
    contamination: f64,
    #[arg(long, default_value_t = 0.01)]
    min_margin: f64,
}

#[derive(clap::Args)]
struct RetainArgs {
    /// Candidate pair files, comma separated. One day's output per file.
    #[arg(long, value_delimiter = ',', required = true)]
    input: Vec<String>,
    /// Corpus the bank keeps. Its index sits beside it as `<out>.idx.json`.
    #[arg(long)]
    out: String,
    /// Rows the bank may hold. Better rows push out worse ones.
    #[arg(long, default_value_t = 60_000)]
    budget: usize,
    /// Most rows from one label day.
    #[arg(long, default_value_t = 4_000)]
    per_day: usize,
    /// Most rows from one group.
    #[arg(long, default_value_t = 500)]
    per_group: usize,
    /// Most rows anchored on one document.
    #[arg(long, default_value_t = 3)]
    per_anchor: usize,
    /// Most rows per task.
    #[arg(long, default_value_t = 12_000)]
    per_task: usize,
    /// Smallest score a row may carry.
    #[arg(long, default_value_t = 0.45)]
    min_score: f64,
    /// Smallest Hamming distance a new anchor needs, exact up to 7.
    #[arg(long, default_value_t = 4)]
    novelty_bits: u32,
    /// Shortest anchor, in words.
    #[arg(long, default_value_t = 5)]
    min_words: usize,
}

#[derive(clap::Args)]
struct BankArgs {
    /// The corpus whose `<name>.idx.json` sits beside it.
    #[arg(long)]
    out: String,
    /// Print this many of the best rows.
    #[arg(long, default_value_t = 5)]
    top: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Adapter {
    Shade,
    Jde,
    Static,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Operator {
    CurrentToPBest,
    Rand1,
    Best1,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum CrossoverKind {
    Binomial,
    Exponential,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Scalar {
    Tchebycheff,
    Pbi,
    WeightedSum,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Method {
    De,
    Moead,
    Both,
}

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    match Cli::parse().command {
        Command::List => list(),
        Command::De(args) => run_de(args),
        Command::Moead(args) => run_moead(args),
        Command::Forecast(args) => run_forecast(args),
        Command::Synth(args) => run_synth(args),
        Command::Bench(args) => run_bench(args),
        Command::Clean(args) => run_clean(args),
        Command::Pairs(args) => run_pairs(args),
        Command::Dataset(args) => run_dataset(args),
        Command::Detect(args) => run_detect(args),
        Command::Transformer(args) => run_transformer(args),
    }
}

fn default_dim(problem: &str) -> usize {
    CATALOGUE
        .iter()
        .find(|(name, _, _)| *name == problem)
        .map(|(_, dim, _)| *dim)
        .unwrap_or(30)
}

fn list() -> Result<(), Box<dyn Error>> {
    println!("{:<12} {:>4}  description", "problem", "dim");
    for (name, dim, description) in CATALOGUE {
        println!("{name:<12} {dim:>4}  {description}");
    }
    Ok(())
}

fn run_de(args: DeArgs) -> Result<(), Box<dyn Error>> {
    let dim = if args.dim == 0 {
        default_dim(&args.problem)
    } else {
        args.dim
    };
    let problem = Benchmark::named(&args.problem, dim)
        .ok_or_else(|| format!("unknown problem {} or bad dimension {dim}", args.problem))?;
    if problem.is_multi_objective() {
        return Err(format!(
            "{} has {} objectives, use the moead command",
            problem.name(),
            problem.objectives()
        )
        .into());
    }
    let config = DeConfig {
        population: args.population,
        budget: args.budget,
        strategy: match args.strategy {
            Operator::CurrentToPBest => Strategy::CurrentToPBest1,
            Operator::Rand1 => Strategy::Rand1Bin,
            Operator::Best1 => Strategy::Best1Bin,
        },
        adaptation: match args.adaptation {
            Adapter::Shade => Adaptation::Shade,
            Adapter::Jde => Adaptation::Jde,
            Adapter::Static => Adaptation::Static,
        },
        crossover: match args.crossover {
            CrossoverKind::Binomial => Crossover::Binomial,
            CrossoverKind::Exponential => Crossover::Exponential,
        },
        f: args.f,
        cr: args.cr,
        p_best: args.p_best,
        p_best_final: args.p_best_final,
        reduce_population: !args.no_reduce,
        stall_generations: args.stall,
        seed: args.seed,
        ..DeConfig::default()
    };
    let started = Instant::now();
    let out = de::minimize(&problem, &config);
    let elapsed = started.elapsed();
    let config_used = &config;
    if args.json {
        let report = serde_json::json!({
            "problem": problem.name(),
            "dim": problem.dim(),
            "seed": args.seed,
            "budget": config_used.resolved_budget(problem.dim()),
            "evaluations": out.evaluations,
            "generations": out.generations,
            "restarts": out.restarts,
            "best_objective": out.best_objective(),
            "feasible": out.best.is_feasible(),
            "violation": out.best.violation(),
            "x": out.best_x,
            "diversity": out.diversity,
            "population": out.population,
            "history": out.history,
            "seconds": elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("problem      {}", problem.name());
        println!("dimension    {}", problem.dim());
        println!("best         {:.10e}", out.best_objective());
        println!(
            "feasible     {}  (violation {:.3e})",
            out.best.is_feasible(),
            out.best.violation()
        );
        println!("evaluations  {}", out.evaluations);
        println!("generations  {}", out.generations);
        println!("restarts     {}", out.restarts);
        println!("population   {}", out.population);
        println!("diversity    {:.4}", out.diversity);
        println!("seconds      {:.3}", elapsed.as_secs_f64());
        let shown: Vec<String> = out
            .best_x
            .iter()
            .take(6)
            .map(|v| format!("{v:+.4}"))
            .collect();
        println!(
            "x            {}{}",
            shown.join(" "),
            if out.best_x.len() > 6 { " ..." } else { "" }
        );
    }
    Ok(())
}

fn run_moead(args: MoeadArgs) -> Result<(), Box<dyn Error>> {
    let dim = if args.dim == 0 {
        default_dim(&args.problem)
    } else {
        args.dim
    };
    let problem = Benchmark::named(&args.problem, dim)
        .ok_or_else(|| format!("unknown problem {} or bad dimension {dim}", args.problem))?;
    if !problem.is_multi_objective() {
        return Err(format!("{} has one objective, use the de command", problem.name()).into());
    }
    let scalar = match args.scalar {
        Scalar::Tchebycheff => Scalarization::Tchebycheff,
        Scalar::Pbi => Scalarization::Pbi { theta: args.theta },
        Scalar::WeightedSum => Scalarization::WeightedSum,
    };
    let config = MoeadConfig {
        population: args.population,
        divisions: args.divisions,
        neighbourhood: args.neighbourhood,
        delta: args.delta,
        replacements: args.replacements,
        update_probability: args.update_probability,
        update_delay: args.update_delay,
        ri_window: args.ri_window,
        scalar,
        f: args.f,
        cr: args.cr,
        budget: args.budget,
        seed: args.seed,
        ..MoeadConfig::default()
    };
    let started = Instant::now();
    let out = moead::minimize(&problem, &config);
    let elapsed = started.elapsed();
    let front = out.front();
    let kind = problem.kind();
    let reference = reference_front(kind, 500);
    let igd_value = if reference.is_empty() {
        f64::NAN
    } else {
        igd(&front, &reference)
    };
    let point = reference_point(kind);
    let hv = hypervolume_report(&front, &point);

    if let Some(path) = &args.out {
        let mut text = String::from("f1,f2");
        if problem.objectives() > 2 {
            text.push_str(",f3");
        }
        text.push('\n');
        for s in &out.solutions {
            let cells: Vec<String> = s.objectives.iter().map(|v| format!("{v:.10}")).collect();
            text.push_str(&cells.join(","));
            text.push('\n');
        }
        std::fs::write(path, text)?;
    }

    if args.json {
        let report = serde_json::json!({
            "problem": problem.name(),
            "dim": problem.dim(),
            "objectives": problem.objectives(),
            "seed": args.seed,
            "evaluations": out.evaluations,
            "generations": out.generations,
            "population": out.population,
            "points": out.solutions.len(),
            "igd": igd_value,
            "hypervolume": hv.value,
            "hypervolume_error": hv.error,
            "hypervolume_exact": hv.exact,
            "ideal": out.ideal,
            "front": out.solutions.iter().map(|s| &s.objectives).collect::<Vec<_>>(),
            "seconds": elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("problem      {}", problem.name());
        println!("objectives   {}", problem.objectives());
        println!("subproblems  {}", out.population);
        println!("generations  {}", out.generations);
        println!("evaluations  {}", out.evaluations);
        println!("points       {}", out.solutions.len());
        if igd_value.is_finite() {
            println!("igd          {igd_value:.6}");
        }
        println!(
            "hypervolume  {:.6}{}",
            hv.value,
            if hv.exact {
                String::new()
            } else {
                format!(" +/- {:.6}", hv.error)
            }
        );
        println!("seconds      {:.3}", elapsed.as_secs_f64());
        println!("\n{:>12} {:>12}", "first", "second");
        for s in out.solutions.iter().take(12) {
            println!(
                "{:>12.6} {:>12.6}",
                s.objectives[0],
                s.objectives.get(1).copied().unwrap_or(f64::NAN)
            );
        }
        if out.solutions.len() > 12 {
            println!("... {} more points", out.solutions.len() - 12);
        }
    }
    Ok(())
}

fn reference_point(kind: Kind) -> Vec<f64> {
    match kind {
        Kind::Dtlz1 => vec![0.6; 3],
        Kind::Dtlz2 => vec![1.1; 3],
        _ => vec![1.1; 2],
    }
}

fn run_forecast(args: ForecastArgs) -> Result<(), Box<dyn Error>> {
    let dataset = Dataset::load(&args.data, &args.target)?;
    if dataset.n_features() < 2 {
        return Err("need at least two feature columns".into());
    }
    let features = dataset.n_features();
    let rows = dataset.len();

    // A holdout carves the last fraction of rows off before the search runs, so
    // the subsets are chosen on the development prefix and then scored once on
    // rows the search never saw. Both halves are scaled with the development
    // statistics only.
    let fraction = args.holdout.clamp(0.0, 0.9);
    let cut = if fraction > 0.0 {
        (rows as f64 * (1.0 - fraction)).round() as usize
    } else {
        rows
    }
    .min(rows);
    if cut < args.folds + args.embargo + 2 {
        return Err(format!(
            "{cut} development rows is too few for {} folds with an embargo of {}",
            args.folds, args.embargo
        )
        .into());
    }

    let mut head = dataset.slice(0..cut);
    let stats = head.standardize();
    let mut scaled = dataset;
    scaled.apply_standardization(&stats);
    let holdout_fold = if cut < rows {
        Some(Fold {
            train_start: 0,
            train_end: cut,
            test_start: (cut + args.embargo).min(rows),
            test_end: rows,
        })
    } else {
        None
    };
    let shared = Arc::new(scaled.slice(0..cut));

    let problem = SelectionProblem::new(shared.clone(), args.folds, args.embargo, args.lambda);
    if problem.folds.is_empty() {
        return Err(format!("{cut} rows is too few for {} folds", args.folds).into());
    }
    let scaled = Arc::new(scaled);
    let all: Vec<usize> = (0..features).collect();
    let baseline = problem.report(&all);
    let started = Instant::now();

    let mut de_result = None;
    if matches!(args.method, Method::De | Method::Both) {
        let sparse = SparseAccuracy::new(problem.clone(), args.sparsity);
        let config = DeConfig {
            budget: args.budget,
            seed: args.seed,
            ..DeConfig::default()
        };
        let out = de::minimize(&sparse, &config);
        let columns = problem.decode(&out.best_x);
        let report = problem.report(&columns);
        de_result = Some((columns, report, out.best_objective()));
    }

    let mut front_result: Option<FrontRun> = None;
    if matches!(args.method, Method::Moead | Method::Both) {
        let config = MoeadConfig {
            population: 60,
            budget: args.budget,
            seed: args.seed,
            ..MoeadConfig::default()
        };
        let outcome = moead::minimize(&problem, &config);
        let front = outcome.front();
        let knee_index = knee(&front).unwrap_or(0);
        let best_index = outcome
            .solutions
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.objectives[0].total_cmp(&b.1.objectives[0]))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let report_at = |index: usize| {
            let columns = outcome
                .solutions
                .get(index)
                .map(|s| problem.decode(&s.x))
                .unwrap_or_default();
            problem.report(&columns)
        };
        let knee_report = report_at(knee_index);
        let best_report = report_at(best_index);
        front_result = Some(FrontRun {
            outcome,
            knee_index,
            knee_report,
            best_index,
            best_report,
        });
    }
    let elapsed = started.elapsed();

    // Score the chosen subsets once on the held-out rows. This is the only
    // number in the run that the search had no hand in.
    let holdout: Vec<(String, Report)> = match &holdout_fold {
        Some(fold) if fold.test_len() > 0 => {
            let mut scored = vec![(
                "all".to_string(),
                score_over(&scaled, std::slice::from_ref(fold), &all, args.lambda),
            )];
            if let Some((columns, _, _)) = &de_result {
                scored.push((
                    "de".to_string(),
                    score_over(&scaled, std::slice::from_ref(fold), columns, args.lambda),
                ));
            }
            if let Some(run) = &front_result {
                for (label, columns) in [
                    ("moead-knee", &run.knee_report.columns),
                    ("moead-best", &run.best_report.columns),
                ] {
                    scored.push((
                        label.to_string(),
                        score_over(&scaled, std::slice::from_ref(fold), columns, args.lambda),
                    ));
                }
            }
            scored
        }
        _ => Vec::new(),
    };

    if let Some(path) = &args.out {
        if let Some(run) = &front_result {
            let mut text = String::from("mse,feature_fraction\n");
            for f in run.outcome.front() {
                text.push_str(&format!("{:.10},{:.10}\n", f[0], f[1]));
            }
            std::fs::write(path, text)?;
        }
    }

    if args.json {
        let mut models = serde_json::json!({
            "all": {
                "features": baseline.columns.len(),
                "mse": baseline.mse,
                "r2": baseline.r2,
                "ic": baseline.ic,
                "skill": baseline.skill,
            }
        });
        if let Some((columns, report, objective)) = &de_result {
            models["de"] = serde_json::json!({
                "columns": columns,
                "names": report.names,
                "objective": objective,
                "mse": report.mse,
                "r2": report.r2,
                "ic": report.ic,
                "skill": report.skill,
            });
        }
        if let Some(run) = &front_result {
            models["moead"] = serde_json::json!({
                "points": run.outcome.solutions.len(),
                "knee": {
                    "index": run.knee_index,
                    "columns": run.knee_report.columns,
                    "names": run.knee_report.names,
                    "mse": run.knee_report.mse,
                    "r2": run.knee_report.r2,
                    "ic": run.knee_report.ic,
                    "skill": run.knee_report.skill,
                },
                "best": {
                    "index": run.best_index,
                    "columns": run.best_report.columns,
                    "names": run.best_report.names,
                    "mse": run.best_report.mse,
                    "r2": run.best_report.r2,
                    "ic": run.best_report.ic,
                    "skill": run.best_report.skill,
                },
                "front": run.outcome.front(),
            });
        }
        let holdout_json: serde_json::Value = holdout
            .iter()
            .map(|(label, r)| {
                (
                    label.clone(),
                    serde_json::json!({
                        "features": r.columns.len(),
                        "names": r.names,
                        "mse": r.mse,
                        "r2": r.r2,
                        "ic": r.ic,
                        "skill": r.skill,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into();
        let report = serde_json::json!({
            "rows": rows,
            "features": features,
            "folds": problem.folds.len(),
            "holdout_rows": holdout_fold.as_ref().map(|f| f.test_len()).unwrap_or(0),
            "skipped_rows": shared.skipped,
            "models": models,
            "holdout": holdout_json,
            "seconds": elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "rows        {rows}   features {features}   folds {}",
            problem.folds.len()
        );
        println!(
            "{:<10} {:>6} {:>12} {:>8} {:>7} {:>7}",
            "model", "feat", "mse", "r2", "ic", "skill"
        );
        println!(
            "{:<10} {:>6} {:>12.6} {:>8.4} {:>7.3} {:>6.1}%",
            "all",
            baseline.columns.len(),
            baseline.mse,
            baseline.r2,
            baseline.ic,
            baseline.skill * 100.0
        );
        if let Some((columns, report, _)) = &de_result {
            println!(
                "{:<10} {:>6} {:>12.6} {:>8.4} {:>7.3} {:>6.1}%",
                "de",
                columns.len(),
                report.mse,
                report.r2,
                report.ic,
                report.skill * 100.0
            );
            println!("  keeps: {}", report.names.join(", "));
        }
        if let Some(run) = &front_result {
            for (label, report) in [
                ("moead-knee", &run.knee_report),
                ("moead-best", &run.best_report),
            ] {
                println!(
                    "{:<10} {:>6} {:>12.6} {:>8.4} {:>7.3} {:>6.1}%",
                    label,
                    report.columns.len(),
                    report.mse,
                    report.r2,
                    report.ic,
                    report.skill * 100.0
                );
                println!("  keeps: {}", report.names.join(", "));
            }
            println!(
                "  front: {} points, knee {}, best {}",
                run.outcome.solutions.len(),
                run.knee_index,
                run.best_index
            );
        }
        if let Some(fold) = &holdout_fold {
            if !holdout.is_empty() {
                println!(
                    "\nholdout     rows {}..{}   trained on 0..{}   never seen by the search",
                    fold.test_start, fold.test_end, fold.train_end
                );
                println!(
                    "{:<10} {:>6} {:>12} {:>8} {:>7} {:>7}",
                    "model", "feat", "mse", "r2", "ic", "skill"
                );
                for (label, report) in &holdout {
                    println!(
                        "{:<10} {:>6} {:>12.6} {:>8.4} {:>7.3} {:>6.1}%",
                        label,
                        report.columns.len(),
                        report.mse,
                        report.r2,
                        report.ic,
                        report.skill * 100.0
                    );
                }
            }
        }
        println!("seconds     {:.2}", elapsed.as_secs_f64());
    }
    Ok(())
}

/// The MOEA/D run, with both ends of the trade-off scored out of sample: the
/// knee for a sparse model and the best error on the curve.
struct FrontRun {
    outcome: MoeadOutcome,
    knee_index: usize,
    knee_report: Report,
    best_index: usize,
    best_report: Report,
}

fn run_synth(args: SynthArgs) -> Result<(), Box<dyn Error>> {
    let dataset = Dataset::synthetic(args.rows, args.features, args.informative, args.seed);
    let wants_jsonl = args.out.extension().and_then(|e| e.to_str()) == Some("jsonl");
    let body = if wants_jsonl {
        dataset.to_jsonl()
    } else {
        dataset.to_csv()
    };
    std::fs::write(&args.out, body)?;
    println!(
        "wrote {} rows and {} features to {}",
        dataset.len(),
        dataset.n_features(),
        args.out.display()
    );
    Ok(())
}

fn run_bench(args: BenchArgs) -> Result<(), Box<dyn Error>> {
    let runs = args.runs.max(1);
    let mut json = Vec::new();

    println!("# differential evolution, SHADE with population reduction");
    println!(
        "{:<10} {:>4} {:>10} {:>11} {:>11} {:>11} {:>8}",
        "problem", "dim", "budget", "best", "median", "mean", "seconds"
    );
    for (name, dim, _) in CATALOGUE {
        let problem = match Benchmark::named(name, *dim) {
            Some(p) if !p.is_multi_objective() => p,
            _ => continue,
        };
        let mut values = Vec::new();
        let started = Instant::now();
        for run in 0..runs {
            let config = DeConfig {
                budget: args.budget,
                p_best: args.p_best,
                p_best_final: args.p_best_final,
                stall_generations: args.stall,
                seed: run + 1,
                ..DeConfig::default()
            };
            values.push(de::minimize(&problem, &config).best_objective());
        }
        let elapsed = started.elapsed().as_secs_f64() / runs as f64;
        values.sort_by(|a, b| a.total_cmp(b));
        let median = values[values.len() / 2];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let budget = if args.budget == 0 {
            de::DEFAULT_BUDGET_PER_DIM * dim
        } else {
            args.budget
        };
        println!(
            "{:<10} {:>4} {:>10} {:>11.3e} {:>11.3e} {:>11.3e} {:>8.3}",
            name, dim, budget, values[0], median, mean, elapsed
        );
        json.push(serde_json::json!({
            "suite": "de",
            "problem": name,
            "dim": dim,
            "budget": budget,
            "best": values[0],
            "median": median,
            "mean": mean,
            "seconds": elapsed,
        }));
    }

    let moead_budget = if args.budget == 0 {
        moead::DEFAULT_BUDGET
    } else {
        args.budget
    };
    let update = args
        .update_probability
        .unwrap_or(MoeadConfig::default().update_probability);
    let allocation = if args.ri_window > 0 {
        format!("relative improvement, window {}", args.ri_window)
    } else {
        "uniform random subset".to_string()
    };
    println!(
        "\n# MOEA/D, Tchebycheff, 100 subproblems, update probability {update}, \
         {allocation}, {moead_budget} evaluations"
    );
    println!(
        "{:<10} {:>10} {:>11} {:>11} {:>8} {:>8}",
        "problem", "igd", "hypervolume", "points", "seconds", "runs"
    );
    for (name, dim, _) in CATALOGUE {
        let problem = match Benchmark::named(name, *dim) {
            Some(p) if p.is_multi_objective() => p,
            _ => continue,
        };
        let kind = problem.kind();
        let reference = reference_front(kind, 500);
        let point = reference_point(kind);
        let mut igds = Vec::new();
        let mut hvs = Vec::new();
        let mut points = 0usize;
        let started = Instant::now();
        for run in 0..runs {
            let mut config = MoeadConfig {
                budget: moead_budget,
                seed: run + 1,
                ..MoeadConfig::default()
            };
            if let Some(probability) = args.update_probability {
                config.update_probability = probability;
            }
            config.ri_window = args.ri_window;
            let out = moead::minimize(&problem, &config);
            let front = out.front();
            igds.push(igd(&front, &reference));
            hvs.push(hypervolume_report(&front, &point).value);
            points = out.solutions.len();
        }
        let elapsed = started.elapsed().as_secs_f64() / runs as f64;
        let igd_mean = igds.iter().sum::<f64>() / runs as f64;
        let hv_mean = hvs.iter().sum::<f64>() / runs as f64;
        let igd_sd = (igds
            .iter()
            .map(|v| (v - igd_mean) * (v - igd_mean))
            .sum::<f64>()
            / runs as f64)
            .sqrt();
        println!(
            "{:<10} {:>10.5} {:>11.5} {:>11} {:>8.3} {:>8}",
            name, igd_mean, hv_mean, points, elapsed, runs
        );
        json.push(serde_json::json!({
            "suite": "moead",
            "problem": name,
            "dim": dim,
            "igd": igd_mean,
            "igd_sd": igd_sd,
            "hypervolume": hv_mean,
            "points": points,
            "seconds": elapsed,
            "runs": runs,
        }));
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&json)?);
    }
    Ok(())
}

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

fn run_clean(args: CleanArgs) -> Result<(), Box<dyn Error>> {
    let pipeline = xcm_mmm::text::run::Pipeline {
        dedupe: if args.dedupe > 0.0 {
            Some(args.dedupe)
        } else {
            None
        },
        budget: args.budget,
        floor: args.floor,
        theta: args.theta,
        group_cap: args.group_cap,
        target: args.target.clone(),
        rules: xcm_mmm::text::quality::Rules {
            min_words: args.min_words,
            max_words: args.max_words,
            ..Default::default()
        },
        ..Default::default()
    };
    let s = xcm_mmm::text::run::run(&args.input, &args.out, &pipeline)?;
    let loss = if s.chars_in == 0 {
        0.0
    } else {
        100.0 * (1.0 - s.chars_out as f64 / s.chars_in as f64)
    };
    println!(
        "read      {} rows ({} malformed)",
        commas(s.read),
        commas(s.malformed)
    );
    println!(
        "normalise {} chars of text, {:.1} percent removed with the markup",
        commas(s.chars_in),
        loss
    );
    println!(
        "quality   {} kept, {} dropped by rules",
        commas(s.kept + s.dropped_duplicate),
        commas(s.dropped_rules)
    );
    if !s.flags.is_empty() {
        let flags: Vec<String> = s
            .flags
            .iter()
            .map(|(k, v)| format!("{k} {}", commas(*v)))
            .collect();
        println!("flags     {}", flags.join(", "));
    }
    if pipeline.dedupe.is_some() {
        println!(
            "dedupe    {} merged, largest cluster {}, {} survive",
            commas(s.duplicates_merged),
            s.largest_cluster,
            commas(s.kept)
        );
    }
    if s.target_aware {
        println!(
            "target    {} rows in the target set, {} drawn by weight, mean quality {:.3}",
            commas(s.target_rows),
            commas(s.selected),
            s.mean_quality
        );
    }
    if !s.target_aware {
        println!(
            "select    {} kept, {} redundant, {} over cap ({} filled after relaxing it), {:.0} percent feature coverage, mean quality {:.3}",
            commas(s.selected),
            commas(s.redundant),
            commas(s.over_cap),
            commas(s.relaxed),
            s.feature_coverage * 100.0,
            s.mean_quality
        );
    }
    println!(
        "out       {} rows, mean {:.1} words",
        commas(s.selected),
        s.mean_words
    );
    Ok(())
}

fn parse_weights(spec: &str) -> Result<Vec<(xcm_mmm::pairs::Task, f64)>, Box<dyn Error>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, value) = part
            .split_once('=')
            .ok_or_else(|| format!("weight {part:?} is not name=value"))?;
        let task: xcm_mmm::pairs::Task = name.trim().parse()?;
        let weight: f64 = value.trim().parse()?;
        out.push((task, weight));
    }
    Ok(out)
}

fn run_dataset(args: DatasetArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        DatasetCommand::Build(a) => {
            let cfg = xcm_mmm::dataset::BuildConfig {
                budget: a.budget,
                seed: a.seed,
                cap_bytes: a.cap_mb.saturating_mul(1 << 20),
                name: a.name,
            };
            let m = xcm_mmm::dataset::build(&a.input, &a.out, &cfg)?;
            println!(
                "source    {} rows, {} bytes from {}",
                commas(m.source.rows),
                commas(m.source.bytes as usize),
                m.source.first_day
            );
            for split in xcm_mmm::dataset::SPLITS {
                if let Some(c) = m.splits.get(split) {
                    println!(
                        "  {split:<6} {:>7} rows  {:>9} bytes  {} to {}",
                        commas(c.rows),
                        commas(c.bytes as usize),
                        c.first_day,
                        c.last_day
                    );
                }
            }
            let mix: Vec<String> = m
                .tasks
                .iter()
                .map(|(task, n)| format!("{task} {}", commas(*n)))
                .collect();
            println!("tasks     {}", mix.join(", "));
            println!(
                "shipped   {} of {} source rows, {:.1} percent",
                commas(m.rows),
                commas(m.source.rows),
                m.rows as f64 * 100.0 / m.source.rows as f64
            );
            println!("year span {} to {}", m.source.first_day, m.source.last_day);
            for line in &m.checks {
                println!("check     {line}");
            }
            println!("out       {}", a.out);
            if m.checks.iter().any(|c| c.contains("FAILED")) {
                return Err("a shipped check failed; the folder is not publishable".into());
            }
            Ok(())
        }
        DatasetCommand::Check(a) => {
            let faults = xcm_mmm::dataset::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     {} agrees with its manifest", a.dir);
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

fn run_detect(args: DetectArgs) -> Result<(), Box<dyn Error>> {
    use xcm_mmm::detect::{self, Artifact, Axis, Detector, TrainConfig};

    match args.command {
        DetectCommand::Train(a) => {
            let mut cfg = TrainConfig {
                name: a.name,
                folds: a.folds.max(2),
                min_rows: a.min_rows.max(1),
                ..Default::default()
            };
            cfg.lexicon.top_terms = a.top_terms;
            let artifact = detect::build(&a.bank, &a.ads, &a.cues, &a.out, &cfg)?;
            println!(
                "source    {} rows, {} ads, {} cue axes",
                commas(artifact.source.bank_rows),
                commas(artifact.source.ads_rows),
                artifact.source.cues_axes
            );
            for (name, axis) in &artifact.axes {
                match axis {
                    Axis::Lexicon(lex) => println!(
                        "  {:<12} lexicon  {:>4} terms  balanced {:.3}  majority {:.3}  shuffled {:.3}",
                        name,
                        lex.lexicon.terms.len(),
                        lex.metrics.balanced,
                        lex.metrics.majority,
                        lex.metrics.shuffled.unwrap_or(f64::NAN)
                    ),
                    Axis::Index(ind) => println!(
                        "  {:<12} index    prevalence {:.3}  halves agree {:.2}  checks {}/{}",
                        name,
                        ind.audit.prevalence,
                        ind.audit.halves_agree,
                        ind.checks.iter().filter(|c| c.passed).count(),
                        ind.checks.len()
                    ),
                }
            }
            println!("wrote     {}/model.json, manifest.json, card.md", a.out);
            Ok(())
        }
        DetectCommand::Score(a) => {
            let detector = Detector::load(&a.model)?;
            if let Some(input) = a.input {
                let out = a.out.ok_or("--input needs --out")?;
                let rows = detector.score_rows(&input, &out)?;
                println!("scored    {} rows into {out}", commas(rows));
                return Ok(());
            }
            if a.text.is_empty() {
                return Err("give --text or --input".into());
            }
            for text in &a.text {
                println!("text      {}", text.chars().take(72).collect::<String>());
                for reading in detector.score(text) {
                    let hits = reading
                        .hits
                        .iter()
                        .take(a.hits)
                        .cloned()
                        .collect::<Vec<String>>()
                        .join(", ");
                    println!(
                        "  {:<12} {:<7} {:.3}  {}",
                        reading.axis,
                        reading.level,
                        reading.score,
                        if hits.is_empty() {
                            "-".to_string()
                        } else {
                            hits
                        }
                    );
                }
            }
            Ok(())
        }
        DetectCommand::Info(a) => {
            let detector = Detector::load(&a.model)?;
            let artifact: &Artifact = detector.artifact();
            println!("name      {} v{}", artifact.name, artifact.version);
            println!("built     {}", artifact.built_at);
            println!(
                "source    {} {} rows  {} ads  {} cue axes",
                artifact.source.bank,
                commas(artifact.source.bank_rows),
                commas(artifact.source.ads_rows),
                artifact.source.cues_axes
            );
            println!(
                "digests   bank {}  ads {}  cues {}",
                artifact.source.bank_fnv64, artifact.source.ads_fnv64, artifact.source.cues_fnv64
            );
            println!("axis         method   tag                  size    held out");
            for (name, axis) in &artifact.axes {
                match axis {
                    Axis::Lexicon(lex) => println!(
                        "  {:<12} lexicon  {:<20} {:>4}    balanced {:.3} vs majority {:.3}",
                        name,
                        "-",
                        lex.lexicon.terms.len(),
                        lex.metrics.balanced,
                        lex.metrics.majority
                    ),
                    Axis::Index(ind) => println!(
                        "  {:<12} index    {:<20} {:>4}    prevalence {:.3}",
                        name,
                        ind.tag,
                        ind.cues.len(),
                        ind.audit.prevalence
                    ),
                }
            }
            Ok(())
        }
        DetectCommand::Check(a) => {
            let faults = detect::check(&a.dir)?;
            if faults.is_empty() {
                println!("clean     {} rebuilds from its own inputs", a.dir);
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

fn run_transformer(args: TransformerArgs) -> Result<(), Box<dyn Error>> {
    use xcm_mmm::transformer::run as transformer;
    match args.command {
        TransformerCommand::Train(a) => {
            let weights = transformer::parse_task_weights(&a.task_weights)?;
            let epochs = match (a.epochs, &a.resume) {
                (Some(value), _) => value,
                (None, Some(path)) => xcm_mmm::transformer::checkpoint::load(path)?.run.epochs,
                (None, None) => 3,
            };
            let config = xcm_mmm::transformer::train::TrainConfig {
                pairs: a.pairs,
                out: a.out,
                vocab: a.vocab,
                max_len: a.max_len,
                dim: a.dim,
                heads: a.heads,
                ff: a.ff,
                layers: a.layers,
                epochs,
                batch: a.batch,
                lr: a.lr,
                weight_decay: a.weight_decay,
                warmup: a.warmup,
                clip: a.clip,
                dropout: a.dropout,
                seed: a.seed,
                limit: a.limit,
                patience: a.patience,
                min_count: a.min_count,
                task_weights: weights,
                resume: a.resume,
                stop_after: a.stop_after,
                checkpoint: a.checkpoint,
                checkpoint_every: a.checkpoint_every,
            };
            transformer::train_model(&config)?;
        }
        TransformerCommand::Eval(a) => {
            transformer::eval_model(&a.model, &a.pairs, a.limit)?;
        }
        TransformerCommand::Info(a) => {
            transformer::info_model(&a.model)?;
        }
        TransformerCommand::Encode(a) => {
            transformer::encode_text(&a.model, &a.text)?;
        }
        TransformerCommand::Score(a) => {
            let task = match &a.task {
                Some(name) => Some(name.parse::<xcm_mmm::pairs::Task>()?),
                None => None,
            };
            transformer::score_rows(&a.model, &a.rows, &a.out, task)?;
        }
    }
    Ok(())
}

fn run_pairs(args: PairsArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        PairCommand::Mine(a) => {
            let cfg = xcm_mmm::pairs::mine::Config {
                min_margin: a.min_margin,
                max_similarity: a.max_similarity,
                neighbours: a.neighbours,
                max_per_task: a.max_per_task,
                max_chars: a.max_chars,
                seed: a.seed,
                ..Default::default()
            };
            let (report, written) = xcm_mmm::pairs::run::run_mine(&a.rows, &a.out, &cfg)?;
            println!(
                "rows      {} over {} days and {} groups",
                commas(report.rows),
                commas(report.days),
                commas(report.groups)
            );
            println!(
                "pairs     {} rank, {} preference, {} contrast, {} written",
                commas(report.rank),
                commas(report.preference),
                commas(report.contrast),
                commas(written)
            );
            println!(
                "dropped   {} margin, {} trivial, {} short, {} duplicate",
                commas(report.dropped_margin),
                commas(report.dropped_trivial),
                commas(report.dropped_short),
                commas(report.dropped_duplicate)
            );
            println!("out       {}", a.out);
            Ok(())
        }
        PairCommand::Pack(a) => {
            let plan = xcm_mmm::pairs::pack::Plan {
                total: a.total,
                weights: parse_weights(&a.weights)?,
                max_similarity: a.max_similarity,
                seed: a.seed,
            };
            let (report, manifest) = xcm_mmm::pairs::run::run_pack(&a.input, &a.out, &plan)?;
            println!(
                "read      {} rows from {} files ({} malformed, {} duplicate, {} trivial)",
                commas(report.read),
                commas(report.files),
                commas(report.malformed),
                commas(report.duplicate_pairs),
                commas(report.trivial)
            );
            let mix: Vec<String> = report
                .by_task
                .iter()
                .map(|(task, n)| format!("{task} {}", commas(*n)))
                .collect();
            println!("tasks     {}", mix.join(", "));
            let requested: Vec<String> = report
                .realised
                .iter()
                .filter(|(task, _)| plan.weight_of(*task) > 0.0)
                .map(|(task, share)| format!("{task} {:.2}", share))
                .collect();
            println!("mix       {}", requested.join(", "));
            let splits: Vec<String> = report
                .by_split
                .iter()
                .map(|(split, n)| format!("{split} {}", commas(*n)))
                .collect();
            println!("splits    {}", splits.join(", "));
            println!("out       {} rows to {}", commas(report.kept), a.out);
            match a.manifest {
                Some(path) => std::fs::write(&path, format!("{manifest}\n"))?,
                None => println!("manifest  {manifest}"),
            }
            Ok(())
        }
        PairCommand::Audit(a) => {
            let cfg = xcm_mmm::pairs::audit::Config {
                max_similarity: a.max_similarity,
                contamination: a.contamination,
                min_margin: a.min_margin,
            };
            let report = xcm_mmm::pairs::run::run_audit(&a.input, a.eval.as_deref(), &cfg)?;
            println!("pairs     {}", commas(report.pairs));
            for (task, n) in &report.by_task {
                println!("  {task:<10} {}", commas(*n));
            }
            for (split, n) in &report.by_split {
                println!("  split {split:<8} {}", commas(*n));
            }
            println!(
                "margins   mean {:.4}, smallest {:.4}, {} below the floor",
                report.mean_margin,
                report.min_margin,
                commas(report.thin_margin)
            );
            println!(
                "difficulty mean {:.3}, side a longer in {:.0} percent of pairs",
                report.mean_difficulty,
                report.longer_side_a * 100.0
            );
            println!(
                "words     mean {:.1} on side a, {:.1} on side b",
                report.mean_words_a, report.mean_words_b
            );
            if report.contaminated > 0 || report.verbatim_overlap > 0 {
                println!(
                    "overlap   {} pairs share {:.0} percent of an evaluation text, {} share a 13 word run",
                    commas(report.contaminated),
                    cfg.contamination * 100.0,
                    commas(report.verbatim_overlap)
                );
            }
            for line in report.summary() {
                println!("check     {line}");
            }
            if report.ok() {
                Ok(())
            } else {
                Err(format!(
                    "audit found {} leaked documents, {} leaked texts, {} identical pairs, {} contaminated pairs",
                    report.leaked_docs,
                    report.leaked_texts,
                    report.identical_sides,
                    report.contaminated
                )
                .into())
            }
        }
        PairCommand::Retain(a) => {
            let policy = xcm_mmm::pairs::retain::Policy {
                budget: a.budget,
                per_day: a.per_day,
                per_group: a.per_group,
                per_anchor: a.per_anchor,
                per_task: a.per_task,
                min_score: a.min_score,
                novelty_bits: a.novelty_bits,
                min_words: a.min_words,
                ..Default::default()
            };
            let report = xcm_mmm::pairs::retain::update(&a.input, &a.out, policy)?;
            println!(
                "read      {} candidates ({} malformed), {} already banked",
                commas(report.candidates),
                commas(report.malformed),
                commas(report.known)
            );
            println!(
                "gated     {} below floor, {} short, {} duplicate day, {} group, {} anchor, {} task",
                commas(report.below_floor),
                commas(report.short),
                commas(report.rejected_day),
                commas(report.rejected_group),
                commas(report.rejected_anchor),
                commas(report.rejected_task)
            );
            println!(
                "admitted  {} new, {} near duplicate, {} evicted",
                commas(report.admitted),
                commas(report.rejected_novelty),
                commas(report.evicted)
            );
            let tasks: Vec<String> = report
                .by_task
                .iter()
                .map(|(task, n)| format!("{task} {}", commas(*n)))
                .collect();
            println!(
                "bank      {} rows over {} days and {} groups, mean {:.3}, top {:.3}",
                commas(report.bank_rows),
                commas(report.by_day.len()),
                commas(report.groups),
                report.mean_score,
                report.top_score
            );
            println!("tasks     {}", tasks.join(", "));
            println!("out       {}", a.out);
            println!("index     {}", xcm_mmm::pairs::retain::index_path(&a.out));
            println!("ledger    {}", xcm_mmm::pairs::retain::ledger_path(&a.out));
            Ok(())
        }
        PairCommand::Bank(a) => {
            let path = xcm_mmm::pairs::retain::index_path(&a.out);
            let bank = xcm_mmm::pairs::retain::Bank::open(&path)?
                .ok_or_else(|| format!("no index at {path}, run pairs retain first"))?;
            let report = bank.summary();
            println!(
                "bank      {} rows over {} days and {} groups, mean {:.3}, top {:.3}",
                commas(report.bank_rows),
                commas(report.by_day.len()),
                commas(report.groups),
                report.mean_score,
                report.top_score
            );
            println!(
                "policy    budget {}, day {}, group {}, anchor {}, task {}, floor {:.2}, novelty {}",
                commas(bank.policy.budget),
                commas(bank.policy.per_day),
                commas(bank.policy.per_group),
                commas(bank.policy.per_anchor),
                commas(bank.policy.per_task),
                bank.policy.min_score,
                bank.policy.novelty_bits
            );
            let tasks: Vec<String> = report
                .by_task
                .iter()
                .map(|(task, n)| format!("{task} {}", commas(*n)))
                .collect();
            println!("tasks     {}", tasks.join(", "));
            let days: Vec<String> = report
                .by_day
                .iter()
                .take(7)
                .map(|(day, n)| format!("{day} {}", commas(*n)))
                .collect();
            println!("days      {}", days.join(", "));
            for entry in bank.top(a.top) {
                println!(
                    "row       {:.3} {} {} {}",
                    entry.score,
                    entry.task,
                    if entry.group.is_empty() {
                        "-"
                    } else {
                        &entry.group
                    },
                    entry.id
                );
            }
            Ok(())
        }
    }
}
