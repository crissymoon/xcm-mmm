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
    about = "Differential evolution and MOEA/D, with a walk-forward forecasting front end",
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
    /// Print JSON instead of a table.
    #[arg(long)]
    json: bool,
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
    println!(
        "\n# MOEA/D, Tchebycheff, 100 subproblems, {} evaluations",
        moead_budget
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
            let config = MoeadConfig {
                budget: moead_budget,
                seed: run + 1,
                ..MoeadConfig::default()
            };
            let out = moead::minimize(&problem, &config);
            let front = out.front();
            igds.push(igd(&front, &reference));
            hvs.push(hypervolume_report(&front, &point).value);
            points = out.solutions.len();
        }
        let elapsed = started.elapsed().as_secs_f64() / runs as f64;
        let igd_mean = igds.iter().sum::<f64>() / runs as f64;
        let hv_mean = hvs.iter().sum::<f64>() / runs as f64;
        println!(
            "{:<10} {:>10.5} {:>11.5} {:>11} {:>8.3} {:>8}",
            name, igd_mean, hv_mean, points, elapsed, runs
        );
        json.push(serde_json::json!({
            "suite": "moead",
            "problem": name,
            "dim": dim,
            "igd": igd_mean,
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
