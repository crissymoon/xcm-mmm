# Install

[Back to the project](../../README.md)

## Requirements

- Rust 1.75 or newer, through rustup or a distribution package.
- A C toolchain is not needed; the crate is pure Rust and has no build script.
- Four dependencies, all from crates.io: rayon, serde, serde_json and clap.

Check the toolchain:

```bash
rustc --version
cargo --version
```

## Build from source

```bash
git clone https://github.com/crissymoon/xcm-mmm.git
cd xcm-mmm
cargo build --release
./target/release/xcm-mmm list
```

The build takes a few seconds after the dependencies are fetched. The binary is `target/release/xcm-mmm`.

## Install on PATH

```bash
cargo install --path .
```

That puts the binary in `~/.cargo/bin`, which rustup adds to PATH. Then:

```bash
xcm-mmm list
xcm-mmm de --problem sphere --dim 30
```

## Run without installing

```bash
cargo run --release -- de --problem rosenbrock --dim 20
cargo run --release --example forecast
```

`cargo run` rebuilds when a source file changes, which is convenient during development and unnecessary once the binary exists.

## Use as a dependency

The crate is not published on crates.io yet. Point at the repository or a local checkout:

```toml
[dependencies]
xcm-mmm = { git = "https://github.com/crissymoon/xcm-mmm", tag = "v0.1.0" }
```

or, while working on both sides:

```toml
[dependencies]
xcm-mmm = { path = "../xcm-mmm" }
```

The library target is `xcm_mmm` (the hyphen becomes an underscore in Rust).

```rust
use xcm_mmm::de::{minimize, DeConfig};
use xcm_mmm::benchmarks::Benchmark;

let problem = Benchmark::named("sphere", 30).unwrap();
let out = minimize(&problem, &DeConfig::seeded(1));
println!("{:.3e}", out.best_objective());
```

## Platforms

There is no hosted CI run: `scripts/check.sh` runs the whole suite on the machine in front of you. The crate has no platform specific code and no feature flags. The only environment variable that changes behaviour is `RAYON_NUM_THREADS`, which sets the evaluation pool size; it never changes the result.

## Offline builds

With a populated cargo registry cache:

```bash
cargo build --release --offline
```

The four dependencies are already vendored in the cache after the first online build. No network access is needed at run time.

## Uninstall

```bash
cargo uninstall xcm-mmm
```

and remove the checkout. There is no configuration file, no cache and no state outside the directory you run it in.

## Troubleshooting

- **`linker not found` on Linux.** Install a C linker such as `build-essential` or `gcc`; Rust needs one to link the binary even for a pure Rust crate.
- **`error: no matching package named ...`** means the registry is unreachable. Use `--offline` if the crates were fetched before, or restore network access.
- **Old results after a rebuild.** Check that you are running `target/release/xcm-mmm` rather than a stale copy on PATH, and that the seed and budget match.
- **A forecast refuses the file.** The error names the problem: a missing target column, fewer than two numeric feature columns, or a panel too short for the requested folds. The CSV rules are in [Forecasting](FORECASTING.md).
