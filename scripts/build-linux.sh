#!/usr/bin/env bash
# Build a static Linux x86-64 binary of xcm-mmm from macOS or Linux.
#
# The musl target ships its own C runtime as self-contained objects, so the
# only missing piece on a macOS host is a linker that understands ELF. Cargo
# defaults to `cc`, which is Apple clang and rejects every GNU flag, so the
# bundled rust-lld is named explicitly. Nothing else has to be installed:
# `rustup target add x86_64-unknown-linux-musl` supplies libc.a and crt1.o.
#
#   scripts/build-linux.sh            # writes target/x86_64-unknown-linux-musl/release/xcm-mmm
#   OUT=dist/kaggle scripts/build-linux.sh
#
# A static-pie ELF runs on any glibc or musl distribution, which is the point:
# the Kaggle image upgrades its libc between seasons and a dynamically linked
# binary would break without warning.
#
# Two reproducibility settings are part of the build, not an afterthought:
# SOURCE_DATE_EPOCH pins the timestamp the binary reports, and the path remap
# keeps this machine's home directory out of the artefact. Build the same commit
# twice and the two checksums agree; that is what makes the hash in the dataset
# card mean something.
set -euo pipefail

TARGET="${TARGET:-x86_64-unknown-linux-musl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! rustup target list --installed | grep -qx "$TARGET"; then
  echo "adding target $TARGET" >&2
  rustup target add "$TARGET"
fi

# rust-lld lives inside the active toolchain, keyed by the host triple.
TOOLCHAIN="$(rustup show active-toolchain | awk '{print $1}')"
LLD="$(find "${RUSTUP_HOME:-$HOME/.rustup}/toolchains/$TOOLCHAIN" -name rust-lld -type f -print -quit)"
if [ -z "$LLD" ]; then
  echo "rust-lld not found inside $TOOLCHAIN; cannot link $TARGET from this host" >&2
  exit 1
fi

BIN="target/$TARGET/release/xcm-mmm"

# The commit timestamp is the reproducible default; an explicit SOURCE_DATE_EPOCH
# in the environment wins, which is what a CI rebuild from a tarball needs.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct 2>/dev/null || echo 0)}"

# --build-id gives the ELF a GNU note naming the link inputs, so a crash report
# or a debug session can be matched to one artefact. The remap is the same trick
# Debian uses on Rust packages: strip is not a guarantee, and a path leak is the
# usual reason two builds of one commit differ.
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$LLD" \
RUSTFLAGS="-C linker-flavor=ld.lld -C link-self-contained=yes -C strip=symbols \
  -C link-arg=--build-id=sha1 --remap-path-prefix=$ROOT=. \
  --remap-path-prefix=$HOME/.cargo=/cargo" \
  cargo build --release --locked --target "$TARGET"

if [ -n "${OUT:-}" ]; then
  mkdir -p "$OUT"
  cp "$BIN" "$OUT/xcm-mmm"
  BIN="$OUT/xcm-mmm"
fi

# `file` is the cheapest proof the artefact is what the dataset card claims:
# a dynamic ELF would say "dynamically linked", and the notebook's glibc report
# would then be a real risk instead of a note.
file "$BIN"
ls -l "$BIN"
DIGEST="$(shasum -a 256 "$BIN" | awk '{print $1}')"
echo "sha256    $DIGEST"

# The card and the notebook both quote a hash and a commit. Writing them beside
# the artefact means neither has to be retyped, and the notebook can check that
# the binary it just chmodded is the one this file describes rather than one
# that merely shares a name.
if [ -n "${OUT:-}" ]; then
  DIRTY=false
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then DIRTY=true; fi
  COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  cat > "$OUT/provenance.json" <<JSON
{
  "binary": "xcm-mmm",
  "version": "$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)",
  "sha256": "$DIGEST",
  "bytes": $(wc -c < "$BIN" | tr -d ' '),
  "target": "$TARGET",
  "commit": "$COMMIT",
  "dirty": $DIRTY,
  "source_date_epoch": $SOURCE_DATE_EPOCH,
  "rustc": "$(rustc --version)",
  "flags": "lto=thin codegen-units=1 panic=abort strip=symbols build-id=sha1"
}
JSON
  echo "provenance $OUT/provenance.json"
fi
