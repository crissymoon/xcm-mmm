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
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$LLD" \
RUSTFLAGS="-C linker-flavor=ld.lld -C link-self-contained=yes -C strip=symbols" \
  cargo build --release --target "$TARGET"

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
shasum -a 256 "$BIN"
