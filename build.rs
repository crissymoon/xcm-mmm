//! Capture the facts a released binary should be able to state about itself.
//!
//! Three values are baked into the crate here rather than guessed at run time,
//! because a shipped binary has no repository and no toolchain to ask:
//!
//! * the commit it was built from, and whether that tree was dirty;
//! * the target triple it was linked for;
//! * the rustc that compiled it, and the build time.
//!
//! The build time comes from `SOURCE_DATE_EPOCH` first, then from the commit
//! timestamp. Falling back to the commit rather than to `SystemTime::now()` is
//! what keeps two builds of one commit byte identical, which is the property
//! that makes the checksum in the dataset card worth publishing.
//!
//! Nothing here fails the build: a source tarball has no `.git`, and the binary
//! then reports `unknown` rather than refusing to compile.

use std::env;
use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn main() {
    // Each of these changes what the crate prints, so cargo must rebuild when
    // the repository head moves even though no source file changed.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let commit = git(&["rev-parse", "--short=7", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let commit_time = git(&["log", "-1", "--format=%ct"]);
    let dirty = match git(&["status", "--porcelain"]) {
        Some(_) => "true",
        None if Path::new(".git").exists() => "false",
        None => "unknown",
    };

    // SOURCE_DATE_EPOCH is the reproducible-builds convention: same epoch, same
    // bytes. A commit timestamp is the next best deterministic answer.
    let epoch = env::var("SOURCE_DATE_EPOCH")
        .ok()
        .or(commit_time)
        .unwrap_or_else(|| "0".into());

    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unknown".into());

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".into());

    println!("cargo:rustc-env=XCM_COMMIT={commit}");
    println!("cargo:rustc-env=XCM_DIRTY={dirty}");
    println!("cargo:rustc-env=XCM_BUILD_EPOCH={epoch}");
    println!("cargo:rustc-env=XCM_RUSTC={rustc_version}");
    println!("cargo:rustc-env=XCM_TARGET={target}");
}
