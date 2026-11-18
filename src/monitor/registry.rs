//! A versioned artifact directory, with rollback as a command.
//!
//! A rebuild proves an artifact can be reproduced. It does not prove a
//! deployment can be undone. The registry keeps a copy of every published
//! artifact and a pointer to the current one, so a rollback is a file copy
//! rather than a rerun, and a second rollback can return to the version that
//! was current before.
//!
//! What it does not carry is the audit trail a model risk review asks for: a
//! published version records an id and a file name, not the data digest, the
//! seed, the metrics or the parent. That is roadmap item 23, and calling this a
//! registry without saying so would be the kind of brochure this repository is
//! trying not to write.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A versioned artifact directory.
#[derive(Clone, Debug)]
pub struct Registry {
    root: PathBuf,
}

/// One published version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub id: u32,
    pub file: String,
}

impl Registry {
    /// Open or create a registry at `root`.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("versions"))?;
        Ok(Self { root })
    }

    /// Publish a model file as the next version and make it current.
    pub fn publish(&self, model: &str) -> io::Result<Version> {
        let source = Path::new(model);
        if !source.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{model} is not a file"),
            ));
        }
        let next = self
            .list()?
            .last()
            .map(|version| version.id + 1)
            .unwrap_or(1);
        let directory = self.root.join("versions").join(format!("v{next:04}"));
        fs::create_dir_all(&directory)?;
        let name = source
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "model.json".to_string());
        fs::copy(source, directory.join(&name))?;
        self.set_current(next, &name)?;
        Ok(Version {
            id: next,
            file: name,
        })
    }

    /// The current version, if anything was published.
    pub fn current(&self) -> io::Result<Option<Version>> {
        let text = match fs::read_to_string(self.root.join("current.json")) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
    }

    /// Every published version, oldest first.
    pub fn list(&self) -> io::Result<Vec<Version>> {
        let mut out = Vec::new();
        let directory = self.root.join("versions");
        if !directory.is_dir() {
            return Ok(out);
        }
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(id) = name.trim_start_matches('v').parse::<u32>() else {
                continue;
            };
            let mut files = fs::read_dir(entry.path())?
                .filter_map(|file| file.ok())
                .map(|file| file.file_name().to_string_lossy().to_string())
                .filter(|file| file.ends_with(".json"))
                .collect::<Vec<_>>();
            files.sort();
            if let Some(file) = files.into_iter().next() {
                out.push(Version { id, file });
            }
        }
        out.sort_by_key(|version| version.id);
        Ok(out)
    }

    /// Move the current pointer back one version and restore `target`.
    ///
    /// The target file is overwritten with the previous version's bytes, and
    /// the registry is unchanged apart from the pointer, so a second rollback
    /// can return to the version that was current before. Returns the version
    /// that was restored.
    pub fn rollback(&self, target: &str) -> io::Result<Version> {
        let current = self
            .current()?
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "nothing is published"))?;
        let versions = self.list()?;
        let previous = versions
            .iter()
            .rev()
            .find(|version| version.id < current.id)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "there is no earlier version to roll back to",
                )
            })?;
        let source = self
            .root
            .join("versions")
            .join(format!("v{:04}", previous.id))
            .join(&previous.file);
        fs::copy(source, target)?;
        self.set_current(previous.id, &previous.file)?;
        Ok(previous)
    }

    fn set_current(&self, id: u32, file: &str) -> io::Result<()> {
        let version = Version {
            id,
            file: file.to_string(),
        };
        let text = serde_json::to_string(&version).map_err(io::Error::other)?;
        fs::write(self.root.join("current.json"), format!("{text}\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_registry_publishes_and_rolls_back() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-reg-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let first = dir.join("first.json");
        let second = dir.join("second.json");
        fs::write(&first, "{\"version\":1}").unwrap();
        fs::write(&second, "{\"version\":2}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        let a = registry.publish(first.to_str().unwrap()).unwrap();
        let b = registry.publish(second.to_str().unwrap()).unwrap();
        assert_eq!(a.id, 1);
        assert_eq!(b.id, 2);
        assert_eq!(registry.current().unwrap().unwrap().id, 2);
        assert_eq!(registry.list().unwrap().len(), 2);
        let target = dir.join("live.json");
        fs::write(&target, "{\"version\":2}").unwrap();
        let restored = registry.rollback(target.to_str().unwrap()).unwrap();
        assert_eq!(restored.id, 1);
        assert!(fs::read_to_string(&target)
            .unwrap()
            .contains("\"version\":1"));
        assert_eq!(registry.current().unwrap().unwrap().id, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rolling_back_without_history_is_refused() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-reg-one-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.json");
        fs::write(&model, "{}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        assert!(registry.rollback(model.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
