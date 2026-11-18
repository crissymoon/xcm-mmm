//! A versioned artifact directory, with an audit trail and rollback as a command.
//!
//! A rebuild proves an artifact can be reproduced. It does not prove a
//! deployment can be undone, and it does not prove which data the deployed copy
//! was built from. The registry keeps a copy of every published artifact, a
//! pointer to the current one, and a record beside each copy: the SHA-256 of the
//! artifact bytes, the SHA-256 of the data, the rows and range that data
//! covered, the seed when the fit used one, the readings the card claims, and
//! the version it replaced.
//!
//! That record is the model-card and SR 11-7 minimum, and it is the difference
//! between a rollback that is reversible and one that is defensible: `verify`
//! recomputes the digest of the stored bytes, so a version that has been edited
//! since publication says so instead of looking current.
//!
//! Every publish and rollback appends a line to `history.jsonl`. The log is
//! append-only and the registry never rewrites it, so the sequence of decisions
//! survives the pointer moving.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::digest::sha256;

/// What a published version states about itself.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Audit {
    /// SHA-256 of the data the artifact was fitted on.
    #[serde(default)]
    pub data_digest: String,
    /// Rows in that data, as the card counts them.
    #[serde(default)]
    pub data_rows: usize,
    /// The range the data covers, in the words the card uses.
    #[serde(default)]
    pub range: String,
    /// Seed, when the fit drew on randomness. Absent is a claim: the fit is
    /// deterministic from its inputs.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Named readings, formatted once at publication so the trail and the card
    /// cannot drift apart in presentation.
    #[serde(default)]
    pub facts: BTreeMap<String, String>,
}

/// One published version.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub id: u32,
    pub file: String,
    /// SHA-256 of the artifact bytes at publication.
    #[serde(default)]
    pub digest: String,
    /// Size of those bytes.
    #[serde(default)]
    pub bytes: u64,
    /// The version this one replaced, absent for the first.
    #[serde(default)]
    pub parent: Option<u32>,
    /// Publication time, ISO 8601 from the same clock the manifests use.
    #[serde(default)]
    pub published: String,
    /// The audit trail. Defaults are empty rather than absent, so a version
    /// published before the trail existed reads as "not recorded" instead of
    /// failing to load.
    #[serde(default)]
    pub audit: Audit,
}

/// One line of the append-only log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub at: String,
    pub action: String,
    pub version: u32,
    pub note: String,
}

/// A versioned artifact directory.
#[derive(Clone, Debug)]
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    /// Open or create a registry at `root`.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("versions"))?;
        Ok(Self { root })
    }

    /// Publish a model file as the next version, with no audit trail.
    ///
    /// Kept for callers that have nothing to record. It is not the path the
    /// command line takes: `churn publish` builds an [`Audit`] from the fit.
    pub fn publish(&self, model: &str) -> io::Result<Version> {
        self.publish_with(model, Audit::default())
    }

    /// Publish a model file as the next version, recording `audit` beside it.
    pub fn publish_with(&self, model: &str, audit: Audit) -> io::Result<Version> {
        let source = Path::new(model);
        let bytes = match fs::read(source) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("{model} is not a file"),
                ))
            }
            Err(error) => return Err(error),
        };
        let parent = self.current()?.map(|version| version.id);
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
        fs::write(directory.join(&name), &bytes)?;
        let version = Version {
            id: next,
            file: name,
            digest: sha256(&bytes),
            bytes: bytes.len() as u64,
            parent,
            published: crate::dataset::built_at(),
            audit,
        };
        self.write_meta(&version)?;
        self.set_current(&version)?;
        self.record(&Event {
            at: version.published.clone(),
            action: "publish".to_string(),
            version: version.id,
            note: format!(
                "{}{}",
                version.file,
                match parent {
                    Some(parent) => format!(" replacing v{parent:04}"),
                    None => " as the first version".to_string(),
                }
            ),
        })?;
        Ok(version)
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
            if let Some(version) = self.meta(&entry.path(), id)? {
                out.push(version);
            }
        }
        out.sort_by_key(|version| version.id);
        Ok(out)
    }

    /// Recompute the digest of a stored version and compare it with the record.
    ///
    /// `Ok(None)` when the version has no recorded digest, which is what an
    /// artifact published before the trail existed looks like.
    pub fn verify(&self, id: u32) -> io::Result<Option<bool>> {
        let version = self
            .list()?
            .into_iter()
            .find(|version| version.id == id)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("no version v{id:04}"))
            })?;
        if version.digest.is_empty() {
            return Ok(None);
        }
        let bytes = fs::read(self.stored(&version))?;
        Ok(Some(sha256(&bytes) == version.digest))
    }

    /// The append-only log, oldest first.
    pub fn history(&self) -> io::Result<Vec<Event>> {
        let text = match fs::read_to_string(self.root.join("history.jsonl")) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut out = Vec::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            out.push(
                serde_json::from_str(line).map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?,
            );
        }
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
        let bytes = fs::read(self.stored(&previous))?;
        if !previous.digest.is_empty() && sha256(&bytes) != previous.digest {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "v{:04} does not match its recorded digest, so it is not restored",
                    previous.id
                ),
            ));
        }
        fs::write(target, bytes)?;
        self.set_current(&previous)?;
        self.record(&Event {
            at: crate::dataset::built_at(),
            action: "rollback".to_string(),
            version: previous.id,
            note: format!(
                "v{:04} restored to {target} replacing v{:04}",
                previous.id, current.id
            ),
        })?;
        Ok(previous)
    }

    fn stored(&self, version: &Version) -> PathBuf {
        self.root
            .join("versions")
            .join(format!("v{:04}", version.id))
            .join(&version.file)
    }

    fn write_meta(&self, version: &Version) -> io::Result<()> {
        let path = self
            .root
            .join("versions")
            .join(format!("v{:04}", version.id))
            .join("meta.json");
        let text = serde_json::to_string_pretty(version).map_err(io::Error::other)?;
        fs::write(path, format!("{text}\n"))
    }

    /// Read the record beside a version directory, falling back to the file
    /// name for a version published before the record existed.
    fn meta(&self, directory: &Path, id: u32) -> io::Result<Option<Version>> {
        let path = directory.join("meta.json");
        if let Ok(text) = fs::read_to_string(&path) {
            let version: Version = serde_json::from_str(&text)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            return Ok(Some(version));
        }
        let mut files = fs::read_dir(directory)?
            .filter_map(|file| file.ok())
            .map(|file| file.file_name().to_string_lossy().to_string())
            .filter(|file| file.ends_with(".json") && file != "meta.json")
            .collect::<Vec<_>>();
        files.sort();
        Ok(files.into_iter().next().map(|file| Version {
            id,
            file,
            digest: String::new(),
            bytes: 0,
            parent: None,
            published: String::new(),
            audit: Audit::default(),
        }))
    }

    fn set_current(&self, version: &Version) -> io::Result<()> {
        let text = serde_json::to_string(version).map_err(io::Error::other)?;
        fs::write(self.root.join("current.json"), format!("{text}\n"))
    }

    fn record(&self, event: &Event) -> io::Result<()> {
        use std::io::Write;
        let line = serde_json::to_string(event).map_err(io::Error::other)?;
        let path = self.root.join("history.jsonl");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{line}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-reg-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_registry_publishes_and_rolls_back() {
        let dir = scratch("plain");
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
        let dir = scratch("one");
        let model = dir.join("model.json");
        fs::write(&model, "{}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        assert!(registry.rollback(model.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_version_records_the_trail_it_was_published_with() {
        let dir = scratch("audit");
        let model = dir.join("model.json");
        fs::write(&model, "{\"auc\":0.867}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        let mut audit = Audit {
            data_digest: sha256(b"panel"),
            data_rows: 7_043,
            range: "month 1 to 72".to_string(),
            seed: Some(7),
            facts: BTreeMap::new(),
        };
        audit
            .facts
            .insert("held-out AUC".to_string(), "0.867".to_string());
        let first = registry
            .publish_with(model.to_str().unwrap(), audit.clone())
            .unwrap();
        let second = registry
            .publish_with(model.to_str().unwrap(), audit.clone())
            .unwrap();
        assert_eq!(first.parent, None);
        assert_eq!(second.parent, Some(1));
        assert_eq!(first.digest, sha256(b"{\"auc\":0.867}"));
        assert_eq!(first.bytes, 13);
        assert!(!first.published.is_empty());
        // The trail survives the round trip through the record on disk.
        let loaded = registry.list().unwrap();
        assert_eq!(loaded[1].audit, audit);
        assert_eq!(loaded[1].audit.facts["held-out AUC"], "0.867");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_notices_an_edited_version() {
        let dir = scratch("verify");
        let model = dir.join("model.json");
        fs::write(&model, "{\"version\":1}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        let version = registry.publish(model.to_str().unwrap()).unwrap();
        assert_eq!(registry.verify(version.id).unwrap(), Some(true));
        let stored = dir
            .join("registry")
            .join("versions")
            .join(format!("v{:04}", version.id))
            .join(&version.file);
        fs::write(&stored, "{\"version\":1,\"edited\":true}").unwrap();
        assert_eq!(registry.verify(version.id).unwrap(), Some(false));
        // An edited version is not restored either.
        let target = dir.join("live.json");
        fs::write(&target, "{\"version\":1}").unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        assert!(registry.rollback(target.to_str().unwrap()).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_log_records_both_decisions() {
        let dir = scratch("log");
        let model = dir.join("model.json");
        fs::write(&model, "{}").unwrap();
        let registry = Registry::open(dir.join("registry")).unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        registry.publish(model.to_str().unwrap()).unwrap();
        let target = dir.join("live.json");
        fs::write(&target, "{}").unwrap();
        registry.rollback(target.to_str().unwrap()).unwrap();
        let history = registry.history().unwrap();
        let actions: Vec<&str> = history.iter().map(|event| event.action.as_str()).collect();
        assert_eq!(actions, vec!["publish", "publish", "rollback"]);
        assert_eq!(history[2].version, 1);
        assert!(history[2].note.contains("replacing v0002"));
        let _ = fs::remove_dir_all(&dir);
    }
}
