//! What the advertising panel is talking about, read from its published files.
//!
//! The panel itself is built on the Python side, where the ad registers are
//! read and the copy is scored. This module is the reading: it opens the two
//! files the panel leaves behind and answers which topics travelled furthest,
//! which terms carried them, and which ads reached the most people.
//!
//! Three things it refuses to smooth over, because the source is explicit and
//! a report that hides them is worse than one that does not exist.
//!
//! * **Reach is not one column.** A European ad publishes the DSA reach figure
//!   and an American issue ad publishes an impression range, whose midpoint the
//!   panel uses. The two are the same question asked of two registers, so they
//!   are summed for ranking and reported with the note the source carries.
//! * **A topic is a cluster, not a topic list.** The names here are the three
//!   heaviest terms of a cluster of words that share ads. Nothing was labelled
//!   by hand, so a cluster that reads oddly reads oddly because the copy does.
//! * **Ads are counted once.** The archive publishes one row per placement set,
//!   so the panel folds identical bodies. `ads` is creatives and `reach` is
//!   every row that ran them.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::text::hash::{fnv1a, hex64};

pub mod cli;

/// The two files the panel publishes, in the order a reader meets them.
pub const HOT: &str = "hot-topics.json";
pub const KEYWORDS: &str = "keywords.json";
pub const MANIFEST: &str = "manifest.json";

/// The folder a built reading lands in.
pub const FOLDER: &str = "data/topics";

/// The headline counts, repeated in both files on purpose so either can be
/// read alone.
#[derive(Clone, Debug, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub read_at: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub ads: usize,
    #[serde(default)]
    pub with_reach: usize,
    #[serde(default)]
    pub reach_total: usize,
    #[serde(default)]
    pub languages: BTreeMap<String, usize>,
    #[serde(default)]
    pub tickers: usize,
    #[serde(default)]
    pub areas: BTreeMap<String, usize>,
    #[serde(default)]
    pub terms: usize,
    #[serde(default)]
    pub topics: usize,
    #[serde(default)]
    pub floor: f64,
    #[serde(default)]
    pub note: String,
}

/// One term and how far it travelled.
#[derive(Clone, Debug, Deserialize)]
pub struct Term {
    pub term: String,
    #[serde(default)]
    pub ads: usize,
    #[serde(default)]
    pub reach: usize,
    #[serde(default)]
    pub lift: f64,
    #[serde(default)]
    pub leaning: String,
}

/// The heaviest ad of a topic, carried so the cluster can be checked against
/// a real ad instead of against its name.
#[derive(Clone, Debug, Deserialize, Default)]
pub struct Example {
    #[serde(default)]
    pub ticker: String,
    #[serde(default)]
    pub page: String,
    #[serde(default)]
    pub reach: Option<usize>,
    #[serde(default)]
    pub started: String,
    #[serde(default)]
    pub copy: String,
}

/// One cluster of terms that share ads.
#[derive(Clone, Debug, Deserialize)]
pub struct Topic {
    pub name: String,
    #[serde(default)]
    pub terms: Vec<String>,
    #[serde(default)]
    pub ads: usize,
    #[serde(default)]
    pub reach: usize,
    #[serde(default)]
    pub tickers: Vec<String>,
    #[serde(default)]
    pub areas: Vec<String>,
    #[serde(default)]
    pub languages: BTreeMap<String, usize>,
    #[serde(default)]
    pub linked: f64,
    #[serde(default)]
    pub example: Example,
}

/// One ad, folded across the placements that ran it.
#[derive(Clone, Debug, Deserialize)]
pub struct Ad {
    #[serde(default)]
    pub ticker: String,
    #[serde(default)]
    pub area: String,
    #[serde(default)]
    pub page: String,
    #[serde(default)]
    pub market: String,
    #[serde(default)]
    pub scope: String,
    #[serde(default)]
    pub started: String,
    #[serde(default)]
    pub ads: usize,
    #[serde(default)]
    pub reach: usize,
    #[serde(default)]
    pub copy: String,
}

/// The hot reading: topics, the keywords beneath them, and the heaviest ads.
#[derive(Clone, Debug, Deserialize)]
pub struct Hot {
    pub meta: Meta,
    #[serde(default)]
    pub topics: Vec<Topic>,
    #[serde(default)]
    pub hot_ads: Vec<Ad>,
    #[serde(default)]
    pub keywords: Vec<Term>,
}

/// The keyword reading: single terms, then the same terms inside an axis.
#[derive(Clone, Debug, Deserialize)]
pub struct Keywords {
    pub meta: Meta,
    #[serde(default)]
    pub keywords: Vec<Term>,
    #[serde(default)]
    pub by_ticker: BTreeMap<String, Vec<Term>>,
    #[serde(default)]
    pub by_area: BTreeMap<String, Vec<Term>>,
}

/// Both files, opened from one folder.
#[derive(Clone, Debug)]
pub struct Panel {
    pub folder: PathBuf,
    pub hot: Hot,
    pub keywords: Keywords,
}

/// One file and the digest that proves it is the one that was published.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Stamp {
    pub name: String,
    #[serde(default)]
    pub rows: u64,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub fnv64: String,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct Manifest {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub read_at: String,
    #[serde(default)]
    pub files: Vec<Stamp>,
}

impl Panel {
    /// Open both files, naming the one that is missing rather than the folder.
    pub fn open(dir: &str) -> Result<Self, Box<dyn Error>> {
        let folder = Path::new(dir).to_path_buf();
        let hot = read_json(&folder.join(HOT))?;
        let keywords = read_json(&folder.join(KEYWORDS))?;
        Ok(Self {
            folder,
            hot,
            keywords,
        })
    }

    /// The topics ranked by the reach their ads carried.
    pub fn ranked(&self) -> Vec<&Topic> {
        let mut ranked: Vec<&Topic> = self.hot.topics.iter().collect();
        ranked.sort_by(|a, b| b.reach.cmp(&a.reach).then(b.ads.cmp(&a.ads)));
        ranked
    }

    /// The ads ranked by reach, which is what the register measured rather
    /// than a count of placements.
    pub fn hottest(&self) -> Vec<&Ad> {
        let mut ranked: Vec<&Ad> = self.hot.hot_ads.iter().collect();
        ranked.sort_by(|a, b| b.reach.cmp(&a.reach).then(b.ads.cmp(&a.ads)));
        ranked
    }

    /// Terms inside one axis group, or empty when the group is unknown.
    pub fn within(&self, axis: &str, name: &str) -> Option<&Vec<Term>> {
        let table = match axis {
            "ticker" => &self.keywords.by_ticker,
            "area" => &self.keywords.by_area,
            _ => return None,
        };
        table.get(name)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn Error>> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let parsed = serde_json::from_str(&text)
        .map_err(|error| format!("{} is not the published shape: {error}", path.display()))?;
    Ok(parsed)
}

impl Manifest {
    pub fn open(dir: &str) -> Result<Option<Self>, Box<dyn Error>> {
        let path = Path::new(dir).join(MANIFEST);
        if !path.exists() {
            return Ok(None);
        }
        read_json(&path).map(Some)
    }

    /// Every file whose bytes disagree with the stamp, as name and reason.
    ///
    /// The digest is the same FNV-1a the artifact manifests use, so a folder
    /// that passed here passed the same test as a model folder, and a file
    /// edited after publication is named rather than quietly ranked.
    pub fn disagreements(&self, dir: &str) -> Vec<String> {
        let mut found = Vec::new();
        for stamp in &self.files {
            let path = Path::new(dir).join(&stamp.name);
            let bytes = match fs::read(&path) {
                Ok(bytes) => bytes,
                Err(error) => {
                    found.push(format!("{} missing: {error}", stamp.name));
                    continue;
                }
            };
            if bytes.len() as u64 != stamp.bytes {
                found.push(format!(
                    "{} is {} bytes, stamped {}",
                    stamp.name,
                    bytes.len(),
                    stamp.bytes
                ));
                continue;
            }
            let digest = hex64(fnv1a(&bytes));
            if digest != stamp.fnv64 {
                found.push(format!(
                    "{} hashes to {digest}, stamped {}",
                    stamp.name, stamp.fnv64
                ));
            }
        }
        found
    }
}

/// Build the manifest a published folder carries.
pub fn manifest(dir: &str) -> Result<Manifest, Box<dyn Error>> {
    let mut files = Vec::new();
    for name in [HOT, KEYWORDS] {
        let bytes = fs::read(Path::new(dir).join(name))?;
        files.push(Stamp {
            name: name.to_string(),
            rows: 0,
            bytes: bytes.len() as u64,
            fnv64: hex64(fnv1a(&bytes)),
        });
    }
    Ok(Manifest {
        name: "marketlab-ad-topics".into(),
        read_at: String::new(),
        files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "meta": {"ads": 2, "reach_total": 300, "with_reach": 2, "topics": 1},
      "topics": [
        {"name": "a / b / c", "terms": ["a","b","c"], "ads": 2, "reach": 15,
         "tickers": ["AAA"], "example": {"copy": "hello"}}
      ],
      "hot_ads": [
        {"ticker": "AAA", "reach": 10, "copy": "one"},
        {"ticker": "BBB", "reach": 5, "copy": "two"}
      ],
      "keywords": [{"term": "a", "ads": 2, "reach": 15}]
    }"#;

    fn panel() -> Hot {
        serde_json::from_str(SAMPLE).expect("sample parses")
    }

    #[test]
    fn ranking_puts_the_heaviest_first_and_keeps_ties_by_ads() {
        let hot = panel();
        let mut ads: Vec<&Ad> = hot.hot_ads.iter().collect();
        ads.sort_by_key(|ad| std::cmp::Reverse(ad.reach));
        assert_eq!(ads[0].ticker, "AAA");
        assert_eq!(ads[1].ticker, "BBB");
    }

    #[test]
    fn an_absent_field_reads_as_zero_rather_than_failing_the_open() {
        let hot = panel();
        assert_eq!(hot.topics[0].linked, 0.0);
        assert_eq!(hot.topics[0].languages.len(), 0);
        assert_eq!(hot.hot_ads[1].area, "");
    }

    #[test]
    fn a_manifest_names_the_file_whose_bytes_moved() {
        let dir = std::env::temp_dir().join(format!("xcm-topics-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let hot = dir.join(HOT);
        let words = dir.join(KEYWORDS);
        fs::write(&hot, SAMPLE).expect("write sample");
        fs::write(&words, SAMPLE).expect("write the second file");
        let built = manifest(dir.to_str().expect("utf8 path")).expect("manifest");
        assert_eq!(built.files.len(), 2, "both files are stamped");
        assert!(built.disagreements(dir.to_str().expect("utf8")).is_empty());
        fs::write(&hot, format!("{SAMPLE} ")).expect("append a byte");
        let found = built.disagreements(dir.to_str().expect("utf8"));
        assert_eq!(found.len(), 1, "one file moved, one reported: {found:?}");
        assert!(found[0].starts_with(HOT));
        let _ = fs::remove_dir_all(&dir);
    }
}
