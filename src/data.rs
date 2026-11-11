//! Tabular data for the forecasting front end: CSV and JSON Lines readers,
//! standardisation and walk-forward folds.
//!
//! The reader is deliberately small. It accepts numeric columns only, skips a
//! row that carries a missing or non numeric value instead of inventing one,
//! and records how many rows it skipped so a caller can report the cost.

use std::fmt;
use std::fs;
use std::path::Path;

use crate::rng::Rng;

/// Why a dataset could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// The file had no data rows.
    Empty,
    /// The target column named on the command line is not in the header.
    MissingTarget(String),
    /// A value could not be parsed, with its line number.
    BadNumber { line: usize, value: String },
    /// The file could not be read.
    Io(String),
    /// Every row was skipped.
    NoUsableRows,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Empty => write!(f, "the file has no data"),
            DataError::MissingTarget(name) => {
                write!(f, "target column {name:?} is not in the header")
            }
            DataError::BadNumber { line, value } => {
                write!(f, "line {line}: {value:?} is not a number")
            }
            DataError::Io(message) => write!(f, "read failed: {message}"),
            DataError::NoUsableRows => write!(f, "every row was skipped"),
        }
    }
}

impl std::error::Error for DataError {}

/// A feature matrix with a single target column.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Feature names, in column order.
    pub names: Vec<String>,
    /// Rows of features, row major.
    pub features: Vec<Vec<f64>>,
    /// Target value per row.
    pub target: Vec<f64>,
    /// Number of rows dropped because a value was missing or not numeric.
    pub skipped: usize,
}

impl Dataset {
    /// Build a dataset from parts, checking the shapes.
    pub fn new(
        names: Vec<String>,
        features: Vec<Vec<f64>>,
        target: Vec<f64>,
    ) -> Result<Self, DataError> {
        if features.is_empty() {
            return Err(DataError::Empty);
        }
        if features.len() != target.len() {
            return Err(DataError::Empty);
        }
        let p = names.len();
        if features.iter().any(|row| row.len() != p) {
            return Err(DataError::Empty);
        }
        Ok(Self {
            names,
            features,
            target,
            skipped: 0,
        })
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// True when there are no rows.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.names.len()
    }

    /// Read a CSV file with a header row.
    pub fn read_csv(path: impl AsRef<Path>, target: &str) -> Result<Self, DataError> {
        let text = fs::read_to_string(path).map_err(|e| DataError::Io(e.to_string()))?;
        Self::from_csv(&text, target)
    }

    /// Parse CSV text with a header row.
    ///
    /// Columns that do not parse as numbers on every row are dropped. The
    /// target column must exist and must parse on every kept row.
    pub fn from_csv(text: &str, target: &str) -> Result<Self, DataError> {
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let header = lines.next().ok_or(DataError::Empty)?;
        let columns: Vec<&str> = header.split(',').map(str::trim).collect();
        let target_index = columns
            .iter()
            .position(|c| *c == target)
            .ok_or_else(|| DataError::MissingTarget(target.to_string()))?;

        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in lines {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != columns.len() {
                continue;
            }
            rows.push(fields.iter().map(|s| s.to_string()).collect());
        }
        if rows.is_empty() {
            return Err(DataError::Empty);
        }

        // A column counts as numeric when at least four rows in five parse, so
        // an occasional blank cell does not take a whole feature out while a
        // date or label column is still dropped. A row that misses a kept
        // feature is skipped rather than filled in.
        let mut parseable = vec![0usize; columns.len()];
        for fields in &rows {
            for (i, value) in fields.iter().enumerate() {
                if value.parse::<f64>().is_ok() {
                    parseable[i] += 1;
                }
            }
        }
        let threshold = (rows.len() * 4).div_ceil(5);
        let kept: Vec<usize> = (0..columns.len())
            .filter(|&i| i != target_index && parseable[i] >= threshold)
            .collect();
        if kept.is_empty() {
            return Err(DataError::NoUsableRows);
        }

        let names: Vec<String> = kept.iter().map(|&i| columns[i].to_string()).collect();
        let mut features = Vec::with_capacity(rows.len());
        let mut target_values = Vec::with_capacity(rows.len());
        let mut skipped = 0usize;
        for fields in &rows {
            let y = match fields[target_index].parse::<f64>() {
                Ok(v) if v.is_finite() => v,
                _ => {
                    skipped += 1;
                    continue;
                }
            };
            let mut row = Vec::with_capacity(kept.len());
            let mut ok = true;
            for &i in &kept {
                match fields[i].parse::<f64>() {
                    Ok(v) if v.is_finite() => row.push(v),
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                features.push(row);
                target_values.push(y);
            } else {
                skipped += 1;
            }
        }
        if features.is_empty() {
            return Err(DataError::NoUsableRows);
        }
        let mut dataset = Dataset::new(names, features, target_values)?;
        dataset.skipped = skipped;
        Ok(dataset)
    }

    /// Parse JSON Lines, where each line is an object of numeric fields.
    ///
    /// The target must be a number on every kept row. String fields such as a
    /// date are ignored, which is what makes a dated export usable without a
    /// preprocessing step.
    pub fn from_jsonl(text: &str, target: &str) -> Result<Self, DataError> {
        let mut names: Vec<String> = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        let mut skipped = 0usize;

        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let value: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let object = match value.as_object() {
                Some(o) => o,
                None => {
                    skipped += 1;
                    continue;
                }
            };
            let y = match object.get(target).and_then(|v| v.as_f64()) {
                Some(v) if v.is_finite() => v,
                _ => {
                    skipped += 1;
                    continue;
                }
            };
            if names.is_empty() {
                for key in object.keys() {
                    if key != target && object.get(key).and_then(|v| v.as_f64()).is_some() {
                        names.push(key.clone());
                    }
                }
            }
            let mut row = Vec::with_capacity(names.len());
            let mut ok = true;
            for name in &names {
                match object.get(name).and_then(|v| v.as_f64()) {
                    Some(v) if v.is_finite() => row.push(v),
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                rows.push(row);
                ys.push(y);
            } else {
                skipped += 1;
            }
        }
        if rows.is_empty() {
            return Err(DataError::NoUsableRows);
        }
        let mut dataset = Dataset::new(names, rows, ys)?;
        dataset.skipped = skipped;
        Ok(dataset)
    }

    /// Read a CSV or JSON Lines file, choosing the parser from the content.
    ///
    /// The first non-empty line decides: a leading brace means JSON Lines,
    /// anything else is CSV with a header row. Sniffing the content instead of
    /// trusting the file name keeps a misnamed export working, and it turns a
    /// silent "every row was skipped" into a correct parse.
    pub fn load(path: impl AsRef<Path>, target: &str) -> Result<Self, DataError> {
        let text = fs::read_to_string(path).map_err(|e| DataError::Io(e.to_string()))?;
        let first = text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("");
        if first.starts_with('{') {
            Self::from_jsonl(&text, target)
        } else {
            Self::from_csv(&text, target)
        }
    }

    /// Centre every feature and divide by its standard deviation.
    ///
    /// Returns the mean and deviation of each column so a fitted model can be
    /// reported in the original units.
    pub fn standardize(&mut self) -> Vec<(f64, f64)> {
        let mut stats = Vec::with_capacity(self.n_features());
        for j in 0..self.n_features() {
            let column: Vec<f64> = self.features.iter().map(|r| r[j]).collect();
            let mean = crate::linalg::mean(&column);
            let sd = crate::linalg::std_dev(&column);
            stats.push((mean, sd));
            if sd > 0.0 {
                for row in self.features.iter_mut() {
                    row[j] = (row[j] - mean) / sd;
                }
            } else {
                for row in self.features.iter_mut() {
                    row[j] = 0.0;
                }
            }
        }
        stats
    }

    /// A copy of the rows in `rows`, keeping the feature names.
    pub fn slice(&self, rows: std::ops::Range<usize>) -> Dataset {
        let start = rows.start.min(self.len());
        let end = rows.end.min(self.len()).max(start);
        Dataset {
            names: self.names.clone(),
            features: self.features[start..end].to_vec(),
            target: self.target[start..end].to_vec(),
            skipped: 0,
        }
    }

    /// Re-apply a mean and deviation per column, as returned by
    /// [`Dataset::standardize`].
    ///
    /// This is how a held-out window inherits the training window's scaling
    /// instead of being standardised with its own statistics.
    pub fn apply_standardization(&mut self, stats: &[(f64, f64)]) {
        for (j, &(mean, sd)) in stats.iter().enumerate() {
            if j >= self.n_features() {
                break;
            }
            for row in self.features.iter_mut() {
                row[j] = if sd > 0.0 { (row[j] - mean) / sd } else { 0.0 };
            }
        }
    }

    /// A synthetic panel for the examples and the tests.
    ///
    /// The first `informative` features drive the target, the rest are noise.
    /// The target also carries an autoregressive term, so a walk-forward split
    /// is a fair evaluation rather than a random shuffle.
    pub fn synthetic(rows: usize, features: usize, informative: usize, seed: u64) -> Self {
        let rows = rows.max(40);
        let features = features.max(informative + 1);
        let mut rng = Rng::new(seed);
        let names: Vec<String> = (1..=features).map(|j| format!("x{j}")).collect();
        let mut matrix = vec![vec![0.0; features]; rows];
        // Give the informative features an AR(1) shape and some correlation,
        // which is closer to a market panel than white noise.
        for j in 0..features {
            let noise_scale = if j < informative { 0.6 } else { 1.0 };
            let mut previous = rng.normal(0.0, 1.0);
            for row in matrix.iter_mut() {
                previous = 0.7 * previous + noise_scale * rng.normal(0.0, 1.0);
                row[j] = previous;
            }
        }
        let mut target = vec![0.0; rows];
        let mut previous = 0.0;
        for i in 0..rows {
            let mut signal = 0.0;
            for (j, row) in matrix[i].iter().enumerate().take(informative) {
                let sign = if j % 2 == 0 { 1.0 } else { -0.6 };
                signal += sign * row;
            }
            let y = 0.55 * previous
                + 0.8 * signal / (informative as f64).sqrt()
                + rng.normal(0.0, 0.35);
            target[i] = y;
            previous = y;
        }
        Self {
            names,
            features: matrix,
            target,
            skipped: 0,
        }
    }

    /// Write the dataset back out as CSV, target last.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.names.join(","));
        out.push_str(",target\n");
        for (row, y) in self.features.iter().zip(&self.target) {
            let cells: Vec<String> = row.iter().map(|v| format!("{v:.10}")).collect();
            out.push_str(&cells.join(","));
            out.push_str(&format!(",{y:.10}\n"));
        }
        out
    }

    /// Render the dataset as JSON Lines, one object per row.
    ///
    /// The target key is `target`, matching the default the reader expects.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for (row, y) in self.features.iter().zip(&self.target) {
            let mut object = serde_json::Map::with_capacity(self.names.len() + 1);
            for (name, value) in self.names.iter().zip(row) {
                object.insert(name.clone(), serde_json::json!(value));
            }
            object.insert("target".to_string(), serde_json::json!(y));
            out.push_str(&serde_json::Value::Object(object).to_string());
            out.push('\n');
        }
        out
    }
}

/// One train and test window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    /// First row of the training window.
    pub train_start: usize,
    /// One past the last row of the training window.
    pub train_end: usize,
    /// First row of the test window.
    pub test_start: usize,
    /// One past the last row of the test window.
    pub test_end: usize,
}

impl Fold {
    /// Training row indices.
    pub fn train(&self) -> std::ops::Range<usize> {
        self.train_start..self.train_end
    }

    /// Test row indices.
    pub fn test(&self) -> std::ops::Range<usize> {
        self.test_start..self.test_end
    }

    /// Number of test rows.
    pub fn test_len(&self) -> usize {
        self.test_end.saturating_sub(self.test_start)
    }
}

/// Blocked walk-forward folds.
///
/// The rows are cut into `folds + 1` contiguous blocks. Fold `k` trains on
/// everything before its test block and tests on that block, so no test row is
/// ever seen in training. `embargo` drops the last rows of each training window,
/// which is the usual guard against a label that overlaps its own future.
///
/// Returns fewer folds when the data is too short for the request.
pub fn walk_forward(rows: usize, folds: usize, embargo: usize) -> Vec<Fold> {
    if rows < 8 || folds == 0 {
        return Vec::new();
    }
    let folds = folds.min(rows / 4).max(1);
    let block = rows / (folds + 1);
    if block == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(folds);
    for k in 1..=folds {
        let test_start = k * block;
        let test_end = if k == folds { rows } else { (k + 1) * block };
        let train_end = test_start.saturating_sub(embargo);
        if train_end < 12 || test_end <= test_start {
            continue;
        }
        out.push(Fold {
            train_start: 0,
            train_end,
            test_start,
            test_end,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_round_trip() {
        let text = "a,b,target\n1,2,3\n4,5,6\n";
        let d = Dataset::from_csv(text, "target").unwrap();
        assert_eq!(d.n_features(), 2);
        assert_eq!(d.len(), 2);
        assert_eq!(d.target, vec![3.0, 6.0]);
        let again = d.to_csv();
        let d2 = Dataset::from_csv(&again, "target").unwrap();
        assert_eq!(d2.len(), 2);
    }

    #[test]
    fn missing_target_is_reported() {
        let text = "a,b\n1,2\n";
        assert!(matches!(
            Dataset::from_csv(text, "y"),
            Err(DataError::MissingTarget(name)) if name == "y"
        ));
    }

    #[test]
    fn bad_rows_are_skipped_not_invented() {
        let text = "a,b,target\n1,2,3\nx,5,6\n7,8,9\n";
        let d = Dataset::from_csv(text, "target").unwrap();
        // The column a does not parse everywhere, so it is dropped entirely.
        assert_eq!(d.n_features(), 1);
        assert_eq!(d.len(), 3);
        assert_eq!(d.names, vec!["b"]);
    }

    #[test]
    fn jsonl_reads_numbers_and_ignores_dates() {
        let text = r#"{"date":"2024-01-01","x":1.0,"y":2.0,"target":3.0}
{"date":"2024-01-02","x":4.0,"y":5.0,"target":6.0}"#;
        let d = Dataset::from_jsonl(text, "target").unwrap();
        assert_eq!(d.n_features(), 2);
        assert_eq!(d.len(), 2);
        assert_eq!(d.target, vec![3.0, 6.0]);
    }

    #[test]
    fn jsonl_writer_round_trips_through_the_reader() {
        let d = Dataset::synthetic(60, 6, 3, 11);
        let text = d.to_jsonl();
        let back = Dataset::from_jsonl(&text, "target").unwrap();
        assert_eq!(back.len(), d.len());
        assert_eq!(back.n_features(), d.n_features());
        for (a, b) in back.target.iter().zip(&d.target) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn load_sniffs_jsonl_and_csv_by_content_not_name() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // JSON Lines written under a .csv name.
        let jsonl = Dataset::synthetic(50, 5, 2, 3).to_jsonl();
        let misleading_jsonl = dir.join("panel.csv");
        std::fs::write(&misleading_jsonl, &jsonl).unwrap();
        let a = Dataset::load(&misleading_jsonl, "target").unwrap();
        assert_eq!(a.len(), 50);

        // CSV written under a .jsonl name, the case that used to skip every row.
        let csv = Dataset::synthetic(50, 5, 2, 3).to_csv();
        let misleading_csv = dir.join("panel.jsonl");
        std::fs::write(&misleading_csv, &csv).unwrap();
        let b = Dataset::load(&misleading_csv, "target").unwrap();
        assert_eq!(b.len(), 50);
        assert_eq!(b.n_features(), 5);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_held_out_slice_inherits_the_training_scaling() {
        let raw = Dataset::synthetic(300, 4, 2, 9);
        let mut head = raw.slice(0..200);
        let stats = head.standardize();
        let mut tail = raw.slice(200..300);
        tail.apply_standardization(&stats);
        assert_eq!(tail.len(), 100);
        let (mean, sd) = stats[0];
        let expected = (raw.features[200][0] - mean) / sd;
        assert!((tail.features[0][0] - expected).abs() < 1e-12);
        // Standardised with its own statistics the first value would differ.
        let mut own = raw.slice(200..300);
        own.standardize();
        assert!((own.features[0][0] - tail.features[0][0]).abs() > 1e-12);
    }

    #[test]
    fn folds_never_overlap_and_respect_the_embargo() {
        let folds = walk_forward(100, 4, 2);
        assert_eq!(folds.len(), 4);
        for f in &folds {
            assert!(f.train_end <= f.test_start);
            assert_eq!(f.train_end + 2, f.test_start);
            assert!(f.test_len() > 0);
        }
    }

    #[test]
    fn short_data_yields_no_folds() {
        assert!(walk_forward(5, 3, 0).is_empty());
        assert!(walk_forward(100, 0, 0).is_empty());
    }

    #[test]
    fn synthetic_data_has_the_requested_shape() {
        let d = Dataset::synthetic(120, 10, 4, 7);
        assert_eq!(d.len(), 120);
        assert_eq!(d.n_features(), 10);
        assert!(d.target.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn standardizing_leaves_zero_mean_and_unit_variance() {
        let mut d = Dataset::synthetic(200, 5, 2, 3);
        d.standardize();
        for j in 0..5 {
            let column: Vec<f64> = d.features.iter().map(|r| r[j]).collect();
            assert!(crate::linalg::mean(&column).abs() < 1e-9);
            assert!((crate::linalg::std_dev(&column) - 1.0).abs() < 1e-9);
        }
    }
}
