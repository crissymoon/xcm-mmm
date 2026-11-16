//! The causal table and its CSV reader.
//!
//! The fit in the parent module works on [`CausalData`] and never touches a
//! file, so the reader can refuse a malformed cell without anyone losing the
//! fitted model, and the fit can be tested on a synthetic table with no CSV.

use std::collections::BTreeSet;
use std::io;

use crate::churn::csvx;

/// One row of the causal table.
#[derive(Clone, Debug)]
pub struct Row {
    pub features: Vec<f64>,
    pub treatment: bool,
    pub outcome: f64,
}

/// A causal table with named covariates.
#[derive(Clone, Debug)]
pub struct CausalData {
    pub names: Vec<String>,
    pub rows: Vec<Row>,
}

impl CausalData {
    /// Rows in the treated arm.
    pub fn treated(&self) -> usize {
        self.rows.iter().filter(|row| row.treatment).count()
    }

    /// Rows in the control arm.
    pub fn control(&self) -> usize {
        self.rows.len() - self.treated()
    }

    /// Feature matrix for the rows selected by `index`.
    pub fn design(&self, index: &[usize]) -> Vec<Vec<f64>> {
        index
            .iter()
            .map(|&at| self.rows[at].features.clone())
            .collect()
    }
}

/// Read a CSV with a binary treatment column and a numeric outcome.
///
/// Numeric covariates are used as they stand. A non-numeric column with at most
/// twelve levels becomes one indicator per level after the first, which is the
/// baseline. A column with one value per row is treated as an identifier and
/// skipped. A missing or unparseable numeric covariate is a refusal rather than
/// a zero, because a zero lands in the propensity model as data.
pub fn load_csv(path: &str, outcome: &str, treatment: &str) -> io::Result<CausalData> {
    let table = csvx::read(path)?;
    let outcome_at = table.require(outcome)?;
    let treatment_at = table.require(treatment)?;
    if outcome_at == treatment_at {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the outcome and the treatment are the same column",
        ));
    }
    if table.rows.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} holds no rows"),
        ));
    }
    let rows = table.rows.len();
    let mut numeric = Vec::new();
    let mut categorical = Vec::new();
    for column in 0..table.header.len() {
        if column == outcome_at || column == treatment_at {
            continue;
        }
        let values: Vec<&str> = table.rows.iter().map(|row| row[column].trim()).collect();
        if values.iter().all(|value| value.parse::<f64>().is_ok()) {
            numeric.push(column);
            continue;
        }
        let levels: BTreeSet<String> = values
            .iter()
            .filter(|value| !value.is_empty())
            .map(|value| value.to_lowercase())
            .collect();
        if levels.len() >= 2 && levels.len() <= 12 && levels.len() < rows {
            categorical.push((column, levels.into_iter().collect::<Vec<_>>()));
        }
    }
    let mut names = Vec::new();
    for &column in &numeric {
        names.push(table.header[column].clone());
    }
    for (column, levels) in &categorical {
        for level in levels.iter().skip(1) {
            names.push(format!("{}={level}", table.header[*column]));
        }
    }
    if names.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the table has no usable covariates",
        ));
    }
    let mut out = Vec::with_capacity(rows);
    for (number, cells) in table.rows.iter().enumerate() {
        let at = number + 2;
        let outcome = cells[outcome_at].trim().parse::<f64>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row {at} has a non numeric outcome {:?}", cells[outcome_at]),
            )
        })?;
        let treatment = parse_treatment(&cells[treatment_at], at)?;
        let mut features = Vec::with_capacity(names.len());
        for &column in &numeric {
            let value = cells[column].trim().parse::<f64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "row {at} has a non numeric {} {:?}",
                        table.header[column], cells[column]
                    ),
                )
            })?;
            features.push(value);
        }
        for (column, levels) in &categorical {
            let value = cells[*column].trim().to_lowercase();
            for level in levels.iter().skip(1) {
                features.push(f64::from(u8::from(&value == level)));
            }
        }
        out.push(Row {
            features,
            treatment,
            outcome,
        });
    }
    Ok(CausalData { names, rows: out })
}

fn parse_treatment(cell: &str, at: usize) -> io::Result<bool> {
    match cell.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "treated" | "treatment" | "t" => Ok(true),
        "0" | "false" | "no" | "n" | "control" | "c" => Ok(false),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("row {at} has an unrecognised treatment value {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_treatment_letter_is_a_refusal() {
        assert!(parse_treatment("maybe", 3).is_err());
        assert!(parse_treatment(" treated ", 3).unwrap());
        assert!(!parse_treatment("C", 3).unwrap());
    }

    #[test]
    fn an_identifier_column_is_skipped_and_a_level_column_becomes_indicators() {
        let dir = std::env::temp_dir().join(format!("xcm-mmm-table-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("table.csv");
        let mut text = String::from("id,y,t,plan\n");
        for i in 0..40 {
            let plan = if i % 2 == 0 { "basic" } else { "plus" };
            text.push_str(&format!("c{i},{},{},{plan}\n", u8::from(i % 3 == 0), i % 2));
        }
        std::fs::write(&path, text).unwrap();
        let data = load_csv(path.to_str().unwrap(), "y", "t").unwrap();
        assert_eq!(data.names.len(), 1);
        assert_eq!(data.names[0], "plan=plus");
        assert_eq!(data.rows.len(), 40);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
