//! One customer row from the telecom churn table.
//!
//! The Maven table is the useful one for an outcome model: it carries tenure,
//! billing, contract and service features, the churn label, and a free-text
//! churn reason for the 1,869 customers who left. This module reads the columns
//! the model and the emotion pass need and refuses a table that lacks one. It
//! does not impute: a missing numeric is an error, because every one of these
//! columns is filled for every row in the source and a zero would be a guess
//! that lands in the regression as data.
//!
//! `Customer Status` has three values, and they are not symmetric. `Churned` is
//! the outcome. `Stayed` is a customer still present at their tenure. `Joined`
//! is a customer so new that the question has not been asked yet, so they are
//! censored rather than counted as a non-churn: the panel keeps them and the
//! series treats them as at risk from their tenure onwards.

use std::io;

use super::csvx::Table;

/// The columns the panel and the series are built from.
#[derive(Debug)]
pub struct Row {
    pub id: String,
    pub churned: bool,
    pub joined: bool,
    pub tenure: u32,
    pub monthly: f64,
    pub total: f64,
    pub age: u32,
    pub dependents: u32,
    pub referrals: u32,
    pub contract: String,
    pub internet: String,
    pub payment: String,
    pub support: bool,
    pub security: bool,
    pub category: Option<String>,
    pub reason: Option<String>,
}

/// Read every row, refusing a table that does not carry the columns.
pub fn read(table: &Table) -> io::Result<Vec<Row>> {
    let id = table.require("Customer ID")?;
    let status = table.require("Customer Status")?;
    let tenure = table.require("Tenure in Months")?;
    let monthly = table.require("Monthly Charge")?;
    let total = table.require("Total Charges")?;
    let age = table.require("Age")?;
    let dependents = table.require("Number of Dependents")?;
    let referrals = table.require("Number of Referrals")?;
    let contract = table.require("Contract")?;
    let internet = table.require("Internet Type")?;
    let payment = table.require("Payment Method")?;
    let support = table.require("Premium Tech Support")?;
    let security = table.require("Online Security")?;
    let category = table.require("Churn Category")?;
    let reason = table.require("Churn Reason")?;
    let mut rows = Vec::with_capacity(table.rows.len());
    for (number, cells) in table.rows.iter().enumerate() {
        let at = number + 2;
        let status = &cells[status];
        let row = Row {
            id: cells[id].trim().to_string(),
            churned: status.eq_ignore_ascii_case("churned"),
            joined: status.eq_ignore_ascii_case("joined"),
            tenure: number_at(&cells[tenure], "Tenure in Months", at)?.round() as u32,
            monthly: number_at(&cells[monthly], "Monthly Charge", at)?,
            total: number_at(&cells[total], "Total Charges", at)?,
            age: number_at(&cells[age], "Age", at)?.round() as u32,
            dependents: number_at(&cells[dependents], "Number of Dependents", at)?.round() as u32,
            referrals: number_at(&cells[referrals], "Number of Referrals", at)?.round() as u32,
            contract: cells[contract].clone(),
            internet: cells[internet].clone(),
            payment: cells[payment].clone(),
            support: cells[support].eq_ignore_ascii_case("yes"),
            security: cells[security].eq_ignore_ascii_case("yes"),
            category: non_empty(&cells[category]),
            reason: non_empty(&cells[reason]),
        };
        if row.id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("row {at} has no customer id"),
            ));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn non_empty(cell: &str) -> Option<String> {
    let text = cell.trim();
    if text.is_empty() || text.eq_ignore_ascii_case("n/a") {
        None
    } else {
        Some(text.to_string())
    }
}

fn number_at(cell: &str, column: &str, row: usize) -> io::Result<f64> {
    let text = cell.trim();
    if text.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("row {row} has no value for {column}"),
        ));
    }
    text.parse::<f64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("row {row} has a non numeric {column}: {text:?}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::churn::csvx;

    const HEADER: &str = "Customer ID,Age,Number of Dependents,Number of Referrals,Tenure in Months,Contract,Monthly Charge,Total Charges,Customer Status,Churn Category,Churn Reason,Premium Tech Support,Online Security,Internet Type,Payment Method";

    fn table(body: &str) -> Table {
        csvx::parse(&format!("{HEADER}\n{body}")).expect("a table")
    }

    #[test]
    fn a_churned_row_reads_its_reason_and_label() {
        let rows = read(&table(
            "1-AAA,37,1,2,9,Month-to-Month,65.60,593.30,Churned,Competitor,Competitor had better devices,Yes,No,Cable,Credit Card\n",
        ))
        .expect("a row");
        assert_eq!(rows[0].tenure, 9);
        assert!(rows[0].churned);
        assert!(rows[0].support);
        assert!(!rows[0].security);
        assert_eq!(
            rows[0].reason.as_deref(),
            Some("Competitor had better devices")
        );
        assert!((rows[0].monthly - 65.6).abs() < 1e-9);
    }

    #[test]
    fn a_stayed_row_keeps_an_empty_reason_empty() {
        let rows = read(&table(
            "2-BBB,44,0,0,3,Two Year,20.00,60.00,Stayed,,,No,Yes,DSL,Bank Withdrawal\n",
        ))
        .expect("a row");
        assert!(!rows[0].churned);
        assert_eq!(rows[0].reason, None);
        assert_eq!(rows[0].category, None);
    }

    #[test]
    fn a_missing_number_is_a_refusal_not_a_zero() {
        let error = read(&table(
            "3-CCC,30,0,0,,One Year,10,10,Joined,,,No,No,DSL,Credit Card\n",
        ))
        .expect_err("a refusal");
        assert!(error.to_string().contains("Tenure in Months"), "{error}");
    }

    #[test]
    fn a_missing_column_is_a_refusal() {
        let table = csvx::parse("Customer ID,Customer Status\n1,Churned\n").unwrap();
        let error = read(&table).expect_err("a refusal");
        assert!(error.to_string().contains("Tenure in Months"), "{error}");
    }
}
