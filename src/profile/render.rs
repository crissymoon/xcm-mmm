//! The text rendering of a report.
//!
//! The layout is one block per question, because the questions are answered by
//! different models and a reader has to be able to see which model said what.
//! Every line prints a field of the [`Report`]; nothing is recomputed here.

use super::report::{Axis, Report, Semantics};

/// Print the whole report.
pub fn print(report: &Report) {
    header(report);
    emotion(report);
    triggers(report);
    detectors(report);
    semantics(&report.semantics);
    segment(report);
    for note in &report.notes {
        println!("note      {note}");
    }
}

fn header(report: &Report) {
    let shape = &report.shape;
    println!(
        "text      {} chars  {} words  {} sentence{}  {:.1} words/sentence",
        commas(shape.chars),
        commas(shape.words),
        commas(shape.sentences),
        if shape.sentences == 1 { "" } else { "s" },
        shape.words_per_sentence
    );
    let preview: String = report.text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !preview.is_empty() {
        let cut: String = preview.chars().take(72).collect();
        let ellipsis = if preview.chars().count() > 72 {
            "..."
        } else {
            ""
        };
        println!("          \"{cut}{ellipsis}\"");
    }
}

fn emotion(report: &Report) {
    let reading = &report.emotion;
    println!(
        "emotion   {:<12} valence {:+.2}  arousal {:.2}  intensity {:.2}",
        reading.dominant, reading.valence, reading.arousal, reading.intensity
    );
    let mut axes: Vec<(&String, &f64)> = reading.shares.iter().collect();
    axes.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (axis, share) in axes {
        if *share <= 0.0 {
            continue;
        }
        let bar = "#".repeat((share * 32.0).round() as usize);
        println!("  {axis:<12} {share:>5.2}  {bar}");
    }
    if reading.negators + reading.amplifiers + reading.diminishers > 0 {
        println!(
            "  modifiers    {} negator, {} amplifier, {} diminisher",
            reading.negators, reading.amplifiers, reading.diminishers
        );
    }
}

fn triggers(report: &Report) {
    if report.triggers.is_empty() {
        println!("triggers  none: no word in the text read above the neutral prior");
        return;
    }
    println!(
        "triggers  {:<16} {:<11} {:<8} {:>7}  valence",
        "word", "axis", "tier", "weight"
    );
    for trigger in &report.triggers {
        println!(
            "          {:<16} {:<11} {:<8} {:>7.2}  {:+6.2}",
            trigger.word, trigger.axis, trigger.tier, trigger.weight, trigger.valence
        );
    }
}

fn detectors(report: &Report) {
    println!(
        "detectors {:<12} {:>6}  {:<7} evidence",
        "axis", "score", "level"
    );
    for axis in &report.detectors {
        println!(
            "          {:<12} {:>6.3}  {:<7} {}",
            axis.name,
            axis.score,
            axis.level,
            evidence(axis)
        );
    }
}

fn evidence(axis: &Axis) -> String {
    let mut parts: Vec<String> = Vec::new();
    if axis.method == "index" {
        if axis.gate_open {
            let fired: Vec<String> = axis
                .parts
                .iter()
                .filter(|(_, value)| **value > 0.0)
                .map(|(name, value)| format!("{name} {value:.2}"))
                .collect();
            if fired.is_empty() {
                parts.push("gate open, no component fired".to_string());
            } else {
                parts.push(format!("per 100 words: {}", fired.join(", ")));
            }
        } else {
            parts.push(format!("gate closed: no cue from {}", axis.gate.join(", ")));
        }
    }
    if !axis.hits.is_empty() {
        parts.push(format!("fired: {}", axis.hits.join(", ")));
    }
    if parts.is_empty() {
        parts.push("-".to_string());
    }
    parts.join("  ")
}

fn semantics(semantics: &Semantics) {
    println!(
        "semantics flesch {:.2}  grade {:.2}  syllables/word {:.2}  long words {:.1}%  second person {:.1}%  questions {}  exclamations {}",
        semantics.flesch_ease,
        semantics.grade_level,
        semantics.syllables_per_word,
        semantics.long_word_share * 100.0,
        semantics.second_person_share * 100.0,
        semantics.questions,
        semantics.exclamations
    );
    for (name, value) in &semantics.surface {
        let value = if value.abs() < 0.0005 { 0.0 } else { *value };
        println!("          {:<24} {:>7.3}", label(name), value);
    }
}

fn segment(report: &Report) {
    let view = match &report.segment {
        Some(view) => view,
        None => return,
    };
    println!("segment   copy against every behavioural centre");
    for (at, placement) in view.ranked.iter().enumerate() {
        println!(
            "          {} {:>2}  rmse {:.2}  share {:>5.1}%  {}",
            if at == 0 { "*" } else { " " },
            placement.id,
            placement.rmse,
            placement.share * 100.0,
            placement.label
        );
    }
    if let Some(best) = view.ranked.first() {
        for reading in &best.readings {
            println!(
                "          {:<18} text {} {}   segment {}",
                reading.label,
                number(reading.value),
                reading.unit,
                number(reading.segment)
            );
        }
    }
    println!("          observed {}", view.basis.join(", "));
    println!("          unobserved {}", view.unobserved.join(", "));
    println!("          {}", view.note);
    if view.ranked.len() > 1 {
        println!(
            "          nearest and second differ by {:.2} rmse, against distances of {:.2} and {:.2}",
            view.margin, view.ranked[0].rmse, view.ranked[1].rmse
        );
    }
}

/// The words the report uses for the surface readings it knows.
fn label(name: &str) -> &str {
    match name {
        "register" => "abstraction register",
        "nominal" => "nominalisations",
        "agentless" => "agentless passives",
        "citation" => "citations",
        "authority" => "authority words",
        other => other,
    }
}

/// A reading in its own units, without a decimal it does not need.
fn number(value: f64) -> String {
    if (value - value.round()).abs() < 0.05 {
        format!("{:.0}", value)
    } else {
        format!("{value:.1}")
    }
}

fn commas(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (at, digit) in digits.chars().enumerate() {
        if at > 0 && (digits.len() - at) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_take_one_separator_each() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1000), "1,000");
        assert_eq!(commas(235976), "235,976");
    }

    #[test]
    fn the_readings_keep_their_names_when_no_label_is_known() {
        assert_eq!(label("register"), "abstraction register");
        assert_eq!(label("some_new_reading"), "some_new_reading");
    }
}
