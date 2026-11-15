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
    language(report);
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

fn language(report: &Report) {
    let view = &report.language;
    println!(
        "language  direct address {:.1}%  punch {:.1} words/sentence  specificity {:.1} numbers/100w  emotional {:.1}%",
        view.second_person * 100.0,
        view.sentence_words,
        view.numbers_per_100,
        view.emotional_density * 100.0
    );
    println!(
        "          distinct {} of {} words  type/token {:.2}  reuse {:.2}  focus {:.2}{}",
        view.distinct,
        view.words,
        view.type_token,
        view.reuse,
        view.focus,
        if view.top_phrase.is_empty() {
            String::new()
        } else {
            format!("  \"{}\"", view.top_phrase)
        }
    );
    if view.too_short {
        println!("          shorter than one four word shingle: the reuse proxies cannot be read");
    }
}

fn segment(report: &Report) {
    let view = match &report.segment {
        Some(view) => view,
        None => return,
    };
    println!(
        "segment   copy fit against every behavioural centre (text reach {:.0}%)",
        view.reach * 100.0
    );
    for placement in &view.ranked {
        println!(
            "          {} {:>2} {:<34} fit {:>5.1}%  distance {:.2}  panel share {:>5.1}%",
            if placement.id == view.ranked[0].id {
                "*"
            } else {
                " "
            },
            placement.id,
            placement.label,
            placement.fit,
            placement.distance,
            placement.share * 100.0
        );
    }
    if let Some(best) = view.ranked.first() {
        if !best.matches.is_empty() {
            println!(
                "          best {} matches on {}",
                best.id,
                best.matches.join(", ")
            );
        }
        if !best.clashes.is_empty() {
            println!(
                "          best {} differs on {}",
                best.id,
                best.clashes.join(", ")
            );
        }
        for reading in &best.readings {
            let text = format!("{} {}", number(reading.value), reading.unit);
            // The z gap says which way the text leans, which is the part a
            // reader can argue with; the weight says how much it counted.
            let lean = format!(
                "{}{:.2} z  weight {:.2}",
                if reading.gap >= 0.0 { "+" } else { "" },
                reading.gap,
                reading.weight
            );
            println!(
                "          {:<20} text {:<7} {:<22} segment {:<7} {}",
                reading.label,
                text,
                reading.source,
                number(reading.segment),
                lean
            );
        }
    }
    if !view.hook.is_empty() {
        println!("          repeated hook \"{}\"", view.hook);
    }
    println!("          observed {}", view.basis.join(", "));
    println!("          unobserved {}", view.unobserved.join(", "));
    println!("          {}", view.note);
    if view.ranked.len() > 1 {
        println!(
            "          nearest and second differ by {:.2} distance, against distances of {:.2} and {:.2}",
            view.margin, view.ranked[0].distance, view.ranked[1].distance
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
    // `-0.0` prints as "-0" under the integer format, and a segment mean of
    // minus nothing reads as a sign error.
    if (value - value.round()).abs() < 0.05 {
        let rounded = value.round();
        if rounded == 0.0 {
            "0".to_string()
        } else {
            format!("{rounded:.0}")
        }
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
