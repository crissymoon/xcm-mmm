//! Legal English: the words a document is built from, and the plain words that
//! can replace them.
//!
//! The crate already fits a `legal` axis over a labelled bank of sentences: it
//! answers whether a passage *is* legal register. This module answers the two
//! questions a person actually asks next. Which words made it legal, at which
//! offsets, and what does it say in plain English.
//!
//! Three files carry the instrument, and none of them is code:
//!
//! * `terms.jsonl`, the glossary: one row per term, with the plain reading, the
//!   kind, whether the term is avoidable, the authority behind it and the
//!   sources it was read from;
//! * `rewrites.jsonl`, the substitutions: one row per pattern, with the reason
//!   and whether the rule edits or only advises;
//! * `settings.json`, the citation patterns a rewrite must not touch and the
//!   density bands a reading is reported in.
//!
//! `legal scan` reads a document and names every term with its offsets.
//! `legal translate` rewrites it and glosses the terms of art it must keep.
//! `legal terms` lists the glossary. `legal check` recomputes the folder's
//! digests and its coverage, so an edited row is a disagreement rather than a
//! quiet change.
//!
//! The distinction the module is built around is avoidable against necessary.
//! Archaic deictics, doublets and nominals can go. A term of art names a
//! doctrine, and replacing it would change what the document claims, so it is
//! counted separately and explained in place.

pub mod brief;
pub mod cli;
mod cues;
pub mod gaps;
pub mod guard;
pub mod rewrite;
pub mod term;
pub mod translate;

pub use brief::{Brief, Clause, Definition, Kind};
pub use cli::{LegalArgs, LegalCommand};
pub use cues::Cues;
pub use gaps::Gap;
pub use guard::{Guard, Span};
pub use rewrite::{Case, Change, Rule, Rules};
pub use term::{Band, ClauseWords, Hit, Lexicon, Settings, Term};
pub use translate::{Gloss, Legal, Reading, Scan, Stats, TermHit};
