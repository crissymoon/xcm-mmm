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
//! digests, its published counts and its coverage, so an edited row is a
//! disagreement rather than a quiet change, and a count the manifest publishes
//! is one the files still hold.
//!
//! Two commands score the instrument rather than run it. `legal bench` scores
//! the rewrite on `pairs.jsonl`, a legal passage beside the plain passage that
//! says the same thing, and reports a win rate with a Wilson interval, the
//! mean movement towards the plain side with a bootstrap interval, and the same
//! numbers over the pairs a rule actually edited. `legal coverage` splits the
//! sentence bank by register and reports which registers the glossary can read,
//! which it is blind to, and which the bank does not hold at all. A claim about
//! legal English is only as wide as that last table, so it ships beside the
//! glossary instead of in a note nobody reads.
//!
//! The distinction the module is built around is avoidable against necessary.
//! Archaic deictics, doublets and nominals can go. A term of art names a
//! doctrine, and replacing it would change what the document claims, so it is
//! counted separately and explained in place.

pub mod bench;
pub mod brief;
pub mod cli;
mod cues;
pub mod freq;
pub mod gaps;
pub mod guard;
pub mod registers;
pub mod rewrite;
pub mod term;
pub mod translate;

pub use bench::{Case, Mcnemar, Pair, Report, Slice};
pub use brief::{Brief, Clause, Definition, Kind};
pub use cli::{LegalArgs, LegalCommand};
pub use cues::Cues;
pub use freq::Freq;
pub use gaps::Gap;
pub use guard::{Guard, Span};
pub use registers::{Coverage, Register, Row};
pub use rewrite::{Change, Rule, Rules};
pub use term::{Band, ClauseWords, Hit, Lexicon, Rarity, Settings, Term};
pub use translate::{Gloss, Legal, Reading, Scan, Stats, TermHit};
