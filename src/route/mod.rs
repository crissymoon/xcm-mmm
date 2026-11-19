//! Turning a reading into one action: answer, forward, file, hold, ignore.
//!
//! Everything else in this crate answers a question about a text. This module
//! asks what to do about it, and the answer is an action with a cost, not a
//! label with a probability. Three pieces make that possible and each one lives
//! in a file rather than in code:
//!
//! * [`envelope`] reads what the message says about itself, which is the
//!   cheapest and most reliable evidence there is: whether a person wrote it,
//!   whether it went to a list, and whether it has a sender worth answering;
//! * [`policy`] holds the class cues, the cost of every action under every
//!   class, the folder tree, the forward queues and the guard thresholds;
//! * [`decide`] combines them, applies the vetoes and the reject option, and
//!   prints the reason for every line it took.
//!
//! The rule is the expected-cost argmin with an asymmetric loss matrix (Elkan,
//! IJCAI, 2001): the cost of ignoring a demand letter is not the cost of
//! ignoring a newsletter, and encoding that difference is the whole design. The
//! reject option is Chow's (IEEE Transactions on Information Theory, 1970), and
//! it is not decoration: a score that is close to a boundary is held for a
//! person, because a plausible action offered without a gate is over-trusted
//! (Parasuraman and Riley, *Human Factors*, 1997).
//!
//! The guards are not cost terms on purpose. RFC 3834 forbids answering an
//! `Auto-Submitted` message, an empty reverse-path, or a machine mailbox, and no
//! cost matrix should be able to buy its way past a protocol rule. A privileged
//! class must not be forwarded outside the organisation whatever the price, so
//! that is a veto as well.
//!
//! Two things are deliberately left to the caller. Nothing here sends mail:
//! [`decide::reply_draft`] and [`decide::forward_draft`] produce an `eml` body
//! with the headers RFC 3834 requires, and a person or a queue releases it.
//! Nothing here deletes: [`organize`] copies and appends to a ledger, so every
//! filing is additive and the original stays where it landed. A run also cannot
//! honour RFC 3834's seven-day repeat window for the same notice, because that
//! needs history; the ledger is the history, and a service that keeps it can
//! enforce the window when it releases the drafts.
//!
//! The shares are shares of evidence, not probabilities. Nothing in this module
//! has measured whether a share of 0.7 is right seven times in ten, and Guo and
//! others show why a score should not be read that way without calibration
//! (ICML, 2017). The cost matrix carries the behaviour until a labelled routing
//! log exists, and the gate is what keeps an uncalibrated score from acting
//! alone.

pub mod cli;
pub mod decide;
pub mod envelope;
pub mod organize;
pub mod policy;

pub use cli::{RouteArgs, RouteCommand};
pub use decide::{decide, forward_draft, reply_draft, Decision};
pub use envelope::Envelope;
pub use organize::{organize, Summary};
pub use policy::Policy;

/// The folder this module ships, relative to the repository root.
pub const FOLDER: &str = policy::DIR;

#[cfg(test)]
mod tests;
