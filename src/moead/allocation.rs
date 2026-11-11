//! Which subproblems produce an offspring in a generation.
//!
//! Plain MOEA/D-DE updates every subproblem each round. Two partial schedules
//! are supported:
//!
//! * a uniform random subset, the strategy of Lavinas et al. (2020), which
//!   needs no bookkeeping and is the default;
//! * a relative-improvement utility after the dynamic resource allocation of
//!   Zhang, Liu and Li (2009): a subproblem that recently improved its scalar
//!   value earns a higher chance of producing another child, and one that has
//!   been stagnant for a full window is reset to the mean priority so it is
//!   still retried. Lavinas et al. use this as the baseline their random
//!   subset beats, which is why both live here and can be compared directly.
//!
//! Both schedules touch variation only. Replacement is left to the caller: a
//! subproblem that is not selected can still be overwritten by a neighbour's
//! child, and Lavinas et al. name this separation as the reason partial updates
//! help at all.

use crate::rng::Rng;

/// Accumulated relative improvement per subproblem.
#[derive(Clone, Debug)]
pub struct RelativeImprovement {
    /// Generations of stagnation before a subproblem is reset to the mean.
    window: usize,
    /// Priority of each subproblem. Starts at one so the opening generations
    /// are uniform.
    gain: Vec<f64>,
    /// Scalar value of each subproblem at the previous observation.
    previous: Vec<f64>,
    /// Generations since the subproblem last improved its scalar value.
    stale: Vec<usize>,
}

/// Priorities start here, so a run opens with a uniform share.
const INITIAL_GAIN: f64 = 1.0;
/// Floor for a relative improvement denominator, and for the mean priority.
const EPSILON: f64 = 1e-9;

impl RelativeImprovement {
    /// Build the bookkeeping for `n` subproblems with a stagnation `window` in
    /// generations. A window of zero is treated as one.
    pub fn new(n: usize, window: usize) -> Self {
        Self {
            window: window.max(1),
            gain: vec![INITIAL_GAIN; n],
            previous: Vec::with_capacity(n),
            stale: vec![0; n],
        }
    }

    /// Number of subproblems tracked.
    pub fn len(&self) -> usize {
        self.gain.len()
    }

    /// Whether any subproblem is tracked at all.
    pub fn is_empty(&self) -> bool {
        self.gain.is_empty()
    }

    /// Record the scalar value of every subproblem after a round of
    /// replacements, and update the priorities.
    ///
    /// The relative improvement is `(old - new) / |old|`, floored at zero, so a
    /// subproblem at a smaller absolute scale is not favoured or punished by
    /// scale alone. A negative scalar value, which weighted sum can produce, is
    /// handled by the same formula.
    pub fn observe(&mut self, values: &[f64]) {
        debug_assert_eq!(values.len(), self.gain.len());
        if self.previous.len() != values.len() {
            // First observation of a run: no improvement to measure yet.
            self.previous = values.to_vec();
            return;
        }
        let decay = 1.0 - 1.0 / self.window as f64;
        let mean = self.mean_gain();
        for (i, &new) in values.iter().enumerate() {
            let old = self.previous[i];
            let relative = if old.is_finite() {
                (old - new) / old.abs().max(EPSILON)
            } else {
                0.0
            };
            if relative > 0.0 {
                self.gain[i] = self.gain[i] * decay + relative;
                self.stale[i] = 0;
            } else {
                self.stale[i] += 1;
                if self.stale[i] >= self.window {
                    // Stagnant for a whole window: give it the average chance
                    // again, the reset that keeps a decomposed search from
                    // abandoning a subproblem for good.
                    self.gain[i] = self.gain[i].max(mean);
                    self.stale[i] = 0;
                }
            }
        }
        self.previous.copy_from_slice(values);
    }

    /// Draw one round of active subproblems, expecting a `share` of the
    /// population. Each subproblem is drawn independently, with a probability
    /// proportional to its priority, so a prioritised subproblem still misses a
    /// round now and then. At least one subproblem is always returned.
    pub fn select(&self, share: f64, rng: &mut Rng) -> Vec<usize> {
        let scale = share / self.mean_gain().max(EPSILON);
        let mut active = Vec::new();
        for (i, &gain) in self.gain.iter().enumerate() {
            if rng.chance((gain * scale).min(1.0)) {
                active.push(i);
            }
        }
        if active.is_empty() && !self.gain.is_empty() {
            active.push(rng.below(self.gain.len()));
        }
        active
    }

    fn mean_gain(&self) -> f64 {
        if self.gain.is_empty() {
            return INITIAL_GAIN;
        }
        self.gain.iter().sum::<f64>() / self.gain.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_priorities_draw_about_the_requested_share() {
        let state = RelativeImprovement::new(100, 50);
        let mut rng = Rng::new(7);
        let mut total = 0usize;
        for _ in 0..200 {
            total += state.select(0.2, &mut rng).len();
        }
        let mean = total as f64 / 200.0;
        assert!((mean - 20.0).abs() < 3.0, "mean {mean}");
    }

    #[test]
    fn improvement_raises_the_priority_and_stagnation_resets_it() {
        let mut state = RelativeImprovement::new(4, 5);
        state.observe(&[10.0, 10.0, 10.0, 10.0]);
        state.observe(&[5.0, 10.0, 10.0, 10.0]);
        assert!(state.gain[0] > state.gain[1], "{:?}", state.gain);
        // Ten stagnant generations pass the window of five, so the stale
        // subproblems are pulled back up to the mean.
        for _ in 0..10 {
            state.observe(&[5.0, 10.0, 10.0, 10.0]);
        }
        assert!(state.gain[1] >= 1.0, "{:?}", state.gain);
    }

    #[test]
    fn selection_is_never_empty() {
        let mut state = RelativeImprovement::new(8, 5);
        state.observe(&[1.0; 8]);
        state.observe(&[10.0; 8]);
        let mut rng = Rng::new(3);
        for _ in 0..100 {
            assert!(!state.select(0.0, &mut rng).is_empty());
        }
    }
}
