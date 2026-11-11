//! The success-history parameter memory from SHADE.
//!
//! Each generation records the scale factor and crossover rate of every trial
//! that replaced its parent, then folds them into one slot of a small ring
//! buffer. Sampling a slot rather than the whole history keeps the search from
//! collapsing onto one setting too early.

/// Size of the memory, the value used by SHADE and jSO.
pub const DEFAULT_HISTORY: usize = 6;

/// Ring buffer of successful `F` and `CR` values.
#[derive(Clone, Debug)]
pub struct ParameterMemory {
    memory_f: Vec<f64>,
    memory_cr: Vec<f64>,
    success_f: Vec<f64>,
    success_cr: Vec<f64>,
    weights: Vec<f64>,
    next: usize,
}

impl ParameterMemory {
    /// A memory of `size` slots, all starting at 0.5.
    pub fn new(size: usize) -> Self {
        let size = size.max(1);
        Self {
            memory_f: vec![0.5; size],
            memory_cr: vec![0.5; size],
            success_f: Vec::with_capacity(size),
            success_cr: Vec::with_capacity(size),
            weights: Vec::with_capacity(size),
            next: 0,
        }
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.memory_f.len()
    }

    /// True when the memory holds no slots, which cannot happen through `new`.
    pub fn is_empty(&self) -> bool {
        self.memory_f.is_empty()
    }

    /// Mean scale factor in slot `i`.
    pub fn mean_f(&self, i: usize) -> f64 {
        self.memory_f[i % self.len()]
    }

    /// Mean crossover rate in slot `i`.
    pub fn mean_cr(&self, i: usize) -> f64 {
        self.memory_cr[i % self.len()]
    }

    /// Record one successful parameter pair with its improvement weight.
    pub fn record(&mut self, f: f64, cr: f64, weight: f64) {
        if f.is_finite() && f > 0.0 {
            self.success_f.push(f);
        }
        if cr.is_finite() {
            self.success_cr.push(cr.clamp(0.0, 1.0));
        }
        self.weights.push(weight.max(0.0));
    }

    /// Fold the recorded successes into the next slot and clear the buffers.
    ///
    /// Returns true when there was anything to fold.
    pub fn commit(&mut self) -> bool {
        if self.success_f.is_empty() {
            return false;
        }
        let total: f64 = self.weights.iter().sum();
        let weights: Vec<f64> = if total > 0.0 {
            self.weights.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / self.weights.len() as f64; self.weights.len()]
        };

        // F follows the weighted Lehmer mean, which leans towards the larger
        // successful values and keeps progress on flat ridges.
        let numerator: f64 = self
            .success_f
            .iter()
            .zip(&weights)
            .map(|(f, w)| w * f * f)
            .sum();
        let denominator: f64 = self
            .success_f
            .iter()
            .zip(&weights)
            .map(|(f, w)| w * f)
            .sum();
        let mean_f = if denominator > 1e-12 {
            (numerator / denominator).clamp(0.01, 1.0)
        } else {
            self.success_f.iter().sum::<f64>() / self.success_f.len() as f64
        };

        // CR follows the weighted arithmetic mean. The original SHADE paper
        // uses the arithmetic form here, and it is steadier than Lehmer when
        // most successes carry a small weight.
        let mean_cr: f64 = self
            .success_cr
            .iter()
            .zip(&weights)
            .map(|(cr, w)| w * cr)
            .sum();

        let slot = self.next % self.len();
        self.memory_f[slot] = mean_f.clamp(0.01, 1.0);
        self.memory_cr[slot] = mean_cr.clamp(0.0, 1.0);
        self.next = (self.next + 1) % self.len();
        self.success_f.clear();
        self.success_cr.clear();
        self.weights.clear();
        true
    }

    /// Return every slot to 0.5.
    pub fn reset(&mut self) {
        self.memory_f.fill(0.5);
        self.memory_cr.fill(0.5);
        self.success_f.clear();
        self.success_cr.clear();
        self.weights.clear();
        self.next = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_memory_starts_at_one_half() {
        let m = ParameterMemory::new(DEFAULT_HISTORY);
        assert_eq!(m.len(), DEFAULT_HISTORY);
        for i in 0..m.len() {
            assert_eq!(m.mean_f(i), 0.5);
            assert_eq!(m.mean_cr(i), 0.5);
        }
    }

    #[test]
    fn commit_moves_the_ring_forward() {
        let mut m = ParameterMemory::new(3);
        m.record(0.8, 0.2, 1.0);
        assert!(m.commit());
        assert!((m.mean_f(0) - 0.8).abs() < 1e-12);
        m.record(0.4, 0.9, 2.0);
        assert!(m.commit());
        assert!((m.mean_f(1) - 0.4).abs() < 1e-12);
        assert!((m.mean_f(0) - 0.8).abs() < 1e-12, "slot 0 must not shift");
        assert!(!m.commit(), "nothing recorded since the last commit");
    }

    #[test]
    fn lehmer_mean_leans_high() {
        let mut m = ParameterMemory::new(6);
        m.record(0.2, 0.5, 1.0);
        m.record(0.9, 0.5, 1.0);
        m.commit();
        // (0.04 + 0.81) / (0.2 + 0.9) = 0.7727
        assert!((m.mean_f(0) - 0.772_727).abs() < 1e-4);
    }

    #[test]
    fn zero_weights_fall_back_to_the_plain_mean() {
        let mut m = ParameterMemory::new(2);
        m.record(0.3, 0.3, 0.0);
        m.record(0.7, 0.7, 0.0);
        m.commit();
        // Equal weights still go through the Lehmer mean for F, which is
        // (0.09 + 0.49) / (0.3 + 0.7) = 0.58, while CR stays arithmetic.
        assert!((m.mean_f(0) - 0.58).abs() < 1e-12);
        assert!((m.mean_cr(0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reset_clears_everything() {
        let mut m = ParameterMemory::new(4);
        m.record(0.9, 0.9, 1.0);
        m.commit();
        m.reset();
        assert_eq!(m.mean_f(0), 0.5);
        assert!(!m.commit());
    }
}
