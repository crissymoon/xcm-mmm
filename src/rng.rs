//! Deterministic pseudo random numbers.
//!
//! The generators here are deliberately small and self contained so that a run
//! reproduces bit for bit on a different machine and a different toolchain.
//! Nothing in this crate depends on the platform random source, and no generator
//! is seeded from the clock.
//!
//! The core is xoshiro256++ seeded through splitmix64. Two streams derived from
//! one seed are independent for every practical purpose and still reproducible,
//! which is what the parallel evaluation paths need.

/// Splitmix64, used only to expand one seed into generator state.
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// xoshiro256++ generator.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    /// Build a generator from a seed.
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let s = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Self { s, spare: None }
    }

    /// Build an independent stream from a base seed and a stream number.
    ///
    /// Stream 0 is the base seed itself, so a serial run and a parallel run over
    /// the same seed agree on the stream each index receives.
    pub fn stream(seed: u64, stream: u64) -> Self {
        if stream == 0 {
            return Self::new(seed);
        }
        let mut mixer = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Self::new(splitmix64(&mut mixer))
    }

    /// Next raw 64 bit word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = rotl(self.s[0].wrapping_add(self.s[3]), 23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = rotl(self.s[3], 45);
        result
    }

    /// Uniform in the half open interval [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform in the open interval (0, 1), never zero.
    #[inline]
    pub fn open_f64(&mut self) -> f64 {
        loop {
            let u = self.next_f64();
            if u > 0.0 {
                return u;
            }
        }
    }

    /// Uniform integer in `0..n`. Panics when `n` is zero.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        let n = n as u64;
        // Lemire's multiply-shift with a rejection zone.
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = (self.next_u64() as u128) * (n as u128);
                low = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Uniform in `lo..hi`.
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// True with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Standard normal from a cached Box-Muller pair.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u1 = self.open_f64();
            let u2 = self.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            let z0 = r * theta.cos();
            let z1 = r * theta.sin();
            if z0.is_finite() && z1.is_finite() {
                self.spare = Some(z1);
                return z0;
            }
        }
    }

    /// Normal with the given location and scale.
    #[inline]
    pub fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.standard_normal()
    }

    /// Cauchy with the given location and scale, from the inverse CDF.
    #[inline]
    pub fn cauchy(&mut self, loc: f64, scale: f64) -> f64 {
        let u = self.next_f64();
        loc + scale * (std::f64::consts::PI * (u - 0.5)).tan()
    }

    /// In place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// Draw `k` distinct indices below `n`, in draw order.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let k = k.min(n);
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn streams_differ() {
        let mut a = Rng::stream(7, 1);
        let mut b = Rng::stream(7, 2);
        let mut same = 0;
        for _ in 0..1000 {
            if a.next_u64() == b.next_u64() {
                same += 1;
            }
        }
        assert!(same < 3, "streams collided {same} times");
    }

    #[test]
    fn uniforms_stay_in_range() {
        let mut rng = Rng::new(11);
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
            let x = rng.uniform(-3.0, 2.0);
            assert!((-3.0..2.0).contains(&x));
            let i = rng.below(17);
            assert!(i < 17);
        }
    }

    #[test]
    fn below_is_roughly_uniform() {
        let mut rng = Rng::new(3);
        let mut counts = [0usize; 5];
        for _ in 0..50_000 {
            counts[rng.below(5)] += 1;
        }
        for c in counts {
            let dev = (c as f64 - 10_000.0).abs();
            assert!(dev < 500.0, "bucket drifted: {counts:?}");
        }
    }

    #[test]
    fn normal_moments_are_sane() {
        let mut rng = Rng::new(5);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let z = rng.normal(1.0, 2.0);
            sum += z;
            sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!((mean - 1.0).abs() < 0.02, "mean {mean}");
        assert!((var - 4.0).abs() < 0.08, "variance {var}");
    }

    #[test]
    fn sample_indices_are_distinct() {
        let mut rng = Rng::new(13);
        for _ in 0..100 {
            let s = rng.sample_indices(30, 10);
            assert_eq!(s.len(), 10);
            let mut sorted = s.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 10);
            assert!(s.iter().all(|&i| i < 30));
        }
    }

    #[test]
    fn shuffle_keeps_the_multiset() {
        let mut rng = Rng::new(17);
        let mut v: Vec<u32> = (0..64).collect();
        rng.shuffle(&mut v);
        v.sort_unstable();
        assert_eq!(v, (0..64).collect::<Vec<_>>());
    }
}
