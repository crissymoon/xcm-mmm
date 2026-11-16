//! The normal distribution, in the few places the crate needs it.
//!
//! Three callers want the same two functions and none of them wants a
//! dependency: a deflated selection gate turns a maximum of many trial scores
//! into a probability, an influence function interval turns a standard error
//! into a bound, and a logit transformed AUC interval needs the same quantile.
//! Keeping one implementation means the three agree to the last digit.
//!
//! The two functions have different accuracy and the difference is deliberate.
//! [`normal_quantile`] is Acklam's rational approximation, good to about 1e-9 in
//! the argument, and it is not refined against the CDF because a refinement step
//! would only pull it down to the CDF's accuracy. [`normal_cdf`] is the
//! Abramowitz and Stegun 7.1.26 erf, good to about 1.5e-7 in probability, which
//! is far below the resolution of any number these modules report: a deflated
//! p-value is read at three decimals and a confidence bound at four.

/// Euler-Mascheroni constant, used by the expected maximum of `n` normals.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Error function by Abramowitz and Stegun 7.1.26, with the `t^6` term left out
/// because that series carries five coefficients and not six.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x.signum();
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
            + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal density.
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal quantile by Acklam's rational approximation.
///
/// Returns `-inf` below zero, `+inf` above one and `NaN` only for a `NaN`
/// input, so a caller never receives a silent NaN from a probability.
pub fn normal_quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Central region coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    // Tail coefficients, shared by both tails.
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    // The tail ratio is negative on both sides, so the calls differ only in
    // sign: a small probability needs `x < 0` and the ratio already supplies it.
    let tail = |q: f64| -> f64 {
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    };
    if p < LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= HIGH {
        let q = p - 0.5;
        let r = q * q;
        let num = ((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5];
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num * q / den
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

/// Two sided critical value for a nominal interval level, `0.95` giving 1.96.
pub fn z_for_level(level: f64) -> f64 {
    normal_quantile(0.5 + level.clamp(0.0, 1.0) / 2.0)
}

/// Expected maximum of `n` independent standard normals.
///
/// The approximation is the standard one from multiple testing theory, usable
/// within a few thousandths for `n` above two and reported as zero for a single
/// trial: `(1 - gamma) Phi^-1(1 - 1/n) + gamma Phi^-1(1 - 1/(n e))`. A caller
/// comparing a selected maximum against noise needs this number, not zero.
pub fn expected_max(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n = n as f64;
    (1.0 - EULER_GAMMA) * normal_quantile(1.0 - 1.0 / n)
        + EULER_GAMMA * normal_quantile(1.0 - 1.0 / (n * std::f64::consts::E))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cdf_hits_the_known_points() {
        // The series is not exact at zero: it leaves about 1e-9, which is the
        // published accuracy of the approximation and well under any reported
        // digit in this crate.
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.959_963_984_540_054) - 0.975).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.158_655).abs() < 1e-5);
        assert_eq!(normal_cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(normal_cdf(f64::INFINITY), 1.0);
    }

    #[test]
    fn the_quantile_inverts_the_cdf_to_its_own_accuracy() {
        for x in [-2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5] {
            let round = normal_quantile(normal_cdf(x));
            assert!((round - x).abs() < 1e-4, "{x} round tripped to {round}");
        }
    }

    #[test]
    fn the_quantile_matches_its_reference_values() {
        // Reference values to twelve places, which the rational fit reaches.
        assert!((normal_quantile(0.025) + 1.959_963_984_540).abs() < 1e-6);
        assert!((normal_quantile(0.975) - 1.959_963_984_540).abs() < 1e-6);
        assert!((normal_quantile(0.001) + 3.090_232_306_168).abs() < 1e-5);
        assert!(normal_quantile(0.5).abs() < 1e-12);
    }

    #[test]
    fn the_quantile_edge_cases_are_not_nan() {
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        assert!(normal_quantile(f64::NAN).is_nan());
        assert!(normal_quantile(0.001).is_finite());
        assert!(normal_quantile(0.999).is_finite());
    }

    #[test]
    fn the_expected_maximum_grows_with_the_trial_count() {
        assert_eq!(expected_max(1), 0.0);
        assert!(expected_max(2) > 0.5);
        assert!(expected_max(10) > expected_max(2));
        assert!(expected_max(1_000) > expected_max(10));
        // The textbook value for ten trials is about 1.54.
        assert!((expected_max(10) - 1.54).abs() < 0.05, "{}", expected_max(10));
    }

    #[test]
    fn a_ninety_five_percent_level_gives_the_usual_critical_value() {
        assert!((z_for_level(0.95) - 1.959_964).abs() < 1e-4);
        assert!((z_for_level(0.90) - 1.644_854).abs() < 1e-4);
    }
}
