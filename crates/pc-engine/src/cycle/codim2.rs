//! Codim-2 test functions of cycles from their Floquet multipliers: the tests of a cycle
//! branch (SPEC §7) and of the LPC, PD and NS curves (SPEC §8). A curve leaves out the
//! trivial multiplier and its own critical multipliers, so a zero comes from one more
//! multiplier, or pair, on the unit circle.
//!
//! On an NS curve with multipliers exp(+-i theta), kappa = cos theta, and the strong
//! resonances 1:1, 1:2, 1:3 and 1:4 are theta = 0, pi, 2 pi/3 and pi/2, so R1 = kappa - 1,
//! R2 = kappa + 1, R3 = kappa + 1/2 and R4 = kappa.

/// Names of the codim-2 tests on an LPC curve.
pub const LPC_CURVE_LABELS: [&str; 4] = ["LPPD", "LPNS", "CPC", "R1"];
/// Names of the codim-2 tests on a PD curve.
pub const PD_CURVE_LABELS: [&str; 4] = ["LPPD", "PDNS", "GPD", "R2"];
/// Names of the codim-2 tests on an NS curve.
pub const NS_CURVE_LABELS: [&str; 7] = ["R1", "R2", "R3", "R4", "LPNS", "PDNS", "CH"];

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Removes the multiplier closest to z, if there is one.
fn remove_nearest(mu: &mut Vec<(f64, f64)>, z: (f64, f64)) {
    let dist = |m: &(f64, f64)| (m.0 - z.0).hypot(m.1 - z.1);
    if let Some(k) = (0..mu.len()).min_by(|&a, &b| dist(&mu[a]).total_cmp(&dist(&mu[b]))) {
        mu.remove(k);
    }
}

/// The multipliers without the one closest to 1, the trivial multiplier.
pub(super) fn nontrivial(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut mu = eigs.to_vec();
    remove_nearest(&mut mu, (1.0, 0.0));
    mu
}

/// The real part of the product of mu + s over `mu`, 1 over no multipliers.
pub(super) fn shifted_product(mu: &[(f64, f64)], s: f64) -> f64 {
    mu.iter().fold((1.0, 0.0), |p, m| cmul(p, (m.0 + s, m.1))).0
}

/// The real part of the product of mu_i mu_j - 1 over i < j, NaN for fewer than two
/// multipliers.
pub(super) fn pair_product(mu: &[(f64, f64)]) -> f64 {
    if mu.len() < 2 {
        return f64::NAN;
    }
    let mut prod = (1.0, 0.0);
    for i in 0..mu.len() {
        for j in i + 1..mu.len() {
            let q = cmul(mu[i], mu[j]);
            prod = cmul(prod, (q.0 - 1.0, q.1));
        }
    }
    prod.0
}

/// Whether the pair of `mu` whose product is closest to 1 is a complex pair. The pair
/// product also vanishes when two real multipliers have product 1, a neutral saddle cycle,
/// and only a complex pair makes its zero an NS point.
pub(super) fn nearest_pair_is_complex(mu: &[(f64, f64)]) -> bool {
    let mut best: Option<(f64, usize, usize)> = None;
    for a in 0..mu.len() {
        for b in a + 1..mu.len() {
            let q = cmul(mu[a], mu[b]);
            let d = (q.0 - 1.0).hypot(q.1);
            if best.is_none_or(|(e, _, _)| d < e) {
                best = Some((d, a, b));
            }
        }
    }
    let Some((_, a, b)) = best else {
        return false;
    };
    let (p, q) = (mu[a], mu[b]);
    let tol = 1e-10 * p.0.hypot(p.1).max(1.0);
    p.1 != 0.0 && (p.0 - q.0).abs() <= tol && (p.1 + q.1).abs() <= tol
}

/// The two roots of mu^2 - 2 kappa mu + 1, exp(+-i theta) for |kappa| <= 1 and the real
/// pair mu, 1/mu otherwise.
fn kappa_roots(kappa: f64) -> [(f64, f64); 2] {
    let d = kappa * kappa - 1.0;
    if d <= 0.0 {
        let s = (-d).sqrt();
        [(kappa, s), (kappa, -s)]
    } else {
        let s = d.sqrt();
        [(kappa + s, 0.0), (kappa - s, 0.0)]
    }
}

/// The multipliers without the trivial one and the second multiplier 1 of the fold.
fn lpc_rest(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut mu = nontrivial(eigs);
    remove_nearest(&mut mu, (1.0, 0.0));
    mu
}

/// The multipliers without the trivial one and the multiplier -1 of the flip.
fn pd_rest(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut mu = nontrivial(eigs);
    remove_nearest(&mut mu, (-1.0, 0.0));
    mu
}

/// The multipliers without the trivial one and the pair of the roots of
/// mu^2 - 2 kappa mu + 1.
fn ns_rest(kappa: f64, eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut mu = nontrivial(eigs);
    for r in kappa_roots(kappa) {
        remove_nearest(&mut mu, r);
    }
    mu
}

/// The real part of the product of mu + s over `mu`, NaN over no multipliers.
fn rest_product(mu: &[(f64, f64)], s: f64) -> f64 {
    if mu.is_empty() {
        f64::NAN
    } else {
        shifted_product(mu, s)
    }
}

/// LPPD = prod(mu + 1), LPNS = prod(mu_i mu_j - 1), the coefficient b of the fold (CPC)
/// and R1 = prod(mu - 1) over the multipliers of an LPC point other than the double
/// multiplier 1. The products are NaN without multipliers to test, and `coeff` is NaN
/// where the normal form is not defined.
pub(super) fn lpc_curve_tests(eigs: &[(f64, f64)], coeff: f64) -> Vec<f64> {
    let mu = lpc_rest(eigs);
    vec![
        rest_product(&mu, 1.0),
        pair_product(&mu),
        coeff,
        rest_product(&mu, -1.0),
    ]
}

/// The label of a zero of LPC curve test `i`. A zero of LPNS from a real pair of product
/// 1 is a neutral saddle, so it gets none.
pub(super) fn lpc_curve_label(i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
    match i {
        1 => nearest_pair_is_complex(&lpc_rest(eigs)).then_some("LPNS"),
        _ => LPC_CURVE_LABELS.get(i).copied(),
    }
}

/// LPPD = prod(mu - 1), PDNS = prod(mu_i mu_j - 1), the coefficient c of the flip (GPD)
/// and R2 = prod(mu + 1) over the multipliers of a PD point other than 1 and -1. The
/// products are NaN without multipliers to test, and `coeff` is NaN where the normal form
/// is not defined.
pub(super) fn pd_curve_tests(eigs: &[(f64, f64)], coeff: f64) -> Vec<f64> {
    let mu = pd_rest(eigs);
    vec![
        rest_product(&mu, -1.0),
        pair_product(&mu),
        coeff,
        rest_product(&mu, 1.0),
    ]
}

/// The label of a zero of PD curve test `i`. A zero of PDNS from a real pair of product 1
/// is a neutral saddle, so it gets none.
pub(super) fn pd_curve_label(i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
    match i {
        1 => nearest_pair_is_complex(&pd_rest(eigs)).then_some("PDNS"),
        _ => PD_CURVE_LABELS.get(i).copied(),
    }
}

/// R1 to R4 from kappa, then LPNS = prod(mu - 1) and PDNS = prod(mu + 1) over the
/// multipliers of an NS point other than 1 and the roots of mu^2 - 2 kappa mu + 1, then
/// Re d of the torus normal form (CH), which is `coeff` and NaN where the normal form is
/// not defined. LPNS and PDNS are NaN without multipliers to test.
pub(super) fn ns_curve_tests(kappa: f64, eigs: &[(f64, f64)], coeff: f64) -> Vec<f64> {
    let mu = ns_rest(kappa, eigs);
    vec![
        kappa - 1.0,
        kappa + 1.0,
        kappa + 0.5,
        kappa,
        rest_product(&mu, -1.0),
        rest_product(&mu, 1.0),
        coeff,
    ]
}

/// The label of a zero of NS curve test `i`. Each zero is the bifurcation it names.
pub(super) fn ns_curve_label(i: usize) -> Option<&'static str> {
    NS_CURVE_LABELS.get(i).copied()
}

/// For |kappa| > 1, the largest distance of the roots mu, 1/mu of mu^2 - 2 kappa mu + 1
/// from the nontrivial multipliers, relative to max(|root|, 1). The NS equations hold
/// when M has one of the two, so a neutral saddle needs both. `None` for |kappa| <= 1 or
/// without multipliers.
pub(super) fn neutral_saddle_gap(kappa: f64, eigs: &[(f64, f64)]) -> Option<f64> {
    if kappa.abs() <= 1.0 || eigs.is_empty() {
        return None;
    }
    let mut mu = nontrivial(eigs);
    let mut worst = 0.0_f64;
    for r in kappa_roots(kappa) {
        let dist = |m: &(f64, f64)| (m.0 - r.0).hypot(m.1 - r.1);
        let k = (0..mu.len()).min_by(|&a, &b| dist(&mu[a]).total_cmp(&dist(&mu[b])));
        let d = match k {
            Some(k) => dist(&mu.remove(k)) / r.0.abs().max(1.0),
            None => f64::INFINITY,
        };
        worst = if worst.is_nan() || d.is_nan() {
            f64::NAN
        } else {
            worst.max(d)
        };
    }
    Some(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polar(r: f64, th: f64) -> (f64, f64) {
        (r * th.cos(), r * th.sin())
    }

    #[test]
    fn lpc_curve_tests_skip_the_double_multiplier_1() {
        // 1 twice (split by rounding), then -0.5 and the pair 0.8 exp(+-0.3 i).
        let eigs = [
            (1.0 + 1e-9, 0.0),
            (1.0 - 1e-9, 0.0),
            (-0.5, 0.0),
            polar(0.8, 0.3),
            polar(0.8, -0.3),
        ];
        let t = lpc_curve_tests(&eigs, f64::NAN);
        let pair = |a: (f64, f64), b: (f64, f64)| {
            let q = cmul(a, b);
            (q.0 - 1.0, q.1)
        };
        let mu = &eigs[2..];
        let lpns = cmul(
            cmul(pair(mu[0], mu[1]), pair(mu[0], mu[2])),
            pair(mu[1], mu[2]),
        );
        let lppd = 0.5 * ((0.8 * 0.3_f64.cos() + 1.0).powi(2) + (0.8 * 0.3_f64.sin()).powi(2));
        assert!((t[0] - lppd).abs() < 1e-14, "{t:?}");
        assert!((t[1] - lpns.0).abs() < 1e-14, "{t:?}");
        // The double multiplier of the fold may come out as a complex pair.
        let split = [
            (1.0, 1e-6),
            (1.0, -1e-6),
            (-0.5, 0.0),
            polar(0.8, 0.3),
            polar(0.8, -0.3),
        ];
        let s = lpc_curve_tests(&split, f64::NAN);
        assert!((s[0] - lppd).abs() < 1e-14 && (s[1] - lpns.0).abs() < 1e-14);
    }

    #[test]
    fn lpc_curve_zeros_get_their_labels() {
        let lppd = [(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.3, 0.0)];
        assert_eq!(lpc_curve_tests(&lppd, f64::NAN)[0], 0.0);
        assert_eq!(lpc_curve_label(0, &lppd), Some("LPPD"));
        let lpns = [(1.0, 0.0), (1.0, 0.0), polar(1.0, 1.1), polar(1.0, -1.1)];
        assert!(lpc_curve_tests(&lpns, f64::NAN)[1].abs() < 1e-15);
        assert_eq!(lpc_curve_label(1, &lpns), Some("LPNS"));
    }

    #[test]
    fn the_curve_tests_carry_the_coefficient_and_the_resonance_product() {
        // LPC: R1 = prod(mu - 1) over the rest, and the coefficient in slot 2.
        let lpc = [(1.0, 0.0), (1.0, 0.0), (0.4, 0.0), (-0.7, 0.0)];
        let t = lpc_curve_tests(&lpc, -4.0);
        assert_eq!(t.len(), LPC_CURVE_LABELS.len());
        assert_eq!(t[2], -4.0);
        assert!((t[3] - (0.4 - 1.0) * (-0.7 - 1.0)).abs() < 1e-15, "{t:?}");
        assert_eq!(lpc_curve_label(2, &lpc), Some("CPC"));
        assert_eq!(lpc_curve_label(3, &lpc), Some("R1"));
        // Planar: nothing left, so R1 is NaN and the coefficient is kept.
        let planar = lpc_curve_tests(&[(1.0, 0.0), (1.0, 0.0)], 2.5);
        assert!(planar[3].is_nan() && planar[2] == 2.5, "{planar:?}");
        // PD: R2 = prod(mu + 1) over the rest, and the coefficient in slot 2.
        let pd = [(1.0, 0.0), (-1.0, 0.0), (0.4, 0.0), (-0.7, 0.0)];
        let t = pd_curve_tests(&pd, 0.3);
        assert_eq!(t.len(), PD_CURVE_LABELS.len());
        assert_eq!(t[2], 0.3);
        assert!((t[3] - (0.4 + 1.0) * (-0.7 + 1.0)).abs() < 1e-15, "{t:?}");
        assert_eq!(pd_curve_label(2, &pd), Some("GPD"));
        assert_eq!(pd_curve_label(3, &pd), Some("R2"));
        let planar = pd_curve_tests(&[(1.0, 0.0), (-1.0, 0.0)], -1.5);
        assert!(planar[3].is_nan() && planar[2] == -1.5, "{planar:?}");
        // NS: Re d in the last slot, with its label.
        let ns = ns_curve_tests(0.3, &[(1.0, 0.0)], 7.0);
        assert_eq!(ns.len(), NS_CURVE_LABELS.len());
        assert_eq!(ns[6], 7.0);
        assert_eq!(ns_curve_label(6), Some("CH"));
    }

    #[test]
    fn lpc_curve_drops_a_neutral_saddle_zero() {
        let saddle = [(1.0, 0.0), (1.0, 0.0), (4.0, 0.0), (0.25, 0.0)];
        assert_eq!(lpc_curve_tests(&saddle, f64::NAN)[1], 0.0);
        assert_eq!(lpc_curve_label(1, &saddle), None);
    }

    #[test]
    fn pd_curve_tests_skip_the_trivial_and_the_flip_multiplier() {
        let eigs = [
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.5, 0.0),
            polar(0.9, 2.0),
            polar(0.9, -2.0),
        ];
        let t = pd_curve_tests(&eigs, f64::NAN);
        let c = 0.9 * 2.0_f64.cos();
        let lppd = -0.5 * ((c - 1.0).powi(2) + (0.9 * 2.0_f64.sin()).powi(2));
        assert!((t[0] - lppd).abs() < 1e-14, "{t:?}");
        assert!(t[1].is_finite());
        let single = [(1.0, 0.0), (-1.0, 0.0), (0.2, 0.0)];
        let s = pd_curve_tests(&single, f64::NAN);
        assert!((s[0] - (0.2 - 1.0)).abs() < 1e-15 && s[1].is_nan(), "{s:?}");
        let planar = pd_curve_tests(&[(1.0, 0.0), (-1.0, 0.0)], f64::NAN);
        assert!(planar.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn pd_curve_zeros_get_their_labels_and_a_neutral_saddle_none() {
        let lppd = [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.3, 0.0)];
        assert_eq!(pd_curve_tests(&lppd, f64::NAN)[0], 0.0);
        assert_eq!(pd_curve_label(0, &lppd), Some("LPPD"));
        let pdns = [(1.0, 0.0), (-1.0, 0.0), polar(1.0, 0.7), polar(1.0, -0.7)];
        assert!(pd_curve_tests(&pdns, f64::NAN)[1].abs() < 1e-15);
        assert_eq!(pd_curve_label(1, &pdns), Some("PDNS"));
        let saddle = [(1.0, 0.0), (-1.0, 0.0), (-3.0, 0.0), (-1.0 / 3.0, 0.0)];
        assert!(pd_curve_tests(&saddle, f64::NAN)[1].abs() < 1e-15);
        assert_eq!(pd_curve_label(1, &saddle), None);
    }

    #[test]
    fn ns_curve_tests_follow_kappa_and_skip_the_critical_pair() {
        let th: f64 = 1.2;
        let kappa = th.cos();
        let eigs = [
            (1.0, 0.0),
            polar(1.0, th),
            polar(1.0, -th),
            (0.4, 0.0),
            (-0.7, 0.0),
        ];
        let t = ns_curve_tests(kappa, &eigs, f64::NAN);
        let want = [
            kappa - 1.0,
            kappa + 1.0,
            kappa + 0.5,
            kappa,
            (0.4 - 1.0) * (-0.7 - 1.0),
            (0.4 + 1.0) * (-0.7 + 1.0),
        ];
        for (a, b) in t.iter().zip(want) {
            assert!((a - b).abs() < 1e-14, "{t:?}");
        }
        for (i, l) in NS_CURVE_LABELS.iter().enumerate() {
            assert_eq!(ns_curve_label(i), Some(*l));
        }
        // On a neutral saddle the pair mu, 1/mu goes.
        let saddle = [(1.0, 0.0), (2.0, 0.0), (0.5, 0.0), (0.3, 0.0)];
        let s = ns_curve_tests(1.25, &saddle, f64::NAN);
        assert!((s[4] - (0.3 - 1.0)).abs() < 1e-15 && (s[5] - 1.3).abs() < 1e-15);
        // Without other multipliers LPNS and PDNS are undefined; R1 to R4 are not.
        let bare = ns_curve_tests(kappa, &eigs[..3], f64::NAN);
        assert!(bare[..4].iter().all(|v| v.is_finite()));
        assert!(bare[4].is_nan() && bare[5].is_nan());
        let none = ns_curve_tests(kappa, &[], f64::NAN);
        assert!(none[3] == kappa && none[4].is_nan());
    }

    #[test]
    fn ns_curve_resonances_sit_at_their_angles() {
        use std::f64::consts::PI;
        for (i, th) in [(0, 0.0), (1, PI), (2, 2.0 * PI / 3.0), (3, PI / 2.0)] {
            let t = ns_curve_tests(f64::cos(th), &[], f64::NAN);
            assert!(t[i].abs() < 1e-15, "R{}: {}", i + 1, t[i]);
        }
    }

    #[test]
    fn a_neutral_saddle_needs_both_roots() {
        let pair = [(1.0, 0.0), (2.0, 0.0), (0.5 + 1e-9, 0.0), (0.3, 0.0)];
        assert!(neutral_saddle_gap(1.25, &pair).unwrap() < 1e-8);
        // Only mu = 2 is a multiplier, so the NS equations hold without a saddle.
        let one = [(1.0, 0.0), (2.0, 0.0), (0.3, 0.0)];
        assert!((neutral_saddle_gap(1.25, &one).unwrap() - 0.2).abs() < 1e-15);
        assert_eq!(neutral_saddle_gap(0.5, &one), None);
        assert_eq!(neutral_saddle_gap(1.25, &[]), None);
        assert_eq!(
            neutral_saddle_gap(1.25, &[(1.0, 0.0), (2.0, 0.0)]),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn cycle_branch_helpers_keep_their_values() {
        let mu = [polar(0.9, 0.4), polar(0.9, -0.4), (-0.3, 0.0)];
        let pd = cmul(
            cmul((mu[0].0 + 1.0, mu[0].1), (mu[1].0 + 1.0, mu[1].1)),
            (mu[2].0 + 1.0, mu[2].1),
        );
        assert_eq!(shifted_product(&mu, 1.0), pd.0);
        assert_eq!(shifted_product(&[], 1.0), 1.0);
        assert!(pair_product(&mu[..1]).is_nan());
        assert!(nearest_pair_is_complex(&[
            polar(1.0, 0.4),
            polar(1.0, -0.4),
            (0.2, 0.0)
        ]));
        assert!(!nearest_pair_is_complex(&[
            (2.0, 0.0),
            (0.5, 0.0),
            (0.2, 0.0)
        ]));
        assert!(!nearest_pair_is_complex(&[(0.5, 0.0)]));
    }
}
