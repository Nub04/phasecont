//! Characteristic roots of linear delay equations (SPEC section 8c, tests 3 to 8).
//!
//! The completeness tests count the roots with an argument-principle winding number of
//! `det Delta` along a rectangle, computed here with a complex LU of the test's own.

use pc_engine::{CharRoot, EngineError, LinearDde, RootOptions, RootSet, characteristic_roots};
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
struct C(f64, f64);

impl Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
}
impl Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C(self.0 - o.0, self.1 - o.1)
    }
}
impl Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}
impl Div for C {
    type Output = C;
    fn div(self, o: C) -> C {
        let q = o.0 * o.0 + o.1 * o.1;
        C(
            (self.0 * o.0 + self.1 * o.1) / q,
            (self.1 * o.0 - self.0 * o.1) / q,
        )
    }
}
impl C {
    fn abs(self) -> f64 {
        self.0.hypot(self.1)
    }
    fn exp_neg(self, tau: f64) -> C {
        let m = (-self.0 * tau).exp();
        C(m * (self.1 * tau).cos(), -m * (self.1 * tau).sin())
    }
}

fn dde(dim: usize, a0: &[f64], delays: &[(f64, &[f64])]) -> LinearDde {
    LinearDde {
        dim,
        a0: a0.to_vec(),
        delays: delays.iter().map(|(t, a)| (*t, a.to_vec())).collect(),
    }
}

fn run(p: &LinearDde, n: usize, re_min: f64) -> RootSet {
    characteristic_roots(p, &RootOptions { n, re_min }).unwrap()
}

fn fro(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `det Delta(lambda)` by Gaussian elimination with partial pivoting.
fn det_delta(p: &LinearDde, lam: C) -> C {
    let d = p.dim;
    let mut m = vec![C(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            let mut z = C(-p.a0[r * d + c], 0.0);
            if r == c {
                z = z + lam;
            }
            for (tau, a) in &p.delays {
                let e = lam.exp_neg(*tau);
                z = z - C(a[r * d + c] * e.0, a[r * d + c] * e.1);
            }
            m[r * d + c] = z;
        }
    }
    let mut det = C(1.0, 0.0);
    for k in 0..d {
        let piv = (k..d)
            .max_by(|&i, &j| m[i * d + k].abs().total_cmp(&m[j * d + k].abs()))
            .unwrap();
        if piv != k {
            for c in 0..d {
                m.swap(k * d + c, piv * d + c);
            }
            det = C(-det.0, -det.1);
        }
        let pivot = m[k * d + k];
        det = det * pivot;
        for i in k + 1..d {
            let f = m[i * d + k] / pivot;
            for c in k..d {
                m[i * d + c] = m[i * d + c] - f * m[k * d + c];
            }
        }
    }
    det
}

/// Number of zeros of `det Delta` in `[r, B] x [-B, B]`, with `B` from the root bound
/// (Frobenius norms bound the spectral norm, so the bound holds), counted as the winding
/// number of `det Delta` along the boundary.
fn zeros_in_rectangle(p: &LinearDde, r: f64) -> usize {
    let b = fro(&p.a0)
        + p.delays
            .iter()
            .map(|(t, a)| fro(a) * (-r * t).exp())
            .sum::<f64>()
        + 1.0;
    let corners = [(r, -b), (b, -b), (b, b), (r, b), (r, -b)];
    let mut total = 0.0;
    for w in corners.windows(2) {
        let len = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
        let at = |t: f64| {
            C(
                w[0].0 + t * (w[1].0 - w[0].0),
                w[0].1 + t * (w[1].1 - w[0].1),
            )
        };
        let det_at = |t: f64| {
            let z = det_delta(p, at(t));
            assert!(z.abs() > 1e-8, "det Delta is near 0 on the contour");
            z
        };
        // A uniform grid first, so that a whole number of turns between two samples cannot
        // pass unseen, then bisection until every phase increment is below pi/4.
        let m = (len / 0.01).ceil() as usize;
        let mut pts: Vec<(f64, C)> = (0..=m)
            .map(|i| {
                let t = i as f64 / m as f64;
                (t, det_at(t))
            })
            .collect();
        let mut i = 0;
        let mut inserted = 0;
        while i + 1 < pts.len() {
            let q = pts[i + 1].1 / pts[i].1;
            if q.1.atan2(q.0).abs() >= PI / 4.0 {
                inserted += 1;
                assert!(inserted < 200_000, "contour sampling did not converge");
                let tm = 0.5 * (pts[i].0 + pts[i + 1].0);
                pts.insert(i + 1, (tm, det_at(tm)));
            } else {
                i += 1;
            }
        }
        for pair in pts.windows(2) {
            let q = pair[1].1 / pair[0].1;
            total += q.1.atan2(q.0);
        }
    }
    let k = total / (2.0 * PI);
    assert!(
        (k - k.round()).abs() < 0.05 && k.round() >= 0.0,
        "winding number {k} is not a count"
    );
    k.round() as usize
}

/// `|Delta v| / (|v| s)` is not available outside the engine, so the tests with a closed
/// form check the characteristic equation itself, scaled by the size of its terms.
fn scalar_residual(a0: f64, delays: &[(f64, f64)], re: f64, im: f64) -> f64 {
    let lam = C(re, im);
    let mut z = lam - C(a0, 0.0);
    let mut s = lam.abs() + a0.abs();
    for (tau, a) in delays {
        let e = lam.exp_neg(*tau);
        z = z - C(a * e.0, a * e.1);
        s += a.abs() * (-re * tau).exp();
    }
    z.abs() / s
}

fn all_residuals_small(set: &RootSet) {
    for r in &set.roots {
        assert!(r.residual <= 1e-12, "residual {:e} at {r:?}", r.residual);
    }
}

/// Prints the root count and the largest residual, for `-- --nocapture`.
fn report(name: &str, set: &RootSet, winding: usize) {
    let worst = set.roots.iter().fold(0.0_f64, |m, r| m.max(r.residual));
    eprintln!(
        "{name}: n={} roots={} winding={winding} unrefined={} max residual={worst:e}",
        set.n,
        set.roots.len(),
        set.unrefined.len()
    );
}

fn sorted_ok(roots: &[CharRoot]) {
    for w in roots.windows(2) {
        assert!(
            w[0].re > w[1].re || (w[0].re == w[1].re && w[0].im >= w[1].im),
            "roots are not sorted: {:?} then {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test3_neutral_pair_on_the_imaginary_axis() {
    let p = dde(1, &[0.0], &[(1.0, &[-PI / 2.0])]);
    let set = run(&p, 20, -3.0);
    all_residuals_small(&set);
    sorted_ok(&set.roots);
    assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
    assert!(set.roots[0].re.abs() <= 1e-12, "{:?}", set.roots[0]);
    for sign in [1.0, -1.0] {
        let hit = set
            .roots
            .iter()
            .any(|r| r.re.abs() <= 1e-12 && (r.im - sign * PI / 2.0).abs() <= 1e-12);
        assert!(hit, "no root near {}i pi/2", sign);
    }
    let top: Vec<&CharRoot> = set.roots.iter().filter(|r| r.re > -1e-6).collect();
    assert_eq!(top.len(), 2, "{top:?}");
}

#[test]
fn test4_omega_constant_and_a_small_negative_coefficient() {
    let p = dde(1, &[0.0], &[(1.0, &[1.0])]);
    let set = run(&p, 24, -3.0);
    all_residuals_small(&set);
    let omega = 0.567_143_290_409_783_8;
    assert_eq!(set.roots[0].im, 0.0);
    assert!(
        (set.roots[0].re - omega).abs() <= 1e-14,
        "{:?} vs {omega}",
        set.roots[0]
    );
    let p = dde(1, &[0.0], &[(1.0, &[-0.1])]);
    let set = run(&p, 24, -3.0);
    all_residuals_small(&set);
    let top = set.roots[0];
    assert_eq!(top.im, 0.0);
    assert!(top.re < 0.0 && top.re > -0.2, "{top:?}");
    assert!(scalar_residual(0.0, &[(1.0, -0.1)], top.re, top.im) <= 1e-14);
}

fn rotation(a: f64, b: f64) -> [f64; 4] {
    [a, -b, b, a]
}

fn check_complex_scalar(set: &RootSet, mu0: C, mu1: C, tau: f64) {
    for r in &set.roots {
        let lam = C(r.re, r.im);
        let e = lam.exp_neg(tau);
        let s = lam.abs() + mu0.abs() + mu1.abs() * (-r.re * tau).exp();
        let direct = (lam - mu0 - mu1 * e).abs() / s;
        let conj = (lam - C(mu0.0, -mu0.1) - C(mu1.0, -mu1.1) * e).abs() / s;
        assert!(
            direct.min(conj) <= 1e-12,
            "{r:?} fails both scalar equations: {direct:e}, {conj:e}"
        );
    }
}

#[test]
fn test5_rotation_systems_match_the_complex_scalar_equation() {
    let (a, b, tau) = (0.4, 0.9, 0.7);
    let a1 = rotation(a, b);
    let p = dde(2, &[0.0; 4], &[(tau, &a1)]);
    let r = -3.0;
    let set = run(&p, 40, r);
    all_residuals_small(&set);
    assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
    check_complex_scalar(&set, C(0.0, 0.0), C(a, b), tau);
    let w = zeros_in_rectangle(&p, r);
    report("test5 rotation", &set, w);
    assert_eq!(set.roots.len(), w);

    let (pp, q) = (-0.5, 0.3);
    let a0 = rotation(pp, q);
    let p = dde(2, &a0, &[(tau, &a1)]);
    let set = run(&p, 40, r);
    all_residuals_small(&set);
    assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
    check_complex_scalar(&set, C(pp, q), C(a, b), tau);
    let w = zeros_in_rectangle(&p, r);
    report("test5 commuting", &set, w);
    assert_eq!(set.roots.len(), w);
}

#[test]
fn test6_two_delays_with_a_node_and_an_off_node_delay() {
    let (a, b, t1) = (-0.8, 0.5, 0.5);
    let p = dde(1, &[0.0], &[(t1, &[a]), (2.0 * t1, &[b])]);
    let r = -4.0;
    let s20 = run(&p, 20, r);
    let s21 = run(&p, 21, r);
    for set in [&s20, &s21] {
        all_residuals_small(set);
        assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
        sorted_ok(&set.roots);
        for root in &set.roots {
            let res = scalar_residual(0.0, &[(t1, a), (2.0 * t1, b)], root.re, root.im);
            assert!(res <= 1e-12, "{root:?}: {res:e}");
        }
    }
    assert_eq!(s20.roots.len(), s21.roots.len());
    for (x, y) in s20.roots.iter().zip(&s21.roots) {
        assert!(
            (x.re - y.re).abs() <= 1e-11 && (x.im - y.im).abs() <= 1e-11,
            "{x:?} vs {y:?}"
        );
    }
    let w = zeros_in_rectangle(&p, r);
    report("test6 n=20", &s20, w);
    report("test6 n=21", &s21, w);
    assert_eq!(s20.roots.len(), w);
}

#[test]
fn test7_completeness_for_noncommuting_systems() {
    let a0 = [0.5, 1.0, -1.0, -0.3];
    let a1 = [-1.2, 0.4, 0.7, -0.8];
    let a2 = [0.2, -0.5, 0.3, 0.1];
    let p = dde(2, &a0, &[(1.0, &a1), (1.7, &a2)]);
    let r = -2.0;
    let set = run(&p, 80, r);
    all_residuals_small(&set);
    assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
    let w = zeros_in_rectangle(&p, r);
    report("test7 d=2", &set, w);
    assert_eq!(set.roots.len(), w);

    let b0 = [0.3, -1.1, 0.2, 0.9, -0.2, 0.4, -0.3, 0.5, -0.6];
    let b1 = [-0.8, 0.3, 0.5, 0.2, -1.0, -0.4, 0.6, 0.1, -0.9];
    let p = dde(3, &b0, &[(1.3, &b1)]);
    let r = -2.5;
    let set = run(&p, 100, r);
    all_residuals_small(&set);
    assert!(set.unrefined.is_empty(), "{:?}", set.unrefined);
    let w = zeros_in_rectangle(&p, r);
    report("test7 d=3", &set, w);
    assert_eq!(set.roots.len(), w);
}

#[test]
fn test8_bad_problems() {
    let good = dde(1, &[0.0], &[(1.0, &[-1.0])]);
    let ok = RootOptions {
        n: 10,
        re_min: -1.0,
    };
    assert!(characteristic_roots(&good, &ok).is_ok());
    let is_bad = |r: Result<RootSet, EngineError>| matches!(r, Err(EngineError::BadProblem(_)));
    // Wrong matrix length.
    let mut p = good.clone();
    p.a0 = vec![0.0, 0.0];
    assert!(is_bad(characteristic_roots(&p, &ok)));
    // No delay.
    let mut p = good.clone();
    p.delays.clear();
    assert!(is_bad(characteristic_roots(&p, &ok)));
    // Non-positive and non-finite delays.
    for tau in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut p = good.clone();
        p.delays[0].0 = tau;
        assert!(is_bad(characteristic_roots(&p, &ok)), "tau {tau}");
    }
    // NaN in a matrix.
    let mut p = good.clone();
    p.delays[0].1[0] = f64::NAN;
    assert!(is_bad(characteristic_roots(&p, &ok)));
    let mut p = good.clone();
    p.a0[0] = f64::INFINITY;
    assert!(is_bad(characteristic_roots(&p, &ok)));
    // n = 0, too large an order, NaN re_min, dim = 0.
    assert!(is_bad(characteristic_roots(
        &good,
        &RootOptions { n: 0, re_min: -1.0 }
    )));
    assert!(is_bad(characteristic_roots(
        &good,
        &RootOptions {
            n: 4000,
            re_min: -1.0
        }
    )));
    assert!(is_bad(characteristic_roots(
        &good,
        &RootOptions {
            n: usize::MAX,
            re_min: -1.0
        }
    )));
    assert!(is_bad(characteristic_roots(
        &good,
        &RootOptions {
            n: 10,
            re_min: f64::NAN
        }
    )));
    let p = dde(0, &[], &[(1.0, &[])]);
    assert!(is_bad(characteristic_roots(&p, &ok)));
}

/// The strip `re_min <= Re` holds roots with |Im| up to the contour height, and
/// `A_n` resolves only the modes below about `n*pi/tau_max`. A run with too small
/// an `n` returns a subset of the true roots, all with small residuals, and no
/// `unrefined` entry. Raising `n` completes the set (SPEC 8c, test 7).
#[test]
fn test7b_a_small_n_returns_a_genuine_subset() {
    let a0 = [0.5, 1.0, -1.0, -0.3];
    let a1 = [-1.2, 0.4, 0.7, -0.8];
    let a2 = [0.2, -0.5, 0.3, 0.1];
    let p = dde(2, &a0, &[(1.0, &a1), (1.7, &a2)]);
    let r = -3.0;
    let w = zeros_in_rectangle(&p, r);
    let mut counts = Vec::new();
    for n in [40, 80, 120] {
        let set = run(&p, n, r);
        all_residuals_small(&set);
        report(&format!("test7b n={n}"), &set, w);
        assert!(
            set.roots.len() <= w,
            "n={n}: {} roots, winding {w}",
            set.roots.len()
        );
        counts.push(set.roots.len());
    }
    assert!(
        counts[0] < w,
        "n=40 was expected to miss roots, got {}",
        counts[0]
    );
    assert_eq!(counts[1], w);
    assert_eq!(counts[2], w);
}
