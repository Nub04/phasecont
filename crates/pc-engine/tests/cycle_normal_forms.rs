//! Periodic normal form coefficients of cycle bifurcations (SPEC §8a). Every model here
//! is built so that the coefficient has a closed form that shares no code with the
//! collocation, and every period at the bifurcation is pi, so a missing factor of T
//! shows up. The runs use ntst = 40 and ncol = 4.

mod common;

use common::{attractor_orbit, load};
use pc_engine::{
    Branch, CycleBifurcation, CycleCurveStart, CycleSetup, CycleStart, EngineError, Evaluator,
    ModelEval, Options, Point, SpecialPoint, StopReason, build, continue_cycle, continue_lpc_curve,
    continue_ns_curve, continue_pd_curve, curve_point_normal_form, lc_point_normal_form,
};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const NTST: usize = 40;
const NCOL: usize = 4;

fn model_of(name: &str) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, Evaluator::Jit).unwrap()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The LC run from `setup`, which must finish without warnings.
fn lc_run(m: &mut dyn ModelEval, setup: CycleSetup, opts: &Options) -> Branch {
    let cancel = AtomicBool::new(false);
    let b = continue_cycle(m, &CycleStart::Orbit(setup), opts, &cancel, &mut |_| ()).unwrap();
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    b
}

/// The one special point labeled `label` of `b`.
fn only_special<'a>(b: &'a Branch, label: &str) -> &'a SpecialPoint {
    let found: Vec<_> = b.specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(found.len(), 1, "{:?}", b.specials);
    found[0]
}

/// The exact cycle of squared radius `s` of `cyc_lpc` at the N m + 1 points of a uniform
/// mesh with N m = `nm`, and its period 2 pi / (1 + s).
fn polar_orbit(s: f64, nm: usize) -> (Vec<f64>, f64) {
    let (omega, r) = (1.0 + s, s.sqrt());
    let period = 2.0 * PI / omega;
    let orbit = (0..=nm)
        .flat_map(|k| {
            let a = omega * period * k as f64 / nm as f64;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    (orbit, period)
}

/// The LPC of `cyc_lpc` at mu = -1, found by an LC run from the unstable cycle at
/// mu = -1/2 (s = 1 - sqrt(1/2)) in the direction of falling mu. `shear` is the k of the
/// coordinates X = x + k y, Y = y of `cyc_lpc_lin`, and 0 for `cyc_lpc` itself.
fn lpc_run(m: &mut dyn ModelEval, shear: f64, ntst: usize) -> Branch {
    let s0 = 1.0 - 0.5_f64.sqrt();
    let (mut orbit, period) = polar_orbit(s0, ntst * NCOL);
    for p in orbit.as_chunks_mut::<2>().0 {
        p[0] += shear * p[1];
    }
    let setup = CycleSetup::new(0, vec![-0.5], ntst, NCOL, orbit, period);
    let opts = Options {
        direction: -1.0,
        max_newton: 15,
        hmax: 0.2,
        bounds: (-3.0, 0.5),
        ..Options::default()
    };
    lc_run(m, setup, &opts)
}

fn lpc_point(m: &mut dyn ModelEval, shear: f64, ntst: usize) -> Point {
    only_special(&lpc_run(m, shear, ntst), "LPC").point.clone()
}

fn check_lpc(name: &str, shear: f64) {
    let mut m = model_of(name);
    let pt = lpc_point(m.as_mut(), shear, NTST);
    assert!((pt.alpha + 1.0).abs() <= 1e-8, "LPC at mu = {}", pt.alpha);
    let nf = lc_point_normal_form(m.as_mut(), CycleBifurcation::Lpc, 0, &[0.0], &pt, NCOL).unwrap();
    println!("{name}: b = {} against -4", nf.coefficient);
    assert!(nf.coefficient2.is_none());
    assert!(
        rel(nf.coefficient, -4.0) <= 1e-5,
        "b = {} against -4",
        nf.coefficient
    );
}

/// SPEC §8a test 1: b = -2 omega0 / sigma = -4 at the fold of `cyc_lpc`. Measured on the
/// first run: b = -4.000000000000008, a relative error of 2e-15.
#[test]
fn lpc_coefficient_of_the_sheared_fold() {
    check_lpc("cyc_lpc", 0.0);
}

/// The same fold in non-orthogonal coordinates: b does not change, the adjoint does.
#[test]
fn lpc_coefficient_is_invariant_under_a_linear_map() {
    check_lpc("cyc_lpc_lin", 0.7);
}

/// The cycle z = r exp(i Om t), w = 0 of `cyc_pd` at squared radius `beta`, with Om = 2 and
/// the period pi, at the N m + 1 points of a uniform mesh with N m = `nm`.
fn ring_orbit(beta: f64, nm: usize) -> (Vec<f64>, f64) {
    let r = beta.sqrt();
    let orbit = (0..=nm)
        .flat_map(|k| {
            let a = 2.0 * PI * k as f64 / nm as f64;
            [r * a.cos(), r * a.sin(), 0.0, 0.0]
        })
        .collect();
    (orbit, PI)
}

/// The LC run of a `cyc_pd` or `cyc_ns` model from the ring at beta = 0.2 upward. Both
/// bifurcations are at beta = 1/4.
fn ring_run(m: &mut dyn ModelEval, ntst: usize) -> Branch {
    let (orbit, period) = ring_orbit(0.2, ntst * NCOL);
    let setup = CycleSetup::new(0, vec![0.2], ntst, NCOL, orbit, period);
    let opts = Options {
        direction: 1.0,
        max_newton: 15,
        hmax: 0.03,
        bounds: (0.05, 0.4),
        ..Options::default()
    };
    lc_run(m, setup, &opts)
}

fn ring_point(m: &mut dyn ModelEval, ntst: usize, label: &str) -> Point {
    only_special(&ring_run(m, ntst), label).point.clone()
}

fn pd_point(m: &mut dyn ModelEval, ntst: usize) -> Point {
    ring_point(m, ntst, "PD")
}

fn check_pd(name: &str) {
    let mut m = model_of(name);
    let pt = pd_point(m.as_mut(), NTST);
    assert!((pt.alpha - 0.25).abs() <= 1e-8, "PD at beta = {}", pt.alpha);
    let nf = lc_point_normal_form(m.as_mut(), CycleBifurcation::Pd, 0, &[0.0], &pt, NCOL).unwrap();
    let exact = -1.0 / PI;
    println!("{name}: c = {} against {exact}", nf.coefficient);
    assert!(nf.coefficient2.is_none());
    assert!(
        rel(nf.coefficient, exact) <= 1e-5,
        "c = {} against {exact}",
        nf.coefficient
    );
}

/// SPEC §8a test 2: c = -gamma / T = -1/pi at the flip of `cyc_pd`.
#[test]
fn pd_coefficient_of_the_constructed_flip() {
    check_pd("cyc_pd");
}

/// The same flip after a nonlinear change of coordinates that makes h2 nonzero (alpha1
/// stays zero by symmetry).
#[test]
fn pd_coefficient_is_invariant_under_a_nonlinear_change() {
    check_pd("cyc_pd_shear");
}

/// The same flip with an amplitude dependent frequency, which makes alpha1 nonzero.
#[test]
fn pd_coefficient_with_a_nonzero_period_coefficient() {
    check_pd("cyc_pd_phase");
}

/// The torus bifurcation of a `cyc_ns` model: the point and its coefficient d, which is
/// -2 gam / T = -(2 + i) / pi in every variant (the coordinate changes and the change of the
/// return time leave Re d alone, and the runs show they leave Im d alone as well).
fn check_ns(name: &str) {
    let mut m = model_of(name);
    let pt = ring_point(m.as_mut(), NTST, "NS");
    assert!((pt.alpha - 0.25).abs() <= 1e-8, "NS at beta = {}", pt.alpha);
    let nf = lc_point_normal_form(m.as_mut(), CycleBifurcation::Ns, 0, &[0.0], &pt, NCOL).unwrap();
    let c2 = nf.coefficient2.expect("an NS point has Im d");
    let (re, im) = (-2.0 / PI, -1.0 / PI);
    println!(
        "{name}: d = {} + {c2} i against {re} + {im} i",
        nf.coefficient
    );
    assert!(
        rel(nf.coefficient, re) <= 1e-5,
        "Re d = {} against {re}",
        nf.coefficient
    );
    assert!(rel(c2, im) <= 1e-5, "Im d = {c2} against {im}");
}

/// SPEC §8a test 3: d = -2 gam / T at the torus bifurcation of `cyc_ns`. Measured on the
/// first run: Re d and Im d within 3e-15 relative of the closed form.
#[test]
fn ns_coefficient_of_the_constructed_torus_bifurcation() {
    check_ns("cyc_ns");
}

/// The same bifurcation after a nonlinear change of coordinates that makes h20 nonzero
/// (max |h20| = 0.19, alpha1 and h11 are zero).
#[test]
fn ns_coefficient_is_invariant_under_a_nonlinear_change() {
    check_ns("cyc_ns_shear");
}

/// The same bifurcation after a change of coordinates that makes h11 nonzero (max |h11| =
/// 0.19): B(v, conj v) then has a component across the cycle, and alpha1 stays zero.
#[test]
fn ns_coefficient_with_a_nonzero_h11() {
    check_ns("cyc_ns_radial");
}

/// The same bifurcation with an amplitude dependent frequency, which makes alpha1 nonzero
/// (0.2546 here) and leaves h11 zero.
#[test]
fn ns_coefficient_with_a_nonzero_period_coefficient() {
    check_ns("cyc_ns_phase");
}

/// The run itself stores the coefficient at each LPC, PD and NS point, the same value the
/// evaluator gives for the stored point.
#[test]
fn a_cycle_run_attaches_the_coefficient_to_an_lpc_point() {
    let mut m = model_of("cyc_lpc");
    let b = lpc_run(m.as_mut(), 0.0, NTST);
    let ex = &only_special(&b, "LPC").extras;
    let c = ex.coefficient.expect("the LPC point has b");
    assert!(rel(c, -4.0) <= 1e-5, "b = {c}");
    assert!(ex.coefficient2.is_none());
    assert!(ex.normalization.is_some());
}

#[test]
fn a_cycle_run_attaches_the_coefficient_to_a_pd_point() {
    let mut m = model_of("cyc_pd");
    let b = ring_run(m.as_mut(), NTST);
    let ex = &only_special(&b, "PD").extras;
    let c = ex.coefficient.expect("the PD point has c");
    assert!(rel(c, -1.0 / PI) <= 1e-5, "c = {c}");
    assert!(ex.coefficient2.is_none());
    assert!(ex.normalization.is_some());
}

#[test]
fn a_cycle_run_attaches_both_parts_of_d_to_an_ns_point() {
    let mut m = model_of("cyc_ns");
    let b = ring_run(m.as_mut(), NTST);
    let ex = &only_special(&b, "NS").extras;
    let (re, im) = (
        ex.coefficient.expect("Re d"),
        ex.coefficient2.expect("Im d"),
    );
    assert!(rel(re, -2.0 / PI) <= 1e-5, "Re d = {re}");
    assert!(rel(im, -1.0 / PI) <= 1e-5, "Im d = {im}");
    assert!(ex.normalization.is_some());
}

/// The first period doubling of Rössler (a = b = 0.2) found by the LC run in c from the
/// attractor at c = 2.5, on a mesh of `ntst` intervals.
fn rossler_pd(m: &mut dyn ModelEval, ntst: usize) -> Point {
    let p = vec![0.2, 0.2, 2.5];
    let (orbit, period) = attractor_orbit(
        m,
        &p,
        vec![1.0, 1.0, 1.0],
        (1000.0, 0.005),
        (1, 0.0),
        ntst * NCOL,
    );
    let setup = CycleSetup::new(2, p, ntst, NCOL, orbit, period);
    let opts = Options {
        max_newton: 15,
        hmax: 0.05,
        bounds: (2.0, 2.9),
        ..Options::default()
    };
    only_special(&lc_run(m, setup, &opts), "PD").point.clone()
}

fn rossler_c(m: &mut dyn ModelEval, ntst: usize) -> (f64, f64) {
    let pt = rossler_pd(m, ntst);
    let nf = lc_point_normal_form(m, CycleBifurcation::Pd, 2, &[0.2, 0.2, 0.0], &pt, NCOL).unwrap();
    (pt.alpha, nf.coefficient)
}

/// SPEC §8a test 4: the first period doubling of Rössler is supercritical, since the doubled
/// cycle exists for c above it (`cycle_continuation.rs`), so c < 0. The sign comes from the
/// collocation alone. The value is c = -0.0037643658 on the ntst = 80 mesh and differs
/// from it by 1.2e-7 relative at ntst = 20 and by 4.6e-9 at ntst = 40.
#[test]
fn rossler_first_flip_is_supercritical() {
    let mut m = model_of("rossler");
    let (c80, coef80) = rossler_c(m.as_mut(), 80);
    assert!((c80 - 2.832445028).abs() <= 1e-6, "PD at c = {c80}");
    assert!(coef80 < 0.0, "c = {coef80}");
    let d: Vec<f64> = [20, 40]
        .into_iter()
        .map(|ntst| {
            let (_, coef) = rossler_c(m.as_mut(), ntst);
            assert!(coef < 0.0, "ntst {ntst}: c = {coef}");
            rel(coef, coef80)
        })
        .collect();
    assert!(d[0] <= 1e-6 && d[1] <= 1e-8 && d[1] < d[0], "{d:?}");
}

/// SPEC §8a test 5: each doubling of the mesh cuts the error by more than 10. For PD and NS
/// the measured factors are 200 and 285 (order 2 ncol = 8 of the collocation at the mesh
/// points), and the error at ntst = 40 is 4e-14. The LPC error is at rounding level from
/// ntst = 20, so only its first step is a rate.
#[test]
fn the_error_falls_with_the_mesh() {
    let errors = |name: &str, kind: CycleBifurcation| -> Vec<f64> {
        [10, 20, 40]
            .into_iter()
            .map(|ntst| {
                let mut m = model_of(name);
                let (pt, exact) = match kind {
                    CycleBifurcation::Lpc => (lpc_point(m.as_mut(), 0.0, ntst), -4.0),
                    CycleBifurcation::Pd => (ring_point(m.as_mut(), ntst, "PD"), -1.0 / PI),
                    CycleBifurcation::Ns => (ring_point(m.as_mut(), ntst, "NS"), -2.0 / PI),
                };
                let nf = lc_point_normal_form(m.as_mut(), kind, 0, &[0.0], &pt, NCOL).unwrap();
                rel(nf.coefficient, exact)
            })
            .collect()
    };
    for (name, kind) in [
        ("cyc_lpc", CycleBifurcation::Lpc),
        ("cyc_pd", CycleBifurcation::Pd),
        ("cyc_ns", CycleBifurcation::Ns),
    ] {
        let e = errors(name, kind);
        println!("{name}: errors at ntst 10, 20, 40: {e:?}");
        assert!(e[1] < e[0] / 10.0, "{name}: {e:?}");
        assert!(e[2] < e[1] / 10.0 || e[2] < 1e-13, "{name}: {e:?}");
    }
}

// SPEC §8a test 6: the coefficient along a curve of cycle bifurcations. Each model has a
// parameter nu that moves the coefficient through zero along the whole curve, with a closed
// form for it, so the curve runs on exact starts and the coefficient is compared at every
// curve point. The curve runs use ntst = 20.

const CURVE_NTST: usize = 20;

/// `orbit` with zeros appended to each point, from `from` to `to` coordinates.
fn pad(orbit: &[f64], from: usize, to: usize) -> Vec<f64> {
    orbit
        .chunks(from)
        .flat_map(|c| {
            let mut v = c.to_vec();
            v.resize(to, 0.0);
            v
        })
        .collect()
}

/// A start on a uniform mesh with the two parameters `params[0]` and `params[1]` free.
fn curve_start(params: Vec<f64>, (orbit, period): (Vec<f64>, f64)) -> CycleCurveStart {
    CycleCurveStart {
        free: [0, 1],
        params,
        orbit,
        period,
        mesh: (0..=CURVE_NTST)
            .map(|j| j as f64 / CURVE_NTST as f64)
            .collect(),
        ntst: CURVE_NTST,
        ncol: NCOL,
    }
}

type CurveRun = fn(
    &mut dyn ModelEval,
    &CycleCurveStart,
    &Options,
    &AtomicBool,
    &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError>;

/// The curve from `start` in the direction `dir` until the second parameter leaves `bounds`.
fn curve_run(
    name: &str,
    run: CurveRun,
    start: &CycleCurveStart,
    dir: f64,
    bounds: (f64, f64),
) -> (Box<dyn ModelEval>, Branch) {
    let mut m = model_of(name);
    let opts = Options {
        direction: dir,
        max_newton: 15,
        hmax: 0.1,
        bounds,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let b = run(m.as_mut(), start, &opts, &cancel, &mut |_| ()).unwrap();
    assert_eq!(b.stop, StopReason::Bounds, "{:?}", b.warnings);
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    (m, b)
}

fn lpc_curve(
    m: &mut dyn ModelEval,
    s: &CycleCurveStart,
    o: &Options,
    c: &AtomicBool,
    f: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    continue_lpc_curve(m, s, o, c, f)
}

fn pd_curve(
    m: &mut dyn ModelEval,
    s: &CycleCurveStart,
    o: &Options,
    c: &AtomicBool,
    f: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    continue_pd_curve(m, s, o, c, f)
}

fn ns_curve(
    m: &mut dyn ModelEval,
    s: &CycleCurveStart,
    o: &Options,
    c: &AtomicBool,
    f: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    continue_ns_curve(m, s, o, c, f)
}

/// The coefficient at `pt`, with Im d as the second part for NS.
fn curve_coefficient(
    m: &mut dyn ModelEval,
    kind: CycleBifurcation,
    start: &CycleCurveStart,
    pt: &Point,
) -> (f64, Option<f64>) {
    let nf = curve_point_normal_form(m, kind, start.free, &start.params, pt, NCOL).unwrap();
    (nf.coefficient, nf.coefficient2)
}

/// c = -nu / T along the PD curve of `cyc_gpd`, which is the line beta = 1/4. The sign of c
/// is opposite to the sign of nu on both sides of the generalized period doubling at nu = 0.
#[test]
fn the_flip_coefficient_changes_sign_along_a_gpd_curve() {
    let start = curve_start(vec![0.25, 0.5], ring_orbit(0.25, CURVE_NTST * NCOL));
    let (mut m, b) = curve_run("cyc_gpd", pd_curve, &start, -1.0, (-0.5, 0.6));
    let (mut worst, mut sides) = (0.0_f64, (0, 0));
    for pt in &b.points {
        let (c, _) = curve_coefficient(m.as_mut(), CycleBifurcation::Pd, &start, pt);
        assert!((pt.u[pt.u.len() - 1] - 0.25).abs() <= 1e-8, "beta");
        worst = worst.max((c + pt.alpha / PI).abs());
        if pt.alpha.abs() > 0.02 {
            assert!(c * pt.alpha < 0.0, "nu = {}: c = {c}", pt.alpha);
            sides = (
                sides.0 + (pt.alpha > 0.0) as usize,
                sides.1 + (pt.alpha < 0.0) as usize,
            );
        }
    }
    println!(
        "cyc_gpd: {} points, worst |c + nu/pi| = {worst:.2e}",
        b.points.len()
    );
    assert!(sides.0 >= 3 && sides.1 >= 3, "{sides:?}");
    assert!(worst <= 1e-6, "{worst:e}");
}

/// Re d = -2 nu / T and Im d = -1/pi along the NS curve of `cyc_ch`, the line beta = 1/4,
/// with the sign change of Re d at the Chenciner point nu = 0.
#[test]
fn the_torus_coefficient_changes_sign_along_a_ch_curve() {
    let start = curve_start(vec![0.25, 0.5], ring_orbit(0.25, CURVE_NTST * NCOL));
    let (mut m, b) = curve_run("cyc_ch", ns_curve, &start, -1.0, (-0.5, 0.6));
    let (mut worst, mut sides) = (0.0_f64, (0, 0));
    for pt in &b.points {
        let (re, im) = curve_coefficient(m.as_mut(), CycleBifurcation::Ns, &start, pt);
        let im = im.expect("Im d");
        worst = worst
            .max((re + 2.0 * pt.alpha / PI).abs())
            .max((im + 1.0 / PI).abs());
        if pt.alpha.abs() > 0.02 {
            assert!(re * pt.alpha < 0.0, "nu = {}: Re d = {re}", pt.alpha);
            sides = (
                sides.0 + (pt.alpha > 0.0) as usize,
                sides.1 + (pt.alpha < 0.0) as usize,
            );
        }
    }
    println!(
        "cyc_ch: {} points, worst error = {worst:.2e}",
        b.points.len()
    );
    assert!(sides.0 >= 3 && sides.1 >= 3, "{sides:?}");
    assert!(worst <= 1e-6, "{worst:e}");
}

/// On the fold curve of `cyc_cpc`, mu = -2 q^3 and nu = 3 q^2 with the cycle of squared
/// radius 1 + q and period 2 pi / (2 + q), the coefficient is b = -6 (1 + q) q (2 + q). The
/// curve runs through the cusp at q = 0 and b changes sign there.
#[test]
fn the_fold_coefficient_changes_sign_at_the_cusp_of_cycles() {
    let q0 = 0.3_f64;
    let start = curve_start(
        vec![-2.0 * q0.powi(3), 3.0 * q0 * q0],
        polar_orbit(1.0 + q0, CURVE_NTST * NCOL),
    );
    let (mut m, b) = curve_run("cyc_cpc", lpc_curve, &start, -1.0, (-0.5, 0.3));
    let (mut worst, mut sides) = (0.0_f64, (0, 0));
    for pt in &b.points {
        let n = pt.u.len();
        let (period, mu) = (pt.u[n - 2], pt.u[n - 1]);
        let q = 2.0 * PI / period - 2.0;
        assert!((mu + 2.0 * q.powi(3)).abs() <= 1e-8, "mu = {mu} at q = {q}");
        assert!(
            (pt.alpha - 3.0 * q * q).abs() <= 1e-8,
            "nu = {} at q = {q}",
            pt.alpha
        );
        let (coef, _) = curve_coefficient(m.as_mut(), CycleBifurcation::Lpc, &start, pt);
        let exact = -6.0 * (1.0 + q) * q * (2.0 + q);
        worst = worst.max((coef - exact).abs());
        if q.abs() > 0.02 {
            assert!(coef * q < 0.0, "q = {q}: b = {coef}");
            sides = (sides.0 + (q > 0.0) as usize, sides.1 + (q < 0.0) as usize);
        }
    }
    println!(
        "cyc_cpc: {} points, worst |b - exact| = {worst:.2e}",
        b.points.len()
    );
    assert!(sides.0 >= 3 && sides.1 >= 3, "{sides:?}");
    assert!(worst <= 1e-5, "{worst:e}");
}

/// The multipliers of `pt` left after removing, for each `z` in `drop` in turn, the multiplier
/// nearest to `z` (the trivial one and the critical ones).
fn rest_after(pt: &Point, drop: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut mu = pt.eigenvalues.clone();
    for z in drop {
        let d = |m: &(f64, f64)| (m.0 - z.0).hypot(m.1 - z.1);
        let k = (0..mu.len())
            .min_by(|&a, &b| d(&mu[a]).total_cmp(&d(&mu[b])))
            .unwrap();
        mu.remove(k);
    }
    mu
}

/// The real part of the product of (mu + s) over `mu`.
fn product(mu: &[(f64, f64)], s: f64) -> f64 {
    mu.iter()
        .fold((1.0, 0.0), |p, m| {
            let q = (m.0 + s, m.1);
            (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0)
        })
        .0
}

/// On the fold line mu = -1 of `cyc_r1` the multipliers are 1, 1 and exp(nu T) with T = pi,
/// so the product of mu - 1 over the multipliers other than the pair at 1 is exp(nu T) - 1,
/// which changes sign at nu = 0. This is the multiplier test the R1 label uses, checked on
/// the curve points away from nu = 0, where the third multiplier is not near 1.
#[test]
fn the_r1_product_changes_sign_along_a_fold_curve() {
    let orbit = polar_orbit(1.0, CURVE_NTST * NCOL);
    let start = curve_start(vec![-1.0, 0.5], (pad(&orbit.0, 2, 3), orbit.1));
    let (_, b) = curve_run("cyc_r1", lpc_curve, &start, -1.0, (-0.5, 0.6));
    let mut sides = (0, 0);
    for pt in &b.points {
        let nu = pt.alpha;
        if nu.abs() < 0.1 {
            continue;
        }
        let rest = rest_after(pt, &[(1.0, 0.0), (1.0, 0.0)]);
        assert_eq!(rest.len(), 1, "{:?}", pt.eigenvalues);
        let got = product(&rest, -1.0);
        let exact = (nu * PI).exp() - 1.0;
        assert!(
            (got - exact).abs() <= 1e-6,
            "nu = {nu}: {got} against {exact}"
        );
        sides = (sides.0 + (nu > 0.0) as usize, sides.1 + (nu < 0.0) as usize);
    }
    assert!(sides.0 >= 2 && sides.1 >= 2, "{sides:?}");
}

/// On the flip line beta = 1/4 of `cyc_r2` the multipliers are 1, exp(-pi/2), -1, -exp(-pi),
/// -exp(nu pi) and -exp((nu - 1) pi). The product of mu + 1 over those other than 1 and -1
/// is (1 + exp(-pi/2)) (1 - exp(-pi)) (1 - exp(nu pi)) (1 - exp((nu - 1) pi)), which changes
/// sign at nu = 0. This is the multiplier test the R2 label uses, checked away from nu = 0.
#[test]
fn the_r2_product_changes_sign_along_a_flip_curve() {
    // The real multipliers -exp(nu pi) and -exp((nu - 1) pi) have product 1 at nu = 1/2, a
    // zero of the PDNS test at the first point. Like an exact zero there, it is neither
    // reported nor warned about.
    let orbit = ring_orbit(0.25, CURVE_NTST * NCOL);
    let start = curve_start(vec![0.25, 0.5], (pad(&orbit.0, 4, 6), orbit.1));
    let (_, b) = curve_run("cyc_r2", pd_curve, &start, -1.0, (-0.5, 0.6));
    let mut sides = (0, 0);
    for pt in &b.points {
        let nu = pt.alpha;
        if nu.abs() < 0.1 {
            continue;
        }
        let rest = rest_after(pt, &[(1.0, 0.0), (-1.0, 0.0)]);
        assert_eq!(rest.len(), 4, "{:?}", pt.eigenvalues);
        let got = product(&rest, 1.0);
        let exact = (1.0 + (-PI / 2.0).exp())
            * (1.0 - (-PI).exp())
            * (1.0 - (nu * PI).exp())
            * (1.0 - ((nu - 1.0) * PI).exp());
        assert!(
            (got - exact).abs() <= 1e-6 * exact.abs().max(1.0),
            "nu = {nu}: {got} against {exact}"
        );
        sides = (sides.0 + (nu > 0.0) as usize, sides.1 + (nu < 0.0) as usize);
    }
    assert!(sides.0 >= 2 && sides.1 >= 2, "{sides:?}");
}

// SPEC §8a, E7b: the codim-2 points that need the coefficients. Each test runs the same
// curve as the sign-change test above and checks the located special point against the
// closed form, which puts the zero at nu = 0 (and at the cusp for CPC).

/// The one special point labeled `label` of `b`, with its two parameters (a1, a2).
fn located(b: &Branch, label: &str) -> (Point, f64, f64) {
    let pt = only_special(b, label).point.clone();
    let a1 = pt.u[pt.u.len() - 1];
    (pt.clone(), a1, pt.alpha)
}

fn count(b: &Branch, label: &str) -> usize {
    b.specials.iter().filter(|s| s.label == label).count()
}

/// The CPC on the fold curve of `cyc_cpc` is at (mu, nu) = (0, 0) on the cycle r = 1.
#[test]
fn the_cpc_is_located_on_the_fold_curve_of_the_cusp_model() {
    let q0 = 0.3_f64;
    let start = curve_start(
        vec![-2.0 * q0.powi(3), 3.0 * q0 * q0],
        polar_orbit(1.0 + q0, CURVE_NTST * NCOL),
    );
    let (_, b) = curve_run("cyc_cpc", lpc_curve, &start, -1.0, (-0.5, 0.3));
    let (pt, mu, nu) = located(&b, "CPC");
    let r = pt.u[0].hypot(pt.u[1]);
    println!("CPC at mu = {mu:.3e}, nu = {nu:.3e}, r = {r:.8}");
    assert!(mu.abs() < 1e-6 && nu.abs() < 1e-6, "{mu:e} {nu:e}");
    assert!((r - 1.0).abs() < 1e-4, "{r}");
}

/// The GPD on the flip line beta = 1/4 of `cyc_gpd` is at nu = 0.
#[test]
fn the_gpd_is_located_on_the_flip_curve_of_the_gpd_model() {
    let start = curve_start(vec![0.25, 0.5], ring_orbit(0.25, CURVE_NTST * NCOL));
    let (_, b) = curve_run("cyc_gpd", pd_curve, &start, -1.0, (-0.5, 0.6));
    let (_, beta, nu) = located(&b, "GPD");
    println!("GPD at beta = {beta:.10}, nu = {nu:.3e}");
    assert!((beta - 0.25).abs() < 1e-8, "{beta}");
    assert!(nu.abs() < 1e-5, "{nu:e}");
}

/// The CH on the torus line beta = 1/4 of `cyc_ch` is at nu = 0.
#[test]
fn the_ch_is_located_on_the_ns_curve_of_the_ch_model() {
    let start = curve_start(vec![0.25, 0.5], ring_orbit(0.25, CURVE_NTST * NCOL));
    let (_, b) = curve_run("cyc_ch", ns_curve, &start, -1.0, (-0.5, 0.6));
    let (pt, nu) = (
        &only_special(&b, "CH").point,
        only_special(&b, "CH").point.alpha,
    );
    let beta = pt.u[pt.u.len() - 2];
    println!("CH at beta = {beta:.10}, nu = {nu:.3e}");
    assert!((beta - 0.25).abs() < 1e-8, "{beta}");
    assert!(nu.abs() < 1e-5, "{nu:e}");
}

/// The R1 point on the fold line mu = -1 of `cyc_r1` is at nu = 0, where the third
/// multiplier reaches 1. The constant b = -4 gives no CPC along the way.
#[test]
fn the_r1_is_located_on_the_fold_curve_of_the_r1_model() {
    let orbit = polar_orbit(1.0, CURVE_NTST * NCOL);
    let start = curve_start(vec![-1.0, 0.5], (pad(&orbit.0, 2, 3), orbit.1));
    let (_, b) = curve_run("cyc_r1", lpc_curve, &start, -1.0, (-0.5, 0.6));
    let (_, mu, nu) = located(&b, "R1");
    println!("R1 at mu = {mu:.10}, nu = {nu:.3e}");
    assert!((mu + 1.0).abs() < 1e-8, "{mu}");
    assert!(nu.abs() < 1e-5, "{nu:e}");
    assert_eq!(count(&b, "CPC"), 0, "{:?}", b.specials);
}

/// With no multiplier left after the trivial one and the second 1, as on a planar model,
/// the R1 test is NaN and no R1 point is found, while the CPC test is defined.
#[test]
fn the_r1_test_is_nan_on_a_planar_model() {
    let q0 = 0.3_f64;
    let start = curve_start(
        vec![-2.0 * q0.powi(3), 3.0 * q0 * q0],
        polar_orbit(1.0 + q0, CURVE_NTST * NCOL),
    );
    let (_, b) = curve_run("cyc_cpc", lpc_curve, &start, -1.0, (-0.5, 0.3));
    assert_eq!(count(&b, "R1"), 0, "{:?}", b.specials);
    for pt in &b.points {
        assert_eq!(pt.codim2.len(), 4, "{:?}", pt.codim2);
        assert!(pt.codim2[3].is_nan(), "R1 test {:?}", pt.codim2);
        assert!(pt.codim2[2].is_finite(), "CPC test {:?}", pt.codim2);
    }
}

/// The R2 points on the flip line beta = 1/4 of `cyc_r2` are at nu = 0 (run down from 0.5)
/// and at nu = 1 (run up from 0.5), the two zeros of the product of mu + 1. The start at
/// nu = 0.5 is a PDNS zero, which the runs skip without a warning.
#[test]
fn the_r2_points_are_located_on_the_flip_curve_of_the_r2_model() {
    let orbit = ring_orbit(0.25, CURVE_NTST * NCOL);
    let start = curve_start(vec![0.25, 0.5], (pad(&orbit.0, 4, 6), orbit.1));
    for (dir, bounds, at) in [(-1.0, (-0.5, 0.6), 0.0), (1.0, (0.3, 1.2), 1.0)] {
        let (_, b) = curve_run("cyc_r2", pd_curve, &start, dir, bounds);
        let (_, beta, nu) = located(&b, "R2");
        println!("R2 at beta = {beta:.10}, nu = {nu:.3e}");
        assert!((beta - 0.25).abs() < 1e-8, "{beta}");
        assert!((nu - at).abs() < 1e-5, "{nu:e} against {at}");
    }
}
