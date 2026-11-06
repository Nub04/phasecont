//! Periodic normal form coefficients of cycle bifurcations (SPEC §8a). Every model here
//! is built so that the coefficient has a closed form that shares no code with the
//! collocation, and every period at the bifurcation is pi, so a missing factor of T
//! shows up. The runs use ntst = 40 and ncol = 4.

mod common;

use common::load;
use pc_engine::{
    Branch, CycleBifurcation, CycleSetup, CycleStart, Evaluator, ModelEval, Options, Point,
    SpecialPoint, build, continue_cycle, lc_point_normal_form,
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
fn lpc_run(m: &mut dyn ModelEval, shear: f64) -> Branch {
    let s0 = 1.0 - 0.5_f64.sqrt();
    let (mut orbit, period) = polar_orbit(s0, NTST * NCOL);
    for p in orbit.as_chunks_mut::<2>().0 {
        p[0] += shear * p[1];
    }
    let setup = CycleSetup::new(0, vec![-0.5], NTST, NCOL, orbit, period);
    let opts = Options {
        direction: -1.0,
        max_newton: 15,
        hmax: 0.2,
        bounds: (-3.0, 0.5),
        ..Options::default()
    };
    lc_run(m, setup, &opts)
}

fn lpc_point(m: &mut dyn ModelEval, shear: f64) -> Point {
    only_special(&lpc_run(m, shear), "LPC").point.clone()
}

fn check_lpc(name: &str, shear: f64) {
    let mut m = model_of(name);
    let pt = lpc_point(m.as_mut(), shear);
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
    let b = lpc_run(m.as_mut(), 0.0);
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
