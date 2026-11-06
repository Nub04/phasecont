//! Periodic normal form coefficients of cycle bifurcations (SPEC §8a). Every model here
//! is built so that the coefficient has a closed form that shares no code with the
//! collocation, and every period at the bifurcation is pi, so a missing factor of T
//! shows up. The runs use ntst = 40 and ncol = 4.

mod common;

use common::load;
use pc_engine::{
    CycleBifurcation, CycleSetup, CycleStart, Evaluator, ModelEval, Options, Point, build,
    continue_cycle, lc_point_normal_form,
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

/// The one special point labeled `label` of the LC run from `setup`.
fn lc_special(m: &mut dyn ModelEval, setup: CycleSetup, opts: &Options, label: &str) -> Point {
    let cancel = AtomicBool::new(false);
    let b = continue_cycle(m, &CycleStart::Orbit(setup), opts, &cancel, &mut |_| ()).unwrap();
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let found: Vec<_> = b.specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(found.len(), 1, "{:?}", b.specials);
    found[0].point.clone()
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
fn lpc_point(m: &mut dyn ModelEval, shear: f64) -> Point {
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
    lc_special(m, setup, &opts, "LPC")
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
