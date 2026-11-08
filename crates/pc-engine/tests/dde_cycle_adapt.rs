//! Mesh adaptation along a branch of delay cycles (SPEC section 8p).
//!
//! The oracle of tests 1 and 2 is the manufactured model STEEP, whose exact orbit
//! `y1 = cos t`, `y2 = sin t`, `x = tanh(20 cos t)`, `T = 2 pi` has two steep parts for every
//! `c` and `tau`. The smooth model of tests 3 and 4 is the oscillator of section 8n. Run
//! `cargo test --release --test dde_cycle_adapt -- --ignored --nocapture` to print the
//! measured values behind the bounds.

use pc_engine::{
    DdeCycleBranch, DdeCyclePoint, DdeCycleSetup, DdeCycleSpecial, DdeCycleStart,
    DdeFloquetOptions, DdeMeshAdapt, DdeModel, EngineError, Evaluator, Options, StopReason,
    build_dde, dde_continue_cycle, dde_continue_cycle_adapt, dde_continue_cycle_floquet,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const NTST: usize = 20;
const NCOL: usize = 4;
/// The steepness of STEEP.
const G: f64 = 20.0;

// Bounds of the tables of section 8p: about 4 times the values measured in a release build
// (`print_tables`), except where the table says otherwise.
/// Table 1: the error of the last point as a fraction of the error of the uniform mesh.
const T1_ERROR_FRACTION: f64 = 0.4;
/// Table 1: the largest `|T - 2 pi|` along the run.
const T1_PERIOD_TOL: f64 = 1e-7;
/// Table 1: the least movement of a mesh point at the last point.
const T1_MOVE_MIN: f64 = 0.03;
/// Table 2: the bound on `trivial_error` and on the distance of the second multiplier.
const T2_TRIVIAL_TOL: f64 = 1e-11;
const T2_SECOND_TOL: f64 = 6e-9;
/// Table 3: the smooth orbit on an adapting run.
const T3_PERIOD_TOL: f64 = 7e-8;
const T3_ERROR_TOL: f64 = 9e-7;
const T3_MOVE_MAX: f64 = 0.12;
/// Table 1: the agreement of the two evaluators.
const JIT_TOL: f64 = 1e-9;

const STEEP: &str = "par c=0.5, tau=1\n\
    x'=-x+c*(delay(x,tau)-tanh(20*delay(y1,tau)))+tanh(20*y1)+20*(1-tanh(20*y1)^2)*(-y2)\n\
    y1'=y1-y2-y1*(y1^2+y2^2)\n\
    y2'=y1+y2-y2*(y1^2+y2^2)\n\
    done\n";

fn osc_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         done\n"
    )
}

/// Section 8o, test 2: the oscillator plus the rotating block; parameters `(mu, tau, s, g)`.
fn pd_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}, s=0.4, g=0.3\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         y'=s*(-delay(x,tau)*y+x*z)-0.3*y\n\
         z'=s*(x*y+delay(x,tau)*z)-0.3*z\n\
         done\n"
    )
}

/// Section 8o, test 3: the cubic model; parameters `(a, tau)`.
fn lpc_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par a=0.25, tau={tau:?}\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         done\n"
    )
}

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    match build_dde(&l.expanded, &l.info, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

fn setup(params: &[f64], free: usize, ntst: usize) -> DdeCycleSetup {
    DdeCycleSetup {
        params: params.to_vec(),
        free,
        ntst,
        ncol: NCOL,
    }
}

/// The positions `s_q` of the base points of `mesh`: `s_j + (l / m) delta_j`, then 1.
fn positions(mesh: &[f64]) -> Vec<f64> {
    let ntst = mesh.len() - 1;
    let mut s = Vec::with_capacity(ntst * NCOL + 1);
    for j in 0..ntst {
        for l in 0..NCOL {
            s.push(mesh[j] + (l as f64 / NCOL as f64) * (mesh[j + 1] - mesh[j]));
        }
    }
    s.push(1.0);
    s
}

fn uniform(ntst: usize) -> Vec<f64> {
    (0..=ntst).map(|j| j as f64 / ntst as f64).collect()
}

/// The profile with `n` values per base point of `mesh`; `f(c, s)` is component `c` at `s`.
fn profile_on(mesh: &[f64], n: usize, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    positions(mesh)
        .iter()
        .flat_map(|&s| (0..n).map(move |c| (c, s)))
        .map(|(c, s)| f(c, s))
        .collect()
}

/// The exact STEEP orbit at `s` in `[0, 1]`: `x`, `y1`, `y2`.
fn steep_exact(c: usize, s: f64) -> f64 {
    let t = 2.0 * PI * s;
    match c {
        0 => (G * t.cos()).tanh(),
        1 => t.cos(),
        _ => t.sin(),
    }
}

fn osc_exact(_: usize, s: f64) -> f64 {
    (2.0 * PI * s).sin()
}

/// The largest difference from the exact orbit over the base points of the mesh of `p`.
fn error_of(p: &DdeCyclePoint, n: usize, exact: impl Fn(usize, f64) -> f64) -> f64 {
    let want = profile_on(&p.mesh, n, exact);
    max_diff(&p.profile, &want)
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter()
        .zip(b)
        .fold(0.0, |s: f64, (x, y)| s.max((x - y).abs()))
}

/// The largest `|s_j - j / N|`.
fn drift(mesh: &[f64]) -> f64 {
    let n = mesh.len() - 1;
    mesh.iter().enumerate().fold(0.0, |a: f64, (j, &s)| {
        a.max((s - j as f64 / n as f64).abs())
    })
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pair_bits(v: &[(f64, f64)]) -> Vec<(u64, u64)> {
    v.iter().map(|z| (z.0.to_bits(), z.1.to_bits())).collect()
}

fn fl(nhist: usize, count: usize) -> DdeFloquetOptions {
    DdeFloquetOptions { nhist, count }
}

fn opts(bounds: (f64, f64), direction: f64, hmax: f64, max_steps: usize) -> Options {
    Options {
        bounds,
        direction,
        hmax,
        max_steps,
        max_newton: 10,
        ..Options::default()
    }
}

/// STEEP at `c = 0.5`, from the exact orbit on the uniform mesh, `c` upward to 0.7.
struct Steep {
    m: DdeModel,
    st: DdeCycleSetup,
    start: DdeCycleStart,
    o: Options,
}

fn steep(which: Evaluator) -> Steep {
    Steep {
        m: dde(STEEP, which),
        st: setup(&[0.5, 1.0], 0, NTST),
        start: DdeCycleStart {
            profile: profile_on(&uniform(NTST), 3, steep_exact),
            period: 2.0 * PI,
        },
        o: opts((0.3, 0.7), 1.0, 0.1, 60),
    }
}

fn adapt_run(
    c: &mut Steep,
    fl: Option<&DdeFloquetOptions>,
    every: usize,
    what: &str,
) -> DdeCycleBranch {
    let cancel = AtomicBool::new(false);
    dde_continue_cycle_adapt(
        &mut c.m,
        &c.st,
        &c.start,
        &c.o,
        fl,
        &DdeMeshAdapt { every },
        &cancel,
    )
    .unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn valid_mesh(mesh: &[f64], ntst: usize, what: &str) {
    assert_eq!(mesh.len(), ntst + 1, "{what}: length");
    assert_eq!(mesh[0], 0.0, "{what}: first");
    assert_eq!(mesh[ntst], 1.0, "{what}: last");
    assert!(
        mesh.windows(2).all(|w| w[0] < w[1]),
        "{what}: not increasing {mesh:?}"
    );
}

/// The interval with the smallest length: its midpoint in `t = 2 pi s`.
fn smallest_interval_t(mesh: &[f64]) -> f64 {
    let (j, _) =
        mesh.windows(2)
            .map(|w| w[1] - w[0])
            .enumerate()
            .fold(
                (0, f64::INFINITY),
                |a, (j, d)| if d < a.1 { (j, d) } else { a },
            );
    2.0 * PI * 0.5 * (mesh[j] + mesh[j + 1])
}

// Test 1: the mesh follows the steep parts.
#[test]
fn the_mesh_follows_the_steep_parts() {
    let mut c = steep(Evaluator::Jit);
    let b = adapt_run(&mut c, None, 1, "steep run");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    assert!(b.points.len() > 2, "{} points", b.points.len());
    for (i, p) in b.points.iter().enumerate() {
        valid_mesh(&p.mesh, NTST, &format!("point {i}"));
        assert!(
            (p.period - 2.0 * PI).abs() <= T1_PERIOD_TOL,
            "point {i}: T = {}",
            p.period
        );
    }
    let first = &b.points[0];
    assert_eq!(
        bits(&first.mesh),
        bits(&uniform(NTST)),
        "first mesh is uniform"
    );
    let e0 = error_of(first, 3, steep_exact);
    let last = b.points.last().unwrap();
    let e_last = error_of(last, 3, steep_exact);
    assert!(
        e_last <= T1_ERROR_FRACTION * e0,
        "error {e_last:e} against the uniform {e0:e}"
    );
    assert!(
        drift(&last.mesh) >= T1_MOVE_MIN,
        "the mesh moved by {:e}",
        drift(&last.mesh)
    );
    let t = smallest_interval_t(&last.mesh);
    let near = |t0: f64| (t - t0).abs() < 0.3;
    assert!(
        near(PI / 2.0) || near(3.0 * PI / 2.0),
        "the smallest interval is at t = {t}"
    );
}

/// The root of `lambda + 1 = c exp(-lambda)` near -0.3, by Newton's method.
fn lambda0(c: f64) -> f64 {
    let mut l = -0.3_f64;
    for _ in 0..50 {
        let f = l + 1.0 - c * (-l).exp();
        let df = 1.0 + c * (-l).exp();
        l -= f / df;
    }
    l
}

// Test 2: the multipliers on an adapted mesh.
#[test]
fn the_multipliers_on_an_adapted_mesh() {
    let mut c = steep(Evaluator::Jit);
    let plain = adapt_run(&mut c, None, 1, "steep run");
    let b = adapt_run(&mut c, Some(&fl(16, 2)), 1, "steep run with floquet");
    assert_eq!(b.points.len(), plain.points.len(), "points");
    for (i, (p, q)) in b.points.iter().zip(&plain.points).enumerate() {
        assert_eq!(p.alpha.to_bits(), q.alpha.to_bits(), "point {i}: alpha");
        assert_eq!(p.period.to_bits(), q.period.to_bits(), "point {i}: period");
        assert_eq!(bits(&p.profile), bits(&q.profile), "point {i}: profile");
        assert_eq!(bits(&p.mesh), bits(&q.mesh), "point {i}: mesh");
        assert_eq!(p.multipliers.len(), 2, "point {i}: multipliers");
        assert!(
            p.trivial_error <= T2_TRIVIAL_TOL,
            "point {i}: trivial error {:e}",
            p.trivial_error
        );
        let want = (2.0 * PI * lambda0(p.alpha)).exp();
        let second = p.multipliers[1];
        assert!(
            (second.0 - want).hypot(second.1) <= T2_SECOND_TOL,
            "point {i}: second multiplier {second:?} against {want}"
        );
    }
}

// Test 3: a smooth orbit stays accurate on an adapting run.
#[test]
fn a_smooth_orbit_stays_accurate_on_an_adapted_mesh() {
    let tau = PI / 2.0;
    let mut m = dde(&osc_source(), Evaluator::Jit);
    let st = setup(&[0.3, tau], 0, NTST);
    let start = DdeCycleStart {
        profile: profile_on(&uniform(NTST), 1, osc_exact),
        period: 2.0 * PI,
    };
    let o = opts((0.1, 0.9), 1.0, 0.2, 200);
    let cancel = AtomicBool::new(false);
    let b = dde_continue_cycle_adapt(
        &mut m,
        &st,
        &start,
        &o,
        None,
        &DdeMeshAdapt { every: 1 },
        &cancel,
    )
    .expect("mu line");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    let mut worst = 0.0_f64;
    for (i, p) in b.points.iter().enumerate() {
        valid_mesh(&p.mesh, NTST, &format!("point {i}"));
        assert!(
            (p.period - 2.0 * PI).abs() <= T3_PERIOD_TOL,
            "point {i}: T = {}",
            p.period
        );
        let e = error_of(p, 1, osc_exact);
        assert!(e <= T3_ERROR_TOL, "point {i}: error {e:e}");
        worst = worst.max(drift(&p.mesh));
    }
    assert!(worst <= T3_MOVE_MAX, "the mesh drifted by {worst:e}");
}

fn same_point(p: &DdeCyclePoint, q: &DdeCyclePoint) -> bool {
    p.alpha.to_bits() == q.alpha.to_bits()
        && p.period.to_bits() == q.period.to_bits()
        && bits(&p.profile) == bits(&q.profile)
        && p.h.to_bits() == q.h.to_bits()
        && p.newton_iters == q.newton_iters
        && pair_bits(&p.multipliers) == pair_bits(&q.multipliers)
        && p.trivial_error.to_bits() == q.trivial_error.to_bits()
}

fn same_special(a: &DdeCycleSpecial, b: &DdeCycleSpecial) -> bool {
    a.label == b.label && a.index == b.index && same_point(&a.point, &b.point)
}

/// `adapt` with `every = usize::MAX` is the fixed-mesh run, with a uniform mesh on each point.
fn check_uniform_path(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    f: Option<&DdeFloquetOptions>,
    what: &str,
) {
    let cancel = AtomicBool::new(false);
    let fixed = dde_continue_cycle_floquet(m, st, start, o, f, &cancel)
        .unwrap_or_else(|e| panic!("{what}: fixed: {e}"));
    let adapt = dde_continue_cycle_adapt(
        m,
        st,
        start,
        o,
        f,
        &DdeMeshAdapt { every: usize::MAX },
        &cancel,
    )
    .unwrap_or_else(|e| panic!("{what}: adapt: {e}"));
    assert_eq!(fixed.points.len(), adapt.points.len(), "{what}: points");
    assert_eq!(fixed.stop, adapt.stop, "{what}: stop");
    assert_eq!(
        fixed.specials.len(),
        adapt.specials.len(),
        "{what}: specials"
    );
    let want = uniform(st.ntst);
    for (i, (p, q)) in fixed.points.iter().zip(&adapt.points).enumerate() {
        assert!(same_point(p, q), "{what}: point {i} differs");
        assert!(p.mesh.is_empty(), "{what}: fixed point {i} has a mesh");
        assert_eq!(bits(&q.mesh), bits(&want), "{what}: mesh of point {i}");
    }
    for (i, (p, q)) in fixed.specials.iter().zip(&adapt.specials).enumerate() {
        assert!(same_special(p, q), "{what}: special {i} differs");
        assert!(
            p.point.mesh.is_empty(),
            "{what}: fixed special {i} has a mesh"
        );
        assert_eq!(
            bits(&q.point.mesh),
            bits(&want),
            "{what}: mesh of special {i}"
        );
    }
    // `continue_cycle` is the same run without Floquet options.
    if f.is_none() {
        let plain = dde_continue_cycle(m, st, start, o, &cancel).unwrap();
        assert!(plain.points.iter().all(|p| p.mesh.is_empty()), "{what}");
        assert!(
            plain.specials.iter().all(|s| s.point.mesh.is_empty()),
            "{what}"
        );
    }
}

// Test 4: the uniform path.
#[test]
fn a_run_with_no_adaptation_due_is_the_fixed_mesh_run() {
    let tau = PI / 2.0;
    for direction in [1.0, -1.0] {
        let mut m = dde(&osc_source(), Evaluator::Jit);
        let st = setup(&[0.3, tau], 0, NTST);
        let start = DdeCycleStart {
            profile: profile_on(&uniform(NTST), 1, osc_exact),
            period: 2.0 * PI,
        };
        let o = opts((0.1, 0.9), direction, 0.2, 200);
        check_uniform_path(
            &mut m,
            &st,
            &start,
            &o,
            None,
            &format!("mu line {direction}"),
        );
    }
    // Section 8o, test 2.
    let mut m = dde(&pd_source(), Evaluator::Jit);
    let st = setup(&[0.3, tau, 0.4, 0.3], 2, NTST);
    let start = DdeCycleStart {
        profile: profile_on(&uniform(NTST), 3, |c, s| {
            if c == 0 { osc_exact(c, s) } else { 0.0 }
        }),
        period: 2.0 * PI,
    };
    let o = opts((0.3, 0.8), 1.0, 0.1, 400);
    check_uniform_path(&mut m, &st, &start, &o, Some(&fl(16, 6)), "pd run");
    // Section 8o, test 3.
    let r = 1.5_f64.sqrt();
    let mut m = dde(&lpc_source(), Evaluator::Jit);
    let st = setup(&[0.25, tau], 0, NTST);
    let start = DdeCycleStart {
        profile: profile_on(&uniform(NTST), 1, |_, s| r * osc_exact(0, s)),
        period: 2.0 * PI,
    };
    let o = opts((-0.5, 0.4), -1.0, 0.1, 400);
    check_uniform_path(&mut m, &st, &start, &o, Some(&fl(12, 4)), "lpc run");
}

// Test 7: errors.
#[test]
fn errors_are_reported_before_the_run() {
    let mut c = steep(Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let run = |c: &mut Steep, o: &Options, f: Option<&DdeFloquetOptions>, every: usize| {
        let cancel = AtomicBool::new(false);
        dde_continue_cycle_adapt(
            &mut c.m,
            &c.st,
            &c.start,
            o,
            f,
            &DdeMeshAdapt { every },
            &cancel,
        )
    };
    let o = c.o.clone();
    match run(&mut c, &o, None, 0) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.starts_with("adapt: "), "every 0: {msg}");
        }
        other => panic!("every 0: expected BadProblem, got {other:?}"),
    }
    // One interval: the profile of `ntst = 1` has the right length.
    let one = setup(&[0.5, 1.0], 0, 1);
    let start1 = DdeCycleStart {
        profile: profile_on(&uniform(1), 3, steep_exact),
        period: 2.0 * PI,
    };
    match dde_continue_cycle_adapt(
        &mut c.m,
        &one,
        &start1,
        &o,
        None,
        &DdeMeshAdapt { every: 1 },
        &cancel,
    ) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.starts_with("adapt: "), "ntst 1: {msg}");
        }
        other => panic!("ntst 1: expected BadProblem, got {other:?}"),
    }
    // A bad step control beats a bad adapt, and a bad floquet beats a bad adapt.
    let bad_step = Options {
        hmax: -1.0,
        ..o.clone()
    };
    match run(&mut c, &bad_step, None, 0) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.contains("step control"), "step control: {msg}");
        }
        other => panic!("step control: expected BadProblem, got {other:?}"),
    }
    match run(&mut c, &o, Some(&fl(0, 1)), 0) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.starts_with("floquet: "), "floquet: {msg}");
        }
        other => panic!("floquet: expected BadProblem, got {other:?}"),
    }
    // A set cancel flag.
    let set = AtomicBool::new(true);
    let r = dde_continue_cycle_adapt(
        &mut c.m,
        &c.st,
        &c.start,
        &o,
        None,
        &DdeMeshAdapt { every: 1 },
        &set,
    );
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    set.store(false, Ordering::Relaxed);
}

// Test 8: the interpreter and the compiled evaluator agree.
#[test]
fn jit_and_fallback_agree() {
    let mut a = steep(Evaluator::Jit);
    let mut b = steep(Evaluator::Fallback);
    let ra = adapt_run(&mut a, None, 1, "jit");
    let rb = adapt_run(&mut b, None, 1, "fallback");
    assert_eq!(ra.points.len(), rb.points.len(), "points");
    let rel = |x: f64, y: f64| (x - y).abs() / (1.0 + y.abs());
    for (i, (p, q)) in ra.points.iter().zip(&rb.points).enumerate() {
        assert!(rel(p.alpha, q.alpha) <= JIT_TOL, "point {i}: alpha");
        assert!(rel(p.period, q.period) <= JIT_TOL, "point {i}: period");
        for (x, y) in p.profile.iter().zip(&q.profile) {
            assert!(rel(*x, *y) <= JIT_TOL, "point {i}: profile {x} against {y}");
        }
        for (x, y) in p.mesh.iter().zip(&q.mesh) {
            assert!(rel(*x, *y) <= JIT_TOL, "point {i}: mesh {x} against {y}");
        }
    }
}

/// Prints the values behind the bounds of section 8p.
#[test]
#[ignore = "prints the measured tables of SPEC section 8p"]
fn print_tables() {
    let mut c = steep(Evaluator::Jit);
    let b = adapt_run(&mut c, None, 1, "table 1");
    let e0 = error_of(&b.points[0], 3, steep_exact);
    let worst_t = b
        .points
        .iter()
        .fold(0.0, |a: f64, p| a.max((p.period - 2.0 * PI).abs()));
    let last = b.points.last().unwrap();
    println!(
        "table 1: points {}, e0 {:.3e}, error at the last point {:.3e} ({:.3} of e0), max |T - 2pi| {:.3e}, drift {:.3e}, smallest interval at t = {:.3}",
        b.points.len(),
        e0,
        error_of(last, 3, steep_exact),
        error_of(last, 3, steep_exact) / e0,
        worst_t,
        drift(&last.mesh),
        smallest_interval_t(&last.mesh)
    );
    for (i, p) in b.points.iter().enumerate() {
        println!(
            "  point {i}: alpha {:.4}, error {:.3e}, drift {:.3e}",
            p.alpha,
            error_of(p, 3, steep_exact),
            drift(&p.mesh)
        );
    }
    let f = adapt_run(&mut c, Some(&fl(16, 2)), 1, "table 2");
    for (i, p) in f.points.iter().enumerate() {
        let want = (2.0 * PI * lambda0(p.alpha)).exp();
        let second = p
            .multipliers
            .get(1)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        println!(
            "table 2: point {i}: trivial error {:.3e}, second multiplier error {:.3e}",
            p.trivial_error,
            (second.0 - want).hypot(second.1)
        );
    }
    let tau = PI / 2.0;
    let mut m = dde(&osc_source(), Evaluator::Jit);
    let st = setup(&[0.3, tau], 0, NTST);
    let start = DdeCycleStart {
        profile: profile_on(&uniform(NTST), 1, osc_exact),
        period: 2.0 * PI,
    };
    let o = opts((0.1, 0.9), 1.0, 0.2, 200);
    let cancel = AtomicBool::new(false);
    let s = dde_continue_cycle_adapt(
        &mut m,
        &st,
        &start,
        &o,
        None,
        &DdeMeshAdapt { every: 1 },
        &cancel,
    )
    .expect("mu line");
    let (mut dt, mut de, mut dm) = (0.0_f64, 0.0_f64, 0.0_f64);
    let fixed = dde_continue_cycle(&mut m, &st, &start, &o, &cancel).expect("fixed mu line");
    let want = profile_on(&uniform(NTST), 1, osc_exact);
    let uni = fixed
        .points
        .iter()
        .fold(0.0_f64, |a, p| a.max(max_diff(&p.profile, &want)));
    println!(
        "table 3: uniform-mesh max error {uni:.3e} on {} points",
        fixed.points.len()
    );
    for p in &s.points {
        dt = dt.max((p.period - 2.0 * PI).abs());
        de = de.max(error_of(p, 1, osc_exact));
        dm = dm.max(drift(&p.mesh));
    }
    println!(
        "table 3: points {}, max |T - 2pi| {dt:.3e}, max error {de:.3e}, max drift {dm:.3e}",
        s.points.len()
    );
}
