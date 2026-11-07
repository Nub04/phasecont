//! Continuation of periodic orbits of a delay equation (SPEC section 8n).
//!
//! The model `x' = -x(t - tau) + mu x (1 - x^2 - x(t - tau)^2)` has the exact orbit
//! `x = sin t`, `T = 2 pi`, for every `mu` at `tau = pi/2 + 2 pi K`, which makes the
//! `mu`-line an exact family (tests 1 and 5). The Hopf branch in `tau` at `mu = 0.3` reaches
//! that orbit at `tau = pi/2` (tests 2 and 3); that part is a numerical oracle.

use pc_engine::{
    DdeCycle, DdeCycleBranch, DdeCycleSetup, DdeCycleStart, DdeHopfCycle, DdeModel, EngineError,
    Evaluator, Options, StopReason, build_dde, dde_continue_cycle, dde_correct_cycle,
    dde_cycle_residual, dde_hopf_cycle,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const TAU_H: f64 = 1.327_237_281_892_678_3;
const OMEGA_H: f64 = 0.953_939_201_416_945_7;
const NTST: usize = 20;
const NCOL: usize = 4;

/// The period and the amplitude of the Hopf cycle at `eps = 0.05` from a scratch calculation
/// with another discretization (SPEC section 8n, test 3).
const SCRATCH_PERIOD: f64 = 6.584_130_506_595_359;
const SCRATCH_AMP: f64 = 0.050_001_5;

fn osc_source(tau: f64) -> String {
    format!(
        "par mu=0.3, tau={tau:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
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

fn tau_k(k: usize) -> f64 {
    PI / 2.0 + 2.0 * PI * k as f64
}

fn setup(params: &[f64], free: usize, ntst: usize, ncol: usize) -> DdeCycleSetup {
    DdeCycleSetup {
        params: params.to_vec(),
        free,
        ntst,
        ncol,
    }
}

/// Base point `q` of the uniform mesh is `q / (N m)`.
fn base(ntst: usize, ncol: usize) -> Vec<f64> {
    let nq = ntst * ncol;
    (0..=nq).map(|q| q as f64 / nq as f64).collect()
}

fn sine(ntst: usize, ncol: usize) -> Vec<f64> {
    base(ntst, ncol)
        .iter()
        .map(|s| (2.0 * PI * s).sin())
        .collect()
}

fn perturbed(ntst: usize, ncol: usize) -> Vec<f64> {
    base(ntst, ncol)
        .iter()
        .map(|s| 1.2 * (2.0 * PI * s).sin() + 0.05 * (4.0 * PI * s).sin())
        .collect()
}

fn bad<T: std::fmt::Debug>(r: Result<T, EngineError>, what: &str) {
    match r {
        Err(EngineError::BadProblem(_)) => {}
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    }
}

/// Options for a run: the bounds on alpha, the direction, the largest step and the cap.
fn run_opts(bounds: (f64, f64), direction: f64, hmax: f64, max_steps: usize) -> Options {
    Options {
        bounds,
        direction,
        hmax,
        max_steps,
        max_newton: 10,
        ..Options::default()
    }
}

/// The same with the tolerances of test 7.
fn tight_opts(bounds: (f64, f64), direction: f64, hmax: f64, max_steps: usize) -> Options {
    Options {
        tol_x: 1e-11,
        tol_f: 1e-10,
        ..run_opts(bounds, direction, hmax, max_steps)
    }
}

fn start_of(profile: Vec<f64>, period: f64) -> DdeCycleStart {
    DdeCycleStart { profile, period }
}

fn go(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
) -> Result<DdeCycleBranch, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_continue_cycle(m, st, start, o, &cancel)
}

fn go_ok(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    what: &str,
) -> DdeCycleBranch {
    go(m, st, start, o).unwrap_or_else(|e| panic!("{what}: {e}"))
}

/// The `mu`-line: `OSC` at `tau = pi/2 + 2 pi K`, `mu` free, from the exact orbit at 0.3.
fn mu_line(k: usize, bounds: (f64, f64), direction: f64, which: Evaluator) -> DdeCycleBranch {
    let mut m = dde(&osc_source(tau_k(k)), which);
    let st = setup(&[0.3, tau_k(k)], 0, NTST, NCOL);
    let o = run_opts(bounds, direction, 0.2, 200);
    go_ok(
        &mut m,
        &st,
        &start_of(sine(NTST, NCOL), 2.0 * PI),
        &o,
        &format!("mu-line K={k} direction {direction}"),
    )
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter()
        .zip(b)
        .fold(0.0, |s: f64, (x, y)| s.max((x - y).abs()))
}

fn max_of(v: &[f64]) -> f64 {
    v.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x))
}

fn min_of(v: &[f64]) -> f64 {
    v.iter().fold(f64::INFINITY, |a, &x| a.min(x))
}

/// The periodic trapezoid rule of `u^2` over one period, on the base points.
fn mean_square(profile: &[f64]) -> f64 {
    let q = profile.len() - 1;
    profile[..q].iter().map(|v| v * v).sum::<f64>() / q as f64
}

/// The `eps = 0.05` Hopf cycle at `tau_H`, with the evaluator `which`.
fn hopf_cycle_at(eps: f64, which: Evaluator) -> DdeCycle {
    let mut m = dde(&osc_source(TAU_H), which);
    let cancel = AtomicBool::new(false);
    dde_hopf_cycle(
        &mut m,
        &setup(&[0.3, TAU_H], 1, NTST, NCOL),
        &DdeHopfCycle {
            x: vec![0.0],
            omega: OMEGA_H,
            eps,
        },
        &Options {
            max_newton: 30,
            ..Options::default()
        },
        &cancel,
    )
    .unwrap_or_else(|e| panic!("hopf cycle eps = {eps}: {e}"))
}

/// Test 2: from the Hopf cycle at `eps = 0.05` in `tau` up to `pi/2`.
fn hopf_to_exact(which: Evaluator, o: &Options) -> (DdeCycle, DdeCycleBranch) {
    let h = hopf_cycle_at(0.05, which.clone());
    let mut m = dde(&osc_source(TAU_H), which);
    // The corrector of the start holds alpha at the value in `params`, the one of the cycle.
    let st = setup(&[0.3, h.alpha], 1, NTST, NCOL);
    let b = go_ok(
        &mut m,
        &st,
        &start_of(h.profile.clone(), h.period),
        o,
        "hopf to exact",
    );
    (h, b)
}

fn hopf_opts() -> Options {
    run_opts((1.0, PI / 2.0), 1.0, 0.5, 500)
}

/// Test 3: from the exact orbit at `tau = pi/2` back to `alpha_min`.
fn backwards(alpha_min: f64, which: Evaluator) -> DdeCycleBranch {
    let mut m = dde(&osc_source(tau_k(0)), which);
    let st = setup(&[0.3, tau_k(0)], 1, NTST, NCOL);
    let o = run_opts((alpha_min, 2.0), -1.0, 0.5, 500);
    go_ok(
        &mut m,
        &st,
        &start_of(sine(NTST, NCOL), 2.0 * PI),
        &o,
        "backwards",
    )
}

// ---------------------------------------------------------------------------------------
// Test 1: the mu-line, an exact family.

/// `(K, max |T - 2 pi|, max |u - sin|)` over every point of the four runs.
const TABLE_1: [(usize, f64, f64); 2] = [(0, 5e-11, 1.4e-7), (1, 5e-11, 1.4e-7)];

#[test]
fn mu_line_is_an_exact_family() {
    for (k, dt, du) in TABLE_1 {
        let exact = sine(NTST, NCOL);
        for (direction, bound) in [(1.0, 0.9), (-1.0, 0.1)] {
            let b = mu_line(k, (0.1, 0.9), direction, Evaluator::Jit);
            let what = format!("K={k} direction {direction}");
            assert_eq!(b.stop, StopReason::Bounds, "{what}: {:?}", b.stop);
            let last = b.points.last().expect("points");
            assert!(
                (last.alpha - bound).abs() <= 1e-9,
                "{what}: last alpha {} against bound {bound}",
                last.alpha
            );
            assert!(b.points.len() > 2, "{what}: {} points", b.points.len());
            for (i, p) in b.points.iter().enumerate() {
                assert!(
                    (p.period - 2.0 * PI).abs() <= dt,
                    "{what} point {i}: T = {} off by {:e}, bound {dt:e}",
                    p.period,
                    (p.period - 2.0 * PI).abs()
                );
                let e = max_diff(&p.profile, &exact);
                assert!(
                    e <= du,
                    "{what} point {i}: profile error {e:e} above {du:e}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Test 2: the Hopf branch to the exact orbit.

/// `|T - 2 pi|`, `|max u - 1|`, `|min u + 1|`, `|mean u^2 - 0.5|` at the last point.
const TABLE_2: [f64; 4] = [5e-11, 1e-12, 1e-12, 6e-9];

#[test]
fn hopf_branch_reaches_the_exact_orbit() {
    let (_, b) = hopf_to_exact(Evaluator::Jit, &hopf_opts());
    assert_eq!(b.stop, StopReason::Bounds, "{:?}", b.stop);
    let last = b.points.last().expect("points");
    assert!(
        (last.alpha - PI / 2.0).abs() <= 1e-9,
        "last alpha {} against pi/2",
        last.alpha
    );
    let got = [
        (last.period - 2.0 * PI).abs(),
        (max_of(&last.profile) - 1.0).abs(),
        (min_of(&last.profile) + 1.0).abs(),
        (mean_square(&last.profile) - 0.5).abs(),
    ];
    for (i, (g, t)) in got.iter().zip(TABLE_2).enumerate() {
        assert!(g <= &t, "table 2 entry {i}: {g:e} above {t:e}");
    }
    let amp: Vec<f64> = b
        .points
        .iter()
        .map(|p| max_of(&p.profile) - min_of(&p.profile))
        .collect();
    for (i, w) in amp.windows(2).enumerate() {
        assert!(
            w[1] > w[0],
            "amplitude {} then {} at points {i} and {}",
            w[0],
            w[1],
            i + 1
        );
    }
}

// ---------------------------------------------------------------------------------------
// Test 3: the same path backwards.

/// Differences of the period and of the half amplitude of the last point from the
/// `eps = 0.05` cycle of `hopf_cycle`, then from the scratch values.
const TABLE_3: [f64; 4] = [1e-12, 3e-12, 1e-12, 1e-7];

#[test]
fn the_path_runs_backwards_to_the_hopf_cycle() {
    let h = hopf_cycle_at(0.05, Evaluator::Jit);
    let b = backwards(h.alpha, Evaluator::Jit);
    assert_eq!(b.stop, StopReason::Bounds, "{:?}", b.stop);
    let last = b.points.last().expect("points");
    assert!(
        (last.alpha - h.alpha).abs() <= 1e-9,
        "last alpha {} against {}",
        last.alpha,
        h.alpha
    );
    let half = |p: &[f64]| 0.5 * (max_of(p) - min_of(p));
    let got = [
        (last.period - h.period).abs(),
        (half(&last.profile) - half(&h.profile)).abs(),
        (last.period - SCRATCH_PERIOD).abs(),
        (half(&last.profile) - SCRATCH_AMP).abs(),
    ];
    for (i, (g, t)) in got.iter().zip(TABLE_3).enumerate() {
        assert!(g <= &t, "table 3 entry {i}: {g:e} above {t:e}");
    }
}

// ---------------------------------------------------------------------------------------
// Test 4: every point is a solution.

fn check_points(b: &DdeCycleBranch, params: &[f64], free: usize, what: &str) {
    let mut m = dde(&osc_source(params[1]), Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = Options {
        max_newton: 30,
        ..Options::default()
    };
    for (i, p) in b.points.iter().enumerate() {
        let mut ps = params.to_vec();
        ps[free] = p.alpha;
        let st = setup(&ps, free, NTST, NCOL);
        let mut x = p.profile.clone();
        x.push(p.period);
        x.push(p.alpha);
        let r = dde_cycle_residual(&mut m, &st, &p.profile, &x)
            .unwrap_or_else(|e| panic!("{what} point {i}: {e}"));
        let worst = r.iter().fold(0.0, |a: f64, v| a.max(v.abs()));
        assert!(worst <= 1e-9, "{what} point {i}: residual {worst:e}");
        let c = dde_correct_cycle(
            &mut m,
            &st,
            &start_of(p.profile.clone(), p.period),
            &o,
            &cancel,
        )
        .unwrap_or_else(|e| panic!("{what} point {i}: correct_cycle {e}"));
        assert!(c.iters <= 3, "{what} point {i}: {} iterations", c.iters);
        assert!(
            (c.period - p.period).abs() <= 1e-9,
            "{what} point {i}: period {} against {}",
            c.period,
            p.period
        );
    }
}

#[test]
fn every_point_is_a_solution() {
    for k in [0, 1] {
        let b = mu_line(k, (0.1, 0.9), 1.0, Evaluator::Jit);
        check_points(&b, &[0.3, tau_k(k)], 0, &format!("mu-line K={k}"));
    }
    let (h, b) = hopf_to_exact(Evaluator::Jit, &hopf_opts());
    check_points(&b, &[0.3, h.alpha], 1, "hopf to exact");
    let b = backwards(h.alpha, Evaluator::Jit);
    check_points(&b, &[0.3, tau_k(0)], 1, "backwards");
}

// ---------------------------------------------------------------------------------------
// Test 5: direction and step control.

#[test]
fn direction_sets_the_side_of_the_first_step() {
    let up = mu_line(0, (0.1, 0.9), 1.0, Evaluator::Jit);
    let down = mu_line(0, (0.1, 0.9), -1.0, Evaluator::Jit);
    assert!(up.points[1].alpha > up.points[0].alpha, "direction +1");
    assert!(down.points[1].alpha < down.points[0].alpha, "direction -1");
}

#[test]
fn max_steps_and_a_start_outside_the_bounds() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let st = setup(&[0.3, tau_k(0)], 0, NTST, NCOL);
    let start = start_of(sine(NTST, NCOL), 2.0 * PI);
    let o = run_opts((f64::NEG_INFINITY, f64::INFINITY), 1.0, 0.2, 3);
    let b = go_ok(&mut m, &st, &start, &o, "max_steps 3");
    assert_eq!(b.points.len(), 4);
    assert_eq!(b.stop, StopReason::MaxSteps, "{:?}", b.stop);
    let o = run_opts((0.5, 0.9), 1.0, 0.2, 50);
    let b = go_ok(&mut m, &st, &start, &o, "start outside the bounds");
    assert_eq!(b.points.len(), 1);
    assert_eq!(b.stop, StopReason::Bounds, "{:?}", b.stop);
}

// ---------------------------------------------------------------------------------------
// Test 6: errors.

#[test]
fn bad_setups_and_options_are_bad_problems() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let pr = [0.3, tau_k(0)];
    let o = run_opts((0.1, 0.9), 1.0, 0.2, 50);
    let good = |ntst: usize, ncol: usize| start_of(sine(ntst, ncol), 6.0);
    let mut run = |st: DdeCycleSetup, start: DdeCycleStart, o: &Options| go(&mut m, &st, &start, o);
    bad(
        run(setup(&pr, 0, 0, 3), start_of(vec![0.0], 6.0), &o),
        "ntst 0",
    );
    bad(run(setup(&pr, 0, 10, 1), good(10, 3), &o), "ncol 1");
    bad(run(setup(&pr, 0, 10, 8), good(10, 3), &o), "ncol 8");
    bad(
        run(setup(&pr, 2, 10, 3), good(10, 3), &o),
        "free = n params",
    );
    bad(
        run(setup(&[0.3], 0, 10, 3), good(10, 3), &o),
        "short params",
    );
    bad(
        run(setup(&[0.3, tau_k(0), 1.0], 0, 10, 3), good(10, 3), &o),
        "long params",
    );
    bad(
        run(setup(&[f64::NAN, tau_k(0)], 0, 10, 3), good(10, 3), &o),
        "NaN parameter",
    );
    bad(
        run(setup(&[0.3, 0.0], 0, 10, 3), good(10, 3), &o),
        "tau 0 in params",
    );
    bad(run(setup(&pr, 0, 10, 3), good(10, 4), &o), "profile length");
    let mut nan = sine(10, 3);
    nan[7] = f64::NAN;
    bad(
        run(setup(&pr, 0, 10, 3), start_of(nan, 6.0), &o),
        "NaN profile",
    );
    for period in [0.0, -1.0, f64::NAN] {
        bad(
            run(setup(&pr, 0, 10, 3), start_of(sine(10, 3), period), &o),
            "bad period",
        );
    }
    bad(
        run(setup(&pr, 0, 1000, 4), start_of(vec![0.0; 4001], 6.0), &o),
        "dim above the limit",
    );
    bad(
        run(
            setup(&pr, 0, usize::MAX / 2, 4),
            start_of(vec![0.0], 6.0),
            &o,
        ),
        "ntst * ncol overflows",
    );
    // A delay time as the free parameter needs a lower bound above 0.
    for lo in [0.0, -1.0, f64::NAN] {
        let o = run_opts((lo, 2.0), 1.0, 0.2, 50);
        match run(setup(&pr, 1, 10, 3), good(10, 3), &o) {
            Err(EngineError::BadProblem(msg)) => {
                assert!(msg.contains("bounds"), "message for {lo}: {msg}")
            }
            other => panic!("bounds.0 = {lo}: expected BadProblem, got {other:?}"),
        }
    }
    // Bad step control.
    let with = |f: &dyn Fn(&mut Options)| {
        let mut o = run_opts((0.1, 0.9), 1.0, 0.2, 50);
        f(&mut o);
        o
    };
    let cases: [(&str, Options); 6] = [
        ("h0 = 0", with(&|o| o.h0 = 0.0)),
        ("hmin > hmax", with(&|o| o.hmin = 1.0)),
        ("hmax inf", with(&|o| o.hmax = f64::INFINITY)),
        ("hmax NaN", with(&|o| o.hmax = f64::NAN)),
        ("max_newton 0", with(&|o| o.max_newton = 0)),
        ("hmin 0", with(&|o| o.hmin = 0.0)),
    ];
    for (what, o) in &cases {
        bad(run(setup(&pr, 0, 10, 3), good(10, 3), o), what);
    }
}

#[test]
fn a_set_cancel_flag_stops_the_call() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let cancel = AtomicBool::new(true);
    let r = dde_continue_cycle(
        &mut m,
        &setup(&[0.3, tau_k(0)], 0, 10, 3),
        &start_of(sine(10, 3), 2.0 * PI),
        &run_opts((0.1, 0.9), 1.0, 0.2, 50),
        &cancel,
    );
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
}

#[test]
fn a_start_that_does_not_converge_is_newton_failed() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let o = Options {
        max_newton: 1,
        ..run_opts((0.1, 0.9), 1.0, 0.2, 50)
    };
    let r = go(
        &mut m,
        &setup(&[0.3, tau_k(0)], 0, NTST, NCOL),
        &start_of(perturbed(NTST, NCOL), 6.0),
        &o,
    );
    assert!(matches!(r, Err(EngineError::NewtonFailed { .. })), "{r:?}");
}

// ---------------------------------------------------------------------------------------
// Test 7: the two evaluators.

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs()),
            "{what}[{i}]: {x} against {y}"
        );
    }
}

fn same_branch(a: &DdeCycleBranch, b: &DdeCycleBranch, what: &str) {
    assert_eq!(a.points.len(), b.points.len(), "{what}: number of points");
    for (i, (p, q)) in a.points.iter().zip(&b.points).enumerate() {
        close(
            &p.profile,
            &q.profile,
            1e-10,
            &format!("{what} point {i} profile"),
        );
        close(
            &[p.period, p.alpha],
            &[q.period, q.alpha],
            1e-10,
            &format!("{what} point {i} T, alpha"),
        );
    }
}

#[test]
fn jit_and_fallback_agree() {
    let line = |which: Evaluator| {
        let mut m = dde(&osc_source(tau_k(0)), which);
        go_ok(
            &mut m,
            &setup(&[0.3, tau_k(0)], 0, NTST, NCOL),
            &start_of(sine(NTST, NCOL), 2.0 * PI),
            &tight_opts((0.1, 0.9), 1.0, 0.2, 6),
            "mu-line",
        )
    };
    same_branch(&line(Evaluator::Jit), &line(Evaluator::Fallback), "mu-line");
    let o = tight_opts((1.0, PI / 2.0), 1.0, 0.5, 6);
    let (_, a) = hopf_to_exact(Evaluator::Jit, &o);
    let (_, b) = hopf_to_exact(Evaluator::Fallback, &o);
    same_branch(&a, &b, "hopf branch");
}

// ---------------------------------------------------------------------------------------
// Not a test: prints the numbers for the tables of SPEC section 8n.
//   cargo test -p pc-engine --test dde_cycle_branch print_tables -- --ignored --nocapture

#[test]
#[ignore]
fn print_tables() {
    let exact = sine(NTST, NCOL);
    for k in [0, 1] {
        for (direction, _) in [(1.0, 0.9), (-1.0, 0.1)] {
            let b = mu_line(k, (0.1, 0.9), direction, Evaluator::Jit);
            let dt = b
                .points
                .iter()
                .fold(0.0, |a: f64, p| a.max((p.period - 2.0 * PI).abs()));
            let du = b
                .points
                .iter()
                .fold(0.0, |a: f64, p| a.max(max_diff(&p.profile, &exact)));
            println!(
                "T1 K={k} dir={direction} points={} stop={:?} dT={dt:e} du={du:e}",
                b.points.len(),
                b.stop
            );
        }
    }
    let (h, b) = hopf_to_exact(Evaluator::Jit, &hopf_opts());
    let last = b.points.last().expect("points");
    println!(
        "T2 alpha0={:?} points={} stop={:?} alpha={:?} dT={:e} dmax={:e} dmin={:e} dms={:e}",
        h.alpha,
        b.points.len(),
        b.stop,
        last.alpha,
        (last.period - 2.0 * PI).abs(),
        (max_of(&last.profile) - 1.0).abs(),
        (min_of(&last.profile) + 1.0).abs(),
        (mean_square(&last.profile) - 0.5).abs()
    );
    let b = backwards(h.alpha, Evaluator::Jit);
    let last = b.points.last().expect("points");
    let half = |p: &[f64]| 0.5 * (max_of(p) - min_of(p));
    println!(
        "T3 points={} stop={:?} dT={:e} damp={:e} dT_scratch={:e} damp_scratch={:e} \
         period={:?} half={:?}",
        b.points.len(),
        b.stop,
        (last.period - h.period).abs(),
        (half(&last.profile) - half(&h.profile)).abs(),
        (last.period - SCRATCH_PERIOD).abs(),
        (half(&last.profile) - SCRATCH_AMP).abs(),
        last.period,
        half(&last.profile)
    );
}
