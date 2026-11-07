//! The Hopf curve of delay equilibria in two parameters (SPEC section 8f, tests 2 to 6).
//!
//! Every oracle is a closed form. Hutchinson: r tau = pi/2 and omega = r on the whole curve.
//! Hayes: kappa = 4 - a^2 and cos(sqrt(kappa) tau) = -a/2. The BT model and the ZH model
//! are written so that the curve and the codim-2 points are known in x or y; the numbers
//! are in the comments of the tests that use them.

use pc_engine::{
    DdeHopfBranch, DdeHopfSetup, DdeModel, EngineError, Evaluator, Options, StopReason, build_dde,
    dde_continue_hopf,
};
use pc_ode::{MapLoader, Model, ModelInfo, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const HUTCH: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";
const HAYES: &str = "par tau=1, a=1\nx'=-a*x-2*delay(x,tau)\ndone\n";
const BT: &str = "par a=-2, tau=1, b=1, c=-0.25\nx'=c-a*x-b*delay(x,tau)-x*x\ndone\n";
const ZH: &str = "par a=0, tau=1, b=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";

fn loaded(src: &str) -> (Model, ModelInfo) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    (l.expanded, l.info)
}

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let (m, i) = loaded(src);
    match build_dde(&m, &i, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

fn both() -> [Evaluator; 2] {
    [Evaluator::Jit, Evaluator::Fallback]
}

fn opts(hmax: f64, bounds: (f64, f64), direction: f64) -> Options {
    Options {
        hmax,
        bounds,
        direction,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        ..Options::default()
    }
}

fn setup(free: [usize; 2], params: &[f64], x0: &[f64], omega: f64) -> DdeHopfSetup {
    DdeHopfSetup {
        free,
        params: params.to_vec(),
        x0: x0.to_vec(),
        omega,
    }
}

fn run(
    src: &str,
    which: Evaluator,
    s: &DdeHopfSetup,
    o: &Options,
) -> Result<DdeHopfBranch, EngineError> {
    let mut m = dde(src, which);
    dde_continue_hopf(&mut m, s, o, &AtomicBool::new(false))
}

fn ok(src: &str, which: Evaluator, s: &DdeHopfSetup, o: &Options) -> DdeHopfBranch {
    match run(src, which, s, o) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn labels(b: &DdeHopfBranch) -> Vec<&'static str> {
    b.specials.iter().map(|s| s.label).collect()
}

/// The error of a run that must not give a branch.
fn err(src: &str, s: &DdeHopfSetup, o: &Options) -> EngineError {
    match run(src, Evaluator::Jit, s, o) {
        Ok(b) => panic!(
            "expected an error, got a branch of {} points",
            b.points.len()
        ),
        Err(e) => e,
    }
}

fn is_bad_problem(e: &EngineError) -> bool {
    matches!(e, EngineError::BadProblem(_))
}

// ---- Hutchinson: r = pi / (2 tau) ----

fn hutch_setup() -> DdeHopfSetup {
    setup([0, 1], &[1.57, 1.0], &[1.001], 1.57)
}

fn hutch_checks(b: &DdeHopfBranch) {
    assert!(b.points.len() > 10, "{} points", b.points.len());
    for p in &b.points {
        let [r, tau] = p.alpha;
        assert!((r * tau - PI / 2.0).abs() <= 1e-8, "r tau = {}", r * tau);
        assert!((p.kappa - r * r).abs() <= 1e-8, "kappa = {}", p.kappa);
        assert!(
            (p.omega.unwrap() - r).abs() <= 1e-8,
            "omega = {:?}",
            p.omega
        );
        assert!((p.x[0] - 1.0).abs() <= 1e-8, "x = {:?}", p.x);
    }
    assert!(b.specials.is_empty(), "{:?}", b.specials);
}

#[test]
fn hutchinson_curve_has_r_tau_pi_over_two_going_up_in_tau() {
    for which in both() {
        let b = ok(HUTCH, which, &hutch_setup(), &opts(0.05, (0.5, 3.0), 1.0));
        assert!(
            (b.points[0].alpha[0] - PI / 2.0).abs() <= 1e-10,
            "start r = {}",
            b.points[0].alpha[0]
        );
        hutch_checks(&b);
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        let last = b.points.last().unwrap();
        assert!(
            (last.alpha[1] - 3.0).abs() <= 1e-8,
            "tau = {}",
            last.alpha[1]
        );
        assert!(
            b.points.windows(2).all(|w| w[1].alpha[1] > w[0].alpha[1]),
            "tau must grow along direction 1"
        );
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn hutchinson_curve_goes_down_in_tau_in_the_other_direction() {
    for which in both() {
        let b = ok(HUTCH, which, &hutch_setup(), &opts(0.05, (0.5, 3.0), -1.0));
        hutch_checks(&b);
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        let last = b.points.last().unwrap();
        assert!(
            (last.alpha[1] - 0.5).abs() <= 1e-8,
            "tau = {}",
            last.alpha[1]
        );
        assert!(
            b.points.windows(2).all(|w| w[1].alpha[1] < w[0].alpha[1]),
            "tau must fall along direction -1"
        );
    }
}

// ---- Hayes: x' = -a x - 2 x(t - tau), kappa = 4 - a^2 ----

fn hayes_setup() -> DdeHopfSetup {
    // free = [tau, a]
    setup([0, 1], &[1.15, 1.0], &[0.0001], 1.7)
}

fn hayes_checks(b: &DdeHopfBranch) {
    assert!(b.points.len() > 10, "{} points", b.points.len());
    for p in &b.points {
        let [tau, a] = p.alpha;
        assert!(
            (p.kappa - (4.0 - a * a)).abs() <= 1e-8,
            "kappa = {}",
            p.kappa
        );
        let w = p.kappa.sqrt();
        assert!(((w * tau).cos() + a / 2.0).abs() <= 1e-8, "cos at a = {a}");
        assert!(p.x[0].abs() <= 1e-10, "x = {:?}", p.x);
    }
    assert!(b.specials.is_empty(), "{:?}", b.specials);
}

#[test]
fn hayes_curve_runs_to_both_bounds_of_a() {
    for which in both() {
        let up = ok(
            HAYES,
            which.clone(),
            &hayes_setup(),
            &opts(0.05, (-1.5, 1.5), 1.0),
        );
        let want = 2.0 * PI / (3.0 * 3f64.sqrt());
        assert!(
            (up.points[0].alpha[0] - want).abs() <= 1e-9,
            "start tau = {}",
            up.points[0].alpha[0]
        );
        hayes_checks(&up);
        assert!(matches!(up.stop, StopReason::Bounds), "{:?}", up.stop);
        assert!((up.points.last().unwrap().alpha[1] - 1.5).abs() <= 1e-8);
        let down = ok(HAYES, which, &hayes_setup(), &opts(0.05, (-1.5, 1.5), -1.0));
        hayes_checks(&down);
        assert!(matches!(down.stop, StopReason::Bounds), "{:?}", down.stop);
        assert!((down.points.last().unwrap().alpha[1] + 1.5).abs() <= 1e-8);
    }
}

// ---- Bogdanov-Takens: x' = c - a x - b x(t - tau) - x^2, c = -1/4, b = 1 ----
//
// Equilibria satisfy a = c/x - b - x. With a_eff = a + 2x = c/x + x - b, the Hopf pair
// i w satisfies cos(w tau) = -a_eff / b and w = b sin(w tau). For w^2 < 0 (mu = sqrt(-kappa))
// the same equations hold with cosh and sinh. The point x = 1/2 has a_eff = -b, the BT point
// at (a, tau) = (-2, 1).

fn bt_setup() -> DdeHopfSetup {
    // free = [a, tau], on the curve at x = 1: a = -2.25, tau = 1.3613444250345883.
    setup(
        [0, 1],
        &[-2.24, 1.3613444250345883, 1.0, -0.25],
        &[1.001],
        0.97,
    )
}

#[test]
fn bt_curve_has_one_bt_point_where_kappa_changes_sign() {
    for which in both() {
        let b = ok(BT, which, &bt_setup(), &opts(0.05, (0.8, 1.5), -1.0));
        assert_eq!(labels(&b), ["BT"], "{:?}", b.specials);
        let s = &b.specials[0];
        assert!((s.alpha[0] + 2.0).abs() <= 1e-8, "a = {}", s.alpha[0]);
        assert!((s.alpha[1] - 1.0).abs() <= 1e-8, "tau = {}", s.alpha[1]);
        assert!((s.x[0] - 0.5).abs() <= 1e-8, "x = {:?}", s.x);
        assert!(s.kappa.abs() <= 1e-8, "kappa = {}", s.kappa);
        let crossings = b
            .points
            .windows(2)
            .filter(|w| (w[0].kappa > 0.0) != (w[1].kappa > 0.0))
            .count();
        assert_eq!(crossings, 1, "kappa must change sign once");
        let (before, after) = (&b.points[s.index - 1], &b.points[s.index]);
        assert!(before.kappa * after.kappa < 0.0 || s.kappa.abs() <= 1e-8);
        for p in &b.points {
            let (a, tau) = (p.alpha[0], p.alpha[1]);
            let x = p.x[0];
            assert!(
                (a - (-0.25 / x - 1.0 - x)).abs() <= 1e-8,
                "a = {a}, x = {x}"
            );
            let a_eff = -0.25 / x + x - 1.0;
            if p.kappa > 0.0 {
                let w = p.kappa.sqrt();
                assert!(((w * tau).cos() + a_eff).abs() <= 1e-7, "cos at x = {x}");
                assert!((w - (w * tau).sin()).abs() <= 1e-7, "sin at x = {x}");
                assert!((p.omega.unwrap() - w).abs() <= 1e-12);
            } else if p.kappa < 0.0 {
                let mu = (-p.kappa).sqrt();
                assert!(((mu * tau).cosh() + a_eff).abs() <= 1e-7, "cosh at x = {x}");
                assert!((mu - (mu * tau).sinh()).abs() <= 1e-7, "sinh at x = {x}");
                assert!(p.omega.is_none());
            }
        }
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        let last = b.points.last().unwrap();
        assert!(
            (last.alpha[1] - 0.8).abs() <= 1e-8,
            "tau = {}",
            last.alpha[1]
        );
        assert!(last.x[0] > 0.2 && last.x[0] < 0.25, "x = {:?}", last.x);
    }
}

// ---- Zero-Hopf: x' = -(a + y) x - b x(t - tau), y' = a - y^2, b = 1 ----
//
// The equilibria are x = 0, y = +-sqrt(a). With a_eff = a + y = y^2 + y the Hopf pair has
// kappa = 1 - a_eff^2 and tau = arccos(-a_eff) / sqrt(kappa). det Delta(0) = (a + y + b)(2 y)
// vanishes at y = 0, which is (a, tau) = (0, pi/2).

fn zh_setup() -> DdeHopfSetup {
    // free = [a, tau], near y = -1/4: a = 1/16, tau = 1.4071360574511174.
    setup(
        [0, 1],
        &[0.07, 1.4071360574511174, 1.0],
        &[0.001, -0.249],
        0.98,
    )
}

#[test]
fn zh_curve_has_one_zh_point_at_y_zero() {
    for which in both() {
        let b = ok(ZH, which, &zh_setup(), &opts(0.05, (1.3, 2.0), 1.0));
        assert!(
            (b.points[0].alpha[0] - 0.0625).abs() <= 1e-9,
            "start a = {}",
            b.points[0].alpha[0]
        );
        assert_eq!(labels(&b), ["ZH"], "{:?}", b.specials);
        let s = &b.specials[0];
        assert!(s.alpha[0].abs() <= 1e-8, "a = {}", s.alpha[0]);
        assert!(
            (s.alpha[1] - PI / 2.0).abs() <= 1e-8,
            "tau = {}",
            s.alpha[1]
        );
        assert!(s.x.iter().all(|v| v.abs() <= 1e-8), "u = {:?}", s.x);
        for p in &b.points {
            let (a, tau) = (p.alpha[0], p.alpha[1]);
            let (x, y) = (p.x[0], p.x[1]);
            assert!((a - y * y).abs() <= 1e-8, "a = {a}, y = {y}");
            assert!(x.abs() <= 1e-8, "x = {x}");
            let a_eff = a + y;
            assert!((p.kappa - (1.0 - a_eff * a_eff)).abs() <= 1e-8, "kappa");
            assert!(((p.kappa.sqrt() * tau).cos() + a_eff).abs() <= 1e-8, "cos");
        }
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        let last = b.points.last().unwrap();
        assert!(
            (last.alpha[1] - 2.0).abs() <= 1e-8,
            "tau = {}",
            last.alpha[1]
        );
    }
}

// ---- start errors ----

#[test]
fn bad_starts_are_bad_problems() {
    let o = opts(0.05, (0.5, 3.0), 1.0);
    let good = hutch_setup();
    let cases: Vec<(&str, DdeHopfSetup)> = vec![
        (
            "free twice",
            DdeHopfSetup {
                free: [0, 0],
                ..good.clone()
            },
        ),
        (
            "free out of range",
            DdeHopfSetup {
                free: [0, 2],
                ..good.clone()
            },
        ),
        (
            "x0 length",
            DdeHopfSetup {
                x0: vec![1.0, 1.0],
                ..good.clone()
            },
        ),
        (
            "params length",
            DdeHopfSetup {
                params: vec![1.57],
                ..good.clone()
            },
        ),
        (
            "omega zero",
            DdeHopfSetup {
                omega: 0.0,
                ..good.clone()
            },
        ),
        (
            "omega NaN",
            DdeHopfSetup {
                omega: f64::NAN,
                ..good.clone()
            },
        ),
        (
            "omega tiny",
            DdeHopfSetup {
                omega: 1e-9,
                ..good.clone()
            },
        ),
    ];
    for (name, s) in cases {
        let e = err(HUTCH, &s, &o);
        assert!(is_bad_problem(&e), "{name}: {e:?}");
    }
}

#[test]
fn a_delay_time_as_second_parameter_needs_a_positive_lower_bound() {
    // free = [r, tau]: alpha2 is the delay time.
    let s = hutch_setup();
    for bounds in [(f64::NEG_INFINITY, f64::INFINITY), (0.0, 3.0)] {
        let e = err(HUTCH, &s, &opts(0.05, bounds, 1.0));
        assert!(is_bad_problem(&e), "{bounds:?}: {e:?}");
    }
}

#[test]
fn a_failed_polish_is_a_newton_error() {
    let s = hutch_setup();
    let mut o = opts(0.05, (0.5, 3.0), 1.0);
    o.max_newton = 1;
    let e = err(HUTCH, &s, &o);
    assert!(matches!(e, EngineError::NewtonFailed { .. }), "{e:?}");
    // The trivial equilibrium has no Hopf pair: the system has no solution near it.
    let zero = DdeHopfSetup {
        x0: vec![0.0],
        ..hutch_setup()
    };
    let e = err(HUTCH, &zero, &opts(0.05, (0.5, 3.0), 1.0));
    assert!(
        matches!(
            e,
            EngineError::NewtonFailed { .. } | EngineError::Singular(_) | EngineError::NotFinite(_)
        ),
        "{e:?}"
    );
}

#[test]
fn a_start_outside_the_bounds_is_one_point() {
    for which in both() {
        let b = ok(HUTCH, which, &hutch_setup(), &opts(0.05, (0.5, 0.9), 1.0));
        assert_eq!(b.points.len(), 1);
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
    }
}

#[test]
fn a_cancelled_run_is_cancelled() {
    let mut m = dde(HUTCH, Evaluator::Jit);
    let cancel = AtomicBool::new(true);
    let e = dde_continue_hopf(
        &mut m,
        &hutch_setup(),
        &opts(0.05, (0.5, 3.0), 1.0),
        &cancel,
    )
    .unwrap_err();
    assert!(matches!(e, EngineError::Cancelled), "{e:?}");
}
