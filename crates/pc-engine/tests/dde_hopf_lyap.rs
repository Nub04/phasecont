//! The first Lyapunov coefficient along a Hopf curve of delay equilibria (SPEC section 8g-2).
//!
//! The oracle is Wright's equation with a cubic term, `u' = -a u(t-1) (1 + u) + b u^3`. The
//! origin is an equilibrium for every `(a, b)` and the cubic term does not touch the
//! linearization, so the Hopf curve is the line `a = pi/2`, `omega = pi/2`. The cubic term
//! adds `3 b p q^2 qbar = 3 b / Delta'(i omega)` to `c1`, with `Delta'(i omega) = 1 + i pi/2`,
//! so on the curve
//!
//! ```text
//! l1(b) = l1(0) + s b,    s = 6 / (pi (1 + pi^2/4)),    l1(0) = -0.21413092244177392,
//! ```
//!
//! and the generalized Hopf point is at `b* = -l1(0) / s`. `l1(0)` is the value of the
//! polynomial formula of SPEC section 8g, recomputed with numpy.

use pc_engine::{
    DdeHopfBranch, DdeHopfSetup, DdeModel, Evaluator, Options, StopReason, build_dde,
    dde_continue_hopf, dde_continue_hopf_with_lyapunov,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const WRIGHT3: &str = "par a=1.5, b=0\nu'=-a*delay(u,1)*(1+u)+b*u^3\ndone\n";
const BT: &str = "par a=-2, tau=1, b=1, c=-0.25\nx'=c-a*x-b*delay(x,tau)-x*x\ndone\n";
const ZH: &str = "par a=0, tau=1, b=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";

const L1_AT_ZERO: f64 = -0.214_130_922_441_773_92;

fn slope() -> f64 {
    6.0 / (PI * (1.0 + PI * PI / 4.0))
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

fn wright_setup() -> DdeHopfSetup {
    setup([0, 1], &[1.57, 0.0], &[0.001], 1.57)
}

fn with_l1(src: &str, which: Evaluator, s: &DdeHopfSetup, o: &Options) -> DdeHopfBranch {
    let mut m = dde(src, which);
    match dde_continue_hopf_with_lyapunov(&mut m, s, o, &AtomicBool::new(false)) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn without_l1(src: &str, which: Evaluator, s: &DdeHopfSetup, o: &Options) -> DdeHopfBranch {
    let mut m = dde(src, which);
    match dde_continue_hopf(&mut m, s, o, &AtomicBool::new(false)) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn labels(b: &DdeHopfBranch) -> Vec<&'static str> {
    b.specials.iter().map(|s| s.label).collect()
}

#[test]
fn l1_follows_the_closed_form_along_the_wright_cubic_curve() {
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        let b = with_l1(
            WRIGHT3,
            which,
            &wright_setup(),
            &opts(0.05, (-1.0, 1.0), 1.0),
        );
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        assert!(b.points.len() > 20, "{} points", b.points.len());
        for p in &b.points {
            let [a, bb] = p.alpha;
            assert!((a - PI / 2.0).abs() <= 1e-8, "a = {a}");
            assert!(
                (p.kappa - PI * PI / 4.0).abs() <= 1e-8,
                "kappa = {}",
                p.kappa
            );
            let want = L1_AT_ZERO + slope() * bb;
            let got = p.l1.expect("l1 is defined at every point of this curve");
            assert!((got - want).abs() <= 1e-7, "b = {bb}: l1 {got} vs {want}");
        }
    }
}

#[test]
fn a_zero_of_l1_is_one_gh_point_at_the_closed_form_b() {
    let b_star = -L1_AT_ZERO / slope();
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        let b = with_l1(
            WRIGHT3,
            which,
            &wright_setup(),
            &opts(0.05, (-1.0, 1.0), 1.0),
        );
        assert_eq!(labels(&b), ["GH"], "{:?}", b.specials);
        let gh = &b.specials[0];
        assert!((gh.alpha[1] - b_star).abs() <= 1e-6, "b = {}", gh.alpha[1]);
        assert!(
            (gh.alpha[0] - PI / 2.0).abs() <= 1e-8,
            "a = {}",
            gh.alpha[0]
        );
        assert!(
            (gh.kappa - PI * PI / 4.0).abs() <= 1e-8,
            "kappa = {}",
            gh.kappa
        );
        assert!(gh.x[0].abs() <= 1e-8, "x = {:?}", gh.x);
        // The index is the first point after the zero, so the points around it have
        // opposite signs of l1.
        let (before, after) = (b.points[gh.index - 1].l1, b.points[gh.index].l1);
        assert!(
            before.unwrap() < 0.0 && after.unwrap() > 0.0,
            "{before:?} {after:?}"
        );
    }
}

#[test]
fn the_other_direction_finds_the_same_gh_point() {
    let b_star = -L1_AT_ZERO / slope();
    let s = setup([0, 1], &[1.57, 0.9], &[0.001], 1.57);
    let b = with_l1(WRIGHT3, Evaluator::Jit, &s, &opts(0.05, (-1.0, 1.0), -1.0));
    assert_eq!(labels(&b), ["GH"], "{:?}", b.specials);
    assert!((b.specials[0].alpha[1] - b_star).abs() <= 1e-6);
}

#[test]
fn without_the_option_l1_is_none_and_the_curve_is_the_same() {
    let o = opts(0.05, (-1.0, 1.0), 1.0);
    let on = with_l1(WRIGHT3, Evaluator::Jit, &wright_setup(), &o);
    let off = without_l1(WRIGHT3, Evaluator::Jit, &wright_setup(), &o);
    assert!(off.specials.is_empty(), "{:?}", off.specials);
    assert!(off.points.iter().all(|p| p.l1.is_none()));
    assert_eq!(on.points.len(), off.points.len());
    for (p, q) in on.points.iter().zip(&off.points) {
        assert_eq!((&p.alpha, &p.x, p.kappa), (&q.alpha, &q.x, q.kappa));
    }
    assert_eq!(on.warnings, off.warnings);
}

#[test]
fn bt_and_zh_are_still_found_and_their_poles_give_no_gh() {
    // l1 has a pole at BT (omega -> 0) and at ZH (det Delta(0) -> 0). The locator drops a
    // sign change through a pole, so the label lists match those of the plain runs.
    let cases = [
        (
            BT,
            setup(
                [0, 1],
                &[-2.24, 1.3613444250345883, 1.0, -0.25],
                &[1.001],
                0.97,
            ),
            opts(0.05, (0.8, 1.5), -1.0),
            "BT",
        ),
        (
            ZH,
            setup(
                [0, 1],
                &[0.07, 1.4071360574511174, 1.0],
                &[0.001, -0.249],
                0.98,
            ),
            opts(0.05, (1.3, 2.0), 1.0),
            "ZH",
        ),
    ];
    for (src, s, o, want) in cases {
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            let on = with_l1(src, which.clone(), &s, &o);
            let off = without_l1(src, which, &s, &o);
            assert_eq!(labels(&on), [want], "{:?}", on.specials);
            assert_eq!(labels(&off), [want]);
            assert_eq!(on.points.len(), off.points.len());
            let at = on.specials[0].index;
            assert_eq!(at, off.specials[0].index);
            assert!(
                (on.specials[0].alpha[0] - off.specials[0].alpha[0]).abs() <= 1e-10,
                "{want} moved"
            );
            // Where kappa <= 0 there is no frequency, so no l1.
            for p in &on.points {
                if p.kappa <= 0.0 {
                    assert!(p.l1.is_none(), "{want}: l1 at kappa = {}", p.kappa);
                }
            }
        }
    }
}
