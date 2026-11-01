//! The model tests of SPEC §14 for curves of cycles: the LPC curve of Morris-Lecar in
//! (I, phi) and the PD curve of Rössler in (c, a). Each curve must meet the LPC or PD
//! points that LC runs find at the end values of its second parameter, and its points
//! must match the RK4 shooting oracle of `common::shoot`, which shares no code with the
//! collocation.

mod common;

use common::{Defining, load, rk4, shoot};
use pc_engine::{
    Branch, CycleCurveStart, CycleSetup, CycleStart, Evaluator, ModelEval, Options, Point,
    StopReason, build, continue_cycle, continue_lpc_curve, continue_pd_curve,
};
use std::sync::atomic::AtomicBool;

fn model_of(name: &str) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, Evaluator::Jit).unwrap()
}

/// Morris-Lecar parameters (I, phi, C, gL, gK, gCa, vL, vK, vCa, v1, v2, v3, v4).
fn ml_params(i: f64, phi: f64) -> Vec<f64> {
    vec![
        i, phi, 20.0, 2.0, 8.0, 4.4, -60.0, -84.0, 120.0, -1.2, 18.0, 2.0, 30.0,
    ]
}

/// RK4 with step `dt` from `x` for time `settle` onto the attractor at `params`, then
/// on to two upward crossings of x[coord] = level, which give the period. Returns the
/// orbit from the first crossing at the N m + 1 points of a uniform mesh with N m =
/// `nm`, and the period.
fn attractor_orbit(
    m: &mut dyn ModelEval,
    params: &[f64],
    mut x: Vec<f64>,
    (settle, dt): (f64, f64),
    (coord, level): (usize, f64),
    nm: usize,
) -> (Vec<f64>, f64) {
    let mut p = params.to_vec();
    p.push(0.0);
    for _ in 0..(settle / dt) as usize {
        rk4(m, &p, 1.0, &mut x, dt);
    }
    let (mut t, mut at, mut first) = (0.0, Vec::new(), Vec::new());
    while at.len() < 2 {
        let y0 = x[coord];
        rk4(m, &p, 1.0, &mut x, dt);
        t += dt;
        if y0 < level && x[coord] >= level {
            at.push(t);
            if first.is_empty() {
                first = x.clone();
            }
        }
        assert!(t <= settle, "no cycle at {params:?}");
    }
    let period = at[1] - at[0];
    let sub = 20;
    let mut y = first;
    let mut orbit = y.clone();
    for _ in 0..nm {
        for _ in 0..sub {
            rk4(m, &p, period, &mut y, 1.0 / (nm * sub) as f64);
        }
        orbit.extend(&y);
    }
    (orbit, period)
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

/// (T, a1) of a point of a curve of cycles.
fn period_a1(pt: &Point) -> (f64, f64) {
    let k = pt.u.len();
    (pt.u[k - 2], pt.u[k - 1])
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// Checks a curve point against the LC special point `sp` of a run in a1 at the same
/// a2: a1 and T must match to 1e-6 relative.
fn check_lc(pt: &Point, sp: &Point) {
    let (t, a1) = period_a1(pt);
    let t_lc = *sp.u.last().unwrap();
    assert!(
        rel(a1, sp.alpha) <= 1e-6,
        "a1 {a1} on the curve, {} on the LC run",
        sp.alpha
    );
    assert!(
        rel(t, t_lc) <= 1e-6,
        "T {t} on the curve, {t_lc} on the LC run"
    );
}

/// Checks `pt`, a point of a curve of cycles of a model with `n` states in the free
/// parameters `free` = [a1, a2], against the shooting oracle, started from a guess off by
/// 1e-4 in T and a1 with the section through u(0) of the point. T, a1 and u(0) must
/// match to 1e-6 relative.
fn check_oracle(
    m: &mut dyn ModelEval,
    params: &[f64],
    free: [usize; 2],
    pt: &Point,
    n: usize,
    cond: Defining,
) {
    let (t, a1) = period_a1(pt);
    let mut p = params.to_vec();
    p[free[1]] = pt.alpha;
    p[free[0]] = a1 * (1.0 + 1e-4);
    p.push(0.0);
    let u0 = &pt.u[..n];
    let (s_u0, s_t, s_a1) = shoot(m, &p, free[0], u0, t * (1.0 - 1e-4), 20_000, cond);
    println!(
        "oracle at a2 = {}: T {s_t} vs {t}, a1 {s_a1} vs {a1}",
        pt.alpha
    );
    assert!(
        rel(s_a1, a1) <= 1e-6,
        "a1 {a1} on the curve, {s_a1} by shooting"
    );
    assert!(rel(s_t, t) <= 1e-6, "T {t} on the curve, {s_t} by shooting");
    let scale = u0.iter().fold(1.0_f64, |s, v| s.max(v.abs()));
    for (a, b) in s_u0.iter().zip(u0) {
        assert!(
            (a - b).abs() <= 1e-6 * scale,
            "u(0) {u0:?} on the curve, {s_u0:?}"
        );
    }
}

/// Checks the run of a curve to the bound `end` of a2.
fn check_run(c: &Branch, end: f64) {
    assert_eq!(c.stop, StopReason::Bounds, "{:?}", c.warnings);
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    assert!((c.points.last().unwrap().alpha - end).abs() <= 1e-12);
}

/// The LC run of Morris-Lecar in I from the attractor at I = 100 finds the fold of
/// cycles below it, at phi = 0.04 and at phi = 0.06. The LPC curve from the first fold,
/// in (I, phi), ends at phi = 0.06 on the second, and its points match shooting.
#[test]
fn morris_lecar_lpc_curve_meets_lc_runs_and_shooting() {
    let mut m = model_of("morris_lecar");
    let (ntst, ncol) = (160, 4);
    let lc_opts = Options {
        direction: -1.0,
        max_newton: 15,
        hmax: 0.5,
        bounds: (80.0, 120.0),
        max_steps: 200,
        ..Options::default()
    };
    let mut folds = Vec::new();
    for phi in [0.04, 0.06] {
        let p = ml_params(100.0, phi);
        let (orbit, period) = attractor_orbit(
            m.as_mut(),
            &p,
            vec![-20.0, 0.1],
            (2000.0, 0.01),
            (0, 0.0),
            ntst * ncol,
        );
        let setup = CycleSetup::new(0, p, ntst, ncol, orbit, period);
        folds.push(lc_special(m.as_mut(), setup, &lc_opts, "LPC"));
    }
    let params = ml_params(100.0, 0.04);
    let start = CycleCurveStart::from_point([0, 1], &params, &folds[0], ntst, ncol);
    let opts = Options {
        max_newton: 15,
        hmax: 0.5,
        bounds: (0.02, 0.06),
        max_steps: 200,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let c = continue_lpc_curve(m.as_mut(), &start, &opts, &cancel, &mut |_| ()).unwrap();
    check_run(&c, 0.06);
    assert_eq!(c.points[0].alpha, 0.04);
    check_lc(&c.points[0], &folds[0]);
    let last = c.points.last().unwrap();
    check_lc(last, &folds[1]);
    // The fold moves up in I and the period falls as phi grows.
    assert!(period_a1(last).1 > period_a1(&c.points[0]).1 + 1.0);
    assert!(period_a1(last).0 < period_a1(&c.points[0]).0 - 10.0);
    for pt in [&c.points[c.points.len() / 2], last] {
        check_oracle(m.as_mut(), &params, [0, 1], pt, 2, Defining::Fold);
    }
}

/// The LC run of Rössler in c finds a period doubling at a = 0.2 (from c = 2.5), at
/// a = 0.25 (from c = 2) and at a = 0.15 (from c = 3). The PD curve from the first, in
/// (c, a), ends on the other two in either direction, and its ends match shooting.
#[test]
fn rossler_pd_curve_meets_lc_runs_and_shooting() {
    let mut m = model_of("rossler");
    let (ntst, ncol) = (40, 4);
    let mut flips = Vec::new();
    for (a, c0, bounds) in [
        (0.2, 2.5, (2.0, 2.9)),
        (0.25, 2.0, (1.8, 2.5)),
        (0.15, 3.0, (2.9, 4.2)),
    ] {
        let p = vec![a, 0.2, c0];
        let (orbit, period) = attractor_orbit(
            m.as_mut(),
            &p,
            vec![1.0, 1.0, 1.0],
            (1000.0, 0.005),
            (1, 0.0),
            ntst * ncol,
        );
        let setup = CycleSetup::new(2, p, ntst, ncol, orbit, period);
        let opts = Options {
            max_newton: 15,
            hmax: 0.05,
            bounds,
            ..Options::default()
        };
        flips.push(lc_special(m.as_mut(), setup, &opts, "PD"));
    }
    let params = vec![0.2, 0.2, 2.5];
    let start = CycleCurveStart::from_point([2, 0], &params, &flips[0], ntst, ncol);
    let cancel = AtomicBool::new(false);
    for (direction, end, flip) in [(1.0, 0.25, &flips[1]), (-1.0, 0.15, &flips[2])] {
        let opts = Options {
            direction,
            max_newton: 15,
            hmax: 0.05,
            bounds: (0.15, 0.25),
            ..Options::default()
        };
        let c = continue_pd_curve(m.as_mut(), &start, &opts, &cancel, &mut |_| ()).unwrap();
        check_run(&c, end);
        assert_eq!(c.points[0].alpha, 0.2);
        check_lc(&c.points[0], &flips[0]);
        let last = c.points.last().unwrap();
        check_lc(last, flip);
        check_oracle(m.as_mut(), &params, [2, 0], last, 3, Defining::Flip);
    }
}
