//! Curves of cycles in two parameters (SPEC §8): the defining systems against finite
//! differences, and each curve against the closed form of an analytic model, on both
//! evaluators.
//!
//! On `bautin_lpc.ode`, r' = r (b1 + b2 r^2 - r^4) and theta' = 1 + g r^2, the cycles are
//! circles with b1 + b2 r^2 - r^4 = 0, and two of them meet in a fold where b2 = 2 r^2.
//! The LPC curve is b1 = -b2^2 / 4, with r^2 = b2 / 2 and T = 2 pi / (1 + g b2 / 2).
//!
//! On `pd_twist.ode` the unit circle in the (x, y) plane is a cycle of period
//! T = 2 pi / (1 + a2) for all parameters, with multipliers -exp(T (a1 - a2^2)),
//! -exp(T l2) and 1. The PD curve is a1 = a2^2.
//!
//! On `ns4.ode` the circle r^2 = al in the (x, y) plane, with z = 0, is a cycle of period
//! 2 pi with multipliers 1, exp(-4 pi al) and exp(2 pi (be + c al)) exp(+-i 2 pi (w2 + e al)).
//! The NS curve is be = -c al with kappa = cos(2 pi (w2 + e al)).
//! Run with `-- --nocapture` to see the errors.

mod common;

use common::{load, rk4};
use pc_engine::{
    CycleCurveStart, CycleSetup, CycleStart, EngineError, Evaluator, LPC_CURVE_LABELS, LpcProblem,
    ModelEval, NS_CURVE_LABELS, NsProblem, Options, PD_CURVE_LABELS, PdProblem, Point, Problem,
    StopReason, build, continue_cycle, continue_lpc_curve, continue_ns_curve, continue_pd_curve,
};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(name: &str, which: &Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which.clone()).unwrap()
}

fn opts(bounds: (f64, f64)) -> Options {
    Options {
        max_newton: 15,
        hmax: 0.05,
        bounds,
        ..Options::default()
    }
}

/// The Bautin cycle of radius r at (b1, b2), with g = 0.5, on a uniform mesh of `ntst`
/// intervals of `ncol` points, starting on the x axis.
fn bautin_start(b1: f64, b2: f64, r: f64, ntst: usize, ncol: usize) -> CycleCurveStart {
    let pts = ntst * ncol + 1;
    let orbit = (0..pts)
        .flat_map(|k| {
            let t = 2.0 * PI * k as f64 / (pts - 1) as f64;
            [r * t.cos(), r * t.sin()]
        })
        .collect();
    CycleCurveStart {
        free: [0, 1],
        params: vec![b1, b2, 0.5],
        orbit,
        period: 2.0 * PI / (1.0 + 0.5 * r * r),
        mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
        ntst,
        ncol,
    }
}

/// The start on the exact LPC curve at b2.
fn bautin_lpc(b2: f64, ntst: usize, ncol: usize) -> CycleCurveStart {
    bautin_start(-b2 * b2 / 4.0, b2, (b2 / 2.0).sqrt(), ntst, ncol)
}

/// Largest error of the dense Jacobian of `p` at x against central differences, relative
/// to the largest entry of each row.
fn jacobian_error(p: &mut dyn Problem, x: &[f64]) -> f64 {
    let dim = p.dim();
    let rows = dim - 1;
    let mut jac = vec![0.0; rows * dim];
    p.jacobian(x, &mut jac).unwrap();
    let h = 1e-6;
    let mut fd = vec![0.0; rows * dim];
    let (mut fp, mut fm) = (vec![0.0; rows], vec![0.0; rows]);
    for q in 0..dim {
        let mut xs = x.to_vec();
        xs[q] = x[q] + h;
        p.residual(&xs, &mut fp).unwrap();
        xs[q] = x[q] - h;
        p.residual(&xs, &mut fm).unwrap();
        for r in 0..rows {
            fd[r + rows * q] = (fp[r] - fm[r]) / (2.0 * h);
        }
    }
    let mut worst = 0.0_f64;
    for r in 0..rows {
        let scale = (0..dim)
            .map(|q| fd[r + rows * q].abs())
            .fold(0.0, f64::max)
            .max(1e-300);
        for q in 0..dim {
            worst = worst.max((jac[r + rows * q] - fd[r + rows * q]).abs() / scale);
        }
    }
    worst
}

/// The dense Jacobian of the LPC system, with its G row, matches central differences on the
/// LPC curve and off it.
#[test]
fn lpc_jacobian_matches_finite_differences() {
    for which in BOTH {
        let mut m = model_of("bautin_lpc", &which);
        let mut p = LpcProblem::new(m.as_mut(), &bautin_lpc(1.0, 5, 3)).unwrap();
        let x0 = p.start().to_vec();
        let x1: Vec<f64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let i = i as f64;
                v + 1e-2 * ((0.7 * i).sin() + 0.3 * (1.3 * i).cos())
            })
            .collect();
        for (name, x) in [("on the curve", &x0), ("off the curve", &x1)] {
            let err = jacobian_error(&mut p, x);
            println!("{which:?} {name}: {err:.2e}");
            assert!(err < 1e-6, "{which:?} {name}: {err:.2e}");
        }
    }
}

/// From b2 = 1 the curve follows b1 = -b2^2 / 4 with r^2 = b2 / 2 and the exact period up
/// to the bound b2 = 1.5, with quadratic Newton convergence.
#[test]
fn lpc_curve_follows_the_bautin_fold() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("bautin_lpc", &which);
        let cancel = AtomicBool::new(false);
        let mut seen = 0;
        let branch = continue_lpc_curve(
            m.as_mut(),
            &bautin_lpc(1.0, ntst, ncol),
            &opts((0.5, 1.5)),
            &cancel,
            &mut |_| seen += 1,
        )
        .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert_eq!(seen, branch.points.len());
        assert!(branch.points.len() > 5);
        let last = branch.points.last().unwrap();
        assert!((last.alpha - 1.5).abs() < 1e-9, "{which:?}: {}", last.alpha);
        // The planar model has no multipliers besides the two at 1, so LPPD, LPNS and R1
        // are undefined. The CPC slot holds b, which has the closed form
        // b = -2 b2 (1 + b2 / 4) here (polar coordinates, SPEC 8a) and no zero on the
        // curve, so there is no special point.
        assert!(
            branch.specials.is_empty(),
            "{which:?}: {:?}",
            branch.specials
        );
        for pt in &branch.points {
            assert_eq!(pt.codim2.len(), LPC_CURVE_LABELS.len());
            for i in [0, 1, 3] {
                assert!(pt.codim2[i].is_nan(), "{:?}", pt.codim2);
            }
            let b = -2.0 * pt.alpha * (1.0 + pt.alpha / 4.0);
            assert!(
                (pt.codim2[2] - b).abs() < 1e-6 * b.abs(),
                "{which:?}: b {} against {b}",
                pt.codim2[2]
            );
        }
        let (mut e_b1, mut e_t, mut e_r, mut iters) = (0.0_f64, 0.0_f64, 0.0_f64, 0);
        for pt in &branch.points {
            let k = pt.u.len() - 2;
            let (b2, b1, t) = (pt.alpha, pt.u[k + 1], pt.u[k]);
            e_b1 = e_b1.max((b1 + b2 * b2 / 4.0).abs());
            e_t = e_t.max((t - 2.0 * PI / (1.0 + 0.25 * b2)).abs());
            assert_eq!(pt.mesh.len(), ntst + 1);
            for j in 0..=ntst {
                let i = 2 * j * ncol;
                let r2 = pt.u[i] * pt.u[i] + pt.u[i + 1] * pt.u[i + 1];
                e_r = e_r.max((r2 - b2 / 2.0).abs());
            }
            iters = iters.max(pt.newton_iters);
        }
        println!("{which:?}: b1 {e_b1:.2e}, T {e_t:.2e}, r^2 {e_r:.2e}, iters {iters}");
        assert!(e_b1 < 1e-8 && e_t < 1e-8 && e_r < 1e-8, "{which:?}");
        assert!(iters <= 4, "{which:?}: {iters} Newton iterations");
    }
}

/// The curve also runs to smaller b2 and from a start off the curve, which the start
/// corrects at fixed b2.
#[test]
fn lpc_curve_runs_down_from_a_start_off_the_curve() {
    let mut m = model_of("bautin_lpc", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let start = bautin_start(-0.24, 1.0, 0.72, 20, 4);
    let o = Options {
        direction: -1.0,
        ..opts((0.5, 1.5))
    };
    let branch = continue_lpc_curve(m.as_mut(), &start, &o, &cancel, &mut |_| ()).unwrap();
    assert_eq!(branch.stop, StopReason::Bounds);
    let first = &branch.points[0];
    let k = first.u.len() - 2;
    assert!((first.alpha - 1.0).abs() < 1e-12);
    assert!((first.u[k + 1] + 0.25).abs() < 1e-8, "{}", first.u[k + 1]);
    assert!((branch.points.last().unwrap().alpha - 0.5).abs() < 1e-9);
}

/// Starts that cannot define a curve are refused.
#[test]
fn lpc_curve_refuses_bad_starts() {
    let mut m = model_of("bautin_lpc", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = opts((0.5, 1.5));
    let mut same = bautin_lpc(1.0, 10, 4);
    same.free = [1, 1];
    let mut out_of_range = bautin_lpc(1.0, 10, 4);
    out_of_range.free = [0, 3];
    let mut short = bautin_lpc(1.0, 10, 4);
    short.orbit.pop();
    let mut no_period = bautin_lpc(1.0, 10, 4);
    no_period.period = 0.0;
    for (name, start) in [
        ("same parameters", same),
        ("index out of range", out_of_range),
        ("short orbit", short),
        ("zero period", no_period),
    ] {
        let r = continue_lpc_curve(m.as_mut(), &start, &o, &cancel, &mut |_| ());
        assert!(
            matches!(r, Err(EngineError::BadProblem(_))),
            "{name}: {r:?}"
        );
    }
}

/// The pd_twist cycle, the unit circle at speed 1 + a2 from (1, 0, 0), with l2 = -1, on a
/// uniform mesh of `ntst` intervals of `ncol` points.
fn pd_start(a1: f64, a2: f64, ntst: usize, ncol: usize) -> CycleCurveStart {
    let pts = ntst * ncol + 1;
    let orbit = (0..pts)
        .flat_map(|k| {
            let t = 2.0 * PI * k as f64 / (pts - 1) as f64;
            [t.cos(), t.sin(), 0.0]
        })
        .collect();
    CycleCurveStart {
        free: [0, 1],
        params: vec![a1, a2, -1.0],
        orbit,
        period: 2.0 * PI / (1.0 + a2),
        mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
        ntst,
        ncol,
    }
}

/// The monodromy of pd_twist at (1, 0, 0) is diag(-exp(T l1), 1, -exp(T l2)) with
/// l1 = a1 - a2^2, by central differences of an RK4 flow over one period.
#[test]
fn pd_twist_has_the_stated_multipliers() {
    let mut m = model_of("pd_twist", &Evaluator::Jit);
    let (a1, a2, l2) = (0.3, 0.2, -1.0);
    // The model reads the time after the parameters.
    let p = [a1, a2, l2, 0.0];
    let t = 2.0 * PI / (1.0 + a2);
    let want = [-(t * (a1 - a2 * a2)).exp(), 1.0, -(t * l2).exp()];
    let steps = 2000;
    let mut flow = |x0: [f64; 3]| {
        let mut x = x0.to_vec();
        for _ in 0..steps {
            rk4(m.as_mut(), &p, t, &mut x, 1.0 / steps as f64);
        }
        x
    };
    let h = 1e-6;
    let mut worst = 0.0_f64;
    for c in 0..3 {
        let (mut xp, mut xm) = ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        xp[c] += h;
        xm[c] -= h;
        let (fp, fm) = (flow(xp), flow(xm));
        for r in 0..3 {
            let d = (fp[r] - fm[r]) / (2.0 * h);
            let e = if r == c { want[c] } else { 0.0 };
            worst = worst.max((d - e).abs());
        }
    }
    println!("monodromy error {worst:.2e}");
    assert!(worst < 1e-7, "{worst:.2e}");
}

/// The dense Jacobian of the PD system, with its G row, matches central differences on the
/// PD curve and off it.
#[test]
fn pd_jacobian_matches_finite_differences() {
    for which in BOTH {
        let mut m = model_of("pd_twist", &which);
        let mut p = PdProblem::new(m.as_mut(), &pd_start(0.09, 0.3, 5, 3)).unwrap();
        let x0 = p.start().to_vec();
        let x1: Vec<f64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let i = i as f64;
                v + 1e-2 * ((0.7 * i).sin() + 0.3 * (1.3 * i).cos())
            })
            .collect();
        for (name, x) in [("on the curve", &x0), ("off the curve", &x1)] {
            let err = jacobian_error(&mut p, x);
            println!("{which:?} {name}: {err:.2e}");
            assert!(err < 1e-6, "{which:?} {name}: {err:.2e}");
        }
    }
}

/// From a2 = 0.2 the curve follows a1 = a2^2 on the unit circle with the exact period up
/// to the bound a2 = 0.5, with quadratic Newton convergence.
#[test]
fn pd_curve_follows_a1_equals_a2_squared() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("pd_twist", &which);
        let cancel = AtomicBool::new(false);
        let mut seen = 0;
        let branch = continue_pd_curve(
            m.as_mut(),
            &pd_start(0.04, 0.2, ntst, ncol),
            &opts((0.1, 0.5)),
            &cancel,
            &mut |_| seen += 1,
        )
        .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert_eq!(seen, branch.points.len());
        assert!(branch.points.len() > 5);
        let last = branch.points.last().unwrap();
        assert!((last.alpha - 0.5).abs() < 1e-9, "{which:?}: {}", last.alpha);
        assert!(
            branch.specials.is_empty(),
            "{which:?}: {:?}",
            branch.specials
        );
        let (mut e_a1, mut e_t, mut e_r, mut iters) = (0.0_f64, 0.0_f64, 0.0_f64, 0);
        let mut e_lppd = 0.0_f64;
        for pt in &branch.points {
            let k = pt.u.len() - 2;
            let (a2, a1, t) = (pt.alpha, pt.u[k + 1], pt.u[k]);
            // Without 1 and -1 the multipliers are -exp(-T), so LPPD = -exp(-T) - 1, and
            // PDNS, a product over pairs, is undefined.
            assert_eq!(pt.codim2.len(), PD_CURVE_LABELS.len());
            e_lppd = e_lppd.max((pt.codim2[0] + (-t).exp() + 1.0).abs());
            assert!(pt.codim2[1].is_nan(), "{:?}", pt.codim2);
            e_a1 = e_a1.max((a1 - a2 * a2).abs());
            e_t = e_t.max((t - 2.0 * PI / (1.0 + a2)).abs());
            assert_eq!(pt.mesh.len(), ntst + 1);
            for j in 0..=ntst {
                let u = &pt.u[3 * j * ncol..3 * j * ncol + 3];
                let r2 = u[0] * u[0] + u[1] * u[1];
                e_r = e_r.max((r2 - 1.0).abs()).max(u[2].abs());
            }
            iters = iters.max(pt.newton_iters);
        }
        println!(
            "{which:?}: a1 {e_a1:.2e}, T {e_t:.2e}, circle {e_r:.2e}, LPPD {e_lppd:.2e}, \
             iters {iters}"
        );
        assert!(e_a1 < 1e-8 && e_t < 1e-8 && e_r < 1e-8, "{which:?}");
        assert!(e_lppd < 1e-6, "{which:?}: {e_lppd:.2e}");
        assert!(iters <= 4, "{which:?}: {iters} Newton iterations");
    }
}

/// The curve also runs to smaller a2 and from a start off the curve, which the start
/// corrects at fixed a2.
#[test]
fn pd_curve_runs_down_from_a_start_off_the_curve() {
    let mut m = model_of("pd_twist", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = Options {
        direction: -1.0,
        ..opts((0.1, 0.5))
    };
    let branch = continue_pd_curve(
        m.as_mut(),
        &pd_start(0.06, 0.2, 20, 4),
        &o,
        &cancel,
        &mut |_| (),
    )
    .unwrap();
    assert_eq!(branch.stop, StopReason::Bounds);
    let first = &branch.points[0];
    let k = first.u.len() - 2;
    assert!((first.alpha - 0.2).abs() < 1e-12);
    assert!((first.u[k + 1] - 0.04).abs() < 1e-8, "{}", first.u[k + 1]);
    assert!((branch.points.last().unwrap().alpha - 0.1).abs() < 1e-9);
}

/// Starts that cannot define a curve are refused.
#[test]
fn pd_curve_refuses_bad_starts() {
    let mut m = model_of("pd_twist", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = opts((0.1, 0.5));
    let mut same = pd_start(0.04, 0.2, 10, 4);
    same.free = [1, 1];
    let mut out_of_range = pd_start(0.04, 0.2, 10, 4);
    out_of_range.free = [0, 3];
    let mut short = pd_start(0.04, 0.2, 10, 4);
    short.orbit.pop();
    let mut no_period = pd_start(0.04, 0.2, 10, 4);
    no_period.period = 0.0;
    for (name, start) in [
        ("same parameters", same),
        ("index out of range", out_of_range),
        ("short orbit", short),
        ("zero period", no_period),
    ] {
        let r = continue_pd_curve(m.as_mut(), &start, &o, &cancel, &mut |_| ());
        assert!(
            matches!(r, Err(EngineError::BadProblem(_))),
            "{name}: {r:?}"
        );
    }
}

/// The ns4 cycle, the circle of radius sqrt(al) at speed 1 from (sqrt(al), 0, 0, 0), with
/// c = 0.5, on a uniform mesh of `ntst` intervals of `ncol` points. The free parameters are
/// be and al, in that order.
fn ns_start(al: f64, be: f64, w2: f64, e: f64, ntst: usize, ncol: usize) -> CycleCurveStart {
    let pts = ntst * ncol + 1;
    let r = al.sqrt();
    let orbit = (0..pts)
        .flat_map(|k| {
            let t = 2.0 * PI * k as f64 / (pts - 1) as f64;
            [r * t.cos(), r * t.sin(), 0.0, 0.0]
        })
        .collect();
    CycleCurveStart {
        free: [1, 0],
        params: vec![al, be, 0.5, w2, e],
        orbit,
        period: 2.0 * PI,
        mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
        ntst,
        ncol,
    }
}

/// The exact kappa of the ns4 NS curve at al, with w2 = 0.2 and e = 0.5.
fn ns_kappa(al: f64) -> f64 {
    (2.0 * PI * (0.2 + 0.5 * al)).cos()
}

/// The monodromy of ns4 at (sqrt(al), 0, 0, 0) is diag(exp(-4 pi al), 1, rho R(phi)) with
/// rho = exp(2 pi (be + c al)) and R(phi) the rotation by phi = 2 pi (w2 + e al), by central
/// differences of an RK4 flow over one period.
#[test]
fn ns4_has_the_stated_multipliers() {
    let mut m = model_of("ns4", &Evaluator::Jit);
    let (al, be, c, w2, e) = (0.2, -0.05, 0.5, 0.2, 0.5);
    // The model reads the time after the parameters.
    let p = [al, be, c, w2, e, 0.0];
    let t = 2.0 * PI;
    let (rho, phi) = ((t * (be + c * al)).exp(), t * (w2 + e * al));
    let want = [
        [(-2.0 * t * al).exp(), 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, rho * phi.cos(), -rho * phi.sin()],
        [0.0, 0.0, rho * phi.sin(), rho * phi.cos()],
    ];
    let steps = 2000;
    let mut flow = |x0: [f64; 4]| {
        let mut x = x0.to_vec();
        for _ in 0..steps {
            rk4(m.as_mut(), &p, t, &mut x, 1.0 / steps as f64);
        }
        x
    };
    let h = 1e-6;
    let x0 = [al.sqrt(), 0.0, 0.0, 0.0];
    let mut worst = 0.0_f64;
    for col in 0..4 {
        let (mut xp, mut xm) = (x0, x0);
        xp[col] += h;
        xm[col] -= h;
        let (fp, fm) = (flow(xp), flow(xm));
        for (row, w) in want.iter().enumerate() {
            let d = (fp[row] - fm[row]) / (2.0 * h);
            worst = worst.max((d - w[col]).abs());
        }
    }
    println!("monodromy error {worst:.2e}");
    assert!(worst < 1e-7, "{worst:.2e}");
}

/// The dense Jacobian of the NS system, with its two G rows, matches central differences on
/// the NS curve and off it.
#[test]
fn ns_jacobian_matches_finite_differences() {
    for which in BOTH {
        let mut m = model_of("ns4", &which);
        let mut p = NsProblem::new(m.as_mut(), &ns_start(0.2, -0.1, 0.2, 0.5, 5, 3)).unwrap();
        let x0 = p.start().to_vec();
        let x1: Vec<f64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let i = i as f64;
                v + 1e-2 * ((0.7 * i).sin() + 0.3 * (1.3 * i).cos())
            })
            .collect();
        for (name, x) in [("on the curve", &x0), ("off the curve", &x1)] {
            let err = jacobian_error(&mut p, x);
            println!("{which:?} {name}: {err:.2e}");
            assert!(err < 1e-6, "{which:?} {name}: {err:.2e}");
        }
    }
}

/// The start takes kappa from the critical pair of multipliers.
#[test]
fn ns_start_takes_kappa_from_the_multipliers() {
    let mut m = model_of("ns4", &Evaluator::Jit);
    // Off the curve the pair has modulus exp(0.04 pi), and kappa is still cos(phi).
    let p = NsProblem::new(m.as_mut(), &ns_start(0.2, -0.08, 0.2, 0.5, 40, 4)).unwrap();
    let x0 = p.start();
    let k = x0.len() - 4;
    println!("kappa {:.15}, want {:.15}", x0[k + 2], ns_kappa(0.2));
    assert!((x0[k + 2] - ns_kappa(0.2)).abs() < 1e-9, "{}", x0[k + 2]);
    assert_eq!(&x0[k..k + 2], &[2.0 * PI, -0.08]);
    assert_eq!(x0[k + 3], 0.2);
}

/// From al = 0.2 the curve follows be = -al / 2 with kappa = cos(2 pi (0.2 + al / 2)) on the
/// exact circle and period up to the bound al = 0.5, with quadratic Newton convergence.
#[test]
fn ns_curve_follows_be_equals_minus_c_al() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("ns4", &which);
        let cancel = AtomicBool::new(false);
        let mut seen = 0;
        let branch = continue_ns_curve(
            m.as_mut(),
            &ns_start(0.2, -0.1, 0.2, 0.5, ntst, ncol),
            &opts((0.12, 0.5)),
            &cancel,
            &mut |_| seen += 1,
        )
        .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert_eq!(seen, branch.points.len());
        assert!(branch.points.len() > 5);
        let last = branch.points.last().unwrap();
        assert!((last.alpha - 0.5).abs() < 1e-9, "{which:?}: {}", last.alpha);
        let (mut e_be, mut e_k, mut e_t, mut e_r, mut iters) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0);
        for pt in &branch.points {
            let k = pt.u.len() - 3;
            let (al, t, be, kappa) = (pt.alpha, pt.u[k], pt.u[k + 1], pt.u[k + 2]);
            e_be = e_be.max((be + 0.5 * al).abs());
            e_k = e_k.max((kappa - ns_kappa(al)).abs());
            e_t = e_t.max((t - 2.0 * PI).abs());
            assert_eq!(pt.mesh.len(), ntst + 1);
            for j in 0..=ntst {
                let u = &pt.u[4 * j * ncol..4 * j * ncol + 4];
                let r2 = u[0] * u[0] + u[1] * u[1];
                e_r = e_r.max((r2 - al).abs()).max(u[2].abs()).max(u[3].abs());
            }
            iters = iters.max(pt.newton_iters);
        }
        println!(
            "{which:?}: be {e_be:.2e}, kappa {e_k:.2e}, T {e_t:.2e}, circle {e_r:.2e}, \
             iters {iters}"
        );
        assert!(
            e_be < 1e-8 && e_k < 1e-8 && e_t < 1e-8 && e_r < 1e-8,
            "{which:?}"
        );
        assert!(iters <= 4, "{which:?}: {iters} Newton iterations");
    }
}

/// From al = 0.06 to 0.5 kappa = cos(2 pi (0.2 + al / 2)) falls from 0.13 to -0.95 and
/// crosses the 1:4 resonance at al = 0.1 and the 1:3 resonance at al = 4/15. Without 1
/// and the critical pair the multiplier is exp(-4 pi al), which gives LPNS and PDNS.
#[test]
fn ns_curve_finds_the_resonances_r4_and_r3() {
    for which in BOTH {
        let mut m = model_of("ns4", &which);
        let cancel = AtomicBool::new(false);
        let branch = continue_ns_curve(
            m.as_mut(),
            &ns_start(0.06, -0.03, 0.2, 0.5, 40, 4),
            &opts((0.05, 0.5)),
            &cancel,
            &mut |_| (),
        )
        .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert!(
            branch.warnings.is_empty(),
            "{which:?}: {:?}",
            branch.warnings
        );
        let labels: Vec<_> = branch.specials.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["R4", "R3"], "{which:?}");
        for (sp, (al, kappa)) in branch.specials.iter().zip([(0.1, 0.0), (4.0 / 15.0, -0.5)]) {
            let u = &sp.point.u;
            println!(
                "{which:?} {}: al {:.12} (want {al:.12}), kappa {:.2e}",
                sp.label,
                sp.point.alpha,
                u[u.len() - 1] - kappa
            );
            assert!((sp.point.alpha - al).abs() < 1e-8, "{which:?} {}", sp.label);
            assert!(
                (u[u.len() - 1] - kappa).abs() < 1e-8,
                "{which:?} {}",
                sp.label
            );
        }
        let (mut e_lpns, mut e_pdns) = (0.0_f64, 0.0_f64);
        for pt in &branch.points {
            assert_eq!(pt.codim2.len(), NS_CURVE_LABELS.len());
            let mu = (-4.0 * PI * pt.alpha).exp();
            e_lpns = e_lpns.max((pt.codim2[4] - (mu - 1.0)).abs());
            e_pdns = e_pdns.max((pt.codim2[5] - (mu + 1.0)).abs());
        }
        println!("{which:?}: LPNS {e_lpns:.2e}, PDNS {e_pdns:.2e}");
        assert!(e_lpns < 1e-6 && e_pdns < 1e-6, "{which:?}");
    }
}

/// The curve also runs to smaller al and from a start off the curve, which the start
/// corrects at fixed al.
#[test]
fn ns_curve_runs_down_from_a_start_off_the_curve() {
    let mut m = model_of("ns4", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = Options {
        direction: -1.0,
        ..opts((0.12, 0.5))
    };
    let branch = continue_ns_curve(
        m.as_mut(),
        &ns_start(0.2, -0.08, 0.2, 0.5, 20, 4),
        &o,
        &cancel,
        &mut |_| (),
    )
    .unwrap();
    assert_eq!(branch.stop, StopReason::Bounds);
    let first = &branch.points[0];
    let k = first.u.len() - 3;
    assert!((first.alpha - 0.2).abs() < 1e-12);
    assert!((first.u[k + 1] + 0.1).abs() < 1e-8, "{}", first.u[k + 1]);
    assert!((first.u[k + 2] - ns_kappa(0.2)).abs() < 1e-8);
    assert!((branch.points.last().unwrap().alpha - 0.12).abs() < 1e-9);
}

/// Starts that cannot define a curve are refused, among them a cycle without a complex
/// pair of multipliers.
#[test]
fn ns_curve_refuses_bad_starts() {
    let mut m = model_of("ns4", &Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let o = opts((0.12, 0.5));
    let mut same = ns_start(0.2, -0.1, 0.2, 0.5, 10, 4);
    same.free = [1, 1];
    let mut out_of_range = ns_start(0.2, -0.1, 0.2, 0.5, 10, 4);
    out_of_range.free = [0, 5];
    let mut short = ns_start(0.2, -0.1, 0.2, 0.5, 10, 4);
    short.orbit.pop();
    let mut no_period = ns_start(0.2, -0.1, 0.2, 0.5, 10, 4);
    no_period.period = 0.0;
    for (name, start) in [
        ("same parameters", same),
        ("index out of range", out_of_range),
        ("short orbit", short),
        ("zero period", no_period),
    ] {
        let r = continue_ns_curve(m.as_mut(), &start, &o, &cancel, &mut |_| ());
        assert!(
            matches!(r, Err(EngineError::BadProblem(_))),
            "{name}: {r:?}"
        );
    }
    // The pd_twist multipliers are all real.
    let mut twist = model_of("pd_twist", &Evaluator::Jit);
    let r = continue_ns_curve(
        twist.as_mut(),
        &pd_start(0.04, 0.2, 10, 4),
        &opts((0.1, 0.5)),
        &cancel,
        &mut |_| (),
    );
    match r {
        Err(EngineError::BadProblem(msg)) => assert!(msg.contains("complex"), "{msg}"),
        other => panic!("no complex pair: {other:?}"),
    }
}

/// The one special point labelled `label` of an LC run in `free` from the orbit of `start`,
/// within `bounds` and in `direction`.
fn lc_special(
    m: &mut dyn ModelEval,
    start: &CycleCurveStart,
    free: usize,
    bounds: (f64, f64),
    direction: f64,
    label: &str,
) -> Point {
    let setup = CycleSetup::new(
        free,
        start.params.clone(),
        start.ntst,
        start.ncol,
        start.orbit.clone(),
        start.period,
    );
    let o = Options {
        direction,
        ..opts(bounds)
    };
    let cancel = AtomicBool::new(false);
    let branch = continue_cycle(m, &CycleStart::Orbit(setup), &o, &cancel, &mut |_| ()).unwrap();
    assert_eq!(branch.stop, StopReason::Bounds, "{:?}", branch.warnings);
    let found: Vec<_> = branch
        .specials
        .iter()
        .filter(|s| s.label == label)
        .collect();
    assert_eq!(found.len(), 1, "{label}: {:?}", branch.specials);
    found[0].point.clone()
}

/// The LC run in b1 at b2 = 1 from the outer cycle r^2 = 0.7 turns at the LPC point
/// b1 = -1/4, and the LPC curve from that point follows b1 = -b2^2 / 4.
#[test]
fn lpc_curve_starts_at_the_lpc_point_of_a_cycle_branch() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("bautin_lpc", &which);
        let orbit = bautin_start(-0.21, 1.0, 0.7_f64.sqrt(), ntst, ncol);
        let lpc = lc_special(m.as_mut(), &orbit, 0, (-0.3, -0.15), -1.0, "LPC");
        println!("{which:?}: LPC at b1 = {:.12}", lpc.alpha);
        assert!((lpc.alpha + 0.25).abs() < 1e-6, "{which:?}: {}", lpc.alpha);
        let start = CycleCurveStart::from_point([0, 1], &orbit.params, &lpc, ntst, ncol);
        let cancel = AtomicBool::new(false);
        let branch =
            continue_lpc_curve(m.as_mut(), &start, &opts((0.5, 1.5)), &cancel, &mut |_| ())
                .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert!((branch.points[0].alpha - 1.0).abs() < 1e-12);
        let e_b1 = branch
            .points
            .iter()
            .map(|pt| (pt.u[pt.u.len() - 1] + pt.alpha * pt.alpha / 4.0).abs())
            .fold(0.0, f64::max);
        println!(
            "{which:?}: b1 {e_b1:.2e} over {} points",
            branch.points.len()
        );
        assert!(e_b1 < 1e-8, "{which:?}: {e_b1:.2e}");
    }
}

/// The LC run in a1 at a2 = 0.2 finds the PD point a1 = 0.04, and the PD curve from that
/// point follows a1 = a2^2.
#[test]
fn pd_curve_starts_at_the_pd_point_of_a_cycle_branch() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("pd_twist", &which);
        let orbit = pd_start(0.0, 0.2, ntst, ncol);
        let pd = lc_special(m.as_mut(), &orbit, 0, (-0.05, 0.1), 1.0, "PD");
        println!("{which:?}: PD at a1 = {:.12}", pd.alpha);
        assert!((pd.alpha - 0.04).abs() < 1e-6, "{which:?}: {}", pd.alpha);
        let start = CycleCurveStart::from_point([0, 1], &orbit.params, &pd, ntst, ncol);
        let cancel = AtomicBool::new(false);
        let branch =
            continue_pd_curve(m.as_mut(), &start, &opts((0.1, 0.5)), &cancel, &mut |_| ()).unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert!((branch.points[0].alpha - 0.2).abs() < 1e-12);
        let e_a1 = branch
            .points
            .iter()
            .map(|pt| (pt.u[pt.u.len() - 1] - pt.alpha * pt.alpha).abs())
            .fold(0.0, f64::max);
        println!(
            "{which:?}: a1 {e_a1:.2e} over {} points",
            branch.points.len()
        );
        assert!(e_a1 < 1e-8, "{which:?}: {e_a1:.2e}");
    }
}

/// The LC run in be at al = 0.2 finds the NS point be = -0.1, and the NS curve from that
/// point follows be = -al / 2 with the exact kappa.
#[test]
fn ns_curve_starts_at_the_ns_point_of_a_cycle_branch() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("ns4", &which);
        let orbit = ns_start(0.2, -0.15, 0.2, 0.5, ntst, ncol);
        let ns = lc_special(m.as_mut(), &orbit, 1, (-0.2, -0.05), 1.0, "NS");
        println!("{which:?}: NS at be = {:.12}", ns.alpha);
        assert!((ns.alpha + 0.1).abs() < 1e-6, "{which:?}: {}", ns.alpha);
        let start = CycleCurveStart::from_point([1, 0], &orbit.params, &ns, ntst, ncol);
        let cancel = AtomicBool::new(false);
        let branch =
            continue_ns_curve(m.as_mut(), &start, &opts((0.12, 0.5)), &cancel, &mut |_| ())
                .unwrap();
        assert_eq!(branch.stop, StopReason::Bounds, "{which:?}");
        assert!((branch.points[0].alpha - 0.2).abs() < 1e-12);
        let (mut e_be, mut e_k) = (0.0_f64, 0.0_f64);
        for pt in &branch.points {
            let k = pt.u.len() - 3;
            e_be = e_be.max((pt.u[k + 1] + 0.5 * pt.alpha).abs());
            e_k = e_k.max((pt.u[k + 2] - ns_kappa(pt.alpha)).abs());
        }
        println!(
            "{which:?}: be {e_be:.2e}, kappa {e_k:.2e} over {} points",
            branch.points.len()
        );
        assert!(e_be < 1e-8 && e_k < 1e-8, "{which:?}");
    }
}

/// A special point of the wrong type does not start a curve: the Bautin LPC point has no
/// multiplier -1, the ns4 NS point no double multiplier 1, and the pd_twist PD point no
/// complex pair.
#[test]
fn curves_refuse_special_points_of_another_type() {
    let (ntst, ncol) = (20, 4);
    let cancel = AtomicBool::new(false);

    let mut bautin = model_of("bautin_lpc", &Evaluator::Jit);
    let orbit = bautin_start(-0.21, 1.0, 0.7_f64.sqrt(), ntst, ncol);
    let lpc = lc_special(bautin.as_mut(), &orbit, 0, (-0.3, -0.15), -1.0, "LPC");
    let start = CycleCurveStart::from_point([0, 1], &orbit.params, &lpc, ntst, ncol);
    let r = continue_pd_curve(
        bautin.as_mut(),
        &start,
        &opts((0.5, 1.5)),
        &cancel,
        &mut |_| (),
    );
    println!("PD curve from an LPC point: {r:?}");
    assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");

    let mut ns4 = model_of("ns4", &Evaluator::Jit);
    let orbit = ns_start(0.2, -0.15, 0.2, 0.5, ntst, ncol);
    let ns = lc_special(ns4.as_mut(), &orbit, 1, (-0.2, -0.05), 1.0, "NS");
    let start = CycleCurveStart::from_point([1, 0], &orbit.params, &ns, ntst, ncol);
    let r = continue_lpc_curve(
        ns4.as_mut(),
        &start,
        &opts((0.12, 0.5)),
        &cancel,
        &mut |_| (),
    );
    println!("LPC curve from an NS point: {r:?}");
    assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");

    let mut twist = model_of("pd_twist", &Evaluator::Jit);
    let orbit = pd_start(0.0, 0.2, ntst, ncol);
    let pd = lc_special(twist.as_mut(), &orbit, 0, (-0.05, 0.1), 1.0, "PD");
    let start = CycleCurveStart::from_point([0, 1], &orbit.params, &pd, ntst, ncol);
    let r = continue_ns_curve(
        twist.as_mut(),
        &start,
        &opts((0.1, 0.5)),
        &cancel,
        &mut |_| (),
    );
    println!("NS curve from a PD point: {r:?}");
    match r {
        Err(EngineError::BadProblem(msg)) => assert!(msg.contains("complex"), "{msg}"),
        other => panic!("NS curve from a PD point: {other:?}"),
    }
}
