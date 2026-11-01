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
//! Run with `-- --nocapture` to see the errors.

mod common;

use common::{load, rk4};
use pc_engine::{
    CycleCurveStart, EngineError, Evaluator, LpcProblem, ModelEval, Options, PdProblem, Problem,
    StopReason, build, continue_lpc_curve, continue_pd_curve,
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
        let (mut e_a1, mut e_t, mut e_r, mut iters) = (0.0_f64, 0.0_f64, 0.0_f64, 0);
        for pt in &branch.points {
            let k = pt.u.len() - 2;
            let (a2, a1, t) = (pt.alpha, pt.u[k + 1], pt.u[k]);
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
        println!("{which:?}: a1 {e_a1:.2e}, T {e_t:.2e}, circle {e_r:.2e}, iters {iters}");
        assert!(e_a1 < 1e-8 && e_t < 1e-8 && e_r < 1e-8, "{which:?}");
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
