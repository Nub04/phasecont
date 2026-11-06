//! The homoclinic HOM curve (SPEC §8b): the defining system against finite differences
//! and the curve against the closed forms of Sandstede's model, on both evaluators.
//!
//! On `sandstede.ode` at mu = mut = alpha = 0 the plane z = 0 is invariant and holds the
//! orbit x = 1 - w^2, y = -w (1 - w^2), w' = (1 - w^2)(b - a w) / 2, homoclinic to the
//! origin for |a| < b. The origin is an equilibrium for all parameters, with eigenvalues
//! a +- sqrt(b^2 + 4 mut^2) and c; at mut = 0 they are a - b, a + b and c.
//! Run with `-- --nocapture` to see the measured values.

mod common;

use common::load;
use pc_engine::{
    Evaluator, HOM_CURVE_LABELS, HomProblem, HomStart, HomStartKind, ModelEval, Options, Problem,
    StopReason, build, continue_hom_curve,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

/// Parameter indices of `sandstede.ode`.
const A: usize = 0;
const MU: usize = 3;
const MUT: usize = 4;

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

/// w(t) of w' = (1 - w^2)(b - a w) / 2 with w(0) = 0, by RK4 with about `per_unit` steps
/// per time unit, at the times `ts` (sorted ascending, any sign).
fn w_at(a: f64, b: f64, ts: &[f64], per_unit: usize) -> Vec<f64> {
    let f = |w: f64| (1.0 - w * w) * (b - a * w) / 2.0;
    let step = |w: f64, h: f64| {
        let k1 = f(w);
        let k2 = f(w + h / 2.0 * k1);
        let k3 = f(w + h / 2.0 * k2);
        let k4 = f(w + h * k3);
        w + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    };
    ts.iter()
        .map(|&t| {
            let steps = ((t.abs() * per_unit as f64).ceil() as usize).max(1);
            let h = t / steps as f64;
            (0..steps).fold(0.0, |w, _| step(w, h))
        })
        .collect()
}

/// The plane orbit of Sandstede's model, u(tau) = x(-T/2 + T tau) with z = 0, at the base
/// points of a uniform mesh of `ntst` intervals of `ncol` points.
fn plane_orbit(a: f64, b: f64, period: f64, ntst: usize, ncol: usize) -> Vec<f64> {
    let pts = ntst * ncol + 1;
    let ts: Vec<f64> = (0..pts)
        .map(|k| period * (k as f64 / (pts - 1) as f64 - 0.5))
        .collect();
    w_at(a, b, &ts, 1000)
        .into_iter()
        .flat_map(|w| [1.0 - w * w, -w * (1.0 - w * w), 0.0])
        .collect()
}

/// The `"orbit"` start of test 1 at (a, b, c) and T, free (mut, a).
fn m1_start(a: f64, period: f64, ntst: usize, ncol: usize) -> HomStart {
    HomStart {
        kind: HomStartKind::Orbit,
        free: [MUT, A],
        params: vec![a, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        orbit: plane_orbit(a, 1.0, period, ntst, ncol),
        period,
        mesh: None,
        ntst,
        ncol,
    }
}

/// Largest relative entry error of the dense Jacobian against central differences, scaled
/// by the largest entry of each row.
fn jacobian_error(p: &mut dyn Problem, x: &[f64]) -> f64 {
    let dim = p.dim();
    let rows = dim - 1;
    let mut jac = vec![0.0; rows * dim];
    p.jacobian(x, &mut jac).unwrap();
    let mut fd = vec![0.0; rows * dim];
    let (mut fp, mut fm) = (vec![0.0; rows], vec![0.0; rows]);
    for q in 0..dim {
        let h = 1e-6 * (1.0 + x[q].abs());
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

/// The dense Jacobian of the HOM system matches central differences at the start and at a
/// point moved off the curve in every unknown.
#[test]
fn hom_jacobian_matches_finite_differences() {
    for which in BOTH {
        let mut m = model_of("sandstede", &which);
        let mut s = m1_start(-0.25, 10.0, 6, 3);
        s.params[MU] = 0.05;
        let mut p = HomProblem::new(m.as_mut(), &s, &Options::default()).unwrap();
        let x0 = p.start().to_vec();
        let x1: Vec<f64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.01 * ((i as f64 * 0.7).sin()))
            .collect();
        for x in [&x0, &x1] {
            let e = jacobian_error(&mut p, x);
            println!("{which:?}: HOM Jacobian error {e:e}");
            assert!(e < 1e-6, "{which:?}: error {e:e}");
        }
    }
}

/// Test 1 of SPEC §8b: the M1 curve in (mut, a) from a = -0.25 to 0.25 on meshes of
/// T / 8 intervals. x0 and z stay at zero, NSA = 2a is the only label, and |mut| is the
/// truncation error at T = 20 and at the rounding floor from T = 30 on.
#[test]
fn sandstede_curve_has_one_neutral_saddle() {
    let runs = [(20.0, 160, 1e-10), (30.0, 240, 1e-13), (40.0, 320, 1e-13)];
    for which in BOTH {
        let mut first = f64::NAN;
        // The fallback evaluator is slow on these meshes, so it runs T = 20 only.
        let runs = if which == Evaluator::Jit {
            &runs[..]
        } else {
            &runs[..1]
        };
        for &(period, nt, bound) in runs {
            let mut m = model_of("sandstede", &which);
            let start = m1_start(-0.25, period, nt, 4);
            let cancel = AtomicBool::new(false);
            let b = continue_hom_curve(
                m.as_mut(),
                &start,
                &opts((-0.26, 0.26)),
                &cancel,
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(
                b.stop,
                StopReason::Bounds,
                "{which:?} T={period}: {:?}",
                b.stop
            );
            let n = 3;
            let ext = n * (nt * 4 + 1);
            let (mut x0max, mut zmax, mut mutmax) = (0.0_f64, 0.0_f64, 0.0_f64);
            for pt in &b.points {
                let x0 = &pt.u[ext + 1..ext + 1 + n];
                x0max = x0.iter().fold(x0max, |a, v| a.max(v.abs()));
                zmax = (0..ext / n).fold(zmax, |a, j| a.max(pt.u[j * n + 2].abs()));
                mutmax = mutmax.max(pt.u[ext + 1 + n].abs());
            }
            let labels: Vec<(&str, f64)> = b
                .specials
                .iter()
                .map(|s| (s.label.as_str(), s.point.alpha))
                .collect();
            println!(
                "{which:?} T={period}: {} points, |x0| {x0max:e}, |z| {zmax:e}, |mut| \
                 {mutmax:e}, labels {labels:?}",
                b.points.len()
            );
            assert!(x0max < 1e-12, "|x0| = {x0max:e}");
            assert!(zmax < 1e-12, "|z| = {zmax:e}");
            assert_eq!(labels.len(), 1, "{labels:?}");
            assert_eq!(labels[0].0, "NSA");
            assert!(labels[0].1.abs() < 1e-10, "NSA at a = {}", labels[0].1);
            assert!(mutmax < bound, "T={period}: |mut| = {mutmax:e}");
            if period == 20.0 {
                first = mutmax;
            } else {
                assert!(
                    mutmax < first / 100.0,
                    "|mut| {mutmax:e} did not fall from {first:e}"
                );
            }
            assert_eq!(b.points[0].codim2.len(), HOM_CURVE_LABELS.len());
        }
    }
}
