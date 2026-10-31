//! The collocation problem for periodic orbits (SPEC §7): the Jacobian against central
//! differences, the condensed bordered solve against a dense LU, and SPEC §9 acceptance
//! Tests 3 and 4 on the Hopf normal form, each on both evaluators.
//!
//! On `hopf_c.ode` with c = -1 the orbit at alpha > 0 is the circle of radius sqrt(alpha),
//! run counterclockwise with period 2 pi, and its nontrivial Floquet multiplier is
//! exp(-4 pi alpha). Run with `-- --nocapture` to see the errors.

mod common;

use common::load;
use pc_engine::{
    CycleProblem, CycleSetup, EngineError, Evaluator, ModelEval, Options, PeriodicOrbit, Problem,
    build, periodic_orbit,
};
use pc_linalg::Lu;
use std::f64::consts::PI;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load("hopf_c");
    build(&model, &info, which).unwrap()
}

/// Profile at the N m + 1 base points of a uniform mesh: radius `r(theta)` along the angle
/// theta = 2 pi tau, point-major.
fn profile(ntst: usize, ncol: usize, r: impl Fn(f64) -> f64) -> Vec<f64> {
    let nm = ntst * ncol;
    let mut u = Vec::with_capacity(2 * (nm + 1));
    for k in 0..=nm {
        let th = 2.0 * PI * k as f64 / nm as f64;
        u.push(r(th) * th.cos());
        u.push(r(th) * th.sin());
    }
    u
}

fn circle(alpha: f64, ntst: usize, ncol: usize, r: f64, period: f64) -> CycleSetup {
    CycleSetup::new(
        0,
        vec![alpha, -1.0],
        ntst,
        ncol,
        profile(ntst, ncol, |_| r),
        period,
    )
}

/// The unknowns (u, T, alpha) of a setup.
fn unknowns(s: &CycleSetup) -> Vec<f64> {
    let mut x = s.profile.clone();
    x.push(s.period);
    x.push(s.params[s.free]);
    x
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// A generic point: a perturbed, non-periodic circle that solves no equation.
fn generic_setup() -> (CycleSetup, Vec<f64>) {
    let (ntst, ncol) = (3, 3);
    let mut s = CycleSetup::new(
        0,
        vec![0.4, -1.0],
        ntst,
        ncol,
        profile(ntst, ncol, |th| 0.7 + 0.1 * (3.0 * th).sin()),
        6.1,
    );
    let mut x = unknowns(&s);
    for (k, v) in x.iter_mut().enumerate() {
        *v += 0.01 * ((k * 7 % 11) as f64 - 5.0) / 5.0;
    }
    s.profile = x[..x.len() - 2].to_vec();
    s.profile[0] += 0.02;
    (s, x)
}

#[test]
fn jacobian_matches_central_differences() {
    for which in BOTH {
        let mut m = model_of(which.clone());
        let (s, x) = generic_setup();
        let mut p = CycleProblem::new(m.as_mut(), &s).unwrap();
        let dim = p.dim();
        assert_eq!(dim, 2 * (3 * 3 + 1) + 2);
        assert_eq!(p.jacobian_len(), 0);
        let mut jac = vec![0.0; (dim - 1) * dim];
        p.jacobian(&x, &mut jac).unwrap();
        let scale = max_abs(&jac);
        let (mut fp, mut fm) = (vec![0.0; dim - 1], vec![0.0; dim - 1]);
        let mut worst = 0.0_f64;
        for col in 0..dim {
            let h = 1e-6 * x[col].abs().max(1.0);
            let mut xp = x.clone();
            xp[col] += h;
            p.residual(&xp, &mut fp).unwrap();
            xp[col] -= 2.0 * h;
            p.residual(&xp, &mut fm).unwrap();
            for row in 0..dim - 1 {
                let fd = (fp[row] - fm[row]) / (2.0 * h);
                worst = worst.max((fd - jac[row + (dim - 1) * col]).abs());
            }
        }
        println!("[{which:?}] Jacobian: max |FD - J| {worst:.3e}, max |J| {scale:.3e}");
        assert!(worst <= 1e-7 * scale, "[{which:?}] {worst:e}");
    }
}

#[test]
fn bordered_solve_matches_a_dense_lu() {
    let mut m = model_of(Evaluator::Jit);
    let (s, x) = generic_setup();
    let mut p = CycleProblem::new(m.as_mut(), &s).unwrap();
    let dim = p.dim();
    let mut jac = vec![0.0; (dim - 1) * dim];
    p.jacobian(&x, &mut jac).unwrap();
    let border: Vec<f64> = (0..dim)
        .map(|k| ((k * 5 % 13) as f64 - 6.0) / 6.0)
        .collect();
    let rhs: Vec<f64> = (0..dim).map(|k| ((k * 3 % 7) as f64 - 3.0) / 3.0).collect();
    let mut sol = vec![0.0; dim];
    let det = p.bordered_solve(&[], &border, &rhs, &mut sol).unwrap();
    assert!(det.is_nan());
    let mut a = vec![0.0; dim * dim];
    for j in 0..dim {
        a[dim * j..dim * j + dim - 1].copy_from_slice(&jac[(dim - 1) * j..(dim - 1) * (j + 1)]);
        a[dim * j + dim - 1] = border[j];
    }
    let mut want = rhs.clone();
    Lu::new(&a, dim).unwrap().solve(&mut want).unwrap();
    let err = sol
        .iter()
        .zip(&want)
        .fold(0.0_f64, |e, (a, b)| e.max((a - b).abs()));
    println!(
        "bordered solve: max error {err:.3e}, max |x| {:.3e}",
        max_abs(&want)
    );
    assert!(err <= 1e-10 * max_abs(&want), "{err:e}");
}

#[test]
fn bad_setups_are_refused() {
    let mut m = model_of(Evaluator::Jit);
    let good = circle(0.5, 4, 3, 0.7, 6.0);
    assert!(CycleProblem::new(m.as_mut(), &good).is_ok());
    let mut bad = Vec::new();
    let mut s = good.clone();
    s.profile.pop();
    bad.push(s);
    let mut s = good.clone();
    s.ntst = 0;
    bad.push(s);
    let mut s = good.clone();
    s.ncol = 8;
    bad.push(s);
    let mut s = good.clone();
    s.period = 0.0;
    bad.push(s);
    let mut s = good.clone();
    s.period = f64::NAN;
    bad.push(s);
    let mut s = good.clone();
    s.free = 2;
    bad.push(s);
    let mut s = good.clone();
    s.params.pop();
    bad.push(s);
    for s in &bad {
        assert!(
            matches!(
                CycleProblem::new(m.as_mut(), s),
                Err(EngineError::BadProblem(_))
            ),
            "{s:?}"
        );
        assert!(matches!(
            periodic_orbit(m.as_mut(), s, &Options::default()),
            Err(EngineError::BadProblem(_))
        ));
    }
}

/// Largest error of the radius at the mesh points against sqrt(alpha).
fn radius_error(orbit: &PeriodicOrbit, ncol: usize, alpha: f64) -> f64 {
    let pts = orbit.u.len() / 2;
    (0..pts)
        .step_by(ncol)
        .map(|k| (orbit.u[2 * k].hypot(orbit.u[2 * k + 1]) - alpha.sqrt()).abs())
        .fold(0.0, f64::max)
}

fn solve(which: Evaluator, s: &CycleSetup) -> PeriodicOrbit {
    let opts = Options {
        max_newton: 15,
        ..Options::default()
    };
    let mut m = model_of(which);
    periodic_orbit(m.as_mut(), s, &opts).unwrap()
}

#[test]
fn acceptance_3_hopf_normal_form_cycle() {
    let alpha = 0.5;
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let orbit = solve(which.clone(), &circle(alpha, ntst, ncol, 0.6, 6.0));
        assert_eq!(orbit.u.len(), 2 * (ntst * ncol + 1));
        assert_eq!(orbit.mesh.len(), ntst + 1);
        let er = radius_error(&orbit, ncol, alpha);
        let et = (orbit.period - 2.0 * PI).abs();
        assert_eq!(orbit.multipliers.len(), 2);
        let (t, q) = (orbit.multipliers[0], orbit.multipliers[1]);
        let e1 = (t.0 - 1.0).hypot(t.1);
        let want = (-4.0 * PI * alpha).exp();
        let e2 = (q.0 - want).hypot(q.1);
        println!(
            "Test 3 [{which:?}] Newton {} iterations: radius error {er:.3e}, period error \
             {et:.3e}, |mu1 - 1| {e1:.3e}, mu2 {:.15e} (anchor {want:.15e}, error {e2:.3e})",
            orbit.newton_iters, q.0
        );
        assert!(er <= 1e-10, "[{which:?}] radius {er:e}");
        assert!(et <= 1e-10, "[{which:?}] period {et:e}");
        assert!(e1 <= 1e-6, "[{which:?}] trivial multiplier {e1:e}");
        assert!(e2 <= 1e-8, "[{which:?}] nontrivial multiplier {e2:e}");
    }
}

#[test]
fn acceptance_4_mesh_point_error_is_order_2m() {
    let alpha = 0.5;
    for (ncol, meshes) in [(2, [8, 16, 32]), (3, [4, 8, 16]), (4, [4, 8, 16])] {
        for which in BOTH {
            let errs: Vec<f64> = meshes
                .iter()
                .map(|&ntst| {
                    let orbit = solve(which.clone(), &circle(alpha, ntst, ncol, 0.6, 6.0));
                    radius_error(&orbit, ncol, alpha).max((orbit.period - 2.0 * PI).abs())
                })
                .collect();
            for k in 0..errs.len() - 1 {
                let order = (errs[k] / errs[k + 1]).log2();
                println!(
                    "Test 4 [{which:?}] m {ncol}, N {} -> {}: error {:.3e} -> {:.3e}, \
                     order {order:.3}",
                    meshes[k],
                    meshes[k + 1],
                    errs[k],
                    errs[k + 1]
                );
                assert!(errs[k + 1] > 1e-12, "m {ncol}: error at round-off");
                assert!(
                    (order - 2.0 * ncol as f64).abs() <= 0.5,
                    "[{which:?}] m {ncol}: order {order}"
                );
            }
        }
    }
}
