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

/// The plane orbit of Sandstede's model, u(tau) = x(-T/2 + T tau) with `zeros` z
/// components at zero, at the base points of a uniform mesh of `ntst` intervals of `ncol`
/// points.
fn plane_orbit(a: f64, b: f64, period: f64, ntst: usize, ncol: usize, zeros: usize) -> Vec<f64> {
    let pts = ntst * ncol + 1;
    let ts: Vec<f64> = (0..pts)
        .map(|k| period * (k as f64 / (pts - 1) as f64 - 0.5))
        .collect();
    w_at(a, b, &ts, 1000)
        .into_iter()
        .flat_map(|w| {
            let mut v = vec![1.0 - w * w, -w * (1.0 - w * w)];
            v.resize(2 + zeros, 0.0);
            v
        })
        .collect()
}

/// The `"orbit"` start of test 1 at (a, b, c) and T, free (mut, a).
fn m1_start(a: f64, period: f64, ntst: usize, ncol: usize) -> HomStart {
    HomStart {
        kind: HomStartKind::Orbit,
        free: [MUT, A],
        params: vec![a, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        orbit: plane_orbit(a, 1.0, period, ntst, ncol, 1),
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

/// The labels of a branch with their a2 values.
fn labels_of(b: &pc_engine::Branch) -> Vec<(String, f64)> {
    b.specials
        .iter()
        .map(|s| (s.label.clone(), s.point.alpha))
        .collect()
}

/// Test 2 of SPEC §8b: the orbit flip of Sandstede's model at (a, b, c) = (-0.5, 2.5, -1).
/// The (x, y) plane does not depend on z, so the orbit exists for every mu with mut = 0,
/// and its z part is proportional to mu. The leading stable eigenvalue is c = -1, along
/// z, so the orbit flips (OFS) at mu = 0.
#[test]
fn sandstede_orbit_flip_at_mu_zero() {
    let (period, nt, ncol) = (20.0, 160, 4);
    for which in BOTH {
        let mut m = model_of("sandstede", &which);
        let start = HomStart {
            kind: HomStartKind::Orbit,
            free: [MUT, MU],
            params: vec![-0.5, 2.5, -1.0, -0.1, 0.0, 0.0, 0.0, 0.0],
            orbit: plane_orbit(-0.5, 2.5, period, nt, ncol, 1),
            period,
            mesh: None,
            ntst: nt,
            ncol,
        };
        let cancel = AtomicBool::new(false);
        let b = continue_hom_curve(
            m.as_mut(),
            &start,
            &opts((-0.11, 0.1)),
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(b.stop, StopReason::Bounds, "{which:?}: {:?}", b.stop);
        let ext = 3 * (nt * ncol + 1);
        let mutmax = b
            .points
            .iter()
            .fold(0.0_f64, |a, pt| a.max(pt.u[ext + 4].abs()));
        let labels = labels_of(&b);
        println!(
            "{which:?}: {} points, |mut| {mutmax:e}, labels {labels:?}",
            b.points.len()
        );
        assert_eq!(labels.len(), 1, "{labels:?}");
        assert_eq!(labels[0].0, "OFS");
        assert!(labels[0].1.abs() < 1e-10, "OFS at mu = {}", labels[0].1);
    }
}

/// Test 3 of SPEC §8b: the Belyakov point of `belyakov4.ode`. The leading stable
/// eigenvalues c +- sqrt(-delta) of the z block meet at delta = 0 in a Jordan block and
/// turn complex, so BLS changes sign there. TLS (three leading stable) can change sign
/// too, since the stable plane eigenvalue a - b = -1 is below c = -0.4 throughout.
#[test]
fn belyakov_point_at_delta_zero() {
    const DELTA: usize = 4;
    const MUT4: usize = 5;
    let (period, nt, ncol) = (20.0, 160, 4);
    for which in BOTH {
        let mut m = model_of("belyakov4", &which);
        let start = HomStart {
            kind: HomStartKind::Orbit,
            free: [MUT4, DELTA],
            params: vec![0.0, 1.0, -0.4, 0.5, -0.1, 0.0],
            orbit: plane_orbit(0.0, 1.0, period, nt, ncol, 2),
            period,
            mesh: None,
            ntst: nt,
            ncol,
        };
        let cancel = AtomicBool::new(false);
        let b = continue_hom_curve(
            m.as_mut(),
            &start,
            &opts((-0.11, 0.1)),
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(b.stop, StopReason::Bounds, "{which:?}: {:?}", b.stop);
        let ext = 4 * (nt * ncol + 1);
        let mutmax = b
            .points
            .iter()
            .fold(0.0_f64, |a, pt| a.max(pt.u[ext + 5].abs()));
        let labels = labels_of(&b);
        println!(
            "{which:?}: {} points, |mut| {mutmax:e}, labels {labels:?}",
            b.points.len()
        );
        let bls: Vec<_> = labels.iter().filter(|l| l.0 == "BLS").collect();
        assert_eq!(bls.len(), 1, "{labels:?}");
        assert!(bls[0].1.abs() < 1e-8, "BLS at delta = {}", bls[0].1);
        assert!(
            labels.iter().all(|l| l.0 == "BLS" || l.0 == "TLS"),
            "{labels:?}"
        );
    }
}

/// The signed miss distance of the unstable manifold of the saddle x = (-b2 + sqrt(b2^2 -
/// 4 b1)) / 2 of `bt_hom.ode`, by RK4 with step `h` from 1e-9 along the unstable
/// eigenvector on the side with y < 0. The orbit is followed until it crosses the vertical
/// x = x_s again, where the miss is y > 0, or turns back (y falls through 0) with x < x_s,
/// where the miss is x - x_s < 0.
fn bt_miss(b1: f64, b2: f64, h: f64) -> f64 {
    let xs = (-b2 + (b2 * b2 - 4.0 * b1).sqrt()) / 2.0;
    let (a21, a22) = (b2 + 2.0 * xs, xs);
    let lam = (a22 + (a22 * a22 + 4.0 * a21).sqrt()) / 2.0;
    let nv = lam.hypot(1.0);
    let f = |u: [f64; 2]| [u[1], b1 + b2 * u[0] + u[0] * u[0] + u[0] * u[1]];
    let add = |u: [f64; 2], k: [f64; 2], s: f64| [u[0] + s * k[0], u[1] + s * k[1]];
    let mut u = [xs - 1e-9 / nv, -1e-9 * lam / nv];
    for _ in 0..10_000_000 {
        let k1 = f(u);
        let k2 = f(add(u, k1, h / 2.0));
        let k3 = f(add(u, k2, h / 2.0));
        let k4 = f(add(u, k3, h));
        let v = [
            u[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            u[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        ];
        if u[0] < xs && v[0] >= xs {
            let s = (xs - u[0]) / (v[0] - u[0]);
            return u[1] + s * (v[1] - u[1]);
        }
        if u[1] > 0.0 && v[1] <= 0.0 && v[0] < xs {
            let s = u[1] / (u[1] - v[1]);
            return u[0] + s * (v[0] - u[0]) - xs;
        }
        u = v;
    }
    panic!("the shooting did not end at b1 = {b1}, b2 = {b2}");
}

/// b1 of the homoclinic orbit of `bt_hom.ode` at `b2`, by bisection on `bt_miss`.
fn bt_shooting(b2: f64) -> f64 {
    let (mut lo, mut hi) = (-0.3 * b2 * b2, -0.18 * b2 * b2);
    let (flo, fhi) = (bt_miss(lo, b2, 1e-3), bt_miss(hi, b2, 1e-3));
    assert!(flo * fhi < 0.0, "no bracket at b2 = {b2}: {flo:e}, {fhi:e}");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if bt_miss(mid, b2, 1e-3) * flo > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Test 4 of SPEC §8b: the LC branch from the Hopf point of `bt_hom.ode` at b2 = -0.1 runs
/// until its period passes `period_max`, and the HOM curve in (b1, b2) starts from its last
/// point. Each point's b1 is compared with the shooting value at the point's own b2.
#[test]
fn bt_homoclinic_from_a_long_cycle() {
    use pc_engine::{CycleStart, HopfStart, continue_cycle};
    use std::sync::atomic::Ordering;
    let reference = bt_shooting(-0.1);
    println!("shooting b1(-0.1) = {reference:.16}");
    assert!(
        (reference + 0.0023714546120554).abs() < 1e-15,
        "{reference}"
    );
    let (ntst, ncol, period_max) = (80, 4, 100.0);
    for which in BOTH {
        let mut m = model_of("bt_hom", &which);
        let lc = CycleStart::Hopf(HopfStart {
            free: 0,
            params: vec![0.0, -0.1],
            u0: vec![0.0, 0.0],
            ntst,
            ncol,
        });
        let cancel = AtomicBool::new(false);
        let o = Options {
            max_steps: 2000,
            ..opts((-0.01, 0.01))
        };
        let branch = continue_cycle(m.as_mut(), &lc, &o, &cancel, &mut |pt| {
            if pt.u[pt.u.len() - 1] > period_max {
                cancel.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();
        let last = branch.points.last().unwrap();
        let period = last.u[last.u.len() - 1];
        println!(
            "{which:?}: LC {} points, stop {:?}, last b1 {:.16}, period {period}",
            branch.points.len(),
            branch.stop,
            last.alpha
        );
        assert!(period > period_max, "period {period}");
        let start = HomStart {
            kind: HomStartKind::Cycle,
            free: [0, 1],
            params: vec![last.alpha, -0.1],
            orbit: last.u[..last.u.len() - 1].to_vec(),
            period,
            mesh: Some(last.mesh.clone()),
            ntst,
            ncol,
        };
        let cancel = AtomicBool::new(false);
        let b = continue_hom_curve(
            m.as_mut(),
            &start,
            &Options {
                hmax: 0.01,
                ..opts((-0.1, -0.05))
            },
            &cancel,
            &mut |_| {},
        )
        .unwrap();
        let ext = 2 * (ntst * ncol + 1);
        let mut worst = 0.0_f64;
        for pt in &b.points {
            let b1 = pt.u[ext + 3];
            let e = (b1 - bt_shooting(pt.alpha)).abs();
            worst = worst.max(e);
        }
        println!(
            "{which:?}: HOM {} points, stop {:?}, first b1 {:.16}, worst b1 error {worst:e}, \
             labels {:?}",
            b.points.len(),
            b.stop,
            b.points[0].u[ext + 3],
            labels_of(&b)
        );
        assert_eq!(b.stop, StopReason::Bounds, "{which:?}: {:?}", b.stop);
        assert!(worst < 1e-12, "{which:?}: b1 error {worst:e}");
    }
}

/// Test 5 of SPEC §8b, the property part: at random parameters of Sandstede's model and at
/// random points near the start, the dense HOM Jacobian, whose x0 and parameter columns of
/// the boundary rows are forward differences, matches central differences. The cases come
/// from a fixed-seed xorshift generator, so a failure repeats.
#[test]
fn hom_jacobian_matches_central_differences_at_random_points() {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = |lo: f64, hi: f64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        lo + (hi - lo) * (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut m = model_of("sandstede", &Evaluator::Jit);
    let mut worst = 0.0_f64;
    for case in 0..16 {
        let a = uniform(-0.4, 0.4);
        let mut s = m1_start(a, 10.0, 6, 3);
        s.params[2] = uniform(-3.0, -0.5);
        s.params[MU] = uniform(-0.2, 0.2);
        s.params[MUT] = uniform(-0.1, 0.1);
        for k in 5..8 {
            s.params[k] = uniform(-0.5, 0.5);
        }
        let mut p = HomProblem::new(m.as_mut(), &s, &Options::default()).unwrap();
        let x: Vec<f64> = p
            .start()
            .to_vec()
            .into_iter()
            .map(|v| v + uniform(-0.01, 0.01))
            .collect();
        let e = jacobian_error(&mut p, &x);
        assert!(e < 1e-6, "case {case}: error {e:e} at {:?}", s.params);
        worst = worst.max(e);
    }
    println!("random points: worst HOM Jacobian error {worst:e}");
}
