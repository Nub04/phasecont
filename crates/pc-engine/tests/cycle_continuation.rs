//! Continuation of limit cycles (SPEC §7): the start from a Hopf point, the branch on the
//! Hopf normal form against its closed form, the Brusselator cycle against an independent
//! RK4 integration, and the Floquet warning, each on both evaluators where it matters.
//!
//! On `hopf_c.ode` with c = -1 the orbit at alpha > 0 is the circle of radius sqrt(alpha),
//! run counterclockwise with period 2 pi, and its nontrivial Floquet multiplier is
//! exp(-4 pi alpha). The Brusselator has its Hopf point at b = 1 + a^2 with omega = a, at
//! the equilibrium (a, b / a). Run with `-- --nocapture` to see the errors.

mod common;

use common::load;
use pc_engine::{
    Branch, CYCLE_LABELS, CycleSetup, CycleStart, EngineError, Evaluator, HopfStart, ModelEval,
    Options, Point, StopReason, build, continue_cycle,
};
use pc_ir::EntryKind;
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which).unwrap()
}

fn opts(bounds: (f64, f64)) -> Options {
    Options {
        max_newton: 15,
        hmax: 0.05,
        bounds,
        ..Options::default()
    }
}

fn run(m: &mut dyn ModelEval, start: &CycleStart, o: &Options) -> (Branch, usize) {
    let cancel = AtomicBool::new(false);
    let mut seen = 0;
    let branch = continue_cycle(m, start, o, &cancel, &mut |_| seen += 1).unwrap();
    (branch, seen)
}

/// The period, the last entry of `Point::u`.
fn period(pt: &Point) -> f64 {
    pt.u[pt.u.len() - 1]
}

/// Largest error of the radius at the mesh points of a uniform mesh against sqrt(alpha).
fn radius_error(pt: &Point, ncol: usize) -> f64 {
    let pts = (pt.u.len() - 1) / 2;
    (0..pts)
        .step_by(ncol)
        .map(|k| (pt.u[2 * k].hypot(pt.u[2 * k + 1]) - pt.alpha.sqrt()).abs())
        .fold(0.0, f64::max)
}

/// The multiplier closest to 1 and the other one, for n = 2.
fn split(pt: &Point) -> ((f64, f64), (f64, f64)) {
    assert_eq!(pt.eigenvalues.len(), 2);
    let (a, b) = (pt.eigenvalues[0], pt.eigenvalues[1]);
    let d = |m: (f64, f64)| (m.0 - 1.0).hypot(m.1);
    if d(a) <= d(b) { (a, b) } else { (b, a) }
}

fn hopf_c_start(ntst: usize, ncol: usize) -> CycleStart {
    CycleStart::Hopf(HopfStart {
        free: 0,
        params: vec![0.0, -1.0],
        u0: vec![0.0, 0.0],
        ntst,
        ncol,
    })
}

#[test]
fn hopf_normal_form_branch_from_the_hopf_point() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("hopf_c", which.clone());
        let (branch, seen) = run(m.as_mut(), &hopf_c_start(ntst, ncol), &opts((-1.0, 0.5)));
        assert_eq!(
            branch.stop,
            StopReason::Bounds,
            "[{which:?}] {:?}",
            branch.warnings
        );
        assert_eq!(seen, branch.points.len());
        assert!(branch.specials.is_empty(), "{:?}", branch.specials);
        assert!(branch.warnings.is_empty(), "{:?}", branch.warnings);
        let first = &branch.points[0];
        assert!(first.alpha > 0.0 && first.alpha < 1e-3, "{}", first.alpha);
        let (mut er, mut et, mut e1, mut e2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for pt in &branch.points {
            assert_eq!(pt.u.len(), 2 * (ntst * ncol + 1) + 1);
            assert_eq!(pt.tangent.len(), pt.u.len() + 1);
            assert_eq!(pt.codim2.len(), CYCLE_LABELS.len());
            // LPC is the alpha part of the tangent: positive on the whole branch.
            assert!(pt.codim2[0] > 0.0, "LPC test {}", pt.codim2[0]);
            assert!(pt.codim2[2].is_nan(), "NS is not defined for n = 2");
            er = er.max(radius_error(pt, ncol));
            et = et.max((period(pt) - 2.0 * PI).abs());
            let (t, q) = split(pt);
            e1 = e1.max((t.0 - 1.0).hypot(t.1));
            e2 = e2.max((q.0 - (-4.0 * PI * pt.alpha).exp()).hypot(q.1));
        }
        let last = branch.points.last().unwrap();
        println!(
            "[{which:?}] {} points, alpha {:.3e} -> {:.15}: max radius error {er:.3e}, \
             period {et:.3e}, |mu1 - 1| {e1:.3e}, mu2 {e2:.3e}",
            branch.points.len(),
            first.alpha,
            last.alpha
        );
        assert!((last.alpha - 0.5).abs() <= 1e-12);
        // SPEC §9 Test 3 tolerances, here over the whole branch.
        assert!(er <= 1e-10, "[{which:?}] radius {er:e}");
        assert!(et <= 1e-10, "[{which:?}] period {et:e}");
        assert!(e1 <= 1e-6, "[{which:?}] trivial multiplier {e1:e}");
        assert!(e2 <= 1e-8, "[{which:?}] nontrivial multiplier {e2:e}");
    }
}

#[test]
fn orbit_start_runs_backward_with_direction() {
    let (ntst, ncol) = (20, 4);
    let nm = ntst * ncol;
    let mut profile = Vec::new();
    for k in 0..=nm {
        let th = 2.0 * PI * k as f64 / nm as f64;
        profile.push(0.6 * th.cos());
        profile.push(0.6 * th.sin());
    }
    let start = CycleStart::Orbit(CycleSetup::new(
        0,
        vec![0.5, -1.0],
        ntst,
        ncol,
        profile,
        6.0,
    ));
    let mut m = model_of("hopf_c", Evaluator::Jit);
    let o = Options {
        direction: -1.0,
        ..opts((0.1, 1.0))
    };
    let (branch, _) = run(m.as_mut(), &start, &o);
    assert_eq!(branch.stop, StopReason::Bounds);
    // The guess is corrected at fixed alpha first.
    let first = &branch.points[0];
    assert!((first.alpha - 0.5).abs() <= 1e-14, "{}", first.alpha);
    assert!(radius_error(first, ncol) <= 1e-8);
    let last = branch.points.last().unwrap();
    assert!((last.alpha - 0.1).abs() <= 1e-12);
    let er = branch
        .points
        .iter()
        .map(|p| radius_error(p, ncol))
        .fold(0.0, f64::max);
    println!(
        "orbit start: {} points, max radius error {er:.3e}",
        branch.points.len()
    );
    assert!(er <= 1e-8, "{er:e}");
}

/// Brusselator with a = 1: the Hopf point is at b = 2, x = (1, 2), omega = 1.
fn brusselator_start(ntst: usize, ncol: usize) -> CycleStart {
    CycleStart::Hopf(HopfStart {
        free: 1,
        params: vec![1.0, 2.0],
        u0: vec![1.0, 2.0],
        ntst,
        ncol,
    })
}

/// One RK4 step of x' = T f(x) in rescaled time.
fn rk4(m: &mut dyn ModelEval, p: &[f64], t: f64, x: &mut [f64], dt: f64) {
    let n = x.len();
    let mut f = |y: &[f64]| {
        let mut out = vec![0.0; n];
        m.eval(&EntryKind::F, y, p, &[], &mut out).unwrap();
        out.iter().map(|v| t * v).collect::<Vec<f64>>()
    };
    let add = |y: &[f64], k: &[f64], s: f64| -> Vec<f64> {
        y.iter().zip(k).map(|(a, b)| a + s * b).collect()
    };
    let k1 = f(x);
    let k2 = f(&add(x, &k1, dt / 2.0));
    let k3 = f(&add(x, &k2, dt / 2.0));
    let k4 = f(&add(x, &k3, dt));
    for i in 0..n {
        x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

#[test]
fn brusselator_cycle_from_the_hopf_point_closes_under_rk4() {
    let (ntst, ncol) = (40, 4);
    for which in BOTH {
        let mut m = model_of("brusselator", which.clone());
        let (branch, _) = run(
            m.as_mut(),
            &brusselator_start(ntst, ncol),
            &opts((0.0, 2.5)),
        );
        assert_eq!(branch.stop, StopReason::Bounds, "{:?}", branch.warnings);
        assert!(branch.specials.is_empty(), "{:?}", branch.specials);
        let first = &branch.points[0];
        let t0 = period(first);
        println!(
            "[{which:?}] first cycle: b {:.6e}, period {t0:.12} (2 pi {:.12})",
            first.alpha,
            2.0 * PI
        );
        assert!(first.alpha > 2.0 && first.alpha < 2.0 + 1e-3);
        assert!((t0 - 2.0 * PI).abs() <= 1e-3);
        let last = branch.points.last().unwrap();
        assert!((last.alpha - 2.5).abs() <= 1e-12);
        let t = period(last);
        let u0 = [last.u[0], last.u[1]];
        let p = [1.0, last.alpha, 0.0];
        let mut x = u0;
        let steps = 20000;
        for _ in 0..steps {
            rk4(m.as_mut(), &p, t, &mut x, 1.0 / steps as f64);
        }
        let gap = (x[0] - u0[0]).hypot(x[1] - u0[1]);
        let (triv, _) = split(last);
        println!(
            "[{which:?}] b = 2.5: period {t:.12}, RK4 closure gap {gap:.3e}, |mu1 - 1| {:.3e}",
            (triv.0 - 1.0).hypot(triv.1)
        );
        assert!(t > 2.0 * PI);
        assert!(gap <= 1e-7, "[{which:?}] {gap:e}");
    }
}

/// The Hopf normal form is equivariant under rotations, so its discrete orbit keeps the
/// multiplier 1 exactly on any mesh. The Brusselator has no such symmetry.
#[test]
fn coarse_mesh_gives_the_floquet_warning() {
    let mut m = model_of("brusselator", Evaluator::Jit);
    let (branch, _) = run(m.as_mut(), &brusselator_start(3, 2), &opts((0.0, 3.0)));
    println!("{:?} {:?}", branch.stop, branch.warnings);
    assert!(
        branch
            .warnings
            .iter()
            .any(|w| w.contains("trivial Floquet multiplier")),
        "{:?}",
        branch.warnings
    );
}

#[test]
fn bad_starts_are_refused() {
    let mut m = model_of("brusselator", Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let good = HopfStart {
        free: 1,
        params: vec![1.0, 2.0],
        u0: vec![1.0, 2.0],
        ntst: 10,
        ncol: 4,
    };
    let mut bad = Vec::new();
    // b = 0: the Jacobian at (1, 0) has the double eigenvalue -1, so no pair.
    bad.push(HopfStart {
        params: vec![1.0, 0.0],
        u0: vec![1.0, 0.0],
        ..good.clone()
    });
    bad.push(HopfStart {
        u0: vec![1.0],
        ..good.clone()
    });
    bad.push(HopfStart {
        free: 2,
        ..good.clone()
    });
    bad.push(HopfStart {
        ntst: 0,
        ..good.clone()
    });
    bad.push(HopfStart {
        ncol: 9,
        ..good.clone()
    });
    for s in bad {
        let r = continue_cycle(
            m.as_mut(),
            &CycleStart::Hopf(s.clone()),
            &Options::default(),
            &cancel,
            &mut |_| {},
        );
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{s:?}: {r:?}");
    }
}
