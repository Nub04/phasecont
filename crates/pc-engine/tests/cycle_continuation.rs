//! Continuation of limit cycles (SPEC §7): the start from a Hopf point, the branch on the
//! Hopf normal form against its closed form, the Brusselator cycle against an independent
//! RK4 integration, and the Floquet warning, each on both evaluators where it matters.
//!
//! On `hopf_c.ode` with c = -1 the orbit at alpha > 0 is the circle of radius sqrt(alpha),
//! run counterclockwise with period 2 pi, and its nontrivial Floquet multiplier is
//! exp(-4 pi alpha). The Brusselator has its Hopf point at b = 1 + a^2 with omega = a, at
//! the equilibrium (a, b / a). The Rössler cycle with a = b = 0.2 doubles its period
//! between c = 2.5 and 2.9, and the doubled orbit is still stable at c = 3, so RK4 finds
//! it there. Run with `-- --nocapture` to see the errors.

mod common;

use common::{load, rk4};
use pc_engine::{
    Branch, CYCLE_LABELS, CycleSetup, CycleStart, EngineError, Evaluator, HopfStart, ModelEval,
    Options, PdStart, Point, StopReason, build, continue_cycle,
};
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

/// Largest error at the mesh points of `pt` against RK4 from its base point u_0, in
/// rescaled time, with `sub` RK4 steps per mesh interval.
fn mesh_point_error(m: &mut dyn ModelEval, pt: &Point, ncol: usize, sub: usize) -> f64 {
    let t = period(pt);
    let p = [1.0, pt.alpha, 0.0];
    let mut x = [pt.u[0], pt.u[1]];
    let mut err = 0.0_f64;
    for j in 0..pt.mesh.len() - 1 {
        let dt = (pt.mesh[j + 1] - pt.mesh[j]) / sub as f64;
        for _ in 0..sub {
            rk4(m, &p, t, &mut x, dt);
        }
        let at = 2 * (j + 1) * ncol;
        err = err.max((x[0] - pt.u[at]).hypot(x[1] - pt.u[at + 1]));
    }
    err
}

/// At larger b the Brusselator cycle has a fast jump. Mesh adaptation moves the mesh
/// points to it and lowers the error at the mesh points well below that of the uniform
/// mesh (about 4e-3 against 1e-5 at b = 3.5; at b = 5 the uniform mesh with N = 20 no
/// longer gives the right cycle).
#[test]
fn mesh_adaptation_lowers_the_error_on_a_stiff_cycle() {
    let (ntst, ncol, b) = (20, 4, 3.5);
    let mut errs = Vec::new();
    for every in [0, 3] {
        let mut m = model_of("brusselator", Evaluator::Jit);
        let o = Options {
            mesh_adapt: every,
            ..opts((0.0, b))
        };
        let (branch, _) = run(m.as_mut(), &brusselator_start(ntst, ncol), &o);
        assert_eq!(branch.stop, StopReason::Bounds, "{:?}", branch.warnings);
        let last = branch.points.last().unwrap();
        assert!((last.alpha - b).abs() <= 1e-12);
        let mesh = &last.mesh;
        assert_eq!(mesh.len(), ntst + 1);
        assert_eq!((mesh[0], mesh[ntst]), (0.0, 1.0));
        assert!(mesh.windows(2).all(|w| w[1] > w[0]));
        let h: Vec<f64> = mesh.windows(2).map(|w| w[1] - w[0]).collect();
        let (hmin, hmax) = h
            .iter()
            .fold((1.0_f64, 0.0_f64), |(a, z), &x| (a.min(x), z.max(x)));
        let err = mesh_point_error(m.as_mut(), last, ncol, 4000);
        println!(
            "adapt every {every}: {} points, period {:.10}, intervals {hmin:.4e} to {hmax:.4e}, \
             mesh-point error {err:.3e}",
            branch.points.len(),
            period(last)
        );
        if every == 0 {
            assert!(hmax - hmin <= 1e-15);
        } else {
            assert!(hmax / hmin > 2.0, "{hmin} {hmax}");
        }
        errs.push(err);
    }
    assert!(errs[1] < errs[0] / 10.0, "{errs:?}");
}

/// Rössler parameters (a, b, c) with t = 0 appended.
fn rossler_p(c: f64) -> [f64; 4] {
    [0.2, 0.2, c, 0.0]
}

/// RK4 on Rössler at c from (1, 1, 1), first for t = 1000 to reach the attractor, then
/// up to `count` + 1 downward crossings of y = 0, with the crossing times interpolated
/// linearly. Returns the state at the last step and the `count` times between crossings.
fn rossler_crossings(m: &mut dyn ModelEval, c: f64, count: usize) -> ([f64; 3], Vec<f64>) {
    let (p, dt) = (rossler_p(c), 0.005);
    let mut x = [1.0, 1.0, 1.0];
    for _ in 0..200_000 {
        rk4(m, &p, 1.0, &mut x, dt);
    }
    let (mut t, mut at) = (0.0, Vec::new());
    while at.len() <= count {
        let y0 = x[1];
        rk4(m, &p, 1.0, &mut x, dt);
        t += dt;
        if y0 > 0.0 && x[1] <= 0.0 {
            at.push(t - dt + dt * y0 / (y0 - x[1]));
        }
    }
    (x, at.windows(2).map(|w| w[1] - w[0]).collect())
}

/// The RK4 orbit of Rössler at c = 2.5 is the guess of an orbit start; its branch has a PD
/// point, and the run from there follows the doubled cycle to c = 3. That cycle has twice
/// the period at the PD point, closes under RK4 over its period but not over half of it,
/// and its period is the sum of two crossing times of the RK4 attractor at c = 3.
#[test]
fn rossler_switches_to_the_doubled_cycle_at_the_pd_point() {
    let mut m = model_of("rossler", Evaluator::Jit);
    let (x, times) = rossler_crossings(m.as_mut(), 2.5, 1);
    let (ntst, ncol) = (40, 4);
    let nm = ntst * ncol;
    let (t, sub) = (times[0], 8);
    let p = rossler_p(2.5);
    let mut y = x;
    let mut profile = y.to_vec();
    for _ in 0..nm {
        for _ in 0..sub {
            rk4(m.as_mut(), &p, t, &mut y, 1.0 / (nm * sub) as f64);
        }
        profile.extend(y);
    }
    let params = vec![0.2, 0.2, 2.5];
    let start = CycleStart::Orbit(CycleSetup::new(2, params.clone(), ntst, ncol, profile, t));
    let (single, _) = run(m.as_mut(), &start, &opts((2.0, 2.9)));
    assert_eq!(single.stop, StopReason::Bounds, "{:?}", single.warnings);
    let pds: Vec<_> = single.specials.iter().filter(|s| s.label == "PD").collect();
    assert_eq!(pds.len(), 1, "{:?}", single.specials);
    let pd = &pds[0].point;
    let t_pd = period(pd);
    println!(
        "single cycle: period {t:.6} at c = 2.5, PD at c = {:.10}, period {t_pd:.10}",
        pd.alpha
    );
    assert!(pd.alpha > 2.5 && pd.alpha < 2.9, "{}", pd.alpha);

    let s = PdStart::from_point(2, &params, pd, 60, 4);
    let (double, _) = run(m.as_mut(), &CycleStart::Pd(s), &opts((2.0, 3.0)));
    assert_eq!(double.stop, StopReason::Bounds, "{:?}", double.warnings);
    assert!(double.warnings.is_empty(), "{:?}", double.warnings);
    assert!(double.specials.is_empty(), "{:?}", double.specials);
    let first = &double.points[0];
    assert!(first.alpha > pd.alpha, "{} {}", first.alpha, pd.alpha);
    assert!(
        (period(first) - 2.0 * t_pd).abs() <= 1e-2,
        "{}",
        period(first)
    );
    let last = double.points.last().unwrap();
    assert!((last.alpha - 3.0).abs() <= 1e-12);
    let t2 = period(last);
    let p3 = rossler_p(3.0);
    let u0 = [last.u[0], last.u[1], last.u[2]];
    let mut y = u0;
    let steps = 40_000;
    let gap = |y: &[f64; 3]| (0..3).map(|c| (y[c] - u0[c]).powi(2)).sum::<f64>().sqrt();
    let mut half = 0.0;
    for k in 0..steps {
        rk4(m.as_mut(), &p3, t2, &mut y, 1.0 / steps as f64);
        if k + 1 == steps / 2 {
            half = gap(&y);
        }
    }
    let closure = gap(&y);
    let (_, rk) = rossler_crossings(m.as_mut(), 3.0, 2);
    println!(
        "doubled cycle: {} points, c {:.6} -> 3, period {t2:.10}, RK4 closure {closure:.3e}, \
         gap at half the period {half:.3e}, RK4 crossing times {rk:?}",
        double.points.len(),
        first.alpha
    );
    assert!(closure <= 1e-6, "{closure:e}");
    assert!(half > 0.1, "{half:e}");
    assert!((rk[0] - rk[1]).abs() > 0.1, "{rk:?}");
    assert!((t2 - rk[0] - rk[1]).abs() <= 1e-3, "{t2} {rk:?}");
}

#[test]
fn bad_pd_starts_are_refused() {
    let mut m = model_of("rossler", Evaluator::Jit);
    let cancel = AtomicBool::new(false);
    let (ntst, ncol) = (4, 2);
    let nm = ntst * ncol;
    let mut orbit = Vec::new();
    for k in 0..=nm {
        let th = 2.0 * PI * k as f64 / nm as f64;
        orbit.extend([3.0 * th.cos(), 3.0 * th.sin(), 0.1]);
    }
    let good = PdStart {
        free: 2,
        params: vec![0.2, 0.2, 2.8],
        orbit,
        period: 6.0,
        mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
        ntst: 10,
        ncol: 4,
    };
    let mut bad = Vec::new();
    bad.push(PdStart {
        mesh: vec![0.1, 0.25, 0.5, 0.75, 1.0],
        ..good.clone()
    });
    bad.push(PdStart {
        mesh: vec![0.0, 0.5, 0.25, 0.75, 1.0],
        ..good.clone()
    });
    bad.push(PdStart {
        mesh: vec![0.0],
        ..good.clone()
    });
    let mut short = good.clone();
    short.orbit.pop();
    bad.push(short);
    bad.push(PdStart {
        period: 0.0,
        ..good.clone()
    });
    bad.push(PdStart {
        params: vec![0.2, f64::NAN, 2.8],
        ..good.clone()
    });
    bad.push(PdStart {
        free: 3,
        ..good.clone()
    });
    bad.push(PdStart {
        ntst: 0,
        ..good.clone()
    });
    bad.push(PdStart {
        ncol: 9,
        ..good.clone()
    });
    for s in bad {
        let r = continue_cycle(
            m.as_mut(),
            &CycleStart::Pd(s.clone()),
            &Options::default(),
            &cancel,
            &mut |_| {},
        );
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{s:?}: {r:?}");
    }
}

/// 2^46 intervals of 2 points for 2 or 3 states need more than 2^49 bytes for one orbit,
/// more than the address space, so those allocations fail whatever the overcommit
/// setting. A start must refuse them with an error instead of aborting the process.
#[test]
fn starts_with_a_huge_mesh_give_an_error() {
    let cancel = AtomicBool::new(false);
    let huge = 1usize << 46;
    let mut m = model_of("brusselator", Evaluator::Jit);
    let hopf = HopfStart {
        free: 1,
        params: vec![1.0, 2.0],
        u0: vec![1.0, 2.0],
        ntst: huge,
        ncol: 2,
    };
    let r = continue_cycle(
        m.as_mut(),
        &CycleStart::Hopf(hopf),
        &Options::default(),
        &cancel,
        &mut |_| {},
    );
    assert!(
        matches!(&r, Err(EngineError::BadProblem(s)) if s.contains("memory")),
        "{r:?}"
    );
    let mut m = model_of("rossler", Evaluator::Jit);
    let (ntst, ncol) = (4, 2);
    let nm = ntst * ncol;
    let mut orbit = Vec::new();
    for k in 0..=nm {
        let th = 2.0 * PI * k as f64 / nm as f64;
        orbit.extend([3.0 * th.cos(), 3.0 * th.sin(), 0.1]);
    }
    let pd = PdStart {
        free: 2,
        params: vec![0.2, 0.2, 2.8],
        orbit,
        period: 6.0,
        mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
        ntst: huge,
        ncol: 2,
    };
    let r = continue_cycle(
        m.as_mut(),
        &CycleStart::Pd(pd),
        &Options::default(),
        &cancel,
        &mut |_| {},
    );
    assert!(
        matches!(&r, Err(EngineError::BadProblem(s)) if s.contains("memory")),
        "{r:?}"
    );
}
