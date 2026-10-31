//! Refining LP, BP and Hopf points: Jacobians against finite differences, regularity,
//! quadratic convergence and the accuracy of the refined points, on both evaluators.

mod common;

use common::load;
use pc_engine::{
    BpSystem, Branch, DefiningSystem, EngineError, EpSetup, Evaluator, HopfSystem, LpSystem,
    ModelEval, NewtonStep, Options, SpecialPoint, StopReason, build, continue_ep, newton,
    refine_special, refine_specials,
};
use pc_ir::EntryKind;
use pc_linalg::{eigenvalues, singular_values};
use std::sync::atomic::{AtomicBool, Ordering};

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

#[derive(Clone, Copy, Debug)]
enum Kind {
    Lp,
    Bp,
    H,
}

/// Where a system is built: model, parameters, free index, state, alpha and (for H) the
/// eigenvalue.
struct Spot {
    name: &'static str,
    params: Vec<f64>,
    free: usize,
    u: Vec<f64>,
    alpha: f64,
    lambda: (f64, f64),
}

fn spot(name: &'static str, params: &[f64], free: usize, u: &[f64]) -> Spot {
    Spot {
        name,
        params: params.to_vec(),
        free,
        u: u.to_vec(),
        alpha: params[free],
        lambda: (0.0, 0.0),
    }
}

fn model_of(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which.clone()).unwrap()
}

fn with_system<T>(
    kind: Kind,
    s: &Spot,
    which: Evaluator,
    f: impl FnOnce(&mut dyn DefiningSystem, Vec<f64>) -> T,
) -> T {
    let mut m = model_of(s.name, which.clone());
    match kind {
        Kind::Lp => {
            let mut sys = LpSystem::new(m.as_mut(), &s.params, s.free, &s.u, s.alpha).unwrap();
            let z0 = sys.start().to_vec();
            f(&mut sys, z0)
        }
        Kind::Bp => {
            let mut sys = BpSystem::new(m.as_mut(), &s.params, s.free, &s.u, s.alpha).unwrap();
            let z0 = sys.start().to_vec();
            f(&mut sys, z0)
        }
        Kind::H => {
            let mut sys =
                HopfSystem::new(m.as_mut(), &s.params, s.free, &s.u, s.alpha, s.lambda).unwrap();
            let z0 = sys.start().to_vec();
            f(&mut sys, z0)
        }
    }
}

/// Max difference between the exact Jacobian and central differences, relative to
/// max(1, max |J|).
fn fd_error(sys: &mut dyn DefiningSystem, z: &[f64]) -> f64 {
    let d = sys.dim();
    let mut jac = vec![0.0; d * d];
    sys.jacobian(z, &mut jac).unwrap();
    let scale = jac.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    let (mut gp, mut gm) = (vec![0.0; d], vec![0.0; d]);
    for j in 0..d {
        let h = 1e-6 * z[j].abs().max(1.0);
        let (mut zp, mut zm) = (z.to_vec(), z.to_vec());
        zp[j] += h;
        zm[j] -= h;
        sys.residual(&zp, &mut gp).unwrap();
        sys.residual(&zm, &mut gm).unwrap();
        for i in 0..d {
            let fd = (gp[i] - gm[i]) / (2.0 * h);
            worst = worst.max((fd - jac[i + d * j]).abs());
        }
    }
    worst / scale
}

/// A fixed, generic displacement of size `amp`.
fn wiggle(z: &[f64], amp: f64) -> Vec<f64> {
    z.iter()
        .enumerate()
        .map(|(i, v)| v + amp * ((3 * i + 1) as f64).sin())
        .collect()
}

fn jac_eigs(name: &str, params: &[f64], u: &[f64], which: Evaluator) -> Vec<(f64, f64)> {
    let mut m = model_of(name, which.clone());
    let n = m.n();
    let mut p = params.to_vec();
    p.push(0.0);
    let mut a = vec![0.0; n * n];
    m.eval(&EntryKind::JacU, u, &p, &[], &mut a).unwrap();
    eigenvalues(&a, n).unwrap()
}

fn lorenz_hopf_spot(which: Evaluator) -> Spot {
    let (s, r, b) = (10.0_f64, 28.0_f64, 8.0_f64 / 3.0);
    let c = (b * (r - 1.0)).sqrt();
    let params = [s, r, b];
    let u = [c, c, r - 1.0];
    let eigs = jac_eigs("lorenz", &params, &u, which.clone());
    let lambda = *eigs.iter().find(|e| e.1 > 0.0).unwrap();
    Spot {
        lambda,
        ..spot("lorenz", &params, 1, &u)
    }
}

#[test]
fn jacobians_match_central_differences() {
    for which in BOTH {
        let mut cases: Vec<(Kind, Spot)> = vec![
            (Kind::Lp, spot("fold", &[0.1], 0, &[0.4])),
            (Kind::Lp, spot("fold2", &[0.1], 0, &[0.3, 0.1])),
            (Kind::Lp, spot("brusselator", &[1.0, 1.5], 1, &[1.0, 1.5])),
            (
                Kind::Lp,
                spot("lorenz", &[10.0, 28.0, 8.0 / 3.0], 1, &[1.0, 2.0, 3.0]),
            ),
            (Kind::Bp, spot("transcritical", &[0.2], 0, &[0.3])),
            (Kind::Bp, spot("pitchfork", &[0.2], 0, &[0.3])),
            (Kind::Bp, spot("fold2", &[0.1], 0, &[0.3, 0.1])),
            (
                Kind::Bp,
                spot("lorenz", &[10.0, 1.0, 8.0 / 3.0], 1, &[0.0, 0.0, 0.0]),
            ),
            (
                Kind::Bp,
                spot("lorenz", &[10.0, 28.0, 8.0 / 3.0], 1, &[1.0, 2.0, 3.0]),
            ),
        ];
        let mut hc = spot("hopf_c", &[0.3, -1.0], 0, &[0.0, 0.0]);
        hc.lambda = (0.3, 1.0);
        cases.push((Kind::H, hc));
        let mut br = spot("brusselator", &[1.0, 1.5], 1, &[1.0, 1.5]);
        br.lambda = jac_eigs("brusselator", &[1.0, 1.5], &[1.0, 1.5], which.clone())
            .into_iter()
            .find(|e| e.1 > 0.0)
            .unwrap();
        cases.push((Kind::H, br));
        cases.push((Kind::H, lorenz_hopf_spot(which.clone())));
        for (kind, s) in cases {
            let err = with_system(kind, &s, which.clone(), |sys, z0| {
                let z1 = wiggle(&z0, 0.05);
                (fd_error(sys, &z0), fd_error(sys, &z1))
            });
            println!(
                "{kind:?} {} {which:?}: fd error {:.2e} / {:.2e}",
                s.name, err.0, err.1
            );
            assert!(
                err.0 < 1e-6 && err.1 < 1e-6,
                "{kind:?} {} {which:?}: {err:?}",
                s.name
            );
        }
    }
}

struct Case {
    kind: Kind,
    spot: Spot,
    /// Expected (u, alpha) of the converged point.
    exact: Vec<f64>,
}

fn cases(which: Evaluator) -> Vec<Case> {
    let mut hc = spot("hopf_c", &[0.0, -1.0], 0, &[0.0, 0.0]);
    hc.lambda = (0.0, 1.0);
    vec![
        Case {
            kind: Kind::Lp,
            spot: spot("fold", &[0.0], 0, &[0.0]),
            exact: vec![0.0, 0.0],
        },
        Case {
            kind: Kind::Lp,
            spot: spot("fold2", &[0.0], 0, &[0.0, 0.0]),
            exact: vec![0.0, 0.0, 0.0],
        },
        Case {
            kind: Kind::H,
            spot: hc,
            exact: vec![0.0, 0.0, 0.0],
        },
        Case {
            kind: Kind::Bp,
            spot: spot("transcritical", &[0.0], 0, &[0.0]),
            exact: vec![0.0, 0.0],
        },
        Case {
            kind: Kind::Bp,
            spot: spot("pitchfork", &[0.0], 0, &[0.0]),
            exact: vec![0.0, 0.0],
        },
        Case {
            kind: Kind::Bp,
            spot: spot("lorenz", &[10.0, 1.0, 8.0 / 3.0], 1, &[0.0, 0.0, 0.0]),
            exact: vec![0.0, 0.0, 0.0, 1.0],
        },
        Case {
            kind: Kind::H,
            spot: lorenz_hopf_spot(which.clone()),
            exact: Vec::new(),
        },
    ]
}

/// Steps of a log, once below 0.1 and above rounding level, must shrink at least
/// quadratically. Returns how many pairs of steps it checked; the caller requires at
/// least one, so a log too short to test cannot pass.
fn assert_quadratic(log: &[NewtonStep], what: &str) -> usize {
    let mut checked = 0;
    for w in log.windows(2) {
        let (s0, s1) = (w[0].step, w[1].step);
        if s0 < 0.1 && s1 > 1e-14 {
            assert!(s1 <= 20.0 * s0 * s0, "{what}: step {s0:e} then {s1:e}");
            checked += 1;
        }
    }
    checked
}

fn show(log: &[NewtonStep]) -> String {
    log.iter()
        .map(|s| format!("(res {:.2e}, step {:.2e})", s.residual, s.step))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn newton_converges_quadratically_with_a_regular_jacobian() {
    let opts = Options::default();
    let cancel = AtomicBool::new(false);
    for which in BOTH {
        for c in cases(which.clone()) {
            let (z, log, sv) = with_system(c.kind, &c.spot, which.clone(), |sys, z0| {
                // Start away from the solution.
                let start = wiggle(&z0, 0.02);
                let r = newton(sys, &start, &opts, &cancel).unwrap();
                let d = sys.dim();
                let mut jac = vec![0.0; d * d];
                sys.jacobian(&r.z, &mut jac).unwrap();
                let sv = singular_values(&jac, d, d).unwrap();
                (r.z, r.log, sv[d - 1])
            });
            let what = format!("{:?} {} {which:?}", c.kind, c.spot.name);
            println!(
                "{what}: iterations {} smallest sv {sv:.3e}\n  {}",
                log.len(),
                show(&log)
            );
            assert!(sv > 1e-3, "{what}: smallest singular value {sv:e}");
            assert!(log.len() <= 8, "{what}");
            let checked = assert_quadratic(&log, &what);
            assert!(
                checked >= 1,
                "{what}: no pair of steps was checked: {}",
                show(&log)
            );
            if !c.exact.is_empty() {
                for (a, b) in z.iter().zip(&c.exact) {
                    assert!((a - b).abs() < 1e-12, "{what}: {z:?}");
                }
            }
        }
    }
}

fn run(name: &str, which: Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let mut m = model_of(name, which.clone());
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

fn line_opts() -> Options {
    Options {
        h0: 0.07,
        hmax: 0.07,
        bounds: (f64::NEG_INFINITY, 1.5),
        ..Options::default()
    }
}

#[test]
fn fold_lp_is_refined_to_the_origin() {
    for which in BOTH {
        for (name, u0) in [("fold", vec![-1.0]), ("fold2", vec![-0.7, 0.0])] {
            let setup = EpSetup::new(0, vec![-1.0], u0);
            let opts = Options {
                max_steps: 150,
                ..Options::default()
            };
            let b = run(name, which.clone(), &setup, &opts);
            let lps: Vec<_> = b.specials.iter().filter(|s| s.label == "LP").collect();
            assert_eq!(lps.len(), 1, "{name} {which:?}");
            let s = lps[0];
            assert!(s.extras.refined, "{name} {which:?}");
            assert!(
                s.point.alpha.abs() < 1e-12,
                "{name} alpha {}",
                s.point.alpha
            );
            assert!(
                s.point.u.iter().all(|v| v.abs() < 1e-12),
                "{name} {:?}",
                s.point.u
            );
            let min_eig = s
                .point
                .eigenvalues
                .iter()
                .map(|e| e.0.hypot(e.1))
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_eig < 1e-10,
                "{name}: eigenvalues {:?}",
                s.point.eigenvalues
            );
            assert!(b.warnings.is_empty());
        }
    }
}

#[test]
fn hopf_c_h_is_refined_with_omega_one() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5, -1.0], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("hopf_c", which.clone(), &setup, &opts);
        assert_eq!(b.specials.len(), 1);
        let s = &b.specials[0];
        assert_eq!(s.label, "H");
        assert!(s.extras.refined);
        assert!(s.point.alpha.abs() < 1e-12, "alpha {}", s.point.alpha);
        assert!(s.point.u.iter().all(|v| v.abs() < 1e-12));
        let om = s.extras.omega.unwrap();
        assert!((om - 1.0).abs() < 1e-12, "omega {om}");
        let (re, im) = s
            .point
            .eigenvalues
            .iter()
            .find(|e| e.1 > 0.0)
            .copied()
            .unwrap();
        assert!(re.abs() < 1e-12 && (im - 1.0).abs() < 1e-12, "{re} {im}");
    }
}

#[test]
fn bp_is_refined_on_transcritical_pitchfork_and_lorenz() {
    for which in BOTH {
        for name in ["transcritical", "pitchfork"] {
            let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
            let opts = Options {
                bounds: (f64::NEG_INFINITY, 1.0),
                ..line_opts()
            };
            let b = run(name, which.clone(), &setup, &opts);
            assert_eq!(b.specials.len(), 1, "{name} {which:?}");
            let s = &b.specials[0];
            assert_eq!(s.label, "BP");
            assert!(s.extras.refined);
            assert!(
                s.point.alpha.abs() < 1e-12 && s.point.u[0].abs() < 1e-12,
                "{name}"
            );
        }
        let setup = EpSetup::new(1, vec![10.0, 0.5, 8.0 / 3.0], vec![0.0; 3]);
        let b = run("lorenz", which.clone(), &setup, &line_opts());
        let bps: Vec<_> = b.specials.iter().filter(|s| s.label == "BP").collect();
        assert_eq!(
            bps.len(),
            1,
            "{which:?}: {:?}",
            b.specials.iter().map(|s| &s.label).collect::<Vec<_>>()
        );
        assert!(bps[0].extras.refined);
        assert!(
            (bps[0].point.alpha - 1.0).abs() < 1e-12,
            "{}",
            bps[0].point.alpha
        );
        assert!(bps[0].point.u.iter().all(|v| v.abs() < 1e-12));
    }
}

#[test]
fn nsa_is_not_refined() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("neutral_saddle", which.clone(), &setup, &opts);
        assert_eq!(b.specials.len(), 1);
        assert_eq!(b.specials[0].label, "NSA");
        assert!(!b.specials[0].extras.refined);
        assert!(b.warnings.is_empty());
    }
}

fn fake(label: &str, u: &[f64], alpha: f64, eigs: Vec<(f64, f64)>) -> SpecialPoint {
    let mut sp = SpecialPoint {
        label: label.into(),
        index: 7,
        point: pc_engine::Point {
            u: u.to_vec(),
            alpha,
            tangent: vec![0.0; u.len() + 1],
            eigenvalues: eigs,
            tests: [0.0; 3],
            codim2: Vec::new(),
            h: 0.0,
            newton_iters: 0,
            mesh: Vec::new(),
        },
        extras: Default::default(),
    };
    sp.point.tangent[u.len()] = 1.0;
    sp
}

fn empty_branch(specials: Vec<SpecialPoint>) -> Branch {
    Branch {
        points: Vec::new(),
        specials,
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    }
}

#[test]
fn a_failed_refinement_keeps_the_point_and_warns() {
    for which in BOTH {
        let mut m = model_of("fold", which.clone());
        // An H label with no complex pair cannot be refined.
        let sp = fake("H", &[0.3], 0.1, vec![(-0.6, 0.0)]);
        let mut b = empty_branch(vec![sp.clone()]);
        refine_specials(
            m.as_mut(),
            &[0.0],
            0,
            &Options::default(),
            &AtomicBool::new(false),
            &mut b,
        );
        assert_eq!(b.warnings, ["could not refine H near point 7"]);
        assert_eq!(b.specials[0], sp);
        assert!(!b.specials[0].extras.refined);
        assert_eq!(b.stop, StopReason::MaxSteps);

        // A BP label on a fold model: the system has no solution near the point.
        let sp = fake("BP", &[0.5], 1.0, Vec::new());
        let r = refine_special(
            m.as_mut(),
            &[1.0],
            0,
            &sp,
            &Options::default(),
            &AtomicBool::new(false),
        );
        assert!(r.is_err(), "{which:?}: {r:?}");
        assert_ne!(r.unwrap_err(), EngineError::Cancelled);
    }
}

#[test]
fn cancel_ends_the_run_as_cancelled() {
    for which in BOTH {
        let mut m = model_of("fold", which.clone());
        let sp = fake("LP", &[0.01], 0.0, Vec::new());
        let mut b = empty_branch(vec![sp.clone()]);
        let flag = AtomicBool::new(false);
        flag.store(true, Ordering::Relaxed);
        refine_specials(m.as_mut(), &[0.0], 0, &Options::default(), &flag, &mut b);
        assert_eq!(b.stop, StopReason::Cancelled);
        assert!(b.warnings.is_empty());
        assert_eq!(b.specials[0], sp);
    }
}

fn fold_lp_branch(u: f64, alpha: f64, h: f64) -> Branch {
    let mut sp = fake("LP", &[u], alpha, Vec::new());
    sp.index = 1;
    sp.point.h = h;
    let mut start = sp.point.clone();
    start.h = 0.0;
    let mut reached = sp.point.clone();
    reached.h = h;
    Branch {
        points: vec![start, reached],
        specials: vec![sp],
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    }
}

#[test]
fn a_refinement_that_moves_far_from_the_located_point_is_rejected() {
    for which in BOTH {
        let mut m = model_of("fold", which.clone());
        let cancel = AtomicBool::new(false);
        // Near the fold the refined point is the origin and the move is small.
        let mut near = fold_lp_branch(0.001, -1e-6, 0.01);
        refine_specials(
            m.as_mut(),
            &[0.0],
            0,
            &Options::default(),
            &cancel,
            &mut near,
        );
        assert!(near.specials[0].extras.refined, "{which:?}");
        assert!(near.warnings.is_empty());
        // Started at (u, alpha) = (-0.7, -0.49) on the same curve, Newton also reaches
        // the origin, which is 0.7 away: more than twice the step length 0.01.
        let mut far = fold_lp_branch(-0.7, -0.49, 0.01);
        let before = far.specials[0].clone();
        refine_specials(
            m.as_mut(),
            &[0.0],
            0,
            &Options::default(),
            &cancel,
            &mut far,
        );
        assert!(!far.specials[0].extras.refined, "{which:?}");
        assert_eq!(far.specials[0], before);
        assert_eq!(far.warnings, ["could not refine LP near point 1"]);
    }
}

/// A model that sets the cancel flag at its first `Dir` entry, which only the defining
/// systems of the refinement ask for.
struct CancelsInRefine<'a> {
    inner: Box<dyn ModelEval>,
    cancel: &'a AtomicBool,
}

impl ModelEval for CancelsInRefine<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn n_p(&self) -> usize {
        self.inner.n_p()
    }
    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        if matches!(kind, EntryKind::Dir(_)) {
            self.cancel.store(true, Ordering::Relaxed);
        }
        self.inner.eval(kind, x, p, v, out)
    }
}

#[test]
fn a_cancel_during_refinement_ends_with_cancelled() {
    for which in BOTH {
        let cancel = AtomicBool::new(false);
        let mut m = CancelsInRefine {
            inner: model_of("fold", which.clone()),
            cancel: &cancel,
        };
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            max_steps: 150,
            ..Options::default()
        };
        let b = continue_ep(&mut m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
        assert!(cancel.load(Ordering::Relaxed), "refinement never started");
        assert_eq!(b.stop, StopReason::Cancelled, "{which:?}");
        assert_eq!(b.specials.len(), 1);
    }
}

#[test]
fn a_run_cancelled_before_refinement_skips_it() {
    for which in BOTH {
        let cancel = AtomicBool::new(false);
        let mut m = model_of("fold", which.clone());
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let mut seen = 0;
        let b = continue_ep(
            m.as_mut(),
            &setup,
            &Options::default(),
            &cancel,
            &mut |_| {
                seen += 1;
                if seen == 100 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        )
        .unwrap();
        assert_eq!(b.stop, StopReason::Cancelled);
        assert!(b.specials.iter().all(|s| !s.extras.refined));
        assert!(b.warnings.is_empty());
    }
}

/// Fails the `fail_at`-th `Dir` evaluation (counting from 0) with `err`.
struct FailsDirAt {
    inner: Box<dyn ModelEval>,
    seen: usize,
    fail_at: usize,
    err: fn() -> EngineError,
}

impl ModelEval for FailsDirAt {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn n_p(&self) -> usize {
        self.inner.n_p()
    }
    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        if matches!(kind, EntryKind::Dir(_)) {
            self.seen += 1;
            if self.seen - 1 == self.fail_at {
                return Err((self.err)());
            }
        }
        self.inner.eval(kind, x, p, v, out)
    }
}

#[test]
fn a_failed_coefficient_keeps_the_refined_point_and_warns() {
    for which in BOTH {
        let cancel = AtomicBool::new(false);
        let opts = Options::default();
        // Count the Dir evaluations of a clean pass; the coefficient step makes the last.
        let mut counter = FailsDirAt {
            inner: model_of("fold", which.clone()),
            seen: 0,
            fail_at: usize::MAX,
            err: || EngineError::Singular("unused".into()),
        };
        let mut b = fold_lp_branch(0.001, -1e-6, 0.01);
        refine_specials(&mut counter, &[0.0], 0, &opts, &cancel, &mut b);
        assert!(b.specials[0].extras.coefficient.is_some(), "{which:?}");
        let total = counter.seen;

        let mut m = FailsDirAt {
            inner: model_of("fold", which.clone()),
            seen: 0,
            fail_at: total - 1,
            err: || EngineError::Singular("injected".into()),
        };
        let mut b = fold_lp_branch(0.001, -1e-6, 0.01);
        refine_specials(&mut m, &[0.0], 0, &opts, &cancel, &mut b);
        let ex = &b.specials[0].extras;
        assert!(ex.refined && ex.coefficient.is_none(), "{which:?}");
        assert_eq!(b.stop, StopReason::MaxSteps);
        assert_eq!(
            b.warnings,
            ["could not compute the <a|l1> coefficient near point 1"]
        );

        let mut m = FailsDirAt {
            inner: model_of("fold", which.clone()),
            seen: 0,
            fail_at: total - 1,
            err: || EngineError::Cancelled,
        };
        let mut b = fold_lp_branch(0.001, -1e-6, 0.01);
        refine_specials(&mut m, &[0.0], 0, &opts, &cancel, &mut b);
        assert_eq!(b.stop, StopReason::Cancelled, "{which:?}");
        assert!(b.warnings.is_empty());
    }
}
