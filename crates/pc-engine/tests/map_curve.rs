//! LP, PD and NS curves of fixed points of maps in two parameters (SPEC section 8w,
//! acceptance tests 1 to 6). Each test states its map and the closed form it expects, and
//! computes the expected multipliers in the test. Points are checked to 1e-8.

mod common;

use common::load;
use pc_engine::{
    Branch, CurveSetup, EngineError, EpSetup, Evaluator, MapAsRoot, MapLpPdProblem, MapNsProblem,
    ModelEval, Options, Point, Problem, Run, RunKind, RunSpec, RunState, SpecialPoint, StopReason,
    build, continue_ep, continue_map_lp_curve, continue_map_ns_curve, continue_map_pd_curve,
};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

type Follow = fn(
    &mut dyn ModelEval,
    &CurveSetup,
    &Options,
    &AtomicBool,
    &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError>;

fn model(name: &str, ev: &Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build(&m, &info, ev.clone()).unwrap()
}

fn curve(
    follow: Follow,
    name: &str,
    ev: &Evaluator,
    setup: &CurveSetup,
    opts: &Options,
) -> Result<Branch, EngineError> {
    let mut m = model(name, ev);
    follow(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
}

/// A fixed point branch of the map `name` through `MapAsRoot`.
fn map_branch(name: &str, ev: &Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let mut inner = model(name, ev);
    let mut g = MapAsRoot::new(inner.as_mut());
    continue_ep(&mut g, setup, opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// The one special point with label `label`.
fn only<'a>(specials: &'a [SpecialPoint], label: &str) -> &'a SpecialPoint {
    let found: Vec<&SpecialPoint> = specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(found.len(), 1, "{label} points among {}", specials.len());
    found[0]
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}, error {:e}",
        (got - want).abs()
    );
}

/// Eigenvalues of the 2x2 matrix [[a, b], [c, d]].
fn eig2(a: f64, b: f64, c: f64, d: f64) -> [(f64, f64); 2] {
    let (h, det) = ((a + d) / 2.0, a * d - b * c);
    let disc = h * h - det;
    if disc >= 0.0 {
        [(h + disc.sqrt(), 0.0), (h - disc.sqrt(), 0.0)]
    } else {
        [(h, (-disc).sqrt()), (h, -(-disc).sqrt())]
    }
}

/// Every expected multiplier has a match among `got` within `tol`.
fn has_multipliers(got: &[(f64, f64)], want: &[(f64, f64)], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?}");
    for w in want {
        assert!(
            got.iter()
                .any(|g| (g.0 - w.0).abs() <= tol && (g.1 - w.1).abs() <= tol),
            "{what}: no multiplier near {w:?} in {got:?}"
        );
    }
}

/// Multipliers (1 + a) e^{+-i th}.
fn rotation(a: f64, th: f64) -> [(f64, f64); 2] {
    [
        ((1.0 + a) * th.cos(), (1.0 + a) * th.sin()),
        ((1.0 + a) * th.cos(), -(1.0 + a) * th.sin()),
    ]
}

/// The fixed point of the Henon map (x, y) -> (1 - a x^2 + y, b x) on the + root.
fn henon_fixed(a: f64, b: f64) -> [f64; 2] {
    let x = if a == 0.0 {
        1.0 / (1.0 - b)
    } else {
        (-(1.0 - b) + ((1.0 - b).powi(2) + 4.0 * a).sqrt()) / (2.0 * a)
    };
    [x, b * x]
}

/// The largest absolute entry of `v`.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// The largest residual of `p` at `x`.
fn residual(p: &mut dyn Problem, x: &[f64]) -> f64 {
    let mut f = vec![0.0; p.dim() - 1];
    p.residual(x, &mut f).unwrap();
    max_abs(&f)
}

/// The analytic Jacobian of `p` at `x` against central differences of its residual, with
/// the borders fixed. Returns the largest error relative to the largest entry.
fn fd_error(p: &mut dyn Problem, x: &[f64]) -> f64 {
    let dim = p.dim();
    let rows = dim - 1;
    let mut jac = vec![0.0; rows * dim];
    p.jacobian(x, &mut jac).unwrap();
    let mut err = 0.0_f64;
    for k in 0..dim {
        let h = 1e-6 * x[k].abs().max(1.0);
        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
        xp[k] += h;
        xm[k] -= h;
        let (mut fp, mut fm) = (vec![0.0; rows], vec![0.0; rows]);
        p.residual(&xp, &mut fp).unwrap();
        p.residual(&xm, &mut fm).unwrap();
        for r in 0..rows {
            let fd = (fp[r] - fm[r]) / (2.0 * h);
            err = err.max((jac[r + rows * k] - fd).abs());
        }
    }
    err / max_abs(&jac)
}

/// map_ns4 at a point that is not on any of the curves: x, the parameters (a, th) free.
const NS4_X: [f64; 4] = [0.1, -0.2, 0.15, 0.05];
const NS4_P: [f64; 7] = [0.05, -0.4, 1.1, 0.95, 1.6, 0.9, 0.3];

#[test]
fn test1_jacobians_match_central_differences() {
    for ev in &BOTH {
        let setup = CurveSetup::new([0, 2], NS4_P.to_vec(), NS4_X.to_vec());
        for (sigma, name) in [(-1.0, "LP"), (1.0, "PD")] {
            let mut m = model("map_ns4", ev);
            let mut p = MapLpPdProblem::new(m.as_mut(), &setup, sigma).unwrap();
            let x = p.start().to_vec();
            let e = fd_error(&mut p, &x);
            eprintln!("{ev:?} {name}: relative Jacobian error {e:e}");
            assert!(e < 1e-6, "{name}: {e:e}");
        }
        let mut m = model("map_ns4", ev);
        let mut p = MapNsProblem::new(m.as_mut(), &setup).unwrap();
        let mut x = p.start().to_vec();
        // A kappa away from the start value, so the bordered matrix is not near singular.
        x[5] = 0.4;
        let e = fd_error(&mut p, &x);
        eprintln!("{ev:?} NS: relative Jacobian error {e:e}");
        assert!(e < 1e-6, "NS: {e:e}");
    }
}

/// Henon (x, y) -> (1 - a x^2 + y, b x) with F_x = [[-2 a x, 1], [b, 0]]. The LP curve is
/// a = -(1 - b)^2 / 4 with x = 2 / (1 - b) and the multipliers 1 and -b. The PD curve is
/// a = 3 (1 - b)^2 / 4 with x = 2 / (3 (1 - b)) and the multipliers -1 and b.
fn henon_lp(b: f64) -> (f64, f64) {
    (-(1.0 - b).powi(2) / 4.0, 2.0 / (1.0 - b))
}

fn henon_pd(b: f64) -> (f64, f64) {
    (3.0 * (1.0 - b).powi(2) / 4.0, 2.0 / (3.0 * (1.0 - b)))
}

/// Checks every point of a Henon LP (`sigma = -1`) or PD (`sigma = 1`) curve, whose points
/// store u = (x, y, a) and alpha = b.
fn check_henon(br: &Branch, sigma: f64, what: &str) {
    assert!(br.points.len() > 5, "{what}: {} points", br.points.len());
    for p in &br.points {
        let b = p.alpha;
        let (a, x) = if sigma < 0.0 {
            henon_lp(b)
        } else {
            henon_pd(b)
        };
        close(p.u[2], a, 1e-8, &format!("{what} a at b = {b}"));
        close(p.u[0], x, 1e-8, &format!("{what} x at b = {b}"));
        close(p.u[1], b * x, 1e-8, &format!("{what} y at b = {b}"));
        has_multipliers(&p.eigenvalues, &eig2(-2.0 * a * x, 1.0, b, 0.0), 1e-8, what);
        has_multipliers(
            &p.eigenvalues,
            &[(-sigma, 0.0), (sigma * b, 0.0)],
            1e-7,
            what,
        );
        assert!(p.tests.iter().all(|t| t.is_nan()), "{what}: {:?}", p.tests);
        assert!(p.codim2.is_empty(), "{what}: {:?}", p.codim2);
    }
    assert!(
        br.specials.is_empty(),
        "{what}: {} specials",
        br.specials.len()
    );
    assert_eq!(br.stop, StopReason::Bounds, "{what}");
}

#[test]
fn test2_henon_lp_and_pd_curves_follow_the_closed_form() {
    let b0 = 0.3;
    for ev in &BOTH {
        for (follow, sigma, name) in [
            (continue_map_lp_curve as Follow, -1.0, "LP"),
            (continue_map_pd_curve as Follow, 1.0, "PD"),
        ] {
            let (a0, x0) = if sigma < 0.0 {
                henon_lp(b0)
            } else {
                henon_pd(b0)
            };
            let setup = CurveSetup::new([0, 1], vec![a0, b0], vec![x0, b0 * x0]);
            for direction in [1.0, -1.0] {
                let opts = Options {
                    direction,
                    bounds: (0.1, 0.6),
                    ..Options::default()
                };
                let br = curve(follow, "map_henon", ev, &setup, &opts).unwrap();
                let what = format!("{ev:?} {name} direction {direction}");
                check_henon(&br, sigma, &what);
                // The default direction is increasing x, which is increasing b on both.
                let last = br.points.last().unwrap().alpha;
                assert!(
                    (direction > 0.0 && last > 0.55) || (direction < 0.0 && last < 0.15),
                    "{what}: ends at b = {last}"
                );
            }
        }
    }
}

/// AUTO's Henon form (x, y) -> (y, a - b x - y^2) with F_x = [[0, 1], [-b, -2y]], so
/// det F_x = b. The NS curve is b = 1 with x = y = -1 + sqrt(1 + a), kappa = -y and the
/// multipliers kappa +- i sqrt(1 - kappa^2). Free [b, a]: points store u = (x, y, b,
/// kappa) and alpha = a.
#[test]
fn test3_henon_auto_ns_curve_is_b_equal_1() {
    for ev in &BOTH {
        let setup = CurveSetup::new([1, 0], vec![0.69, 1.0], vec![0.3, 0.3]);
        for direction in [1.0, -1.0] {
            let opts = Options {
                direction,
                bounds: (-0.5, 2.5),
                ..Options::default()
            };
            let br = curve(continue_map_ns_curve, "map_henon_auto", ev, &setup, &opts).unwrap();
            let what = format!("{ev:?} NS direction {direction}");
            assert!(br.points.len() > 5, "{what}: {} points", br.points.len());
            for p in &br.points {
                let a = p.alpha;
                let y = -1.0 + (1.0 + a).sqrt();
                close(p.u[0], y, 1e-8, &format!("{what} x at a = {a}"));
                close(p.u[1], y, 1e-8, &format!("{what} y at a = {a}"));
                close(p.u[2], 1.0, 1e-8, &format!("{what} b at a = {a}"));
                close(p.u[3], -y, 1e-8, &format!("{what} kappa at a = {a}"));
                let s = (1.0 - y * y).sqrt();
                has_multipliers(&p.eigenvalues, &[(-y, s), (-y, -s)], 1e-8, &what);
                assert!(p.tests.iter().all(|t| t.is_nan()), "{what}: {:?}", p.tests);
            }
            assert!(br.specials.is_empty(), "{what}");
            assert_eq!(br.stop, StopReason::Bounds, "{what}");
            // The default direction is increasing x, which is increasing a.
            let (first, last) = (br.points[0].alpha, br.points.last().unwrap().alpha);
            assert!(
                (direction > 0.0 && last > 2.4) || (direction < 0.0 && last < -0.4),
                "{what}: runs from a = {first} to a = {last}"
            );
        }
    }
}

/// map_ns4 keeps the origin fixed with the multipliers (1 + a) e^{+-i th} and
/// rho e^{+-i om}, rho = 0.95, om = 1.6. The NS curve in (a, th) is a = 0 with
/// kappa = cos(th). Points store u = (x, a, kappa) and alpha = th.
#[test]
fn test4_ns4_ns_curve_is_a_equal_0() {
    for ev in &BOTH {
        let mut params = NS4_P.to_vec();
        params[0] = 0.0;
        let setup = CurveSetup::new([0, 2], params, vec![0.0; 4]);
        for direction in [1.0, -1.0] {
            let opts = Options {
                direction,
                bounds: (0.5, 2.5),
                ..Options::default()
            };
            let br = curve(continue_map_ns_curve, "map_ns4", ev, &setup, &opts).unwrap();
            let what = format!("{ev:?} ns4 direction {direction}");
            assert!(br.points.len() > 5, "{what}: {} points", br.points.len());
            for p in &br.points {
                let th = p.alpha;
                for i in 0..4 {
                    close(p.u[i], 0.0, 1e-8, &format!("{what} x[{i}] at th = {th}"));
                }
                close(p.u[4], 0.0, 1e-8, &format!("{what} a at th = {th}"));
                close(
                    p.u[5],
                    th.cos(),
                    1e-8,
                    &format!("{what} kappa at th = {th}"),
                );
                let [s1, s2] = rotation(0.0, th);
                let [s3, s4] = rotation(0.95 - 1.0, 1.6);
                has_multipliers(&p.eigenvalues, &[s1, s2, s3, s4], 1e-8, &what);
            }
            assert_eq!(br.stop, StopReason::Bounds, "{what}");
            // kappa is the first unknown that moves, and kappa grows as th falls.
            let last = br.points.last().unwrap().alpha;
            assert!(
                (direction > 0.0 && last < 0.6) || (direction < 0.0 && last > 2.4),
                "{what}: ends at th = {last}"
            );
        }
    }
}

/// The start residual of each curve problem from `setup`, printed.
fn start_residual(name: &str, kind: &str, setup: &CurveSetup) -> f64 {
    let mut m = model(name, &Evaluator::Jit);
    let r = match kind {
        "LP" => {
            let mut p = MapLpPdProblem::new(m.as_mut(), setup, -1.0).unwrap();
            let x = p.start().to_vec();
            residual(&mut p, &x)
        }
        "PD" => {
            let mut p = MapLpPdProblem::new(m.as_mut(), setup, 1.0).unwrap();
            let x = p.start().to_vec();
            residual(&mut p, &x)
        }
        _ => {
            let mut p = MapNsProblem::new(m.as_mut(), setup).unwrap();
            let x = p.start().to_vec();
            residual(&mut p, &x)
        }
    };
    eprintln!("{name} {kind}: start residual {r:e}");
    r
}

#[test]
fn test5_curves_start_from_stored_ep_specials_and_nearby() {
    let b = 0.3;
    // Henon PD and LP points of the branches in a at b = 0.3.
    let pd_branch = map_branch(
        "map_henon",
        &Evaluator::Jit,
        &EpSetup::new(0, vec![0.3, b], henon_fixed(0.3, b).to_vec()),
        &Options {
            bounds: (f64::NEG_INFINITY, 0.5),
            ..Options::default()
        },
    );
    let lp_branch = map_branch(
        "map_henon",
        &Evaluator::Jit,
        &EpSetup::new(0, vec![0.0, b], henon_fixed(0.0, b).to_vec()),
        &Options {
            direction: -1.0,
            max_steps: 100,
            ..Options::default()
        },
    );
    let opts = Options {
        bounds: (0.1, 0.6),
        ..Options::default()
    };
    for (branch, follow, sigma, kind) in [
        (&lp_branch, continue_map_lp_curve as Follow, -1.0, "LP"),
        (&pd_branch, continue_map_pd_curve as Follow, 1.0, "PD"),
    ] {
        let sp = only(&branch.specials, kind);
        for shift in [0.0, 1e-8] {
            let setup =
                CurveSetup::new([0, 1], vec![sp.point.alpha + shift, b], sp.point.u.clone());
            let r = start_residual("map_henon", kind, &setup);
            assert!(r < 1e-6, "{kind} shift {shift:e}: {r:e}");
            let br = curve(follow, "map_henon", &Evaluator::Jit, &setup, &opts).unwrap();
            check_henon(&br, sigma, &format!("{kind} from EP, shift {shift:e}"));
        }
    }

    // map_henon_auto: the branch in b at a = 0.69 from b = 0.5 has NS at b = 1.
    let a = 0.69;
    let x = (-1.5 + (2.25_f64 + 4.0 * a).sqrt()) / 2.0;
    let br = map_branch(
        "map_henon_auto",
        &Evaluator::Jit,
        &EpSetup::new(1, vec![a, 0.5], vec![x, x]),
        &Options {
            bounds: (f64::NEG_INFINITY, 1.2),
            ..Options::default()
        },
    );
    let ns = only(&br.specials, "NS");
    close(ns.point.alpha, 1.0, 1e-8, "NS b");
    let opts = Options {
        bounds: (-0.5, 2.5),
        ..Options::default()
    };
    for shift in [0.0, 1e-8] {
        let setup = CurveSetup::new([1, 0], vec![a, ns.point.alpha + shift], ns.point.u.clone());
        let r = start_residual("map_henon_auto", "NS", &setup);
        assert!(r < 1e-6, "NS shift {shift:e}: {r:e}");
        let curve = curve(
            continue_map_ns_curve,
            "map_henon_auto",
            &Evaluator::Jit,
            &setup,
            &opts,
        )
        .unwrap();
        for p in &curve.points {
            close(
                p.u[2],
                1.0,
                1e-8,
                &format!("NS from EP b at a = {}", p.alpha),
            );
        }
        assert_eq!(curve.stop, StopReason::Bounds);
    }

    // map_ns4: the branch in a has NS at a = 0.
    let mut params = NS4_P.to_vec();
    params[0] = -0.1;
    let br = map_branch(
        "map_ns4",
        &Evaluator::Jit,
        &EpSetup::new(0, params.clone(), vec![0.0; 4]),
        &Options {
            bounds: (f64::NEG_INFINITY, 0.1),
            ..Options::default()
        },
    );
    let ns = only(&br.specials, "NS");
    let opts = Options {
        bounds: (0.5, 2.5),
        ..Options::default()
    };
    for shift in [0.0, 1e-8] {
        params[0] = ns.point.alpha + shift;
        let setup = CurveSetup::new([0, 2], params.clone(), ns.point.u.clone());
        let r = start_residual("map_ns4", "NS", &setup);
        assert!(r < 1e-6, "ns4 shift {shift:e}: {r:e}");
        let curve = curve(
            continue_map_ns_curve,
            "map_ns4",
            &Evaluator::Jit,
            &setup,
            &opts,
        )
        .unwrap();
        for p in &curve.points {
            close(
                p.u[4],
                0.0,
                1e-8,
                &format!("ns4 from EP a at th = {}", p.alpha),
            );
        }
        assert_eq!(curve.stop, StopReason::Bounds);
    }
}

fn refused(r: Result<Branch, EngineError>, text: &str) {
    match r {
        Err(EngineError::BadProblem(m)) => assert!(m.contains(text), "{m}"),
        other => panic!("expected a BadProblem with {text:?}, got {other:?}"),
    }
}

#[test]
fn test6_refusals() {
    let opts = Options::default();
    let ev = &Evaluator::Jit;
    // A real pair with product 1: b = 1, x = y = 1.5, a = 5.25 has kappa = -1.5.
    let setup = CurveSetup::new([1, 0], vec![5.25, 1.0], vec![1.5, 1.5]);
    refused(
        curve(continue_map_ns_curve, "map_henon_auto", ev, &setup, &opts),
        "no complex pair of multipliers",
    );
    // Fixed points off the curves.
    let setup = CurveSetup::new([0, 1], vec![0.2, 0.3], henon_fixed(0.2, 0.3).to_vec());
    refused(
        curve(continue_map_lp_curve, "map_henon", ev, &setup, &opts),
        "the start is not near the LP curve",
    );
    refused(
        curve(continue_map_pd_curve, "map_henon", ev, &setup, &opts),
        "the start is not near the PD curve",
    );
    let x = (-1.5 + (2.25 + 4.0 * 0.69_f64).sqrt()) / 2.0;
    let setup = CurveSetup::new([1, 0], vec![0.69, 0.5], vec![x, x]);
    refused(
        curve(continue_map_ns_curve, "map_henon_auto", ev, &setup, &opts),
        "the start is not near the NS curve",
    );
    // One variable has no NS.
    let setup = CurveSetup::new([0, 0], vec![2.5], vec![0.6]);
    refused(
        curve(continue_map_ns_curve, "map_logistic", ev, &setup, &opts),
        "needs at least two variables",
    );
}

/// Runs `spec` to its end.
fn run_to_end(spec: RunSpec) -> (Run, RunState) {
    let run = Run::start(spec);
    let t0 = Instant::now();
    while run.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(30), "timed out");
        std::thread::sleep(Duration::from_millis(2));
    }
    let state = run.poll().state;
    (run, state)
}

fn run_spec(name: &str, kind: RunKind, opts: Options) -> RunSpec {
    let (m, info) = load(name);
    RunSpec::curve(m, info, Evaluator::Jit, kind, opts)
}

#[test]
fn a_run_follows_map_curves_and_refuses_the_wrong_model_kind() {
    let b = 0.3;
    let (a0, x0) = henon_pd(b);
    let setup = CurveSetup::new([0, 1], vec![a0, b], vec![x0, b * x0]);
    let opts = Options {
        bounds: (0.1, 0.6),
        ..Options::default()
    };
    let direct = curve(
        continue_map_pd_curve,
        "map_henon",
        &Evaluator::Jit,
        &setup,
        &opts,
    )
    .unwrap();
    let (run, state) = run_to_end(run_spec(
        "map_henon",
        RunKind::MapPdCurve(setup.clone()),
        opts.clone(),
    ));
    assert_eq!(state, RunState::Done);
    assert_eq!(run.stop(0), Some(StopReason::Bounds));
    let pts = run.points(0, 0, usize::MAX);
    assert_eq!(pts.len(), direct.points.len());
    for (p, q) in pts.iter().zip(&direct.points) {
        assert_eq!((p.alpha, &p.u), (q.alpha, &q.u));
    }

    // A map curve on an ODE, and the ODE curves on a map.
    let ode = CurveSetup::new([0, 1], vec![0.0, 1.0], vec![-1.0]);
    for kind in [
        RunKind::MapLpCurve(ode.clone()),
        RunKind::MapPdCurve(ode.clone()),
        RunKind::MapNsCurve(ode),
    ] {
        let (_, state) = run_to_end(run_spec("cusp", kind, Options::default()));
        assert!(
            matches!(&state, RunState::Failed(EngineError::BadProblem(m)) if m.contains("needs a map")),
            "{state:?}"
        );
    }
    for kind in [
        RunKind::LpCurve(setup.clone()),
        RunKind::HopfCurve(setup.clone()),
    ] {
        let (_, state) = run_to_end(run_spec("map_henon", kind, Options::default()));
        assert!(
            matches!(state, RunState::Failed(EngineError::BadProblem(_))),
            "{state:?}"
        );
    }

    // A map that reads t has no fixed points to follow.
    let shift = CurveSetup::new([0, 1], vec![], vec![0.0]);
    let (_, state) = run_to_end(run_spec(
        "map_shift",
        RunKind::MapLpCurve(shift),
        Options::default(),
    ));
    assert_eq!(state, RunState::Failed(EngineError::NonAutonomous));
}
