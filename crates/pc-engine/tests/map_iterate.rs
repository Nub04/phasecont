//! Period-k points of maps (SPEC section 8y, acceptance tests 1 to 6). Each test states
//! its map and the closed-form value it expects, and computes the expected values in the
//! test.

mod common;

use common::load;
use pc_engine::{
    CurveSetup, EpSetup, Evaluator, IteratedMap, ModelEval, Options, Point, Run, RunKind, RunSpec,
    RunState, SpecialPoint, StopReason, build,
};
use pc_ir::{EntryKind, Slot};
use std::time::{Duration, Instant};

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model(name: &str, ev: &Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build(&m, &info, ev.clone()).unwrap()
}

fn eval(m: &mut dyn ModelEval, kind: &EntryKind, x: &[f64], p: &[f64], v: &[f64]) -> Vec<f64> {
    let len = match kind {
        EntryKind::JacU => m.n() * m.n(),
        EntryKind::JacP => m.n() * m.n_p(),
        _ => m.n(),
    };
    let mut out = vec![0.0; len];
    m.eval(kind, x, p, v, &mut out).unwrap();
    out
}

/// A fixed direction for slot `k` of length `len`.
fn direction(k: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (1.3 * i as f64 + 0.7 * k as f64 + 0.2).sin())
        .collect()
}

/// Every slot list of `U`, `P` and `Full` of length 1 to 5.
fn slot_lists() -> Vec<Vec<Slot>> {
    let mut out: Vec<Vec<Slot>> = Vec::new();
    let mut last: Vec<Vec<Slot>> = vec![Vec::new()];
    for _ in 0..5 {
        let mut next = Vec::new();
        for l in &last {
            for s in [Slot::U, Slot::P, Slot::Full] {
                let mut m = l.clone();
                m.push(s);
                next.push(m);
            }
        }
        out.extend(next.iter().cloned());
        last = next;
    }
    out
}

fn agree(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-10 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g:e}, want {w:e}, error {:e}",
            (g - w).abs()
        );
    }
}

/// Test 1 at one point: every entry of `IteratedMap(base, k)` against the composed model.
fn check_composition(base: &str, k: usize, composed: &str, x: &[f64], p: &[f64]) {
    let lists = slot_lists();
    assert_eq!(lists.len(), 3 + 9 + 27 + 81 + 243);
    for ev in &BOTH {
        let mut inner = model(base, ev);
        let mut want = model(composed, ev);
        let (n, n_p) = (inner.n(), inner.n_p());
        let mut it = IteratedMap::new(inner.as_mut(), k);
        assert_eq!((it.n(), it.n_p(), it.k()), (n, n_p, k));
        assert!(!it.map_residual());
        let tag = format!("{base}^{k} {ev:?} at {x:?}, {p:?}");
        for kind in [EntryKind::F, EntryKind::JacU, EntryKind::JacP] {
            let g = eval(&mut it, &kind, x, p, &[]);
            let w = eval(want.as_mut(), &kind, x, p, &[]);
            agree(&g, &w, &format!("{tag} {kind:?}"));
        }
        for slots in &lists {
            let v: Vec<f64> = slots
                .iter()
                .enumerate()
                .flat_map(|(j, s)| direction(j, s.len(n, n_p)))
                .collect();
            let kind = EntryKind::Dir(slots.clone());
            let g = eval(&mut it, &kind, x, p, &v);
            let w = eval(want.as_mut(), &kind, x, p, &v);
            agree(&g, &w, &format!("{tag} {kind:?}"));
        }
    }
}

#[test]
fn test1_iterated_map_agrees_with_the_composed_models() {
    for (x, r) in [(0.1, 3.2), (0.37, 2.5), (0.81, 3.9)] {
        check_composition("map_logistic", 2, "map_logistic2", &[x], &[r, 0.0]);
        check_composition("map_logistic", 3, "map_logistic3", &[x], &[r, 0.0]);
    }
    for (x, y, a, b) in [
        (0.1, -0.2, 1.4, 0.3),
        (-0.6, 0.4, 0.9, -0.5),
        (0.8, 0.05, 0.3, 1.1),
    ] {
        check_composition("map_henon", 2, "map_henon2", &[x, y], &[a, b, 0.0]);
    }
}

#[test]
fn test1_iterate_one_is_the_map_and_lengths_are_checked() {
    let mut a = model("map_henon", &Evaluator::Jit);
    let mut b = model("map_henon", &Evaluator::Jit);
    let (x, p) = ([0.3, -0.1], [1.4, 0.3, 0.0]);
    let want = eval(a.as_mut(), &EntryKind::JacU, &x, &p, &[]);
    let mut it = IteratedMap::new(b.as_mut(), 0);
    assert_eq!(it.k(), 1);
    assert_eq!(eval(&mut it, &EntryKind::JacU, &x, &p, &[]), want);

    let mut it = IteratedMap::new(b.as_mut(), 2);
    let mut out = vec![0.0; 2];
    let err = it
        .eval(&EntryKind::F, &x, &p[..2], &[], &mut out)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("bad length: p has 2 values, expected 3"),
        "{err}"
    );
    let err = it
        .eval(&EntryKind::Dir(vec![Slot::U]), &x, &p, &[1.0], &mut out)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("bad length: v has 1 values, expected 2"),
        "{err}"
    );
}

/// Runs `spec` to its end.
fn run_to_end(spec: RunSpec) -> (Run, RunState) {
    let run = Run::start(spec);
    let t0 = Instant::now();
    while run.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(60), "timed out");
        std::thread::sleep(Duration::from_millis(2));
    }
    let state = run.poll().state;
    (run, state)
}

fn labels(specials: &[SpecialPoint]) -> Vec<String> {
    specials
        .iter()
        .map(|s| format!("{}@{}", s.label, s.point.alpha))
        .collect()
}

/// The one special point with label `label`.
fn only<'a>(specials: &'a [SpecialPoint], label: &str) -> &'a SpecialPoint {
    let found: Vec<&SpecialPoint> = specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(found.len(), 1, "{label} points: {:?}", labels(specials));
    found[0]
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}, error {:e}",
        (got - want).abs()
    );
}

fn failed(state: RunState, text: &str) {
    match state {
        RunState::Failed(e) => assert!(e.to_string().contains(text), "{e}"),
        other => panic!("expected a failure with {text:?}, got {other:?}"),
    }
}

/// An EP run of the map `name` with free parameter `free`.
fn ep_spec(name: &str, ev: &Evaluator, setup: EpSetup, bounds: (f64, f64)) -> RunSpec {
    let (m, info) = load(name);
    let opts = Options {
        bounds,
        ..Options::default()
    };
    RunSpec::new(m, info, ev.clone(), setup, opts)
}

/// The 2-cycle points x+ and x- of the logistic map at r.
fn logistic_cycle(r: f64) -> [f64; 2] {
    let d = ((r + 1.0) * (r - 3.0)).sqrt();
    [(r + 1.0 + d) / (2.0 * r), (r + 1.0 - d) / (2.0 * r)]
}

/// The logistic map x -> r x (1 - x) from x = 0.6, r = 2.5, with r in [2.5, 3.6].
fn logistic_spec(ev: &Evaluator) -> RunSpec {
    ep_spec(
        "map_logistic",
        ev,
        EpSetup::new(0, vec![2.5], vec![0.6]),
        (2.5, 3.6),
    )
}

fn real_multiplier(p: &Point) -> f64 {
    assert_eq!(p.eigenvalues.len(), 1, "{:?}", p.eigenvalues);
    assert_eq!(p.eigenvalues[0].1, 0.0, "{:?}", p.eigenvalues);
    p.eigenvalues[0].0
}

fn same_rows(a: &[Point], b: &[Point], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        assert_eq!(p.alpha.to_bits(), q.alpha.to_bits(), "{what} row {i} alpha");
        assert_eq!(p.u, q.u, "{what} row {i} u");
        assert_eq!(p.eigenvalues, q.eigenvalues, "{what} row {i} multipliers");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p.tests), bits(&q.tests), "{what} row {i} tests");
    }
}

#[test]
fn test2_logistic_pd_switch_runs_the_2_cycle_at_iterate_2() {
    let r_pd = 1.0 + 6f64.sqrt();
    for ev in &BOTH {
        let mut spec = logistic_spec(ev);
        spec.switch_at = Some("PD1".into());
        let (run, state) = run_to_end(spec);
        assert_eq!(state, RunState::Done, "{ev:?}");
        assert_eq!((run.iterate(0), run.iterate(1)), (Some(1), Some(2)));
        let s0 = run.specials(0);
        let pd0 = only(&s0, "PD");
        close(pd0.point.alpha, 3.0, 1e-8, "PD of the fixed point");
        // Branch 1 is the 2-cycle, with the multiplier F'(x+) F'(x-) = -r^2 + 2r + 4.
        let b1 = run.points(1, 0, usize::MAX);
        assert!(b1.len() > 5, "{} points", b1.len());
        for p in &b1 {
            let r = p.alpha;
            assert!(r > 3.0 - 1e-9, "{ev:?}: r = {r}");
            let near = logistic_cycle(r)
                .iter()
                .map(|c| (p.u[0] - c).abs())
                .fold(f64::INFINITY, f64::min);
            close(near, 0.0, 1e-9, &format!("{ev:?} 2-cycle at r = {r}"));
            close(
                real_multiplier(p),
                -r * r + 2.0 * r + 4.0,
                1e-9,
                &format!("{ev:?} multiplier at r = {r}"),
            );
        }
        assert_eq!(run.stop(1), Some(StopReason::Bounds));
        let s1 = run.specials(1);
        let pd1 = only(&s1, "PD");
        close(pd1.point.alpha, r_pd, 1e-8, "PD of the 2-cycle");
        close(real_multiplier(&pd1.point), -1.0, 1e-8, "PD multiplier");
        // The same PD on the composed model at iterate 1, from the same side of the cycle.
        let side = if pd1.point.u[0] > 0.5 { 0 } else { 1 };
        let r0 = 3.3;
        let (run2, state2) = run_to_end(ep_spec(
            "map_logistic2",
            ev,
            EpSetup::new(0, vec![r0], vec![logistic_cycle(r0)[side]]),
            (3.25, 3.6),
        ));
        assert_eq!(state2, RunState::Done, "{ev:?}");
        let s2 = run2.specials(0);
        let pd2 = only(&s2, "PD");
        close(pd2.point.alpha, pd1.point.alpha, 1e-8, "PD of logistic2");
        close(pd2.point.u[0], pd1.point.u[0], 1e-8, "PD x of logistic2");
        let (c1, c2) = (
            pd1.extras.coefficient.unwrap(),
            pd2.extras.coefficient.unwrap(),
        );
        close(c1, c2, 1e-8 * c2.abs(), "PD coefficient against logistic2");
    }
}

#[test]
fn test3_pd_start_gives_the_switched_branch_and_doubles_again() {
    for ev in &BOTH {
        let mut spec = logistic_spec(ev);
        spec.switch_at = Some("PD1".into());
        let (run, state) = run_to_end(spec);
        assert_eq!(state, RunState::Done, "{ev:?}");
        let pd0 = only(&run.specials(0), "PD").clone();
        let setup = EpSetup::new(0, vec![pd0.point.alpha], pd0.point.u.clone());
        let mut start = ep_spec("map_logistic", ev, setup, (2.5, 3.6));
        start.iterate = 2;
        start.pd_start = true;
        let mut doubled = start.clone();
        let (run2, state2) = run_to_end(start);
        assert_eq!(state2, RunState::Done, "{ev:?}");
        assert_eq!(run2.iterate(0), Some(2));
        same_rows(
            &run2.points(0, 0, usize::MAX),
            &run.points(1, 0, usize::MAX),
            &format!("{ev:?} pd start"),
        );
        let (a, b) = (run2.specials(0), run.specials(1));
        assert_eq!(labels(&a), labels(&b), "{ev:?}");

        // PD1 on the 2-cycle branch runs the 4-cycle at iterate 4.
        doubled.switch_at = Some("PD1".into());
        let (run4, state4) = run_to_end(doubled);
        assert_eq!(state4, RunState::Done, "{ev:?}");
        assert_eq!((run4.iterate(0), run4.iterate(1)), (Some(2), Some(4)));
        let b4 = run4.points(1, 0, usize::MAX);
        assert!(b4.len() > 5, "{} points", b4.len());
        let orbit = |x: f64, r: f64| {
            let mut xs = vec![x];
            for _ in 0..4 {
                let last = *xs.last().unwrap();
                xs.push(r * last * (1.0 - last));
            }
            xs
        };
        for p in &b4 {
            let xs = orbit(p.u[0], p.alpha);
            close(
                xs[4],
                xs[0],
                1e-9,
                &format!("{ev:?} F^4 at r = {}", p.alpha),
            );
        }
        let s4 = run4.specials(1);
        let pd4 = only(&s4, "PD");
        let (x, r) = (pd4.point.u[0], pd4.point.alpha);
        let product: f64 = orbit(x, r)[..4]
            .iter()
            .map(|&xj| r * (1.0 - 2.0 * xj))
            .product();
        close(product, -1.0, 1e-7, "multiplier of the 4-cycle at its PD");
    }
}

/// The + root of the Henon fixed point equation a x^2 + (1 - b) x - 1 = 0, y = b x.
fn henon_fixed(a: f64, b: f64) -> [f64; 2] {
    let x = (-(1.0 - b) + ((1.0 - b).powi(2) + 4.0 * a).sqrt()) / (2.0 * a);
    [x, b * x]
}

/// The PD curve of the Henon 2-cycle, a = (1 - b)^2 + (1 + b)^2 / 4.
fn henon_pd2(b: f64) -> f64 {
    (1.0 - b).powi(2) + (1.0 + b).powi(2) / 4.0
}

fn curve_spec(name: &str, kind: RunKind, iterate: usize, direction: f64) -> RunSpec {
    let (m, info) = load(name);
    let opts = Options {
        direction,
        bounds: (-1.2, 1.2),
        ..Options::default()
    };
    let mut spec = RunSpec::curve(m, info, Evaluator::Jit, kind, opts);
    spec.iterate = iterate;
    spec
}

/// The specials of a curve run in both directions.
fn both_ways(name: &str, setup: &CurveSetup, iterate: usize) -> Vec<SpecialPoint> {
    let mut out = Vec::new();
    for direction in [1.0, -1.0] {
        let (run, state) = run_to_end(curve_spec(
            name,
            RunKind::MapPdCurve(setup.clone()),
            iterate,
            direction,
        ));
        assert_eq!(state, RunState::Done, "{name} direction {direction}");
        assert_eq!(run.iterate(0), Some(iterate));
        assert_eq!(run.stop(0), Some(StopReason::Bounds));
        let pts = run.points(0, 0, usize::MAX);
        assert!(pts.len() > 5, "{name}: {} points", pts.len());
        for p in &pts {
            let b = p.alpha;
            close(
                p.u[2],
                henon_pd2(b),
                1e-9,
                &format!("{name} PD curve at b = {b}"),
            );
        }
        out.extend(run.specials(0));
    }
    out.sort_by(|s, t| s.point.alpha.total_cmp(&t.point.alpha));
    out
}

#[test]
fn test4_henon_2_cycle_and_its_pd_curve() {
    let b = 0.3;
    for ev in &BOTH {
        let a0 = 0.2;
        let mut spec = ep_spec(
            "map_henon",
            ev,
            EpSetup::new(0, vec![a0, b], henon_fixed(a0, b).to_vec()),
            (0.2, 1.2),
        );
        spec.switch_at = Some("PD1".into());
        let (run, state) = run_to_end(spec);
        assert_eq!(state, RunState::Done, "{ev:?}");
        assert_eq!((run.iterate(0), run.iterate(1)), (Some(1), Some(2)));
        let pd0 = only(&run.specials(0), "PD").clone();
        close(pd0.point.alpha, 3.0 * (1.0 - b).powi(2) / 4.0, 1e-8, "PD");
        let b1 = run.points(1, 0, usize::MAX);
        assert!(b1.len() > 5, "{} points", b1.len());
        for p in &b1 {
            let a = p.alpha;
            let (x0, y0) = (p.u[0], p.u[1]);
            let (x1, y1) = (1.0 - a * x0 * x0 + y0, b * x0);
            close(1.0 - a * x1 * x1 + y1, x0, 1e-9, "2-cycle x");
            close(b * x1, y0, 1e-9, "2-cycle y");
            // trace of [[-2a x1, 1], [b, 0]] [[-2a x0, 1], [b, 0]].
            let trace = 4.0 * a * a * x0 * x1 + 2.0 * b;
            let want = 4.0 * ((1.0 - b).powi(2) - a) + 2.0 * b;
            close(trace, want, 1e-9, &format!("{ev:?} trace at a = {a}"));
            let sum: f64 = p.eigenvalues.iter().map(|m| m.0).sum();
            close(sum, want, 1e-9, &format!("{ev:?} multipliers at a = {a}"));
        }
        let pd1 = only(&run.specials(1), "PD").clone();
        close(pd1.point.alpha, henon_pd2(b), 1e-8, "PD of the 2-cycle");

        if *ev != Evaluator::Jit {
            continue;
        }
        let setup = CurveSetup::new([0, 1], vec![pd1.point.alpha, b], pd1.point.u[..2].to_vec());
        let got = both_ways("map_henon", &setup, 2);
        for (a, bb) in [(1.0, 1.0), (4.0, -1.0)] {
            let r2: Vec<&SpecialPoint> = got
                .iter()
                .filter(|s| s.label == "R2" && (s.point.alpha - bb).abs() < 1e-6)
                .collect();
            assert_eq!(r2.len(), 1, "R2 at b = {bb}: {:?}", labels(&got));
            close(r2[0].point.u[2], a, 1e-8, "R2 a");
            close(r2[0].point.alpha, bb, 1e-8, "R2 b");
        }
        let want = both_ways("map_henon2", &setup, 1);
        assert_eq!(
            got.iter().map(|s| &s.label).collect::<Vec<_>>(),
            want.iter().map(|s| &s.label).collect::<Vec<_>>(),
            "{:?} against {:?}",
            labels(&got),
            labels(&want)
        );
        for (s, w) in got.iter().zip(&want) {
            close(
                s.point.alpha,
                w.point.alpha,
                1e-8,
                &format!("{} b", s.label),
            );
            close(s.point.u[2], w.point.u[2], 1e-8, &format!("{} a", s.label));
        }
    }
}

/// A curve at iterate 2 in both directions, with every point checked by `check`.
fn curve2(
    name: &str,
    kind: fn(CurveSetup) -> RunKind,
    setup: CurveSetup,
    bounds: (f64, f64),
    check: &dyn Fn(&Point, &str),
) -> Vec<SpecialPoint> {
    let mut out = Vec::new();
    for ev in &BOTH {
        for direction in [1.0, -1.0] {
            let what = format!("{name} {ev:?} direction {direction}");
            let (m, info) = load(name);
            let opts = Options {
                direction,
                bounds,
                ..Options::default()
            };
            let mut spec = RunSpec::curve(m, info, ev.clone(), kind(setup.clone()), opts);
            spec.iterate = 2;
            let (run, state) = run_to_end(spec);
            assert_eq!(state, RunState::Done, "{what}");
            assert_eq!(run.iterate(0), Some(2), "{what}");
            assert_eq!(run.stop(0), Some(StopReason::Bounds), "{what}");
            let pts = run.points(0, 0, usize::MAX);
            assert!(pts.len() > 5, "{what}: {} points", pts.len());
            let specials = run.specials(0);
            for p in pts.iter().chain(specials.iter().map(|s| &s.point)) {
                check(p, &format!("{what} at alpha = {}", p.alpha));
            }
            assert!(run.warnings(0).is_empty(), "{what}: {:?}", run.warnings(0));
            if *ev == Evaluator::Jit {
                out.extend(specials);
            }
        }
    }
    out
}

/// x -> x + a1 + a2 x - x^3, y -> y/2. The LP curve of F (a1 = -2x^3, a2 = 3x^2) is an LP
/// curve of F^2, whose rest multiplier is 1/4, so LPPD = 1/4 + 1 = 5/4, and whose second
/// derivative along x is (F^2)'' = F''(F(x)) F'(x)^2 + F'(F(x)) F''(x) = 2 F''(x) = -12 x.
#[test]
fn test5_cusp_lp_curve_at_iterate_2() {
    let specials = curve2(
        "map_cusp_cp",
        RunKind::MapLpCurve,
        CurveSetup::new([0, 1], vec![0.25, 0.75], vec![-0.5, 0.0]),
        (-1.0, 1.0),
        &|p, at| {
            let x = p.u[0];
            close(p.u[1], 0.0, 1e-8, &format!("{at} y"));
            close(p.u[2], -2.0 * x.powi(3), 1e-8, &format!("{at} a1"));
            close(p.alpha, 3.0 * x * x, 1e-8, &format!("{at} a2"));
            assert_eq!(p.codim2.len(), 4, "{at}");
            close(p.codim2[0], 1.25, 1e-7, &format!("{at} LPPD"));
            assert!(p.codim2[1].is_nan(), "{at} LPNS {}", p.codim2[1]);
            close(p.codim2[2].abs(), 12.0 * x.abs(), 1e-7, &format!("{at} CP"));
            close(p.codim2[3].abs(), 1.0, 1e-8, &format!("{at} R1"));
        },
    );
    assert_eq!(labels(&specials).len(), 1, "{:?}", labels(&specials));
    let cp = only(&specials, "CP");
    close(cp.point.u[2], 0.0, 1e-8, "CP a1");
    close(cp.point.alpha, 0.0, 1e-8, "CP a2");
}

/// z -> (1 + a1) R(1) z + |z|^2 R(1) (a2 I + J) z. With F(z) = l z + c |z|^2 z, F^2(z) is
/// l^2 z + 2 l c |z|^2 z + ..., so the NS curve of F^2 is a1 = 0 with kappa = cos 2, and its
/// first Lyapunov coefficient is a positive multiple of that of F: CH = a2/sqrt(a2^2 + 1).
#[test]
fn test5_chenciner_ns_curve_at_iterate_2() {
    let specials = curve2(
        "map_ch",
        RunKind::MapNsCurve,
        CurveSetup::new([0, 1], vec![0.0, -0.5], vec![0.0, 0.0]),
        (-1.0, 1.0),
        &|p, at| {
            let a2 = p.alpha;
            for i in 0..3 {
                close(p.u[i], 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            close(p.u[3], 2f64.cos(), 1e-8, &format!("{at} kappa"));
            assert_eq!(p.codim2.len(), 8, "{at}");
            let ch = a2 / (a2 * a2 + 1.0).sqrt();
            close(p.codim2[7], ch, 1e-7, &format!("{at} CH"));
        },
    );
    assert_eq!(labels(&specials).len(), 1, "{:?}", labels(&specials));
    let ch = only(&specials, "CH");
    close(ch.point.u[2], 0.0, 1e-8, "CH a1");
    close(ch.point.alpha, 0.0, 1e-8, "CH a2");
}

#[test]
fn test6_pd_with_a_kernel_of_dimension_3_and_non_pd_starts_are_refused() {
    for ev in &BOTH {
        let flip = || {
            ep_spec(
                "map_flip3",
                ev,
                EpSetup::new(0, vec![-0.5], vec![0.0; 3]),
                (-0.5, 0.5),
            )
        };
        let (run, state) = run_to_end(flip());
        assert_eq!(state, RunState::Done, "{ev:?}");
        let specials = run.specials(0);
        let pd = only(&specials, "PD");
        // det(A + I) = (-a)^3 has a triple root, so the located a is only as close as the
        // cube root of the test function's floor (about 2e-6 here). Every singular value of
        // A + I is then |a| > 1e-6, so the first check refuses the switch. At a = 0 below,
        // every singular value is 0 and the second check refuses the start.
        close(pd.point.alpha, 0.0, 1e-5, "PD of the flip");
        let mut spec = flip();
        spec.switch_at = Some("PD1".into());
        failed(run_to_end(spec).1, "PD is not simple");

        let mut spec = ep_spec(
            "map_flip3",
            ev,
            EpSetup::new(0, vec![0.0], vec![0.0; 3]),
            (-0.5, 0.5),
        );
        spec.iterate = 2;
        spec.pd_start = true;
        failed(
            run_to_end(spec).1,
            "the start is not a simple PD point of F^1",
        );
        let mut spec = logistic_spec(ev);
        spec.iterate = 2;
        spec.pd_start = true;
        failed(
            run_to_end(spec).1,
            "the start is not a simple PD point of F^1",
        );
    }
}

#[test]
fn run_refuses_iterate_and_pd_start_where_they_do_not_apply() {
    let ev = Evaluator::Jit;
    let mut spec = logistic_spec(&ev);
    spec.iterate = 0;
    failed(
        run_to_end(spec).1,
        "iterate: expected a whole number from 1 to 1000000",
    );
    let mut spec = logistic_spec(&ev);
    spec.pd_start = true;
    spec.iterate = 3;
    failed(run_to_end(spec).1, "needs an even iterate");
    // An ODE takes neither.
    let ode = || {
        ep_spec(
            "fold",
            &ev,
            EpSetup::new(0, vec![-1.0], vec![1.0]),
            (-2.0, 2.0),
        )
    };
    let mut spec = ode();
    spec.iterate = 2;
    failed(run_to_end(spec).1, "iterate: only a map takes this key");
    let mut spec = ode();
    spec.pd_start = true;
    failed(run_to_end(spec).1, "start: only an EP run of a map");
}
