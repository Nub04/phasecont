//! Fixed points of maps (SPEC section 8v, acceptance tests 1 to 8). Each test states its
//! map and the closed-form value it expects, and computes the expected multipliers in the
//! test. Points are checked to 1e-8 in alpha and in u, coefficients to a relative 1e-7.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, EpSetup, Evaluator, IvpOptions, MapAsRoot, ModelEval, Options, Run,
    RunKind, RunSpec, RunState, SpecialPoint, build, continue_ep, refine_special,
};
use pc_ir::{EntryKind, Slot};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model(name: &str, ev: &Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build(&m, &info, ev.clone()).unwrap()
}

/// A fixed point branch of the map `name` through `MapAsRoot`.
fn map_branch(name: &str, ev: &Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let mut inner = model(name, ev);
    let mut g = MapAsRoot::new(inner.as_mut());
    continue_ep(&mut g, setup, opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
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

fn rel(got: f64, want: f64, tol: f64, what: &str) {
    close(got, want, tol * want.abs(), what);
}

/// Checks a special point against its expected position.
fn at(sp: &SpecialPoint, alpha: f64, u: &[f64]) {
    close(sp.point.alpha, alpha, 1e-8, &format!("{} alpha", sp.label));
    for (i, (&g, &w)) in sp.point.u.iter().zip(u).enumerate() {
        close(g, w, 1e-8, &format!("{} u[{i}]", sp.label));
    }
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

/// The (x, p) step of one slot direction: `U` moves x, `P` moves the first n_p entries of
/// p and `Full` moves both.
fn embed(slot: Slot, v: &[f64], n: usize, n_p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dp = vec![0.0; n_p + 1];
    match slot {
        Slot::U => (v.to_vec(), dp),
        Slot::P => {
            dp[..n_p].copy_from_slice(v);
            (vec![0.0; n], dp)
        }
        Slot::Full => {
            dp[..n_p].copy_from_slice(&v[n..]);
            (v[..n].to_vec(), dp)
        }
    }
}

fn shifted(x: &[f64], p: &[f64], dx: &[f64], dp: &[f64], s: f64) -> (Vec<f64>, Vec<f64>) {
    (
        x.iter().zip(dx).map(|(a, b)| a + s * b).collect(),
        p.iter().zip(dp).map(|(a, b)| a + s * b).collect(),
    )
}

/// Test 1 at one point: every entry of `MapAsRoot` against `F - x` and its derivatives,
/// with finite differences of the adapter's own entries.
fn check_adapter(name: &str, ev: &Evaluator, x: &[f64], p: &[f64], n_aux: usize) {
    let mut inner = model(name, ev);
    let (n, n_p) = (inner.n(), inner.n_p());
    let fi = eval(inner.as_mut(), &EntryKind::F, x, p, &[]);
    let ji = eval(inner.as_mut(), &EntryKind::JacU, x, p, &[]);
    let pi = eval(inner.as_mut(), &EntryKind::JacP, x, p, &[]);
    let mut ai = vec![0.0; n_aux];
    if n_aux > 0 {
        inner.eval(&EntryKind::Aux, x, p, &[], &mut ai).unwrap();
    }
    let slot_sets: [&[Slot]; 8] = [
        &[Slot::U],
        &[Slot::Full],
        &[Slot::P],
        &[Slot::U, Slot::U],
        &[Slot::U, Slot::P],
        &[Slot::Full, Slot::Full],
        &[Slot::U, Slot::U, Slot::U],
        &[Slot::U, Slot::U, Slot::P],
    ];
    let dirs: Vec<Vec<f64>> = slot_sets
        .iter()
        .map(|s| {
            s.iter()
                .enumerate()
                .flat_map(|(k, sl)| direction(k, sl.len(n, n_p)))
                .collect()
        })
        .collect();
    let inner_dirs: Vec<Vec<f64>> = slot_sets
        .iter()
        .zip(&dirs)
        .map(|(s, v)| eval(inner.as_mut(), &EntryKind::Dir(s.to_vec()), x, p, v))
        .collect();

    let mut g = MapAsRoot::new(inner.as_mut());
    assert!(g.map_residual());
    let tag = format!("{name} {ev:?}");

    // F, JacU and Aux exactly.
    let f = eval(&mut g, &EntryKind::F, x, p, &[]);
    for i in 0..n {
        assert_eq!(f[i].to_bits(), (fi[i] - x[i]).to_bits(), "{tag} F[{i}]");
    }
    let j = eval(&mut g, &EntryKind::JacU, x, p, &[]);
    for c in 0..n {
        for r in 0..n {
            let want = ji[r + n * c] - if r == c { 1.0 } else { 0.0 };
            assert_eq!(j[r + n * c].to_bits(), want.to_bits(), "{tag} JacU");
        }
    }
    let jp = eval(&mut g, &EntryKind::JacP, x, p, &[]);
    assert_eq!(jp, pi, "{tag} JacP");
    if n_aux > 0 {
        let mut a = vec![0.0; n_aux];
        g.eval(&EntryKind::Aux, x, p, &[], &mut a).unwrap();
        assert_eq!(a, ai, "{tag} Aux");
    }

    // JacU and JacP against central differences of G.
    let h = 1e-6;
    for c in 0..n + n_p {
        let mut dx = vec![0.0; n];
        let mut dp = vec![0.0; n_p + 1];
        if c < n {
            dx[c] = 1.0;
        } else {
            dp[c - n] = 1.0;
        }
        let (xa, pa) = shifted(x, p, &dx, &dp, h);
        let (xb, pb) = shifted(x, p, &dx, &dp, -h);
        let fa = eval(&mut g, &EntryKind::F, &xa, &pa, &[]);
        let fb = eval(&mut g, &EntryKind::F, &xb, &pb, &[]);
        for r in 0..n {
            let fd = (fa[r] - fb[r]) / (2.0 * h);
            let got = if c < n {
                j[r + n * c]
            } else {
                jp[r + n * (c - n)]
            };
            close(
                got,
                fd,
                1e-6 * got.abs().max(1.0),
                &format!("{tag} jac {r},{c}"),
            );
        }
    }

    // Each Dir entry against a central difference of the entry with one slot fewer, along
    // the direction of its last slot.
    for (s, (v, vi)) in slot_sets.iter().zip(dirs.iter().zip(&inner_dirs)) {
        let k = s.len();
        let got = eval(&mut g, &EntryKind::Dir(s.to_vec()), x, p, v);
        match s {
            [Slot::U] | [Slot::Full] => {
                for i in 0..n {
                    assert_eq!(got[i].to_bits(), (vi[i] - v[i]).to_bits(), "{tag} {s:?}");
                }
            }
            _ => assert_eq!(&got, vi, "{tag} {s:?}"),
        }
        let last = s[k - 1];
        let split = v.len() - last.len(n, n_p);
        let (dx, dp) = embed(last, &v[split..], n, n_p);
        let (xa, pa) = shifted(x, p, &dx, &dp, h);
        let (xb, pb) = shifted(x, p, &dx, &dp, -h);
        let lower = if k == 1 {
            EntryKind::F
        } else {
            EntryKind::Dir(s[..k - 1].to_vec())
        };
        let ga = eval(&mut g, &lower, &xa, &pa, &v[..split]);
        let gb = eval(&mut g, &lower, &xb, &pb, &v[..split]);
        let scale = v.iter().fold(1.0f64, |m, a| m.max(a.abs()));
        for i in 0..n {
            let fd = (ga[i] - gb[i]) / (2.0 * h);
            close(
                got[i],
                fd,
                1e-6 * scale * got[i].abs().max(1.0),
                &format!("{tag} {s:?}[{i}]"),
            );
        }
    }
}

#[test]
fn test1_adapter_entries_match_f_minus_x_and_finite_differences() {
    for ev in &BOTH {
        // Henon (x, y) -> (1 - a x^2 + y, b x).
        check_adapter("map_henon", ev, &[0.3, -0.2], &[1.4, 0.3, 0.0], 0);
        // map_ns4: a, s, th, rho, om, k, m and t.
        check_adapter(
            "map_ns4",
            ev,
            &[0.1, -0.2, 0.15, 0.05],
            &[0.05, -0.4, 1.1, 0.95, 1.6, 0.9, 0.3, 0.0],
            1,
        );
    }
}

/// Logistic map x -> r x (1 - x) from x = 0 at r = 0.5, switching at BP1.
fn logistic_spec(ev: &Evaluator) -> RunSpec {
    let (m, info) = load("map_logistic");
    let setup = EpSetup::new(0, vec![0.5], vec![0.0]);
    let opts = Options {
        bounds: (0.0, 4.0),
        ..Options::default()
    };
    let mut spec = RunSpec::new(m, info, ev.clone(), setup, opts);
    spec.switch_at = Some("BP1".into());
    spec
}

#[test]
fn test2_logistic_bp_then_pd_with_b_9() {
    for ev in &BOTH {
        let (run, state) = run_to_end(logistic_spec(ev));
        assert_eq!(state, RunState::Done, "{ev:?}");
        // Branch 0 is x = 0, with the multiplier r.
        let b0 = run.points(0, 0, usize::MAX);
        assert!(b0.len() > 5);
        for p in &b0 {
            close(p.u[0], 0.0, 1e-10, "x = 0 branch");
            has_multipliers(&p.eigenvalues, &[(p.alpha, 0.0)], 1e-10, "x = 0 branch");
        }
        let s0 = run.specials(0);
        let bp = only(&s0, "BP");
        at(bp, 1.0, &[0.0]);
        has_multipliers(&bp.point.eigenvalues, &[(1.0, 0.0)], 1e-8, "BP");
        // Branch 1 is x = 1 - 1/r, with the multiplier 2 - r.
        let b1 = run.points(1, 0, usize::MAX);
        assert!(b1.len() > 5);
        for p in &b1 {
            close(p.u[0], 1.0 - 1.0 / p.alpha, 1e-9, "x = 1 - 1/r branch");
            has_multipliers(&p.eigenvalues, &[(2.0 - p.alpha, 0.0)], 1e-8, "x = 1 - 1/r");
        }
        let s1 = run.specials(1);
        let pd = only(&s1, "PD");
        at(pd, 3.0, &[2.0 / 3.0]);
        has_multipliers(&pd.point.eigenvalues, &[(-1.0, 0.0)], 1e-8, "PD");
        // b = (f'')^2/4 + f'''/6 with f'' = -2r = -6 and f''' = 0.
        rel(pd.extras.coefficient.unwrap(), 9.0, 1e-7, "PD b");
        assert_eq!(pd.extras.coefficient2, None);
        assert_eq!(pd.extras.normalization.as_deref(), Some("|q|=1, p'q=1"));
        assert!(run.warnings(1).is_empty(), "{:?}", run.warnings(1));
    }
}

/// The + root of the Henon fixed point equation a x^2 + (1 - b) x - 1 = 0, y = b x.
fn henon_fixed(a: f64, b: f64) -> [f64; 2] {
    let x = if a == 0.0 {
        1.0 / (1.0 - b)
    } else {
        (-(1.0 - b) + ((1.0 - b).powi(2) + 4.0 * a).sqrt()) / (2.0 * a)
    };
    [x, b * x]
}

/// Multipliers of the Henon map at (x, y): F_x = [[-2 a x, 1], [b, 0]].
fn henon_multipliers(a: f64, b: f64, x: f64) -> [(f64, f64); 2] {
    eig2(-2.0 * a * x, 1.0, b, 0.0)
}

#[test]
fn test3_henon_lp_and_pd() {
    let b = 0.3;
    for ev in &BOTH {
        // PD at a = 3 (1 - b)^2 / 4 on the + root, where a x = (1 - b) / 2.
        let setup = EpSetup::new(0, vec![0.3, b], henon_fixed(0.3, b).to_vec());
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.5),
            ..Options::default()
        };
        let br = map_branch("map_henon", ev, &setup, &opts);
        for p in &br.points {
            has_multipliers(
                &p.eigenvalues,
                &henon_multipliers(p.alpha, b, p.u[0]),
                1e-8,
                "Henon branch",
            );
        }
        let pd = only(&br.specials, "PD");
        let a_pd = 3.0 * (1.0 - b) * (1.0 - b) / 4.0;
        at(pd, a_pd, &henon_fixed(a_pd, b));
        has_multipliers(&pd.point.eigenvalues, &[(-1.0, 0.0), (b, 0.0)], 1e-8, "PD");
        rel(
            pd.extras.coefficient.unwrap(),
            0.136159139026,
            1e-7,
            "Henon PD b",
        );
        assert_eq!(br.specials.len(), 1, "{:?}", labels(&br.specials));

        // LP at a = -(1 - b)^2 / 4, from a = 0 downward.
        let setup = EpSetup::new(0, vec![0.0, b], henon_fixed(0.0, b).to_vec());
        let opts = Options {
            direction: -1.0,
            max_steps: 100,
            ..Options::default()
        };
        let br = map_branch("map_henon", ev, &setup, &opts);
        let lp = only(&br.specials, "LP");
        at(lp, -0.1225, &[2.857142857142857, 0.8571428571428571]);
        // At the fold one multiplier is 1 and their product is -b.
        has_multipliers(&lp.point.eigenvalues, &[(1.0, 0.0), (-b, 0.0)], 1e-7, "LP");
        rel(
            lp.extras.coefficient.unwrap(),
            -0.09025670764583925,
            1e-7,
            "Henon LP a",
        );
    }
}

#[test]
fn test4_flip_normal_form_and_conjugate_have_b_equal_s() {
    for ev in &BOTH {
        for (name, params) in [
            ("map_flip", vec![-0.1, 0.7, 0.5, 0.3]),
            ("map_flip_conj", vec![-0.1, 0.7, 0.5, 0.3, 0.8, -0.6]),
        ] {
            let setup = EpSetup::new(0, params, vec![0.0, 0.0]);
            let opts = Options {
                bounds: (f64::NEG_INFINITY, 0.1),
                ..Options::default()
            };
            let br = map_branch(name, ev, &setup, &opts);
            for p in &br.points {
                // The multipliers at the origin are -(1 + a) and lam = 0.5.
                has_multipliers(
                    &p.eigenvalues,
                    &[(-(1.0 + p.alpha), 0.0), (0.5, 0.0)],
                    1e-10,
                    name,
                );
            }
            let pd = only(&br.specials, "PD");
            at(pd, 0.0, &[0.0, 0.0]);
            rel(pd.extras.coefficient.unwrap(), 0.7, 1e-7, name);
            assert_eq!(br.specials.len(), 1, "{name}: {:?}", labels(&br.specials));
        }
    }
}

/// Multipliers (1 + a) e^{+-i th}.
fn rotation(a: f64, th: f64) -> [(f64, f64); 2] {
    [
        ((1.0 + a) * th.cos(), (1.0 + a) * th.sin()),
        ((1.0 + a) * th.cos(), -(1.0 + a) * th.sin()),
    ]
}

fn theta(sp: &SpecialPoint) -> f64 {
    let v: Vec<f64> = sp
        .extras
        .values
        .iter()
        .filter(|(k, _)| k == "theta")
        .map(|(_, v)| *v)
        .collect();
    assert_eq!(v.len(), 1, "values: {:?}", sp.extras.values);
    v[0]
}

#[test]
fn test5_ns_normal_form_and_conjugate_have_d_equal_2s() {
    for ev in &BOTH {
        for (name, params) in [
            ("map_ns", vec![-0.1, -0.4, 1.1]),
            ("map_ns_conj", vec![-0.1, -0.4, 1.1, 0.9]),
        ] {
            let setup = EpSetup::new(0, params, vec![0.0, 0.0]);
            let opts = Options {
                bounds: (f64::NEG_INFINITY, 0.1),
                ..Options::default()
            };
            let br = map_branch(name, ev, &setup, &opts);
            for p in &br.points {
                has_multipliers(&p.eigenvalues, &rotation(p.alpha, 1.1), 1e-10, name);
            }
            let ns = only(&br.specials, "NS");
            at(ns, 0.0, &[0.0, 0.0]);
            has_multipliers(&ns.point.eigenvalues, &rotation(0.0, 1.1), 1e-8, name);
            close(theta(ns), 1.1, 1e-8, "theta");
            rel(ns.extras.coefficient.unwrap(), -0.8, 1e-7, name);
            close(ns.extras.coefficient2.unwrap(), 0.0, 1e-7, name);
            assert_eq!(ns.extras.normalization.as_deref(), Some("<q,q>=1, <p,q>=1"));
            assert_eq!(br.specials.len(), 1, "{name}: {:?}", labels(&br.specials));
        }
    }
}

#[test]
fn test8_ns_at_the_1_3_resonance_keeps_its_label_and_has_no_coefficient() {
    // map_ns with th = 2 pi / 3: the NS point at a = 0 has the multipliers e^{+-2 pi i / 3},
    // so the h2000 solve is singular and the point gets a warning instead of coefficients.
    // With th 1e-3 away the guard does not apply, and d = 2s as in test 5.
    let th3 = 2.0 * PI / 3.0;
    for ev in &BOTH {
        for (th, resonant) in [(th3, true), (th3 + 1e-3, false)] {
            let setup = EpSetup::new(0, vec![-0.1, -0.4, th], vec![0.0, 0.0]);
            let opts = Options {
                bounds: (f64::NEG_INFINITY, 0.1),
                ..Options::default()
            };
            let br = map_branch("map_ns", ev, &setup, &opts);
            let ns = only(&br.specials, "NS");
            at(ns, 0.0, &[0.0, 0.0]);
            assert_eq!(br.specials.len(), 1, "{:?}", labels(&br.specials));
            if resonant {
                assert!(ns.extras.coefficient.is_none());
                assert!(ns.extras.coefficient2.is_none());
                assert!(ns.extras.values.is_empty(), "{:?}", ns.extras.values);
                assert_eq!(br.warnings.len(), 1, "{:?}", br.warnings);
                let w = &br.warnings[0];
                assert!(w.starts_with("NS point at alpha = "), "{w}");
                assert!(w.contains("no normal form coefficient"), "{w}");
                assert!(
                    w.contains("the multipliers are at a strong resonance (1:1, 1:2 or 1:3)"),
                    "{w}"
                );
            } else {
                assert!(br.warnings.is_empty(), "{:?}", br.warnings);
                close(theta(ns), th, 1e-8, "theta");
                rel(ns.extras.coefficient.unwrap(), -0.8, 1e-7, "d");
            }
        }
    }
}

#[test]
fn test6_ns4_takes_theta_from_the_pair_nearest_the_unit_circle() {
    for ev in &BOTH {
        let params = vec![-0.1, -0.4, 1.1, 0.95, 1.6, 0.9, 0.3];
        let setup = EpSetup::new(0, params, vec![0.0; 4]);
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.1),
            ..Options::default()
        };
        let br = map_branch("map_ns4", ev, &setup, &opts);
        let ns = only(&br.specials, "NS");
        at(ns, 0.0, &[0.0; 4]);
        let [s1, s2] = rotation(0.0, 1.1);
        let [s3, s4] = rotation(0.95 - 1.0, 1.6);
        has_multipliers(&ns.point.eigenvalues, &[s1, s2, s3, s4], 1e-8, "ns4");
        close(theta(ns), 1.1, 1e-8, "theta");
        let d = ns.extras.coefficient.unwrap();
        let d2 = ns.extras.coefficient2.unwrap();
        rel(d, -0.8, 1e-7, "ns4 d");
        assert!(d2.abs() < 1e-7, "ns4 coefficient2 {d2:e}");
        // c1 = e^{i theta} (d + i coefficient2).
        let (c, s) = (1.1f64.cos(), 1.1f64.sin());
        let c1 = (c * d - s * d2, s * d + c * d2);
        rel(c1.0, -0.36287689714046184, 1e-7, "Re c1");
        rel(c1.1, -0.7129658880491483, 1e-7, "Im c1");
        assert_eq!(br.specials.len(), 1, "{:?}", labels(&br.specials));
    }
}

/// Multipliers of the delayed logistic map at its fixed point 1 - 1/r: the roots of
/// mu^2 - mu + (r - 1) = 0.
fn delayed_multipliers(r: f64) -> [(f64, f64); 2] {
    eig2(1.0, -(r - 1.0), 1.0, 0.0)
}

#[test]
fn test7_delayed_logistic_ns_and_orbit_half_width() {
    for ev in &BOTH {
        let x0 = 1.0 - 1.0 / 1.8;
        let setup = EpSetup::new(0, vec![1.8], vec![x0, x0]);
        let opts = Options {
            bounds: (0.0, 2.3),
            ..Options::default()
        };
        let br = map_branch("map_delayed_logistic", ev, &setup, &opts);
        for p in &br.points {
            has_multipliers(
                &p.eigenvalues,
                &delayed_multipliers(p.alpha),
                1e-8,
                "branch",
            );
        }
        let ns = only(&br.specials, "NS");
        at(ns, 2.0, &[0.5, 0.5]);
        has_multipliers(&ns.point.eigenvalues, &delayed_multipliers(2.0), 1e-8, "NS");
        close(theta(ns), PI / 3.0, 1e-8, "theta");
        let d = ns.extras.coefficient.unwrap();
        rel(d, -1.0, 1e-7, "d");
        rel(
            ns.extras.coefficient2.unwrap(),
            -(3f64.sqrt()),
            1e-7,
            "coefficient2",
        );
        assert_eq!(br.specials.len(), 1, "{:?}", labels(&br.specials));

        // Past NS the orbit settles on a closed curve of radius about
        // 2 |q_1| sqrt(-beta / d), with |mu| = sqrt(r - 1) = 1 + beta and q = (mu, 1) / |(mu, 1)|.
        let r = 2.02;
        let beta = (r - 1.0f64).sqrt() - 1.0;
        let mu = delayed_multipliers(2.0)[0];
        let mod_mu = (mu.0 * mu.0 + mu.1 * mu.1).sqrt();
        let q1 = mod_mu / (mod_mu * mod_mu + 1.0).sqrt();
        let predicted = 2.0 * q1 * (-beta / d).sqrt();
        close(predicted, 0.14107, 1e-5, "predicted half-width");
        let (mut x, mut y) = (0.51, 0.5);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..200_000 {
            (x, y) = (r * x * (1.0 - y), x);
            if k >= 190_000 {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        let half = (hi - lo) / 2.0;
        assert!(
            (half - predicted).abs() <= 0.02 * predicted,
            "half-width {half} against {predicted}"
        );
    }
}

#[test]
fn test8_neutral_saddle_gives_no_ns_and_no_warning() {
    for ev in &BOTH {
        // (x, y) -> ((2 + al) x + x^2, 0.5 y + x y): the multipliers 2 + al and 0.5 at the
        // origin are real and their product 1 + al/2 passes 1 at al = 0.
        let setup = EpSetup::new(0, vec![-0.2], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.2),
            ..Options::default()
        };
        let br = map_branch("map_neutral_saddle", ev, &setup, &opts);
        assert!(br.specials.is_empty(), "{:?}", labels(&br.specials));
        assert!(br.warnings.is_empty(), "{:?}", br.warnings);
        for p in &br.points {
            has_multipliers(
                &p.eigenvalues,
                &[(2.0 + p.alpha, 0.0), (0.5, 0.0)],
                1e-10,
                "",
            );
            close(p.codim2[1], p.alpha / 2.0, 1e-10, "psi_NS");
        }
        let (first, last) = (&br.points[0], br.points.last().unwrap());
        assert!(first.alpha < -0.1 && last.alpha > 0.1);
    }
}

#[test]
fn test8_one_variable_map_has_no_ns_test() {
    for ev in &BOTH {
        let (run, state) = run_to_end(logistic_spec(ev));
        assert_eq!(state, RunState::Done);
        for b in 0..2 {
            for p in run.points(b, 0, usize::MAX) {
                assert_eq!(p.codim2.len(), 2);
                assert!(p.codim2[1].is_nan(), "psi_NS {}", p.codim2[1]);
                assert!(p.tests[2].is_nan(), "psi_H {}", p.tests[2]);
            }
        }
    }
}

#[test]
fn test8_detect_off_gives_nan_and_no_special() {
    for ev in &BOTH {
        // Logistic on x = 1 - 1/r from r = 2.5 over the PD at r = 3.
        let setup = EpSetup::new(0, vec![2.5], vec![0.6]);
        let on = Options {
            bounds: (0.0, 3.5),
            ..Options::default()
        };
        let br = map_branch("map_logistic", ev, &setup, &on);
        only(&br.specials, "PD");
        assert!(br.points.iter().all(|p| p.codim2[0].is_finite()));
        let off = Options {
            detect_pd: false,
            ..on.clone()
        };
        let br = map_branch("map_logistic", ev, &setup, &off);
        assert!(br.specials.is_empty(), "{:?}", labels(&br.specials));
        assert!(br.points.iter().all(|p| p.codim2[0].is_nan()));

        // Delayed logistic over its NS at r = 2.
        let x0 = 1.0 - 1.0 / 1.8;
        let setup = EpSetup::new(0, vec![1.8], vec![x0, x0]);
        let on = Options {
            bounds: (0.0, 2.3),
            ..Options::default()
        };
        let br = map_branch("map_delayed_logistic", ev, &setup, &on);
        only(&br.specials, "NS");
        let off = Options {
            detect_ns: false,
            ..on.clone()
        };
        let br = map_branch("map_delayed_logistic", ev, &setup, &off);
        assert!(br.specials.is_empty(), "{:?}", labels(&br.specials));
        assert!(br.points.iter().all(|p| p.codim2[1].is_nan()));
        assert!(br.points.iter().all(|p| p.codim2[0].is_finite()));
    }
}

#[test]
fn test8_refine_special_leaves_pd_and_ns_alone() {
    let cases = [
        ("map_logistic", vec![2.5], vec![0.6], 3.5, "PD"),
        (
            "map_delayed_logistic",
            vec![1.8],
            vec![1.0 - 1.0 / 1.8; 2],
            2.3,
            "NS",
        ),
    ];
    for ev in &BOTH {
        for (name, params, u0, hi, label) in &cases {
            let setup = EpSetup::new(0, params.clone(), u0.clone());
            let opts = Options {
                bounds: (0.0, *hi),
                ..Options::default()
            };
            let br = map_branch(name, ev, &setup, &opts);
            let sp = only(&br.specials, label);
            let mut inner = model(name, ev);
            let mut g = MapAsRoot::new(inner.as_mut());
            let r = refine_special(&mut g, params, 0, sp, &opts, &AtomicBool::new(false));
            assert!(matches!(r, Ok(None)), "{name} {label}: {r:?}");
        }
    }
}

#[test]
fn run_refuses_other_kinds_on_a_map_and_a_map_that_reads_t() {
    let (m, info) = load("map_logistic");
    let setup = EpSetup::new(0, vec![2.5], vec![0.1]);
    let mut spec = RunSpec::new(m, info, Evaluator::Jit, setup, Options::default());
    spec.kind = RunKind::Ivp {
        tspan: [0.0, 1.0],
        opts: IvpOptions::default(),
    };
    let (_, state) = run_to_end(spec);
    assert!(
        matches!(state, RunState::Failed(EngineError::BadProblem(_))),
        "{state:?}"
    );

    // x(t+1) = x + t: a fixed point that changes with the iterate number is not defined.
    let (m, info) = load("map_shift");
    let setup = EpSetup::new(0, vec![], vec![0.0]);
    let spec = RunSpec::new(m, info, Evaluator::Jit, setup, Options::default());
    let (_, state) = run_to_end(spec);
    assert_eq!(state, RunState::Failed(EngineError::NonAutonomous));
}
