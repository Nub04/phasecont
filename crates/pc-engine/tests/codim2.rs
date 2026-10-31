//! Codim-2 points on LP and Hopf curves: the cusp, Bogdanov-Takens and Bautin normal forms
//! of SPEC section 9 (Tests 7, 8 and 9).

mod common;

use common::load;
use pc_engine::{
    CurveSetup, EpSetup, Evaluator, ModelEval, Options, SpecialPoint, build, continue_ep,
    continue_hopf_curve, continue_lp_curve,
};
use pc_ir::EntryKind;
use std::sync::atomic::AtomicBool;

fn model_of(name: &str) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, Evaluator::Jit).unwrap()
}

/// F and the state Jacobian of `name` at `u` with parameters `p` (t = 0 is appended).
fn f_and_jac(name: &str, u: &[f64], p: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut m = model_of(name);
    let n = m.n();
    let mut p = p.to_vec();
    p.push(0.0);
    let (mut f, mut j) = (vec![0.0; n], vec![0.0; n * n]);
    m.eval(&EntryKind::F, u, &p, &[], &mut f).unwrap();
    m.eval(&EntryKind::JacU, u, &p, &[], &mut j).unwrap();
    (f, j)
}

#[test]
fn normal_form_models_have_the_expected_origin() {
    // Cusp at b1 = b2 = 0: F = 0 and A = 0.
    let (f, a) = f_and_jac("cusp", &[0.0], &[0.0, 0.0]);
    assert_eq!((f[0], a[0]), (0.0, 0.0));
    // BT: A = [[0, 1], [0, 0]] in column-major order.
    let (f, a) = f_and_jac("bt", &[0.0, 0.0], &[0.0, 0.0, 1.0]);
    assert_eq!(f, [0.0, 0.0]);
    assert_eq!(a, [0.0, 0.0, 1.0, 0.0]);
    // Bautin at b1 = 0: A = [[0, -1], [1, 0]] in column-major order.
    let (f, a) = f_and_jac("bautin", &[0.0, 0.0], &[0.0, 0.5, 1.0]);
    assert_eq!(f, [0.0, 0.0]);
    assert_eq!(a, [0.0, 1.0, -1.0, 0.0]);
}

fn specials<'a>(sp: &'a [SpecialPoint], label: &str) -> Vec<&'a SpecialPoint> {
    sp.iter().filter(|s| s.label == label).collect()
}

/// Test 7: the LP curve of x' = b1 + b2 x - x^3 in (b1, b2) has a cusp at the origin.
#[test]
fn test7_cusp_cp() {
    let mut m = model_of("cusp");
    let cancel = AtomicBool::new(false);
    // The equilibrium branch in b1 at b2 = 1 has folds at x = -+1/sqrt(3); take the first.
    let ep_opts = Options {
        bounds: (-1.0, 1.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, vec![0.0, 1.0], vec![-1.0]),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp = specials(&ep.specials, "LP");
    assert!(!lp.is_empty(), "no LP on the equilibrium branch");
    let u0 = lp[0].point.u.clone();
    let setup = CurveSetup::new([0, 1], vec![lp[0].point.alpha, 1.0], u0);
    // The first tangent has u increasing, which is toward the cusp from x = -1/sqrt(3).
    let opts = Options {
        bounds: (-0.5, 1.5),
        ..Options::default()
    };
    let br = continue_lp_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let cp = specials(&br.specials, "CP");
    assert_eq!(
        cp.len(),
        1,
        "stop {:?}, specials {:?}",
        br.stop,
        br.specials
    );
    // Point.u is (x, b1) on a curve and Point.alpha is b2.
    let pt = &cp[0].point;
    assert!(pt.u[0].abs() < 1e-10, "x = {}", pt.u[0]);
    assert!(pt.u[1].abs() < 1e-10, "b1 = {}", pt.u[1]);
    assert!(pt.alpha.abs() < 1e-10, "b2 = {}", pt.alpha);
    assert!(cp[0].extras.refined, "CP not refined");
    // x' = b1 + b2 x - x^3 has C = -6 and B = 0 at the origin, and p q = 1 with |q| = 1 gives
    // c = -1.
    let c = cp[0].extras.coefficient.expect("cusp coefficient");
    assert!((c + 1.0).abs() < 1e-8, "c = {c}");
    assert!(cp[0].extras.normalization.is_some());
}

/// The LP curve of the Bogdanov-Takens model with x*y coefficient `s`, from the fold of the
/// equilibrium branch in b1 at b2 = 0.5, through b2 = 0.
fn bt_lp_branch(s: f64) -> pc_engine::Branch {
    let mut m = model_of("bt");
    let cancel = AtomicBool::new(false);
    // x' = y, y' = b1 + x^2 on y = 0 has the fold x = 0 at b1 = 0; start on the branch x < 0.
    let ep_opts = Options {
        bounds: (-1.0, 1.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, vec![-1.0, 0.5, s], vec![-1.0, 0.0]),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp = specials(&ep.specials, "LP");
    assert_eq!(lp.len(), 1, "specials {:?}", ep.specials);
    let setup = CurveSetup::new(
        [0, 1],
        vec![lp[0].point.alpha, 0.5, s],
        lp[0].point.u.clone(),
    );
    // The first tangent points to increasing b2; run toward decreasing b2 through the BT.
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    continue_lp_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap()
}

/// Test 8, LP half: the LP curve has a BT at the origin with a = 1 and b = s. The LP curve is
/// x = y = b1 = 0, so it has no CP and no ZH.
fn test8_bt_lp(s: f64) {
    let br = bt_lp_branch(s);
    let bt = specials(&br.specials, "BT");
    assert_eq!(
        bt.len(),
        1,
        "stop {:?}, specials {:?}",
        br.stop,
        br.specials
    );
    let pt = &bt[0].point;
    for (name, v) in [
        ("x", pt.u[0]),
        ("y", pt.u[1]),
        ("b1", pt.u[2]),
        ("b2", pt.alpha),
    ] {
        assert!(v.abs() < 1e-10, "{name} = {v}");
    }
    let (a, b) = (
        bt[0].extras.coefficient.expect("a"),
        bt[0].extras.coefficient2.expect("b"),
    );
    assert!(bt[0].extras.refined, "BT not refined");
    assert!((a - 1.0).abs() < 1e-8, "a = {a}");
    assert!((b - s).abs() < 1e-8, "b = {b}");
    assert!(bt[0].extras.normalization.is_some());
    assert!(specials(&br.specials, "CP").is_empty());
    assert!(specials(&br.specials, "ZH").is_empty());
}

#[test]
fn test8_bt_s_plus() {
    test8_bt_lp(1.0);
}

#[test]
fn test8_bt_s_minus() {
    test8_bt_lp(-1.0);
}

/// The Hopf curve of `name` through the Hopf point of the equilibrium z = 0 or `u0`: the
/// equilibrium branch in b1 with b2 and `s` fixed gives the Hopf start.
fn hopf_start(m: &mut dyn ModelEval, p: Vec<f64>, u0: Vec<f64>, bounds: (f64, f64)) -> CurveSetup {
    let cancel = AtomicBool::new(false);
    let opts = Options {
        bounds,
        ..Options::default()
    };
    let ep = continue_ep(
        m,
        &EpSetup::new(0, p.clone(), u0),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h = specials(&ep.specials, "H");
    assert_eq!(h.len(), 1, "specials {:?}", ep.specials);
    let mut q = p;
    q[0] = h[0].point.alpha;
    CurveSetup::new([0, 1], q, h[0].point.u.clone())
}

/// Test 8, H half: the Hopf curve of the BT model passes through the BT at the origin with
/// kappa = 0, a = 1 and b = s.
fn test8_bt_h(s: f64) {
    let mut m = model_of("bt");
    let cancel = AtomicBool::new(false);
    // Hopf at x = -0.5, b1 = -0.25 and b2 = 0.5 s (trace b2 + s x vanishes, det = -2x > 0).
    let setup = hopf_start(
        &mut *m,
        vec![-1.0, 0.5 * s, s],
        vec![-1.0, 0.0],
        (-1.0, 1.0),
    );
    // The first tangent points to increasing x, which is toward the BT.
    let bounds = if s > 0.0 { (-0.3, 1.0) } else { (-1.0, 0.3) };
    let opts = Options {
        bounds,
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let bt = specials(&br.specials, "BT");
    assert_eq!(
        bt.len(),
        1,
        "stop {:?}, specials {:?}",
        br.stop,
        br.specials
    );
    let pt = &bt[0].point;
    // Point.u is (x, y, b1, kappa) on a Hopf curve and Point.alpha is b2.
    for (name, v) in [
        ("x", pt.u[0]),
        ("y", pt.u[1]),
        ("b1", pt.u[2]),
        ("kappa", pt.u[3]),
        ("b2", pt.alpha),
    ] {
        assert!(v.abs() < 1e-10, "{name} = {v}");
    }
    let (a, b) = (
        bt[0].extras.coefficient.expect("a"),
        bt[0].extras.coefficient2.expect("b"),
    );
    assert!(bt[0].extras.refined, "BT not refined");
    assert!((a - 1.0).abs() < 1e-8, "a = {a}");
    assert!((b - s).abs() < 1e-8, "b = {b}");
    assert!(specials(&br.specials, "GH").is_empty(), "{:?}", br.specials);
    // The curve is x = -b2 s, b1 = -x^2 and kappa = -2 x on both sides of the BT.
    for p in &br.points {
        let (x, kappa, b1, b2) = (p.u[0], p.u[3], p.u[2], p.alpha);
        assert!((b1 + x * x).abs() < 1e-7 && (b2 + s * x).abs() < 1e-7);
        assert!((kappa + 2.0 * x).abs() < 1e-7, "kappa {kappa} at x {x}");
    }
}

#[test]
fn test8_bt_h_s_plus() {
    test8_bt_h(1.0);
}

#[test]
fn test8_bt_h_s_minus() {
    test8_bt_h(-1.0);
}

/// The Hopf curve of a Bautin model (b1 = 0, kappa = 1) from b2 = 0.5 down through the GH at
/// b2 = 0; returns the GH point and the branch.
fn bautin_gh(name: &str, s: f64, extra: &[f64]) -> (pc_engine::Branch, SpecialPoint) {
    let mut m = model_of(name);
    let cancel = AtomicBool::new(false);
    let mut p = vec![-0.5, 0.5, s];
    p.extend_from_slice(extra);
    let setup = hopf_start(&mut *m, p, vec![0.0, 0.0], (-1.0, 1.0));
    // The first tangent has b2 increasing; run toward decreasing b2.
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 0.8),
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let gh = specials(&br.specials, "GH");
    assert_eq!(
        gh.len(),
        1,
        "stop {:?}, specials {:?}",
        br.stop,
        br.specials
    );
    let gh = gh[0].clone();
    (br, gh)
}

/// Test 9: the Hopf curve of the Bautin model has a GH at b2 = 0 with omega = 1 and l2 = 4 s
/// (z = sqrt(2) w for |q| = 1, so z|z|^4 becomes 4 w|w|^4).
fn test9_bautin(s: f64) {
    let (_, gh) = bautin_gh("bautin", s, &[]);
    let pt = &gh.point;
    assert!(pt.alpha.abs() < 1e-10, "b2 = {}", pt.alpha);
    assert!(pt.u[2].abs() < 1e-10, "b1 = {}", pt.u[2]);
    assert!((pt.u[3] - 1.0).abs() < 1e-10, "kappa = {}", pt.u[3]);
    assert!(gh.extras.refined, "GH not refined");
    let l2 = gh.extras.coefficient.expect("l2");
    assert!(l2 * s > 0.0, "l2 = {l2} must have the sign of s");
    assert!((l2 - 4.0 * s).abs() < 1e-6, "l2 = {l2}");
    assert!((gh.extras.omega.expect("omega") - 1.0).abs() < 1e-8);
    assert!(gh.extras.normalization.is_some());
}

#[test]
fn test9_bautin_s_plus() {
    test9_bautin(1.0);
}

#[test]
fn test9_bautin_s_minus() {
    test9_bautin(-1.0);
}

/// A near-identity change of coordinates leaves the Hopf curve (b1 = 0) and l2 unchanged.
#[test]
fn test9_bautin_warped_l2_is_invariant() {
    for s in [1.0, -1.0] {
        let (_, gh) = bautin_gh("bautin_warped", s, &[0.4]);
        assert!(gh.point.alpha.abs() < 1e-10, "b2 = {}", gh.point.alpha);
        assert!(gh.extras.refined, "GH not refined");
        let l2 = gh.extras.coefficient.expect("l2");
        assert!((l2 - 4.0 * s).abs() < 1e-6, "s = {s}, l2 = {l2}");
    }
}

/// The Hopf curve of the Brusselator is b = 1 + a^2 with kappa = a^2 and the equilibrium
/// (a, b/a).
#[test]
fn brusselator_hopf_curve_is_analytic() {
    let mut m = model_of("brusselator");
    let cancel = AtomicBool::new(false);
    let a0 = 1.0;
    let ep_opts = Options {
        bounds: (0.5, 4.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(1, vec![a0, 1.2], vec![a0, 1.2 / a0]),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h = specials(&ep.specials, "H");
    assert_eq!(h.len(), 1, "specials {:?}", ep.specials);
    let setup = CurveSetup::new([0, 1], vec![a0, h[0].point.alpha], h[0].point.u.clone());
    let opts = Options {
        bounds: (1.5, 5.0),
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    assert!(br.points.len() > 5, "stop {:?}", br.stop);
    for p in &br.points {
        let (x, y, a, kappa, b) = (p.u[0], p.u[1], p.u[2], p.u[3], p.alpha);
        assert!((b - (1.0 + a * a)).abs() < 1e-9, "b = {b} at a = {a}");
        assert!((kappa - a * a).abs() < 1e-9, "kappa = {kappa} at a = {a}");
        assert!((x - a).abs() < 1e-9 && (y - b / a).abs() < 1e-9);
    }
}

// ---- Zero-Hopf points (Kuznetsov, SIAM J. Numer. Anal. 36 (1999), section 8) ----
//
// zh.ode is x' = b1 + B x^2 + C rr + K x^3 + L x rr, w' = (b2 + i om + d0 x + e0 x^2 + m rr) w
// with w = y + i z, rr = |w|^2, d0 = dr + i di, e0 = er + i ei and m = mr + i mi. In the
// basis q0 = sigma e_x, q1 = (0, 1, -i)/sqrt(2) with sigma = sign(B), which makes b > 0, the
// bilinear terms are already in normal form, so every h vector is zero, and the G values
// are read off the Taylor coefficients: G200 = 2 sigma B, G011 = 2 sigma C, G110 = sigma d0,
// G300 = 6 K, G111 = 2 L, G210 = 2 e0, G021 = 4 m. Kuznetsov's formulas then give
//
//   b = sigma B,  c = 2 sigma C,  d = sigma (d0 - i om K / B),
//   e = er + (dr / 2) (2 mr / C - 3 K / B + L / C) - mr B / C.

/// B, C, K, L, om, dr, di, er, ei, mr, mi of the zero-Hopf models, after b1 and b2.
const ZH_P: [f64; 11] = [1.5, 0.8, 0.3, 0.6, 2.0, 1.0, 0.4, 0.5, -0.3, -0.4, 0.7];
const ROT: [f64; 3] = [0.7, -0.5, 1.1];
const WARP: [f64; 3] = [0.4, -0.3, 0.2];

/// All parameters of zero-Hopf model `name` in file order.
fn zh_params(name: &str, b1: f64, b2: f64) -> Vec<f64> {
    let mut p = vec![b1, b2];
    p.extend_from_slice(&ZH_P);
    match name {
        "zh_rot" => p.extend_from_slice(&ROT),
        "zh_warped" => p.extend_from_slice(&WARP),
        _ => {}
    }
    p
}

/// The equilibrium with x = `x` on y = z = 0 of zh.ode, in the coordinates of model `name`.
fn zh_state(name: &str, x: f64) -> Vec<f64> {
    match name {
        "zh_rot" => {
            let [_, tb, tc] = ROT;
            vec![
                tc.cos() * tb.cos() * x,
                tc.sin() * tb.cos() * x,
                -tb.sin() * x,
            ]
        }
        "zh_warped" => vec![x, WARP[0] * x * x, WARP[2] * x * x * x],
        _ => vec![x, 0.0, 0.0],
    }
}

/// The b1 of the equilibrium with state `x` on y = z = 0.
fn zh_b1(x: f64) -> f64 {
    -(ZH_P[0] * x * x + ZH_P[2] * x * x * x)
}

/// The hand-derived b, c, d (re, im) and e.
fn zh_expected() -> (f64, f64, (f64, f64), f64) {
    let [bb, cc, k, l, om, dr, di, er, _, mr, _] = ZH_P;
    let sigma = bb.signum();
    let s = 2.0 * mr / cc - 3.0 * k / bb + l / cc;
    (
        sigma * bb,
        2.0 * sigma * cc,
        (sigma * dr, sigma * (di - om * k / bb)),
        er + 0.5 * dr * s - mr * bb / cc,
    )
}

fn value(sp: &SpecialPoint, name: &str) -> f64 {
    sp.extras
        .values
        .iter()
        .find(|(k, _)| k == name)
        .unwrap_or_else(|| panic!("no value {name} in {:?}", sp.extras))
        .1
}

/// The ZH point of the LP curve of model `name`: the equilibrium branch in b1 at b2 = 0.5
/// has the fold x = 0, and the LP curve runs from there to decreasing b2 through b2 = 0.
fn zh_on_lp_curve(name: &str) -> SpecialPoint {
    let mut m = model_of(name);
    let cancel = AtomicBool::new(false);
    let x0 = -1.0;
    let ep_opts = Options {
        bounds: (-2.0, 1.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, zh_params(name, zh_b1(x0), 0.5), zh_state(name, x0)),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp = specials(&ep.specials, "LP");
    assert_eq!(lp.len(), 1, "{name}: specials {:?}", ep.specials);
    let setup = CurveSetup::new(
        [0, 1],
        zh_params(name, lp[0].point.alpha, 0.5),
        lp[0].point.u.clone(),
    );
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    let br = continue_lp_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let zh = specials(&br.specials, "ZH");
    assert_eq!(zh.len(), 1, "{name}: stop {:?}, {:?}", br.stop, br.specials);
    let sp = zh[0].clone();
    // Point.u is (x, y, z, b1) on an LP curve and Point.alpha is b2.
    for (k, v) in sp.point.u.iter().enumerate() {
        assert!(v.abs() < 1e-10, "{name}: u[{k}] = {v}");
    }
    assert!(
        sp.point.alpha.abs() < 1e-10,
        "{name}: b2 = {}",
        sp.point.alpha
    );
    sp
}

/// The ZH point of the Hopf curve of model `name`: the equilibrium branch in b1 at b2 = 0.3
/// has a Hopf point at x = -0.37, from which the Hopf curve runs through x = 0, b2 = 0.
fn zh_on_hopf_curve(name: &str) -> SpecialPoint {
    let mut m = model_of(name);
    let cancel = AtomicBool::new(false);
    let x0 = -1.0;
    let setup = hopf_start(
        &mut *m,
        zh_params(name, zh_b1(x0), 0.3),
        zh_state(name, x0),
        (-2.0, 1.0),
    );
    let opts = Options {
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let zh = specials(&br.specials, "ZH");
    assert_eq!(zh.len(), 1, "{name}: stop {:?}, {:?}", br.stop, br.specials);
    let sp = zh[0].clone();
    // Point.u is (x, y, z, b1, kappa) on a Hopf curve and Point.alpha is b2.
    let n = 3;
    for (k, v) in sp.point.u[..n + 1].iter().enumerate() {
        assert!(v.abs() < 1e-10, "{name}: u[{k}] = {v}");
    }
    let om = ZH_P[4];
    assert!((sp.point.u[n + 1] - om * om).abs() < 1e-9, "kappa");
    assert!(
        sp.point.alpha.abs() < 1e-10,
        "{name}: b2 = {}",
        sp.point.alpha
    );
    sp
}

/// `full` also checks Im d, which a nonlinear change of coordinates may move.
fn check_zh(name: &str, sp: &SpecialPoint, tol: f64, full: bool) {
    let (b, c, d, e) = zh_expected();
    assert!(sp.extras.refined, "{name}: ZH not refined");
    let got = (
        sp.extras.coefficient.expect("b"),
        sp.extras.coefficient2.expect("c"),
        value(sp, "re_d"),
        value(sp, "im_d"),
        value(sp, "e"),
    );
    assert!((got.0 - b).abs() < tol, "{name}: b = {}, want {b}", got.0);
    assert!((got.1 - c).abs() < tol, "{name}: c = {}, want {c}", got.1);
    assert!(
        (got.2 - d.0).abs() < tol,
        "{name}: Re d = {}, want {}",
        got.2,
        d.0
    );
    if full {
        assert!(
            (got.3 - d.1).abs() < tol,
            "{name}: Im d = {}, want {}",
            got.3,
            d.1
        );
    }
    assert!((got.4 - e).abs() < tol, "{name}: e = {}, want {e}", got.4);
    assert!((sp.extras.omega.expect("omega") - ZH_P[4]).abs() < 1e-8);
    let norm = sp.extras.normalization.as_deref().expect("normalization");
    assert!(norm.contains("b>0"), "{norm}");
}

#[test]
fn zh_coefficients_on_the_lp_curve() {
    check_zh("zh", &zh_on_lp_curve("zh"), 1e-8, true);
}

#[test]
fn zh_coefficients_on_the_hopf_curve() {
    check_zh("zh", &zh_on_hopf_curve("zh"), 1e-8, true);
}

/// An orthogonal change of coordinates keeps |q0| = 1, so all four coefficients stay.
#[test]
fn zh_coefficients_are_invariant_under_a_rotation() {
    check_zh("zh_rot", &zh_on_lp_curve("zh_rot"), 1e-8, true);
    check_zh("zh_rot", &zh_on_hopf_curve("zh_rot"), 1e-8, true);
}

/// A triangular near-identity change of coordinates makes B(q0, q1) and h nonzero. b, c,
/// Re d and e are properties of the singularity and do not change.
#[test]
fn zh_coefficients_are_invariant_under_a_nonlinear_warp() {
    check_zh("zh_warped", &zh_on_lp_curve("zh_warped"), 1e-6, false);
    check_zh("zh_warped", &zh_on_hopf_curve("zh_warped"), 1e-6, false);
}

// Test 11: double-Hopf coefficients. hh.ode is two Hopf normal forms with cubic coupling,
// w1 = x1 + i y1 and w2 = x2 + i y2, with
//
//   w1' = (b1 + i om1 + a11 |w1|^2 + a12 |w2|^2) w1,
//   w2' = (b2 + i om2 + a21 |w1|^2 + a22 |w2|^2) w2.
//
// The eigenvector of the pair k is q_k = (1, -i) / sqrt(2) in the plane (x_k, y_k), which
// gives w_k = sqrt(2) z_k and, with Kuznetsov's z-expansion, G2100 = 4 a11, G1011 = 2 a12,
// G1110 = 2 a21 and G0021 = 4 a22. The pair with the larger frequency is pair 1 in the
// normal form, so Re P11, Re P12, Re P21 and Re P22 are 4 a11r, 2 a12r, 2 a21r and 4 a22r
// when om1 > om2, and 4 a22r, 2 a21r, 2 a12r and 4 a11r when om1 < om2.

/// a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i of the double-Hopf models.
const HH_A: [f64; 8] = [-0.8, 0.3, 1.7, -0.4, 0.6, 0.5, -1.1, 0.2];
const HH_ROT: [f64; 4] = [0.7, -0.5, 1.1, 0.4];
const HH_WARP: [f64; 4] = [0.4, -0.3, 0.2, 0.5];

/// All parameters of double-Hopf model `name` in file order.
fn hh_params(name: &str, b: (f64, f64), om: (f64, f64)) -> Vec<f64> {
    let mut p = vec![b.0, b.1, om.0, om.1];
    p.extend_from_slice(&HH_A);
    match name {
        "hh_rot" => p.extend_from_slice(&HH_ROT),
        "hh_warped" => p.extend_from_slice(&HH_WARP),
        _ => {}
    }
    p
}

/// The HH point of the Hopf curve of model `name` through the Hopf point of pair 1 at
/// b2 = 0.4. The origin is an equilibrium for all b, so the Hopf curve is b1 = 0.
fn hh_on_hopf_curve(name: &str, om: (f64, f64)) -> SpecialPoint {
    let mut m = model_of(name);
    let cancel = AtomicBool::new(false);
    let setup = hopf_start(
        &mut *m,
        hh_params(name, (-0.3, 0.4), om),
        vec![0.0; 4],
        (-0.5, 0.5),
    );
    let opts = Options {
        bounds: (-0.5, 1.0),
        direction: -1.0,
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    let hh = specials(&br.specials, "HH");
    assert_eq!(hh.len(), 1, "{name}: stop {:?}, {:?}", br.stop, br.specials);
    let sp = hh[0].clone();
    // Point.u is (x1, y1, x2, y2, b1, kappa) on a Hopf curve and Point.alpha is b2.
    let n = 4;
    for (k, v) in sp.point.u[..n + 1].iter().enumerate() {
        assert!(v.abs() < 1e-10, "{name}: u[{k}] = {v}");
    }
    assert!((sp.point.u[n + 1] - om.0 * om.0).abs() < 1e-9, "kappa");
    assert!(
        sp.point.alpha.abs() < 1e-10,
        "{name}: b2 = {}",
        sp.point.alpha
    );
    sp
}

/// Checks Re P11, Re P12, Re P21 and Re P22 against the hand-derived values.
fn check_hh(name: &str, sp: &SpecialPoint, om: (f64, f64), tol: f64) {
    let [a11, _, a12, _, a21, _, a22, _] = HH_A;
    let (p11, p12, p21, p22) = if om.0 > om.1 {
        (4.0 * a11, 2.0 * a12, 2.0 * a21, 4.0 * a22)
    } else {
        (4.0 * a22, 2.0 * a21, 2.0 * a12, 4.0 * a11)
    };
    assert!(sp.extras.refined, "{name}: HH not refined");
    let got = (
        sp.extras.coefficient.expect("re_p11"),
        value(sp, "re_p12"),
        value(sp, "re_p21"),
        value(sp, "re_p22"),
    );
    for (what, g, w) in [
        ("Re P11", got.0, p11),
        ("Re P12", got.1, p12),
        ("Re P21", got.2, p21),
        ("Re P22", got.3, p22),
    ] {
        assert!((g - w).abs() < tol, "{name}: {what} = {g}, want {w}");
    }
    let (w1, w2) = (om.0.max(om.1), om.0.min(om.1));
    assert!(
        (sp.extras.omega.expect("omega") - w1).abs() < 1e-8,
        "omega1"
    );
    assert!((value(sp, "omega2") - w2).abs() < 1e-8, "omega2");
    let norm = sp.extras.normalization.as_deref().expect("normalization");
    assert!(norm.contains("w1>w2"), "{norm}");
}

#[test]
fn hh_coefficients_on_the_hopf_curve() {
    let om = (2.0, 0.9);
    check_hh("hh", &hh_on_hopf_curve("hh", om), om, 1e-8);
}

/// With om1 < om2 the curve pair is the slow one, so P11 belongs to the other pair.
#[test]
fn hh_pairs_are_ordered_by_frequency() {
    let om = (0.9, 2.0);
    check_hh("hh", &hh_on_hopf_curve("hh", om), om, 1e-8);
}

/// An orthogonal change of coordinates keeps |q| = 1, so the coefficients stay.
#[test]
fn hh_coefficients_are_invariant_under_a_rotation() {
    let om = (2.0, 0.9);
    check_hh("hh_rot", &hh_on_hopf_curve("hh_rot", om), om, 1e-8);
}

/// A triangular near-identity change of coordinates makes the quadratic terms and the h
/// vectors nonzero. The real parts of the cubic coefficients do not change.
#[test]
fn hh_coefficients_are_invariant_under_a_nonlinear_warp() {
    let om = (2.0, 0.9);
    check_hh("hh_warped", &hh_on_hopf_curve("hh_warped", om), om, 1e-6);
}
