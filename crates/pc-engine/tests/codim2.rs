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
    assert!(pt.u[0].abs() < 1e-8, "x = {}", pt.u[0]);
    assert!(pt.u[1].abs() < 1e-8, "b1 = {}", pt.u[1]);
    assert!(pt.alpha.abs() < 1e-8, "b2 = {}", pt.alpha);
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
        assert!(v.abs() < 1e-8, "{name} = {v}");
    }
    let (a, b) = (
        bt[0].extras.coefficient.expect("a"),
        bt[0].extras.coefficient2.expect("b"),
    );
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
        assert!(v.abs() < 1e-8, "{name} = {v}");
    }
    let (a, b) = (
        bt[0].extras.coefficient.expect("a"),
        bt[0].extras.coefficient2.expect("b"),
    );
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
    assert!(pt.alpha.abs() < 1e-8, "b2 = {}", pt.alpha);
    assert!(pt.u[2].abs() < 1e-8, "b1 = {}", pt.u[2]);
    assert!((pt.u[3] - 1.0).abs() < 1e-8, "kappa = {}", pt.u[3]);
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
        assert!(gh.point.alpha.abs() < 1e-8, "b2 = {}", gh.point.alpha);
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
