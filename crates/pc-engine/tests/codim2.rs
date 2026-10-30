//! Codim-2 points on LP and Hopf curves: the cusp, Bogdanov-Takens and Bautin normal forms
//! of SPEC section 9 (Tests 7, 8 and 9).

mod common;

use common::load;
use pc_engine::{
    CurveSetup, EpSetup, Evaluator, ModelEval, Options, SpecialPoint, build, continue_ep,
    continue_lp_curve,
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
