//! Codim-2 points on LP and Hopf curves: the cusp, Bogdanov-Takens and Bautin normal forms
//! of SPEC section 9 (Tests 7, 8 and 9).

mod common;

use common::load;
use pc_engine::{Evaluator, ModelEval, build};
use pc_ir::EntryKind;

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
