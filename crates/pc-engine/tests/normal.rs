//! Fold coefficient and first Lyapunov coefficient on both evaluators, against hand
//! values: fold a = 1, hopf_c l1 = 2c, hopf_scaled (X = 2x, Y = 2y) l1 = c/2.

mod common;

use common::load;
use pc_engine::{
    Branch, EpSetup, Evaluator, ModelEval, Options, build, continue_ep, first_lyapunov,
    fold_coefficient, hopf_eigenvectors,
};
use pc_ir::EntryKind;
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which).unwrap()
}

fn run(name: &str, which: Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let mut m = model_of(name, which);
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

/// The single refined special point with `label`; returns (coefficient, normalization).
fn coefficient(b: &Branch, label: &str) -> (f64, String) {
    let hits: Vec<_> = b.specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(hits.len(), 1, "{label}: {:?}", b.specials);
    let s = hits[0];
    assert!(s.extras.refined);
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    (
        s.extras.coefficient.unwrap(),
        s.extras.normalization.clone().unwrap(),
    )
}

#[test]
fn fold_coefficient_is_one() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            max_steps: 150,
            ..Options::default()
        };
        let b = run("fold", which, &setup, &opts);
        let (a, norm) = coefficient(&b, "LP");
        println!("fold {which:?}: a = {a:.16}");
        assert!((a - 1.0).abs() < 1e-12, "{which:?}: a = {a}");
        assert_eq!(norm, "|q|=1, p'q=1, p'F_alpha>0");
        let mut m = model_of("fold", which);
        let a = fold_coefficient(m.as_mut(), &[0.0], 0, &[0.0], 0.0).unwrap();
        assert!((a - 1.0).abs() < 1e-12, "direct {which:?}: a = {a}");
    }
}

fn hopf_run(name: &str, c: f64, which: Evaluator) -> Branch {
    let setup = EpSetup::new(0, vec![-0.5, c], vec![0.0, 0.0]);
    let opts = Options {
        bounds: (-0.5, 0.5),
        ..Options::default()
    };
    run(name, which, &setup, &opts)
}

#[test]
fn hopf_c_l1_is_two_c() {
    for which in BOTH {
        for c in [-1.0, 1.0] {
            let b = hopf_run("hopf_c", c, which);
            let (l1, norm) = coefficient(&b, "H");
            println!("hopf_c c={c} {which:?}: l1 = {l1:.16}");
            assert!((l1 - 2.0 * c).abs() < 1e-10, "{which:?} c={c}: l1 = {l1}");
            assert_eq!(norm, "<q,q>=1, <p,q>=1");
        }
    }
}

#[test]
fn scaled_coordinates_give_l1_c_over_two() {
    for which in BOTH {
        for c in [-1.0, 1.0] {
            let b = hopf_run("hopf_scaled", c, which);
            let (l1, _) = coefficient(&b, "H");
            println!("hopf_scaled c={c} {which:?}: l1 = {l1:.16}");
            assert!((l1 - 0.5 * c).abs() < 1e-10, "{which:?} c={c}: l1 = {l1}");
            assert_eq!(l1.signum(), c.signum());
        }
    }
}

#[test]
fn lorenz_hopf_has_positive_l1() {
    for which in BOTH {
        let (s, b_par) = (10.0, 8.0 / 3.0);
        let r0 = 20.0;
        let c = (b_par * (r0 - 1.0_f64)).sqrt();
        let setup = EpSetup::new(1, vec![s, r0, b_par], vec![c, c, r0 - 1.0]);
        let opts = Options {
            bounds: (r0, 30.0),
            ..Options::default()
        };
        let br = run("lorenz", which, &setup, &opts);
        let (l1, _) = coefficient(&br, "H");
        let h = br.specials.iter().find(|s| s.label == "H").unwrap();
        println!("lorenz {which:?}: r = {:.14}, l1 = {l1:.10}", h.point.alpha);
        assert!((h.point.alpha - 470.0 / 19.0).abs() < 1e-9);
        assert!(l1 > 0.0, "{which:?}: l1 = {l1}");
        let want = 2.6551897258e-4;
        assert!(((l1 - want) / want).abs() < 1e-6, "{which:?}: l1 = {l1:e}");
    }
}

#[test]
fn hopf_eigenvectors_satisfy_their_equations() {
    for which in BOTH {
        let mut cases: Vec<(&str, Vec<f64>, Vec<f64>, f64)> = vec![
            ("hopf_c", vec![0.0, 1.0], vec![0.0, 0.0], 1.0),
            ("hopf_scaled", vec![0.0, 1.0], vec![0.0, 0.0], 1.0),
        ];
        let (b_par, r) = (8.0_f64 / 3.0, 470.0_f64 / 19.0);
        let c = (b_par * (r - 1.0)).sqrt();
        let om = (2.0 * 10.0 * b_par * (10.0 + 1.0) / (10.0 - b_par - 1.0)).sqrt();
        cases.push(("lorenz", vec![10.0, r, b_par], vec![c, c, r - 1.0], om));
        for (name, params, u, omega) in cases {
            let mut m = model_of(name, which);
            let n = m.n();
            let mut p = params.clone();
            p.push(0.0);
            let mut a = vec![0.0; n * n];
            m.eval(&EntryKind::JacU, &u, &p, &[], &mut a).unwrap();
            let (q, pv) = hopf_eigenvectors(&a, n, omega).unwrap();
            let nrm = |x: &(Vec<f64>, Vec<f64>)| {
                x.0.iter().chain(&x.1).map(|v| v * v).sum::<f64>().sqrt()
            };
            // A q - i omega q and A^T p + i omega p, relative to the vector norm.
            let (mut rq, mut rp) = (0.0_f64, 0.0_f64);
            for i in 0..n {
                let (mut aqr, mut aqi, mut atpr, mut atpi) = (0.0, 0.0, 0.0, 0.0);
                for j in 0..n {
                    aqr += a[i + n * j] * q.0[j];
                    aqi += a[i + n * j] * q.1[j];
                    atpr += a[j + n * i] * pv.0[j];
                    atpi += a[j + n * i] * pv.1[j];
                }
                rq = rq.max((aqr + omega * q.1[i]).abs() + (aqi - omega * q.0[i]).abs());
                rp = rp.max((atpr - omega * pv.1[i]).abs() + (atpi + omega * pv.0[i]).abs());
            }
            let (rq, rp) = (rq / nrm(&q), rp / nrm(&pv));
            let qq: f64 = q.0.iter().chain(&q.1).map(|v| v * v).sum();
            // <p, q> = conj(p)^T q.
            let (mut pr, mut pi) = (0.0, 0.0);
            for i in 0..n {
                pr += pv.0[i] * q.0[i] + pv.1[i] * q.1[i];
                pi += pv.0[i] * q.1[i] - pv.1[i] * q.0[i];
            }
            println!(
                "{name} {which:?}: |Aq-iwq| {rq:.2e}, |A'p+iwp| {rp:.2e}, <q,q> {qq}, <p,q> ({pr}, {pi})"
            );
            assert!(rq < 1e-10 && rp < 1e-10, "{name} {which:?}");
            assert!((qq - 1.0).abs() < 1e-12);
            assert!((pr - 1.0).abs() < 1e-12 && pi.abs() < 1e-12);
        }
    }
}

#[test]
fn singular_a_makes_l1_an_error() {
    // At the fold point A = [0], so A^-1 does not exist and l1 is an error, not a number.
    for which in BOTH {
        let mut m = model_of("fold", which);
        let r = first_lyapunov(m.as_mut(), &[0.0], 0, &[0.0], 0.0, 1.0);
        assert!(r.is_err(), "{which:?}: {r:?}");
    }
}

#[test]
fn bp_gets_no_coefficient() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 1.0),
            h0: 0.07,
            hmax: 0.07,
            ..Options::default()
        };
        let b = run("pitchfork", which, &setup, &opts);
        assert_eq!(b.specials.len(), 1);
        assert_eq!(b.specials[0].label, "BP");
        assert_eq!(b.specials[0].extras.coefficient, None);
        assert_eq!(b.specials[0].extras.normalization, None);
        assert!(b.warnings.is_empty());
    }
}

#[test]
fn near_identity_change_keeps_l1() {
    // hopf_nl has B != 0 but the same linear part and the same l1 = 2c as hopf_c, so
    // this tests the B terms of the formula.
    for which in BOTH {
        for c in [-1.0, 1.0] {
            let b = hopf_run("hopf_nl", c, which);
            let (l1, _) = coefficient(&b, "H");
            println!("hopf_nl c={c} {which:?}: l1 = {l1:.16}");
            assert!((l1 - 2.0 * c).abs() < 1e-10, "{which:?} c={c}: l1 = {l1}");
        }
    }
}

#[test]
fn fold_sign_follows_the_alpha_orientation() {
    // fold: F_alpha = +1 gives a = 1. fold_neg: F_alpha = -1 gives a = -1. Whatever sign
    // null_right returns, one of the two models takes the orientation flip.
    for which in BOTH {
        for (name, want) in [("fold", 1.0), ("fold_neg", -1.0)] {
            let mut m = model_of(name, which);
            let a = fold_coefficient(m.as_mut(), &[0.0], 0, &[0.0], 0.0).unwrap();
            assert!((a - want).abs() < 1e-12, "{name} {which:?}: a = {a}");
        }
        let setup = EpSetup::new(0, vec![1.0], vec![-1.0]);
        let opts = Options {
            direction: -1.0,
            max_steps: 150,
            ..Options::default()
        };
        let b = run("fold_neg", which, &setup, &opts);
        let (a, _) = coefficient(&b, "LP");
        assert!((a + 1.0).abs() < 1e-12, "{which:?}: a = {a}");
    }
}
