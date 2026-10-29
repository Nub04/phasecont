//! SPEC section 9 acceptance tests 1, 2, 5 and 6, each on both evaluators.
//!
//! Every located value is printed with its evaluator and its error against the hand
//! anchor; run with `-- --nocapture` to see them.

mod common;

use common::load;
use pc_engine::{
    Branch, EpSetup, Evaluator, ModelEval, Options, SpecialPoint, StopReason, build, continue_ep,
    special_index, switch_bp,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which.clone()).unwrap()
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

fn with_label<'a>(b: &'a Branch, label: &str) -> Vec<&'a SpecialPoint> {
    b.specials.iter().filter(|s| s.label == label).collect()
}

fn labels(b: &Branch) -> Vec<&str> {
    b.specials.iter().map(|s| s.label.as_str()).collect()
}

/// Prints one located value with its error and asserts the tolerance.
fn check(test: &str, which: Evaluator, what: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs();
    println!("{test} [{which:?}] {what}: got {got:.15e}, anchor {want:.15e}, error {err:.3e}");
    assert!(
        err <= tol,
        "{test} [{which:?}] {what}: error {err:e} above {tol:e}"
    );
}

#[test]
fn acceptance_1_fold_has_one_lp_at_the_origin() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            direction: 1.0,
            max_steps: 100,
            bounds: (-5.0, 5.0),
            ..Options::default()
        };
        let b = run("fold", which.clone(), &setup, &opts);
        let lp = with_label(&b, "LP");
        assert_eq!(lp.len(), 1, "{which:?}: {:?}", labels(&b));
        let p = &lp[0].point;
        check("test1", which.clone(), "LP x", p.u[0], 0.0, 1e-10);
        check("test1", which.clone(), "LP a", p.alpha, 0.0, 1e-10);
        let coef = lp[0].extras.coefficient.expect("LP coefficient");
        check("test1", which.clone(), "LP coefficient a", coef, 1.0, 1e-12);
    }
}

#[test]
fn acceptance_2_hopf_normal_form_has_l1_equal_2c() {
    for which in BOTH {
        for c in [1.0, -1.0] {
            let setup = EpSetup::new(0, vec![-0.5, c], vec![0.0, 0.0]);
            let opts = Options {
                direction: 1.0,
                max_steps: 60,
                bounds: (-1.0, 1.0),
                ..Options::default()
            };
            let b = run("hopf_c", which.clone(), &setup, &opts);
            let h = with_label(&b, "H");
            assert_eq!(h.len(), 1, "{which:?} c={c}: {:?}", labels(&b));
            assert!(
                with_label(&b, "NSA").is_empty(),
                "{which:?} c={c}: {:?}",
                labels(&b)
            );
            let tag = format!("c={c}");
            check(
                "test2",
                which.clone(),
                &format!("{tag} H al"),
                h[0].point.alpha,
                0.0,
                1e-10,
            );
            let omega = h[0].extras.omega.expect("H omega");
            check(
                "test2",
                which.clone(),
                &format!("{tag} omega"),
                omega,
                1.0,
                1e-10,
            );
            let l1 = h[0].extras.coefficient.expect("H l1");
            check(
                "test2",
                which.clone(),
                &format!("{tag} l1"),
                l1,
                2.0 * c,
                1e-10,
            );
            assert_eq!(l1.signum(), c.signum(), "{which:?} {tag}: sign of l1");
        }
    }
}

#[test]
fn acceptance_5_lorenz_pitchfork_and_hopf() {
    let b_par = 8.0 / 3.0;
    for which in BOTH {
        let setup = EpSetup::new(1, vec![10.0, 0.5, b_par], vec![0.0; 3]);
        let opts = Options {
            direction: 1.0,
            hmax: 0.5,
            max_steps: 1000,
            bounds: (0.0, 30.0),
            ..Options::default()
        };
        let b0 = run("lorenz", which.clone(), &setup, &opts);
        let bps = with_label(&b0, "BP");
        assert_eq!(bps.len(), 1, "{which:?}: {:?}", labels(&b0));
        check(
            "test5",
            which.clone(),
            "BP r",
            bps[0].point.alpha,
            1.0,
            1e-10,
        );

        let k = special_index(&b0, "BP1").expect("BP1");
        let mut m = model_of("lorenz", which.clone());
        let next = switch_bp(m.as_mut(), &setup, &b0, k, &opts).unwrap();
        let b1 = run("lorenz", which.clone(), &next, &opts);
        assert_eq!(b1.stop, StopReason::Bounds, "{which:?}: {:?}", b1.stop);
        let last = b1.points.last().unwrap();
        check(
            "test5",
            which.clone(),
            "switched branch end r",
            last.alpha,
            30.0,
            1e-9,
        );

        let (mut dxy, mut dz) = (0.0f64, 0.0f64);
        for p in &b1.points {
            dxy = dxy.max((p.u[0] - p.u[1]).abs());
            dz = dz.max((p.u[2] - p.u[0] * p.u[0] / b_par).abs());
        }
        check("test5", which.clone(), "max |x - y|", dxy, 0.0, 1e-9);
        check("test5", which.clone(), "max |z - x^2/b|", dz, 0.0, 1e-9);

        let h = with_label(&b1, "H");
        assert_eq!(h.len(), 1, "{which:?}: {:?}", labels(&b1));
        check(
            "test5",
            which.clone(),
            "H r",
            h[0].point.alpha,
            470.0 / 19.0,
            1e-9,
        );
        let l1 = h[0].extras.coefficient.expect("H l1");
        println!("test5 [{which:?}] H l1 = {l1:.15e}");
        assert!(l1 > 0.0, "{which:?}: l1 = {l1}");
    }
}

#[test]
fn acceptance_6_brusselator_hopf() {
    for which in BOTH {
        for a in [1.0, 2.0] {
            let b0 = 1.0 + a * a - 0.5;
            let setup = EpSetup::new(1, vec![a, b0], vec![a, b0 / a]);
            let opts = Options {
                direction: 1.0,
                max_steps: 100,
                bounds: (0.0, 1.0 + a * a + 1.0),
                ..Options::default()
            };
            let b = run("brusselator", which.clone(), &setup, &opts);
            let h = with_label(&b, "H");
            assert_eq!(h.len(), 1, "{which:?} a={a}: {:?}", labels(&b));
            let tag = format!("a={a}");
            check(
                "test6",
                which.clone(),
                &format!("{tag} H b"),
                h[0].point.alpha,
                1.0 + a * a,
                1e-9,
            );
            let omega = h[0].extras.omega.expect("H omega");
            check(
                "test6",
                which.clone(),
                &format!("{tag} omega"),
                omega,
                a,
                1e-9,
            );
        }
    }
}
