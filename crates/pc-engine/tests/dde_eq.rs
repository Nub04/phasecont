//! `.ode` delays, equilibria and their stability (SPEC section 8d, tests 1 to 8).
//!
//! The models are written inline. Every closed form below is derived in the SPEC.

use pc_engine::{
    DdeModel, EngineError, EpSetup, Evaluator, Options, RootOptions, build, build_dde,
    dde_equilibrium, dde_linearize, dde_stability,
};
use pc_ir::EntryKind;
use pc_ode::{MapLoader, Model, ModelInfo, Severity, SourceMap, load};
use std::f64::consts::PI;

fn loaded(src: &str) -> (Model, ModelInfo) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    (l.expanded, l.info)
}

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let (m, i) = loaded(src);
    match build_dde(&m, &i, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

/// Both evaluators, since the lowering is shared and the evaluators are not.
fn both() -> [Evaluator; 2] {
    [Evaluator::Jit, Evaluator::Fallback]
}

/// Newton with at most 8 steps and tight tolerances, as the SPEC tests ask.
fn tight() -> Options {
    Options {
        max_newton: 8,
        tol_x: 1e-14,
        tol_f: 1e-14,
        ..Options::default()
    }
}

fn setup(params: &[f64], u0: &[f64]) -> EpSetup {
    EpSetup::new(0, params.to_vec(), u0.to_vec())
}

fn roots(r: f64, x: f64) -> Vec<(f64, f64, f64)> {
    let mut m = dde(
        "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n",
        Evaluator::Jit,
    );
    let set = dde_stability(
        &mut m,
        &[x],
        &[r, 1.0],
        &RootOptions {
            n: 30,
            re_min: -2.0,
        },
    )
    .expect("roots");
    set.roots.iter().map(|c| (c.re, c.im, c.residual)).collect()
}

const LOGISTIC: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

// Test 1.

#[test]
fn logistic_lowers_to_three_parameters_and_finds_one() {
    for which in both() {
        let mut m = dde(LOGISTIC, which);
        assert_eq!(m.n_user_params, 2);
        assert_eq!(m.eval.n(), 1);
        assert_eq!(m.eval.n_p(), 3);
        assert_eq!(m.delays.len(), 1);
        let x = dde_equilibrium(&mut m, &setup(&[1.5, 1.0], &[0.8]), &tight()).expect("newton");
        assert!((x[0] - 1.0).abs() <= 1e-14, "{x:?}");
    }
}

#[test]
fn logistic_linearizes_exactly() {
    for which in both() {
        let mut m = dde(LOGISTIC, which);
        let at1 = dde_linearize(&mut m, &[1.0], &[1.5, 1.0]).expect("linearize");
        assert_eq!(at1.dim, 1);
        assert_eq!(at1.a0, [0.0]);
        assert_eq!(at1.delays, vec![(1.0, vec![-1.5])]);
        let at0 = dde_linearize(&mut m, &[0.0], &[1.5, 1.0]).expect("linearize");
        assert_eq!(at0.a0, [1.5]);
        assert_eq!(at0.delays, vec![(1.0, vec![0.0])]);
    }
}

#[test]
fn logistic_root_at_the_origin_is_r() {
    let rs = roots(1.5, 0.0);
    assert!(
        rs.iter()
            .any(|&(re, im, _)| (re - 1.5).abs() < 1e-12 && im.abs() < 1e-12),
        "{rs:?}"
    );
}

#[test]
fn logistic_stability_follows_r_tau() {
    let lead = |r: f64| {
        let rs = roots(r, 1.0);
        assert!(!rs.is_empty(), "no roots at r = {r}");
        rs.iter().map(|c| c.0).fold(f64::NEG_INFINITY, f64::max)
    };
    assert!(lead(1.5) < -1e-3, "r tau = 1.5 is stable");
    assert!(lead(1.7) > 1e-3, "r tau = 1.7 is unstable");
    assert!(
        lead(PI / 2.0).abs() < 1e-12,
        "r tau = pi/2 is the Hopf value"
    );
    let rs = roots(PI / 2.0, 1.0);
    for sign in [1.0, -1.0] {
        assert!(
            rs.iter().any(|&(re, im, res)| re.abs() < 1e-12
                && (im - sign * PI / 2.0).abs() < 1e-12
                && res <= 1e-12),
            "no root at {sign} i pi/2 in {rs:?}"
        );
    }
}

// Test 2.

const QUADRATIC: &str = "par a=1.5, tau=1\nx'=a-x-0.5*delay(x,tau)^2\ndone\n";

#[test]
fn newton_uses_the_sum_of_the_jacobians() {
    for which in both() {
        let mut m = dde(QUADRATIC, which);
        let lin = dde_linearize(&mut m, &[1.0], &[1.5, 1.0]).expect("linearize");
        assert_eq!(lin.a0, [-1.0]);
        assert_eq!(lin.delays, vec![(1.0, vec![-1.0])]);
        // With A_0 alone the iteration x -> a - x^2/2 has multiplier -1 at the root and
        // would not converge in 8 steps.
        let x = dde_equilibrium(&mut m, &setup(&[1.5, 1.0], &[0.5]), &tight()).expect("newton");
        assert!((x[0] - 1.0).abs() <= 1e-14, "{x:?}");
    }
}

#[test]
fn quadratic_roots_satisfy_the_characteristic_equation() {
    let mut m = dde(QUADRATIC, Evaluator::Jit);
    let set = dde_stability(
        &mut m,
        &[1.0],
        &[1.5, 1.0],
        &RootOptions {
            n: 30,
            re_min: -3.0,
        },
    )
    .expect("roots");
    assert!(set.roots.len() >= 2, "{set:?}");
    for c in &set.roots {
        assert!(c.residual <= 1e-12, "{c:?}");
        // lambda = -1 - e^{-lambda}.
        let e = (-c.re).exp();
        let (gr, gi) = (c.re + 1.0 + e * c.im.cos(), c.im - e * c.im.sin());
        assert!(gr.hypot(gi) <= 1e-11, "{c:?}: {gr} {gi}");
    }
}

// Test 3.

const TWO_STATES: &str = "par tau1=0.7, tau2=1.3\nx'=-delay(y,tau1)\ny'=delay(x,tau2)-2*y\ndone\n";

#[test]
fn two_states_two_delays_come_out_ascending() {
    for which in both() {
        let mut m = dde(TWO_STATES, which);
        assert_eq!(m.delays.len(), 2);
        let lin = dde_linearize(&mut m, &[0.0, 0.0], &[0.7, 1.3]).expect("linearize");
        assert_eq!(lin.dim, 2);
        assert_eq!(lin.a0, [0.0, 0.0, 0.0, -2.0]);
        assert_eq!(
            lin.delays,
            vec![
                (0.7, vec![0.0, -1.0, 0.0, 0.0]),
                (1.3, vec![0.0, 0.0, 1.0, 0.0])
            ]
        );
        // The same model with the values of the two parameters swapped.
        let lin = dde_linearize(&mut m, &[0.0, 0.0], &[1.3, 0.7]).expect("linearize");
        assert_eq!(
            lin.delays,
            vec![
                (0.7, vec![0.0, 0.0, 1.0, 0.0]),
                (1.3, vec![0.0, -1.0, 0.0, 0.0])
            ]
        );
    }
}

// Test 4.

#[test]
fn slots_with_equal_times_merge() {
    let mut m = dde(
        "par tau=1\nx'=-delay(x,tau)-2*delay(y,tau)\ny'=x-y\ndone\n",
        Evaluator::Fallback,
    );
    assert_eq!(m.delays.len(), 2);
    let lin = dde_linearize(&mut m, &[0.0, 0.0], &[1.0]).expect("linearize");
    assert_eq!(lin.a0, [0.0, 0.0, 1.0, -1.0]);
    assert_eq!(lin.delays, vec![(1.0, vec![-1.0, -2.0, 0.0, 0.0])]);
}

#[test]
fn different_times_stay_apart() {
    let mut m = dde("x'=-delay(x,1)-0.5*delay(x,2)\ndone\n", Evaluator::Fallback);
    assert_eq!(m.delays.len(), 2);
    let lin = dde_linearize(&mut m, &[0.0], &[]).expect("linearize");
    assert_eq!(lin.delays, vec![(1.0, vec![-1.0]), (2.0, vec![-0.5])]);
}

#[test]
fn a_repeated_call_is_one_slot_and_a_literal_equal_to_a_parameter_is_two() {
    let m = dde(
        "par tau=1\nx'=-delay(x,tau)-3*delay(x,tau)\ndone\n",
        Evaluator::Fallback,
    );
    assert_eq!(m.delays.len(), 1);
    let mut m = dde(
        "par tau=1\nx'=-delay(x,1)-delay(x,tau)\ndone\n",
        Evaluator::Fallback,
    );
    assert_eq!(m.delays.len(), 2);
    let lin = dde_linearize(&mut m, &[0.0], &[1.0]).expect("linearize");
    assert_eq!(lin.delays, vec![(1.0, vec![-2.0])]);
}

// Test 5.

#[test]
fn slot_order_follows_the_statements() {
    let m = dde(
        "par t1=1, t2=2\nx'=delay(y,t2)+delay(x,t1)\ny'=-y\ndone\n",
        Evaluator::Fallback,
    );
    assert_eq!(m.n_user_params, 2);
    assert_eq!(m.eval.n_p(), 4);
    assert_eq!(m.delays[0].state, 1);
    assert_eq!(m.delays[1].state, 0);
}

// Test 6.

/// F, JacU and JacP of the JIT and of the interpreter agree on a model with three slots,
/// at several values of the hidden parameters. The hidden columns of JacP are included.
#[test]
fn jit_and_fallback_agree_with_three_slots() {
    let src = "par a=0.7, b=1.3\n\
               x'=-delay(y,a)+delay(x,b)*x\n\
               y'=delay(x,b)-2*y+delay(y,0.5)^2\n\
               done\n";
    let mut jit = dde(src, Evaluator::Jit);
    let mut fb = dde(src, Evaluator::Fallback);
    assert_eq!(jit.delays.len(), 3);
    let n_p = jit.eval.n_p();
    assert_eq!(n_p, 5);
    let points: [([f64; 2], [f64; 6]); 3] = [
        ([0.3, -0.4], [0.7, 1.3, 0.1, 0.2, 0.3, 0.0]),
        ([1.0, 2.0], [0.7, 1.3, 1.0, 1.0, 1.0, 5.0]),
        ([-0.5, 0.25], [0.1, 0.2, -2.0, 0.5, 3.0, 0.0]),
    ];
    let len = |k: &EntryKind| match k {
        EntryKind::F => 2,
        EntryKind::JacU => 4,
        EntryKind::JacP => 2 * n_p,
        _ => unreachable!(),
    };
    for kind in [EntryKind::F, EntryKind::JacU, EntryKind::JacP] {
        for (x, p) in &points {
            let mut a = vec![0.0; len(&kind)];
            let mut b = vec![0.0; len(&kind)];
            jit.eval.eval(&kind, x, p, &[], &mut a).expect("jit");
            fb.eval.eval(&kind, x, p, &[], &mut b).expect("fallback");
            for (i, (g, w)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-14 * (1.0 + w.abs()),
                    "{kind:?}[{i}] at {x:?} {p:?}: {g} vs {w}"
                );
            }
        }
    }
}

// Test 7.

#[test]
fn a_delay_time_that_is_not_finite_and_positive_is_a_bad_problem() {
    for which in both() {
        let mut m = dde(LOGISTIC, which);
        for tau in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let p = [1.5, tau];
            let e = dde_equilibrium(&mut m, &setup(&p, &[0.8]), &tight()).unwrap_err();
            assert!(
                matches!(e, EngineError::BadProblem(_)),
                "tau = {tau}: {e:?}"
            );
            let e = dde_stability(
                &mut m,
                &[1.0],
                &p,
                &RootOptions {
                    n: 20,
                    re_min: -1.0,
                },
            )
            .unwrap_err();
            assert!(
                matches!(e, EngineError::BadProblem(_)),
                "tau = {tau}: {e:?}"
            );
            let e = dde_linearize(&mut m, &[1.0], &p).unwrap_err();
            assert!(
                matches!(e, EngineError::BadProblem(_)),
                "tau = {tau}: {e:?}"
            );
        }
    }
}

#[test]
fn wrong_lengths_are_bad_problems() {
    let mut m = dde(LOGISTIC, Evaluator::Fallback);
    // The caller gives the declared parameters only, not the hidden slot or t.
    for p in [&[1.5][..], &[1.5, 1.0, 0.0][..]] {
        let e = dde_equilibrium(&mut m, &setup(p, &[0.8]), &tight()).unwrap_err();
        assert!(matches!(e, EngineError::BadProblem(_)), "{e:?}");
        let e = dde_linearize(&mut m, &[1.0], p).unwrap_err();
        assert!(matches!(e, EngineError::BadProblem(_)), "{e:?}");
    }
    let e = dde_linearize(&mut m, &[1.0, 2.0], &[1.5, 1.0]).unwrap_err();
    assert!(matches!(e, EngineError::BadProblem(_)), "{e:?}");
}

#[test]
fn newton_failure_is_reported() {
    // x' = 1 + delay(x,1)^2 has no equilibrium.
    let mut m = dde("x'=1+delay(x,1)^2\ndone\n", Evaluator::Fallback);
    let e = dde_equilibrium(&mut m, &setup(&[], &[0.5]), &tight()).unwrap_err();
    assert!(matches!(e, EngineError::NewtonFailed { .. }), "{e:?}");
}

// Test 8.

#[test]
fn build_still_refuses_and_build_dde_needs_a_delay() {
    let (m, i) = loaded(LOGISTIC);
    for which in both() {
        let e = build(&m, &i, which).err().expect("build refuses a delay");
        assert!(matches!(e, EngineError::Model(_)), "{e:?}");
    }
    let (m, i) = loaded("par r=1\nx'=-r*x\ndone\n");
    for which in both() {
        let e = build_dde(&m, &i, which).err().expect("no delay");
        assert!(matches!(e, EngineError::Model(_)), "{e:?}");
    }
    // A model without a delay builds as before.
    assert!(build(&m, &i, Evaluator::Fallback).is_ok());
}

#[test]
fn a_model_that_reads_time_is_refused() {
    let (m, i) = loaded("x'=-delay(x,1)+t\ndone\n");
    let e = build_dde(&m, &i, Evaluator::Fallback)
        .err()
        .expect("refused");
    assert!(matches!(e, EngineError::NonAutonomous), "{e:?}");
}
