//! Both evaluators against each other on the test models, and the refusals of `build`.

mod common;

use common::{load, ulp_dist};
use pc_engine::{EngineError, Evaluator, build};
use pc_ir::{EntryKind, Slot};

/// Points per model: the declared state with the declared parameters, then two shifted
/// states. `t` (the last slot of `p`) is 0.3.
fn points(x0: &[f64], par: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut out = Vec::new();
    for (k, s) in [0.0, 0.37, -0.8].into_iter().enumerate() {
        let x: Vec<f64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| v + s * (1.0 + 0.3 * i as f64) + 0.1 * k as f64)
            .collect();
        let mut p: Vec<f64> = par.iter().map(|v| v * (1.0 + 0.2 * s)).collect();
        p.push(0.3);
        out.push((x, p));
    }
    out
}

const MODELS: [(&str, usize, usize); 9] = [
    ("fold", 1, 1),
    ("hopf_c", 2, 2),
    ("lorenz", 3, 3),
    ("brusselator", 2, 2),
    ("transcritical", 1, 1),
    ("pitchfork", 1, 1),
    ("circle", 1, 1),
    ("neutral_saddle", 2, 1),
    ("table_t", 1, 1),
];

#[test]
fn jit_and_fallback_agree() {
    for (name, n, n_p) in MODELS {
        let (model, info) = load(name);
        let mut jit = build(&model, &info, Evaluator::Jit).unwrap();
        let mut fb = build(&model, &info, Evaluator::Fallback).unwrap();
        assert_eq!((jit.n(), jit.n_p()), (n, n_p), "{name}");
        assert_eq!((fb.n(), fb.n_p()), (n, n_p), "{name}");
        let x0 = vec![0.5; n];
        let par = vec![1.3; n_p];
        for (x, p) in points(&x0, &par) {
            for (kind, len, bound) in [
                (EntryKind::F, n, 4u64),
                (EntryKind::JacU, n * n, 4),
                (EntryKind::JacP, n * n_p, 4),
            ] {
                let (mut a, mut b) = (vec![0.0; len], vec![0.0; len]);
                jit.eval(&kind, &x, &p, &[], &mut a).unwrap();
                fb.eval(&kind, &x, &p, &[], &mut b).unwrap();
                for (i, (u, w)) in a.iter().zip(&b).enumerate() {
                    let d = ulp_dist(*u, *w);
                    assert!(
                        d <= bound,
                        "{name} {kind:?}[{i}]: jit {u:e} fallback {w:e} ({d} ulp)"
                    );
                }
            }
            // One directional entry, to cover the `v` argument.
            let kind = EntryKind::Dir(vec![Slot::U, Slot::U]);
            let v: Vec<f64> = (0..2 * n).map(|i| 0.1 + 0.2 * i as f64).collect();
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            jit.eval(&kind, &x, &p, &v, &mut a).unwrap();
            fb.eval(&kind, &x, &p, &v, &mut b).unwrap();
            for (i, (u, w)) in a.iter().zip(&b).enumerate() {
                let d = ulp_dist(*u, *w);
                assert!(
                    d <= 64,
                    "{name} Dir[{i}]: jit {u:e} fallback {w:e} ({d} ulp)"
                );
            }
        }
    }
}

#[test]
fn nonautonomous_is_refused() {
    let (model, info) = load("nonautonomous");
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        assert!(matches!(
            build(&model, &info, which),
            Err(EngineError::NonAutonomous)
        ));
    }
}

#[test]
fn table_that_reads_a_parameter_is_refused() {
    for name in ["table_par", "table_derived"] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            match build(&model, &info, which) {
                Err(EngineError::BadProblem(m)) => assert!(m.contains("table"), "{name}: {m}"),
                other => panic!("{name}: expected BadProblem, got {:?}", other.err()),
            }
        }
    }
}

#[test]
fn table_that_reads_only_t_is_accepted() {
    for name in ["table_t", "table_func_shadow"] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            assert!(build(&model, &info, which).is_ok(), "{name}");
        }
    }
}

#[test]
fn non_finite_output_is_an_error() {
    for name in ["nan_log", "inf_div"] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            let mut m = build(&model, &info, which).unwrap();
            let mut out = [0.0];
            let (x, p) = if name == "nan_log" {
                (-1.0, 0.0)
            } else {
                (0.0, 0.0)
            };
            match m.eval(&EntryKind::F, &[x], &[p], &[], &mut out) {
                Err(EngineError::NotFinite(k)) => assert_eq!(k, "F", "{name}"),
                other => panic!("{name}: expected NotFinite, got {other:?}"),
            }
            // A finite point still works afterwards.
            m.eval(&EntryKind::F, &[2.0], &[0.0], &[], &mut out)
                .unwrap();
        }
    }
}

#[test]
fn wrong_length_is_an_error() {
    let (model, info) = load("fold");
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        let mut m = build(&model, &info, which).unwrap();
        let mut out = [0.0];
        assert!(
            m.eval(&EntryKind::F, &[1.0, 2.0], &[0.0, 0.0], &[], &mut out)
                .is_err()
        );
    }
}
