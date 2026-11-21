//! Both evaluators against each other on the test models, and the refusals of `build`.

mod common;

use common::{load, ulp_dist};
use pc_engine::{EngineError, Evaluator, build, build_allowing_t, build_dde};
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
            // One directional entry, to cover the `v` argument. The two evaluators differ by
            // at most 1 ulp here (measured over these models); 4 matches the other entries.
            let kind = EntryKind::Dir(vec![Slot::U, Slot::U]);
            let v: Vec<f64> = (0..2 * n).map(|i| 0.1 + 0.2 * i as f64).collect();
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            jit.eval(&kind, &x, &p, &v, &mut a).unwrap();
            fb.eval(&kind, &x, &p, &v, &mut b).unwrap();
            for (i, (u, w)) in a.iter().zip(&b).enumerate() {
                let d = ulp_dist(*u, *w);
                assert!(
                    d <= 4,
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
            build(&model, &info, which.clone()),
            Err(EngineError::NonAutonomous)
        ));
    }
}

#[test]
fn build_allowing_t_accepts_a_model_that_uses_t() {
    let (model, info) = load("nonautonomous");
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        assert!(matches!(
            build(&model, &info, which.clone()),
            Err(EngineError::NonAutonomous)
        ));
        let mut m = build_allowing_t(&model, &info, which.clone()).unwrap();
        let n_p = m.n_p();
        let mut out = vec![0.0; m.n()];
        let mut p = vec![0.5; n_p];
        p.push(1.3);
        let x = vec![0.5; m.n()];
        m.eval(&EntryKind::F, &x, &p, &[], &mut out).unwrap();
        p[n_p] = 2.0;
        let mut out2 = vec![0.0; m.n()];
        m.eval(&EntryKind::F, &x, &p, &[], &mut out2).unwrap();
        assert_ne!(out, out2, "time must reach the right-hand side");
    }
}

#[test]
fn table_that_reads_a_parameter_is_refused() {
    for name in [
        "table_par",
        "table_derived",
        "table_via_table",
        "table_fixed_par",
    ] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            match build(&model, &info, which.clone()) {
                Err(EngineError::BadProblem(m)) => assert!(m.contains("table"), "{name}: {m}"),
                other => panic!("{name}: expected BadProblem, got {:?}", other.err()),
            }
        }
    }
}

#[test]
fn table_that_reads_only_t_is_accepted() {
    for name in ["table_t", "table_func_shadow", "table_fixed_const"] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            assert!(build(&model, &info, which.clone()).is_ok(), "{name}");
        }
    }
}

#[test]
fn non_finite_output_is_an_error() {
    for name in ["nan_log", "inf_div"] {
        let (model, info) = load(name);
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            let mut m = build(&model, &info, which.clone()).unwrap();
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
        let mut m = build(&model, &info, which.clone()).unwrap();
        let mut out = [0.0];
        assert!(
            m.eval(&EntryKind::F, &[1.0, 2.0], &[0.0, 0.0], &[], &mut out)
                .is_err()
        );
    }
}

/// SPEC 8af-5, stage 2: the JIT and the fallback run a recursive function and agree; an AOT
/// library still refuses at build with `Model`, before any library is opened.
#[test]
fn recursive_model_runs_on_the_jit_and_the_fallback_but_not_on_aot() {
    // Inline, not in tests/models: the recursion baseline golden lists every file there.
    let mut map = pc_ode::SourceMap::new();
    let top = map.add(
        "pw.ode",
        "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\ninit x=1.5\nx'=pw(x,k)\ndone\n",
    );
    let l = pc_ode::load(top, &mut map, &mut pc_ode::MapLoader::new());
    let (model, info) = (l.expanded, l.info);
    match build(
        &model,
        &info,
        Evaluator::Aot {
            lib: std::path::PathBuf::from("no-such-library"),
            tag: 0,
        },
    ) {
        Err(EngineError::Model(m)) => assert!(m.contains("pw"), "{m}"),
        Err(e) => panic!("{e:?}"),
        Ok(_) => panic!("an AOT build of a recursive model must be refused"),
    }
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        let mut ev = build(&model, &info, which).expect("builds");
        let mut out = [f64::NAN];
        ev.eval(&EntryKind::F, &[1.5], &[3.0, 0.0], &[], &mut out)
            .expect("evaluates");
        assert_eq!(out[0], 3.375);
        let mut j = [f64::NAN];
        ev.eval(&EntryKind::JacU, &[1.5], &[3.0, 0.0], &[], &mut j)
            .expect("evaluates");
        assert_eq!(j[0], 6.75);
    }
}

/// Builds an inline model on the fallback evaluator, which runs the interpreter.
fn fallback_of(src: &str) -> Box<dyn pc_engine::ModelEval> {
    let mut map = pc_ode::SourceMap::new();
    let top = map.add("t.ode", src);
    let l = pc_ode::load(top, &mut map, &mut pc_ode::MapLoader::new());
    build(&l.expanded, &l.info, Evaluator::Fallback).expect("fallback")
}

/// SPEC 8af-5: `pw(x, n)` gives x^n, a JacU of n x^(n-1) and a second derivative of
/// n(n-1) x^(n-2) to 1 ulp on the interpreter too, not only on the DAG evaluator.
#[test]
fn interpreter_gives_pw_and_its_derivatives_to_1_ulp() {
    let mut fb =
        fallback_of("par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n");
    for x in [1.5f64, 0.5] {
        for n in 0..=30i32 {
            let nf = f64::from(n);
            let p = [nf, 0.0];
            let mut f = [f64::NAN];
            let mut j = [f64::NAN];
            let mut d = [f64::NAN];
            fb.eval(&EntryKind::F, &[x], &p, &[], &mut f).unwrap();
            fb.eval(&EntryKind::JacU, &[x], &p, &[], &mut j).unwrap();
            fb.eval(
                &EntryKind::Dir(vec![Slot::U, Slot::U]),
                &[x],
                &p,
                &[1.0, 1.0],
                &mut d,
            )
            .unwrap();
            // x = 1.5 and 0.5 make every power, and so every closed form below, exact.
            let w0 = x.powi(n);
            let w1 = if n >= 1 { nf * x.powi(n - 1) } else { 0.0 };
            let w2 = if n >= 2 {
                nf * (nf - 1.0) * x.powi(n - 2)
            } else {
                0.0
            };
            for (what, g, w) in [("F", f[0], w0), ("JacU", j[0], w1), ("Dir", d[0], w2)] {
                assert!(
                    g == w || ulp_dist(g, w) <= 1,
                    "{what} at x={x} n={n}: {g:?} vs {w:?}"
                );
            }
        }
    }
}

/// SPEC 8af-5: on the interpreter, `sum(999999)` runs 1,000,000 active calls and gives
/// n(n+1)/2, and `sum(1000000)` is the cap error, not a NaN.
#[test]
fn interpreter_reaches_the_cap_and_one_more_fails() {
    // The interpreter keeps its frames on the heap, so a small native stack is enough.
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let mut fb =
                fallback_of("par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n");
            let mut out = [f64::NAN];
            fb.eval(&EntryKind::F, &[0.0], &[999_999.0, 0.0], &[], &mut out)
                .expect("1,000,000 active calls");
            assert_eq!(out[0], 999_999.0 * 1_000_000.0 / 2.0);
            match fb.eval(&EntryKind::F, &[0.0], &[1_000_000.0, 0.0], &[], &mut out) {
                Err(EngineError::CallLimit(m)) => {
                    assert!(m.contains("sum") && m.contains("1000000"), "{m}")
                }
                other => panic!("{other:?}"),
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Builds an inline model on the JIT.
fn jit_of(src: &str) -> Box<dyn pc_engine::ModelEval> {
    let mut map = pc_ode::SourceMap::new();
    let top = map.add("t.ode", src);
    let l = pc_ode::load(top, &mut map, &mut pc_ode::MapLoader::new());
    build(&l.expanded, &l.info, Evaluator::Jit).expect("jit")
}

/// SPEC 8af-5, stage 2: on the JIT, `sum(999999)` runs 1,000,000 active calls, and
/// `sum(1000000)` is the cap error with the function and the cap in its text, not a NaN. The
/// next call on the same evaluator runs again, so the error does not stick.
#[test]
fn jit_reaches_the_cap_and_one_more_fails() {
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let mut ev =
                jit_of("par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n");
            let mut out = [f64::NAN];
            ev.eval(&EntryKind::F, &[0.0], &[999_999.0, 0.0], &[], &mut out)
                .expect("1,000,000 active calls");
            assert_eq!(out[0], 999_999.0 * 1_000_000.0 / 2.0);
            match ev.eval(&EntryKind::F, &[0.0], &[1_000_000.0, 0.0], &[], &mut out) {
                Err(EngineError::CallLimit(m)) => {
                    assert!(m.contains("sum") && m.contains("1000000"), "{m}")
                }
                other => panic!("{other:?}"),
            }
            ev.eval(&EntryKind::F, &[0.0], &[10.0, 0.0], &[], &mut out)
                .expect("the error does not stick");
            assert_eq!(out[0], 55.0);
        })
        .unwrap()
        .join()
        .unwrap();
}

/// SPEC 8af-5, stage 2, delay in a recursive body: `build_dde` runs the model on the JIT and on
/// the fallback evaluator and the two agree to 0 ulp. The parameters are the declared `k`, the
/// hidden slot of `delay(x,1)`, then `t`.
#[test]
fn delay_in_a_recursive_body_runs_on_the_jit_and_agrees_with_the_fallback() {
    let src = "par k=3\nf(n)=if(n<1)then(1)else(delay(x,1)*f(n-1))\nx'=f(k)\ndone\n";
    let mut map = pc_ode::SourceMap::new();
    let top = map.add("dde_rec.ode", src);
    let l = pc_ode::load(top, &mut map, &mut pc_ode::MapLoader::new());
    let mut jit = build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("jit");
    let mut fb = build_dde(&l.expanded, &l.info, Evaluator::Fallback).expect("fallback");
    assert_eq!(jit.n_user_params, 1);
    assert_eq!(jit.delays.len(), 1);
    let kinds = [
        EntryKind::F,
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![Slot::U, Slot::U]),
        EntryKind::Dir(vec![Slot::U, Slot::P]),
    ];
    for (x, k, d) in [(0.7f64, 3.0, 1.5), (-0.4, 2.0, 0.9), (1.1, 0.0, 2.0)] {
        let p = [k, d, 0.0];
        for kind in &kinds {
            let n_p = jit.eval.n_p();
            assert_eq!(n_p, 2);
            let v: Vec<f64> = match kind {
                EntryKind::Dir(slots) => slots
                    .iter()
                    .flat_map(|s| vec![1.0; s.len(1, n_p)])
                    .collect(),
                _ => Vec::new(),
            };
            let out_len = match kind {
                EntryKind::JacP => n_p,
                _ => 1,
            };
            let mut a = vec![f64::NAN; out_len];
            let mut b = vec![f64::NAN; out_len];
            jit.eval.eval(kind, &[x], &p, &v, &mut a).expect("jit eval");
            fb.eval
                .eval(kind, &[x], &p, &v, &mut b)
                .expect("fallback eval");
            for (u, w) in a.iter().zip(&b) {
                assert_eq!(ulp_dist(*u, *w), 0, "{kind:?} x={x} k={k}: {u:e} vs {w:e}");
            }
        }
        let mut f = [f64::NAN];
        jit.eval.eval(&EntryKind::F, &[x], &p, &[], &mut f).unwrap();
        assert_eq!(f[0], d.powi(k as i32), "x={x} k={k}");
    }
}
