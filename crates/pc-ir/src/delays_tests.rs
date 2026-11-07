use super::*;
use crate::dag::NodeId;
use crate::entry::EntryKind;
use crate::error::IrErrorKind as K;
use crate::eval::eval;
use crate::interp::Fallback;
use crate::lower::{IrModel, lower, lower_dde};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn loaded(src: &str) -> pc_ode::Loaded {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    load(top, &mut map, &mut MapLoader::new())
}

/// Loads `src`, asserts the checker found no error, and lowers it with `lower_dde`.
fn dde(src: &str) -> IrModel {
    let l = loaded(src);
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    match lower_dde(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower_dde failed: {e:?}"),
    }
}

fn dde_errs(src: &str) -> Vec<IrError> {
    let l = loaded(src);
    match lower_dde(&l.expanded, &l.info) {
        Ok(_) => panic!("expected lower_dde errors for {src:?}"),
        Err(e) => e,
    }
}

fn run(m: &IrModel, outs: &[NodeId], x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; outs.len()];
    eval(&m.dag, outs, x, p, &[], &mut out);
    out
}

#[test]
fn logistic_has_one_hidden_slot() {
    let m = dde("par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\nx(0)=0.8\ndone\n");
    assert_eq!(m.n_user_params, 2);
    assert_eq!(m.n_p(), 3);
    assert_eq!(m.param_names, ["r", "tau", "delay(x,tau)"]);
    assert_eq!(
        m.delays,
        vec![DelaySlot {
            state: 0,
            time: DelayTime::Param(1)
        }]
    );
    assert_eq!(m.par_values.len(), 3);
    assert_eq!(m.par_values[..2], [1.5, 1.0]);
    assert!(m.par_values[2].is_nan(), "{:?}", m.par_values);
    // The hidden slot stands where the delayed state stood.
    let p = [1.5, 1.0, 0.25, 0.0];
    let got = run(&m, &m.f, &[0.5], &p)[0];
    assert!((got - 1.5 * 0.5 * 0.75).abs() < 1e-15, "{got}");
}

#[test]
fn time_follows_the_hidden_parameters() {
    let m = dde("x'=delay(x,1)*t\ndone\n");
    assert_eq!(m.n_p(), 1);
    assert!(!m.autonomous);
    let got = run(&m, &m.f, &[0.0], &[2.0, 3.0])[0];
    assert_eq!(got, 6.0);
}

#[test]
fn slots_follow_statement_order() {
    let m = dde("par t1=1, t2=2\nx'=delay(y,t2)+delay(x,t1)\ny'=-y\ndone\n");
    assert_eq!(m.n_user_params, 2);
    assert_eq!(
        m.param_names,
        ["t1", "t2", "delay(y,t2)", "delay(x,t1)"].map(String::from)
    );
    assert_eq!(m.delays[0].state, 1);
    assert_eq!(m.delays[0].time, DelayTime::Param(1));
    assert_eq!(m.delays[1].state, 0);
    assert_eq!(m.delays[1].time, DelayTime::Param(0));
}

#[test]
fn equal_pairs_share_a_slot() {
    // The same call twice, and a literal written two ways.
    let m = dde("par tau=1\nx'=-delay(x,tau)-2*delay(x,tau)\ndone\n");
    assert_eq!(m.delays.len(), 1);
    let m = dde("x'=-delay(x,1)-delay(x,1.0)\ndone\n");
    assert_eq!(m.delays.len(), 1);
    assert_eq!(m.param_names, ["delay(x,1)"]);
    // A literal and a parameter with the same value are two slots.
    let m = dde("par tau=1\nx'=-delay(x,1)-delay(x,tau)\ndone\n");
    assert_eq!(m.delays.len(), 2);
    assert_eq!(m.delays[0].time, DelayTime::Lit(1.0));
    assert_eq!(m.delays[1].time, DelayTime::Param(0));
    // Different states and different times are different slots.
    let m = dde("par tau=1\nx'=-delay(x,tau)-delay(y,tau)-delay(x,2)\ny'=x-y\ndone\n");
    assert_eq!(m.delays.len(), 3);
}

#[test]
fn a_number_constant_shares_the_slot_of_its_value() {
    let m = dde("number c=1.5\nx'=-delay(x,c)-delay(x,1.5)\ndone\n");
    assert_eq!(m.delays.len(), 1);
    assert_eq!(m.param_names, ["delay(x,1.5)"]);
    assert_eq!(m.delays[0].time, DelayTime::Lit(1.5));
}

#[test]
fn delays_are_read_in_every_formula_that_may_hold_a_state() {
    // An aux formula, a fixed quantity and a function body naming a global state.
    let m =
        dde("par tau=1\nf(u)=delay(x,tau)*u\nw=delay(x,2)\nx'=f(x)-w\naux a=delay(x,3)\ndone\n");
    let times: Vec<_> = m.delays.iter().map(|d| d.time).collect();
    assert_eq!(
        times,
        [
            DelayTime::Param(0),
            DelayTime::Lit(2.0),
            DelayTime::Lit(3.0)
        ]
    );
    // An unused function still gets its slot, which keeps the numbering fixed.
    let m = dde("par tau=1\nf(u)=delay(x,tau)*u\nx'=-x+delay(x,2)\ndone\n");
    assert_eq!(m.delays.len(), 2);
    assert_eq!(m.delays[0].time, DelayTime::Param(0));
}

#[test]
fn refusals_name_the_call() {
    let cases: &[(&str, &str)] = &[
        // First argument.
        ("par tau=1\nx'=delay(x+1,tau)\ndone\n", "delay(x+1,tau)"),
        ("par a=1, tau=1\nx'=delay(a,tau)\ndone\n", "delay(a,tau)"),
        (
            "par tau=1\nx'=-x\naux w=2*x\ny'=delay(w,tau)\ndone\n",
            "delay(w,tau)",
        ),
        ("par tau=1\nw=2*x\nx'=delay(w,tau)\ndone\n", "delay(w,tau)"),
        // Delay time.
        ("par tau=1\nx'=delay(x,2*tau)\ndone\n", "delay(x,2*tau)"),
        ("x'=delay(x,y)\ny'=-y\ndone\n", "delay(x,y)"),
        ("x'=delay(x,0)\ndone\n", "delay(x,0)"),
        ("x'=delay(x,-1)\ndone\n", "delay(x,-1)"),
        ("number c=0\nx'=delay(x,c)\ndone\n", "delay(x,c)"),
        ("par a=2\n!d=2*a\nx'=delay(x,d)\ndone\n", "delay(x,d)"),
        ("par a=2\nd=2*a\nx'=delay(x,d)\ndone\n", "delay(x,d)"),
        // Argument count.
        ("x'=delay(x)\ndone\n", "delay(x)"),
        ("x'=delay(x,1,2)\ndone\n", "delay(x,1,2)"),
        // A formal argument that hides the state or the parameter.
        (
            "par tau=1\nf(x)=delay(x,tau)\ny'=f(y)\ndone\n",
            "delay(x,tau)",
        ),
        ("f(tau)=delay(x,tau)\nx'=f(1)\ndone\n", "delay(x,tau)"),
        // A table, a derived parameter, a map.
        ("x'=-x\ntable h % 3 0 2 delay(x,1)\ndone\n", "delay(x,1)"),
        ("par a=1\n!d=delay(x,1)\nx'=-x\ndone\n", "delay(x,1)"),
        ("x(t+1)=delay(x,1)\ndone\n", "delay(x,1)"),
    ];
    for (src, call) in cases {
        let e = dde_errs(src);
        assert!(e.iter().all(|e| e.kind == K::Delay), "{src:?}: {e:?}");
        assert!(
            e.iter().any(|e| &src[e.span.start..e.span.end] == *call),
            "{src:?}: no error at {call:?} in {e:?}"
        );
        // The default entry refuses the same model as it refuses any delay.
        let l = loaded(src);
        let plain = lower(&l.expanded, &l.info).err().expect("lower refuses");
        assert!(plain.iter().any(|e| e.kind == K::Delay), "{src:?}");
    }
}

#[test]
fn a_model_without_a_delay_is_an_error_of_lower_dde() {
    let e = dde_errs("par r=1\nx'=-r*x\ndone\n");
    assert_eq!(e.len(), 1);
    assert_eq!(e[0].kind, K::Delay);
}

#[test]
fn lower_still_refuses_every_delay() {
    let src = "par tau=1\nx'=-delay(x,tau)\ndone\n";
    let l = loaded(src);
    let e = lower(&l.expanded, &l.info).err().expect("refused");
    assert_eq!(e.len(), 1);
    assert_eq!(e[0].kind, K::Delay);
    assert_eq!(&src[e[0].span.start..e[0].span.end], "delay(x,tau)");
}

#[test]
fn the_other_refusals_still_apply() {
    let l = loaded("x'=-x+ran(1)+delay(x,1)\ndone\n");
    let e = lower_dde(&l.expanded, &l.info).err().expect("refused");
    assert!(e.iter().any(|e| e.kind == K::SimulationOnly), "{e:?}");
}

/// The three evaluators of the lowered model see one parameter vector. Here the DAG and
/// `Interp` (through `Fallback`) agree on F, JacU and JacP, hidden columns included.
#[test]
fn dag_and_fallback_agree_with_hidden_parameters() {
    let src = "par a=0.7, b=1.3\n\
               f(u)=delay(x,a)*u\n\
               x'=-delay(y,a)+f(x)+delay(x,b)*x\n\
               y'=delay(x,b)-2*y+delay(y,0.5)^2\n\
               aux s=delay(y,0.5)+x\n\
               done\n";
    let l = loaded(src);
    let mut ir = lower_dde(&l.expanded, &l.info).expect("lowers");
    assert_eq!(ir.delays.len(), 4);
    let fb = Fallback::new(&l.expanded, &l.info, &ir);
    let n_p = ir.n_p();
    assert_eq!(n_p, 6);
    let points: [([f64; 2], [f64; 7]); 3] = [
        ([0.3, -0.4], [0.7, 1.3, 0.1, 0.2, 0.3, 0.4, 0.0]),
        ([1.0, 2.0], [0.7, 1.3, 1.0, 1.0, 1.0, 1.0, 5.0]),
        ([-0.5, 0.25], [0.1, 0.2, -2.0, 0.5, 3.0, -1.0, 0.0]),
    ];
    for kind in [
        EntryKind::F,
        EntryKind::Aux,
        EntryKind::JacU,
        EntryKind::JacP,
    ] {
        let entry = ir.entry(&kind).expect("entry");
        for (x, p) in &points {
            let want = run(&ir, &entry.outputs, x, p);
            let mut got = vec![0.0; want.len()];
            fb.eval(&kind, x, p, &[], &mut got);
            if kind == EntryKind::JacP {
                assert_eq!(got.len(), 2 * n_p);
            }
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let tol = 1e-14 * (1.0 + w.abs());
                assert!(
                    (g - w).abs() <= tol,
                    "{kind:?}[{i}] at {x:?} {p:?}: {g} vs {w}"
                );
            }
        }
    }
}

/// A hidden column of JacP is the derivative with respect to the delayed value.
#[test]
fn hidden_jacp_columns_are_the_derivatives_in_the_lags() {
    let mut m = dde("par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n");
    let e = m.entry(&EntryKind::JacP).expect("entry");
    // d f / d r, d f / d tau, d f / d lag at x = 0.5, lag = 0.25.
    let got = run(&m, &e.outputs, &[0.5], &[1.5, 1.0, 0.25, 0.0]);
    assert_eq!(got, [0.5 * 0.75, 0.0, -1.5 * 0.5]);
}
