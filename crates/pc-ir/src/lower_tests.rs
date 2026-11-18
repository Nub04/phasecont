use super::*;
use crate::dag::{Fn2, Op};
use crate::error::IrErrorKind as K;
use crate::eval::eval;
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn loaded(src: &str) -> pc_ode::Loaded {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    load(top, &mut map, &mut MapLoader::new())
}

/// Loads `src`, asserts the checker found no error, and lowers it.
fn ir(src: &str) -> IrModel {
    let l = loaded(src);
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}

fn errs(src: &str) -> Vec<IrError> {
    let l = loaded(src);
    match lower(&l.expanded, &l.info) {
        Ok(_) => panic!("expected lowering errors"),
        Err(e) => e,
    }
}

fn kinds(src: &str) -> Vec<K> {
    errs(src).iter().map(|e| e.kind).collect()
}

fn run(m: &IrModel, outs: &[NodeId], x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; outs.len()];
    eval(&m.dag, outs, x, p, &[], &mut out);
    out
}

fn has_op(m: &IrModel, pred: impl Fn(&Op) -> bool) -> bool {
    (0..m.dag.len()).any(|i| pred(m.dag.op(NodeId(i as u32))))
}

#[test]
fn refuses_no_equations() {
    assert_eq!(kinds("par a=1\ndone\n"), vec![K::NoEquations]);
}

#[test]
fn refuses_simulation_only() {
    let k = kinds("x'=-x\nglobal 1 x-1 {x=0}\ndone\n");
    assert_eq!(k, vec![K::SimulationOnly]);
    let src = "x'=-x+ran(1)\ndone\n";
    let e = errs(src);
    assert_eq!(e.len(), 1);
    assert_eq!(e[0].kind, K::SimulationOnly);
    assert_eq!(&src[e[0].span.start..e[0].span.end], "ran(1)");
    assert!(e[0].to_string().contains("SimulationOnly"));
}

#[test]
fn refuses_global_event_actions() {
    // `arret` and `out_put` are event actions in XPPAUT. The model must not
    // lower, so no path can run it and ignore the action.
    for action in ["arret=1", "out_put=1"] {
        let src = format!("x'=1\nx(0)=0\nglobal 1 x-0.5 {{{action}}}\ndone\n");
        assert_eq!(kinds(&src), vec![K::SimulationOnly], "{action}");
    }
}

#[test]
fn refuses_delay_with_call_span() {
    let src = "x'=-delay(x,1)\ndone\n";
    let e = errs(src);
    assert_eq!(e.len(), 1);
    assert_eq!(e[0].kind, K::Delay);
    assert_eq!(&src[e[0].span.start..e[0].span.end], "delay(x,1)");
}

#[test]
fn refuses_unsupported_statement() {
    let src = "x'=-x\nbndry x-1\ndone\n";
    let e = errs(src);
    assert!(e.iter().all(|e| e.kind == K::Unsupported), "{e:?}");
    assert_eq!(e.len(), 1);
}

#[test]
fn refuses_shift_and_del_shft_by_name() {
    for (src, call, msg) in [
        (
            "x'=-x+shift(x,1)\ndone\n",
            "shift(x,1)",
            "`shift` is not supported",
        ),
        (
            "x'=-x+del_shft(x,2,1)\ndone\n",
            "del_shft(x,2,1)",
            "`del_shft` is not supported",
        ),
    ] {
        let e = errs(src);
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].kind, K::Unsupported);
        assert_eq!(&src[e[0].span.start..e[0].span.end], call);
        assert!(e[0].to_string().contains(msg), "{}", e[0]);
    }
}

#[test]
fn refuses_bad_tables() {
    for hdr in ["1 0 1", "3 1 1", "3 2 1", "2000000 0 1", "3 0 1e999"] {
        let src = format!("x'=-x\ntable h % {hdr} t\ndone\n");
        let l = loaded(&src);
        if let Err(e) = lower(&l.expanded, &l.info) {
            assert!(e.iter().all(|e| e.kind == K::BadTable), "{hdr}: {e:?}");
        } else {
            panic!("{hdr} lowered");
        }
    }
    let k = kinds("x'=-x\ntable h % 2.5 0 1 t\ndone\n");
    assert_eq!(k, vec![K::BadTable]);
}

#[test]
fn refuses_too_large() {
    let l = loaded("x'=sin(x)*cos(x)+exp(x)\ndone\n");
    let e = match lower_with_cap(&l.expanded, &l.info, 3) {
        Ok(_) => panic!("expected TooLarge"),
        Err(e) => e,
    };
    assert!(e.iter().any(|e| e.kind == K::TooLarge));
}

#[test]
fn collects_several_errors() {
    let k = kinds("x'=delay(x,1)\ny'=delay(y,2)\ndone\n");
    assert_eq!(k, vec![K::Delay, K::Delay]);
}

#[test]
fn every_statement_kind() {
    let m = ir("par a=2, b=0.5\n\
                number c=3\n\
                !d=a*b\n\
                w=x*c\n\
                f(u,v)=u+v*arg1\n\
                table h % 3 0 2 t^2\n\
                x'=-a*x+w+d+h(1.5)+f(x,y)\n\
                dy/dt=y^3+x^(-2)+y^a\n\
                aux s=x+y\n\
                x(0)=1\n\
                init y=2\n\
                done\n");
    assert_eq!(m.n(), 2);
    assert_eq!(m.n_p(), 2);
    assert_eq!(m.state_names, ["x", "y"]);
    assert_eq!(m.param_names, ["a", "b"]);
    assert_eq!(m.aux_names, ["s"]);
    assert_eq!(m.init, [1.0, 2.0]);
    assert_eq!(m.par_values, [2.0, 0.5]);
    assert_eq!(m.kind, pc_ode::ModelKind::Ode);
    assert!(!m.smooth);
    let v = run(&m, &m.f, &[1.0, 2.0], &[2.0, 0.5, 0.0]);
    // x' = -2 + 3 + 1 + 2.5 + (1 + 2*1); y' = 8 + 1 + 4
    assert_eq!(v, [7.5, 13.0]);
    assert_eq!(run(&m, &m.aux, &[1.0, 2.0], &[2.0, 0.5, 0.0]), [3.0]);
    // The derived parameter follows p at evaluation time: d = 4 * 0.5.
    let v = run(&m, &m.f, &[1.0, 2.0], &[4.0, 0.5, 0.0]);
    assert_eq!(v[0], -4.0 + 3.0 + 2.0 + 2.5 + 3.0);
}

#[test]
fn map_model() {
    let m = ir("par r=3\nx(t+1)=r*x*(1-x)\nx(0)=0.5\ndone\n");
    assert_eq!(m.kind, pc_ode::ModelKind::Map);
    assert_eq!(m.init, [0.5]);
    assert_eq!(run(&m, &m.f, &[0.5], &[3.0, 0.0]), [0.75]);
    assert!(m.smooth);
}

#[test]
fn missing_values_are_zero() {
    let m = ir("par a, b=2\nx'=a*x\ny'=b\ndone\n");
    assert_eq!(m.par_values, [0.0, 2.0]);
    assert_eq!(m.init, [0.0, 0.0]);
}

#[test]
fn time_slot_and_autonomous() {
    let m = ir("par a=1\nx'=a*t*x\ndone\n");
    assert!(!m.autonomous);
    assert_eq!(run(&m, &m.f, &[2.0], &[1.0, 3.0]), [6.0]);
    assert!(has_op(&m, |o| *o == Op::Param(1)));

    let m = ir("par a=1\nx'=-a*x\nw=t\ndone\n");
    assert!(m.autonomous, "unused fixed quantity must not count");

    let m = ir("x'=-x\nw=t+1\naux q=w\ndone\n");
    assert!(!m.autonomous, "aux through a fixed quantity reads t");

    let m = ir("x'=-x\nf(u)=u*t\naux q=f(2)\ndone\n");
    assert!(!m.autonomous);
}

#[test]
fn pi_and_logic() {
    let m = ir("x'=pi*x\ny'=if(x>1)then(2)else(3)\nz'=(x>0)&(x<2)\nq'=not(x)|0\ndone\n");
    let v = run(&m, &m.f, &[0.5, 0.0, 0.0, 0.0], &[0.0]);
    assert_eq!(v[0], std::f64::consts::PI * 0.5);
    assert_eq!(v[1], 3.0);
    assert_eq!(v[2], 1.0);
    assert_eq!(v[3], 0.0);
}

#[test]
fn builtins_map() {
    let m = ir("x'=ln(x)\ny'=log(x)\nz'=flr(x)+mod(x,2)+max(x,1)+atan2(1,x)\ndone\n");
    assert_eq!(m.f[0], m.f[1]);
    let v = run(&m, &m.f, &[2.5, 0.0, 0.0], &[0.0]);
    assert_eq!(v[0], 2.5f64.ln());
    assert_eq!(v[2], 2.0 + 0.5 + 2.5 + 1.0f64.atan2(2.5));
}

#[test]
fn power_selection() {
    let m = ir("par a=2\nx'=x^3\ny'=x**16\nz'=x^17\nu'=x^2.5\nv'=x^a\nq'=x^(-2)\nw'=x^0\ndone\n");
    assert!(has_op(&m, |o| matches!(o, Op::PowI(_, 3))));
    assert!(has_op(&m, |o| matches!(o, Op::PowI(_, 16))));
    assert!(has_op(&m, |o| matches!(o, Op::PowI(_, -2))));
    // x^17, x^2.5 and x^a go through Pow.
    let pows = (0..m.dag.len())
        .filter(|&i| matches!(m.dag.op(NodeId(i as u32)), Op::Call2(Fn2::Pow, _, _)))
        .count();
    assert_eq!(pows, 3);
    let v = run(&m, &m.f, &[2.0; 7], &[2.0, 0.0]);
    assert_eq!(v[0], 8.0);
    assert_eq!(v[1], 65536.0);
    assert_eq!(v[3], crate::math::pow(2.0, 2.5));
    assert_eq!(v[4], 4.0);
    assert_eq!(v[5], 0.25);
    assert_eq!(v[6], 1.0);
}

#[test]
fn tables_in_declaration_order() {
    let m = ir("par a=2\n\
                table h % 3 0 2 t^2\n\
                table g % 3 0 1 a*t\n\
                table k % 3 0 2 h(t)\n\
                x'=h(x)+g(x)+k(x)\n\
                done\n");
    assert_eq!(m.dag.tables.len(), 3);
    assert_eq!(m.dag.tables[0], [3.0, 0.0, 2.0, 0.0, 1.0, 4.0]);
    assert_eq!(m.dag.tables[1], [3.0, 0.0, 1.0, 0.0, 1.0, 2.0]);
    assert_eq!(m.dag.tables[2], m.dag.tables[0]);
    // x = 1.5: h = 2.5, g = 2 * 1.5 = 3 (extrapolated), k = 2.5.
    assert_eq!(run(&m, &m.f, &[1.5], &[2.0, 0.0]), [8.0]);
    assert!(!m.smooth);
}

#[test]
fn table_formula_may_read_states_and_fixed_quantities() {
    let m = ir("w=x*2\ntable h % 3 0 1 t+w\nx'=h(0.5)\nx(0)=4\ndone\n");
    assert_eq!(m.dag.tables[0], [3.0, 0.0, 1.0, 8.0, 8.5, 9.0]);
    assert!(m.autonomous, "t inside a table is the table variable");
}

#[test]
fn array_models_lower() {
    let m = ir("%[1..3]\nx[j]'=-x[j]+[j]\nx[j](0)=[j]\n%\ndone\n");
    assert_eq!(m.n(), 3);
    assert_eq!(m.init, [1.0, 2.0, 3.0]);
    assert_eq!(run(&m, &m.f, &[1.0, 1.0, 1.0], &[0.0]), [0.0, 1.0, 2.0]);
}

/// A constant index `name[n]` lowers to the symbol `name<n>`. The values are
/// the ones XPPAUT 8.2.2 wrote for the `index-fixed-*` probes.
#[test]
fn constant_index_lowers_to_the_numbered_symbol() {
    // index-fixed-read: r = 7 + 3 + 8 + 100 = 118.
    let m = ir("x'=-x\npar a2=3\nf3=7\nk3=100\ng2(u)=u*2\n\
                aux r=f[3]+a[2]+g[2](4)+k[03]\ndone\n");
    assert_eq!(m.state_names, ["x"]);
    assert_eq!(m.param_names, ["a2"]);
    assert_eq!(m.aux_names, ["r"]);
    assert_eq!(run(&m, &m.aux, &[1.0], &[3.0]), [118.0]);

    // index-fixed-decl: the declared names are y1, z2, q2, a2, n2, f3, d2, g2,
    // h2 and v1, v2 (the block).
    let m = ir(
        "y[1]'=1\ndz[2]/dt=2\nq2'=0\ny1(0)=0\nz2(0)=0\ninit q[2]=4\npar a[2]=3\n\
         number n[2]=6\nf[3]=7\n!d[2]=n2+1\ng[2](u)=u*4\ntable h[2] % 3 0 2 t^2\n\
         aux s=f3+a2+g2(1)+y[1]+z[2]\naux w[2]=y1+3\naux e=d2+h2(1)+q[2]+g2(f[3])\n\
         %[1..2]\nv[j]=n[2]*[j]\n%\naux m=v[2]+v1\ndone\n",
    );
    assert_eq!(m.state_names, ["y1", "z2", "q2"]);
    assert_eq!(m.param_names, ["a2"]);
    assert_eq!(m.aux_names, ["s", "w2", "e", "m"]);
    assert_eq!(run(&m, &m.f, &[0.0, 0.0, 4.0], &[3.0]), [1.0, 2.0, 0.0]);
    assert_eq!(
        run(&m, &m.aux, &[0.0, 0.0, 4.0], &[3.0]),
        [14.0, 3.0, 40.0, 18.0]
    );

    // index-fixed-range-line: flux[3] is flux3 = 3*x, so y' = -3 at x = 1.
    let m = ir("flux[1..3]=[j]*x\ny'=-flux[3]\nx'=0\naux r=flux[2]+flux3\ndone\n");
    assert_eq!(m.state_names, ["y", "x"]);
    assert_eq!(run(&m, &m.f, &[0.0, 1.0], &[]), [-3.0, 0.0]);
    assert_eq!(run(&m, &m.aux, &[0.0, 1.0], &[]), [5.0]);
}

/// An aux that repeats a fixed name keeps its own formula; every other
/// reader of the name gets the fixed quantity. The values are the ones
/// XPPAUT 8.2.2 wrote for the `aux-fixed-*` probes at x = 1.
#[test]
fn aux_repeating_a_fixed_name_reads_the_fixed_quantity() {
    // Aux after the fixed quantity, and before it.
    for src in [
        "x'=-x\ny'=f\nf=2*x+1\naux f=7*x\naux g=f\ndone\n",
        "x'=-x\ny'=f\naux f=7*x\nf=2*x+1\naux g=f\ndone\n",
    ] {
        let m = ir(src);
        assert_eq!(m.aux_names, ["f", "g"], "{src:?}");
        // x' = -1, y' = f = 3; aux f = 7, aux g = f = 3.
        assert_eq!(run(&m, &m.f, &[1.0, 0.0], &[]), [-1.0, 3.0], "{src:?}");
        assert_eq!(run(&m, &m.aux, &[1.0, 0.0], &[]), [7.0, 3.0], "{src:?}");
    }
    // The aux formula reads the fixed quantity of the same name.
    for src in [
        "x'=-x\nf=2*x+1\naux f=f+1\ndone\n",
        "x'=-x\naux f=f+1\nf=2*x+1\ndone\n",
    ] {
        let m = ir(src);
        assert_eq!(m.aux_names, ["f"], "{src:?}");
        assert_eq!(run(&m, &m.aux, &[1.0], &[]), [4.0], "{src:?}");
    }
}
