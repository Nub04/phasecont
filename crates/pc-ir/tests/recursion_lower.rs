//! Acceptance tests of SPEC section 8af-5 that go from `.ode` source through `lower`:
//! closed-form values, derivatives, call counts, the depth cap and the refusals.

use pc_ir::eval::try_eval_entry;
use pc_ir::machine::{self, EvalError, EvalStats, MAX_ACTIVE_CALLS};
use pc_ir::{EntryKind, IrErrorKind, IrModel, Slot, lower, lower_dde};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn load_src(src: &str, dde: bool) -> Result<IrModel, Vec<pc_ir::IrError>> {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errors: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{src}: {errors:?}");
    if dde {
        lower_dde(&l.expanded, &l.info)
    } else {
        lower(&l.expanded, &l.info)
    }
}

fn ir(src: &str) -> IrModel {
    load_src(src, false).unwrap_or_else(|e| panic!("{src}: {e:?}"))
}

/// Evaluates one entry at state `x` with the parameters `p` (t is appended as 0).
fn run(
    ir: &mut IrModel,
    kind: EntryKind,
    x: &[f64],
    p: &[f64],
    v: &[f64],
) -> (Vec<f64>, EvalStats) {
    let e = ir
        .entry(&kind)
        .unwrap_or_else(|err| panic!("{kind:?}: {err:?}"));
    let mut p = p.to_vec();
    p.push(0.0);
    let mut out = vec![f64::NAN; e.outputs.len()];
    let st = try_eval_entry(&ir.dag, &e, x, &p, v, &mut out).expect("evaluates");
    (out, st)
}

fn ulp_gap(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

const PW: &str = "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n";
const FX: &str = "par k=3\nf(n)=if(n<1)then(1)else(x*f(n-1))\nx'=f(k)\ndone\n";

#[test]
fn fact_is_exact_up_to_22() {
    let mut m = ir("par k=3\nfact(n)=if(n<=1)then(1)else(n*fact(n-1))\nx'=fact(k)\ndone\n");
    let mut want: u128 = 1;
    for n in 1..=22u32 {
        want *= u128::from(n);
        let (out, _) = run(&mut m, EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(out[0], want as f64, "{n}!");
    }
}

#[test]
fn sum_reaches_the_cap_and_one_more_fails() {
    // The machine keeps its frames on the heap, so a small native stack is enough.
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(|| {
            let mut m = ir("par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n");
            let (out, st) = run(&mut m, EntryKind::F, &[0.0], &[999_999.0], &[]);
            assert_eq!(out[0], 999_999.0 * 1_000_000.0 / 2.0);
            assert_eq!(st.max_depth, MAX_ACTIVE_CALLS);

            let e = m.entry(&EntryKind::F).unwrap();
            let mut out = [f64::NAN];
            let err = try_eval_entry(&m.dag, &e, &[0.0], &[1_000_000.0, 0.0], &[], &mut out)
                .expect_err("one call past the cap");
            assert!(
                matches!(&err, EvalError::Depth { func, cap } if func == "sum" && *cap == MAX_ACTIVE_CALLS),
                "{err:?}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn pw_gives_the_power_and_its_derivatives_to_1_ulp() {
    let mut m = ir(PW);
    for x in [1.5f64, 0.5] {
        for n in 0..=30i32 {
            let nf = f64::from(n);
            let p = [nf];
            let (f, _) = run(&mut m, EntryKind::F, &[x], &p, &[]);
            let (j, _) = run(&mut m, EntryKind::JacU, &[x], &p, &[]);
            let (d, _) = run(
                &mut m,
                EntryKind::Dir(vec![Slot::U, Slot::U]),
                &[x],
                &p,
                &[1.0, 1.0],
            );
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
                    g == w || ulp_gap(g, w) <= 1,
                    "{what} at x={x} n={n}: {g:?} vs {w:?}"
                );
            }
        }
    }
}

#[test]
fn mutual_recursion_gives_parity() {
    let mut m = ir("par k=3\nev(n)=if(n==0)then(1)else(od(n-1))\n\
                    od(n)=if(n==0)then(0)else(ev(n-1))\nx'=ev(k)\naux q=od(k)\ndone\n");
    for n in 0..40u32 {
        let (f, _) = run(&mut m, EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        let (a, _) = run(&mut m, EntryKind::Aux, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(f[0], f64::from(u8::from(n % 2 == 0)), "ev({n})");
        assert_eq!(a[0], f64::from(u8::from(n % 2 == 1)), "od({n})");
    }
}

#[test]
fn body_that_reads_a_state() {
    let mut m = ir(FX);
    for n in 0..=20i32 {
        let nf = f64::from(n);
        let (f, _) = run(&mut m, EntryKind::F, &[1.5], &[nf], &[]);
        let (j, _) = run(&mut m, EntryKind::JacU, &[1.5], &[nf], &[]);
        assert_eq!(f[0], 1.5f64.powi(n), "n={n}");
        let w1 = if n >= 1 { nf * 1.5f64.powi(n - 1) } else { 0.0 };
        assert!(
            j[0] == w1 || ulp_gap(j[0], w1) <= 1,
            "n={n}: {:?} vs {w1:?}",
            j[0]
        );
    }
}

#[test]
fn derivatives_make_n_plus_1_calls() {
    for n in [100.0, 1000.0] {
        let want = n as u64 + 1;
        // x near 1 keeps x^1000 finite.
        let x = [1.000_1];
        let mut m = ir(PW);
        let (_, st) = run(&mut m, EntryKind::JacU, &x, &[n], &[]);
        assert_eq!(st.calls, want, "JacU of pw at n={n}");
        let (_, st) = run(
            &mut m,
            EntryKind::Dir(vec![Slot::U, Slot::U]),
            &x,
            &[n],
            &[1.0, 1.0],
        );
        assert_eq!(st.calls, want, "Dir[U,U] of pw at n={n}");
        let mut m = ir(FX);
        let (_, st) = run(&mut m, EntryKind::JacU, &x, &[n], &[]);
        assert_eq!(st.calls, want, "JacU of f at n={n}");
    }
}

#[test]
fn nested_if_gives_min_of_n_and_5() {
    let mut m = ir(
        "par k=3\nh(n)=if(n<1)then(0)else(if(n>5)then(h(n-1))else(1+h(n-1)))\n\
                    x'=h(k)\ndone\n",
    );
    for n in 0..30u32 {
        let (f, _) = run(&mut m, EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(f[0], f64::from(n.min(5)), "h({n})");
    }
}

#[test]
fn shared_call_in_two_arms_runs_once_per_frame() {
    let mut m = ir(
        "par k=3\ng(n)=(if(n>0)then(g(n-1))else(0))+(if(n>5)then(g(n-1))else(1))\n\
                    x'=g(k)\ndone\n",
    );
    for n in 0..40u32 {
        let (f, st) = run(&mut m, EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        let want = if n <= 5 {
            f64::from(n + 1)
        } else {
            6.0 * 2f64.powi(n as i32 - 5)
        };
        assert_eq!(f[0], want, "g({n})");
        assert_eq!(st.calls, u64::from(n) + 1, "calls of g({n})");
    }
}

#[test]
fn untaken_arm_makes_no_call() {
    let mut m = ir("par k=-1\nf(n)=if(n<1)then(1)else(x*f(n-1))\n\
                    x'=if(k<0)then(0)else(f(k))\ndone\n");
    let (f, st) = run(&mut m, EntryKind::F, &[2.0], &[-1.0], &[]);
    assert_eq!(f[0], 0.0);
    assert_eq!(st.calls, 0);
    let (f, st) = run(&mut m, EntryKind::F, &[2.0], &[3.0], &[]);
    assert_eq!(f[0], 8.0);
    assert_eq!(st.calls, 4);
}

#[test]
fn table_formula_that_calls_a_recursive_function_is_refused() {
    let src = "f(n)=if(n<1)then(1)else(2*f(n-1))\ntable tb % 3 0 2 f(t)\nx'=tb(x)\ndone\n";
    let Err(errs) = load_src(src, false) else {
        panic!("a table formula renumbers the states")
    };
    assert!(
        errs.iter().any(|e| e.kind == IrErrorKind::BadTable),
        "{errs:?}"
    );
}

#[test]
fn delay_inside_a_recursive_body() {
    let src = "par k=3\nf(n)=if(n<1)then(1)else(delay(x,1)*f(n-1))\nx'=f(k)\ndone\n";
    let mut m = load_src(src, true).expect("lower_dde");
    assert_eq!(m.delays.len(), 1);
    assert!(!m.dag.units().is_empty());
    // p = [k, delay slot], then t.
    let (f, st) = run(&mut m, EntryKind::F, &[0.0], &[3.0, 1.5], &[]);
    assert_eq!(f[0], 1.5f64.powi(3));
    assert_eq!(st.calls, 4);
}

#[test]
fn model_without_recursion_has_no_units() {
    let m = ir("par a=2\nf(u)=u*u+a\nx'=f(x)-f(a)\ndone\n");
    assert!(m.dag.units().is_empty());
}

/// A formal used as an exponent inside a jet body: the Dir row of `x^e` needs the partial
/// of `Arg(e)`, which is 0, and must build instead of being refused.
#[test]
fn dir_of_a_formal_exponent() {
    let mut m = ir("par k=2\nf(e,n)=if(n<=0)then(x^e)else(f(e,n-1))\nx'=f(3,k)\ndone\n");
    for x in [0.7f64, 1.1, 2.0] {
        let (d, _) = run(
            &mut m,
            EntryKind::Dir(vec![Slot::U, Slot::U]),
            &[x],
            &[2.0],
            &[1.0, 1.0],
        );
        let w = 6.0 * x;
        assert!(
            (d[0] - w).abs() <= 1e-14 * w.abs(),
            "x={x}: {:?} vs {w:?}",
            d[0]
        );
        let (j, _) = run(&mut m, EntryKind::JacU, &[x], &[2.0], &[]);
        let w = 3.0 * x * x;
        assert!(
            (j[0] - w).abs() <= 1e-14 * w.abs(),
            "x={x}: {:?} vs {w:?}",
            j[0]
        );
    }
}

// ---- machine::run_unit: the entry point of the JIT helper ----

fn unit_named(ir: &IrModel, name: &str) -> u32 {
    ir.dag
        .units()
        .iter()
        .position(|u| u.name == name)
        .unwrap_or_else(|| panic!("no unit {name}")) as u32
}

#[test]
fn run_unit_matches_the_entry_run_bit_for_bit() {
    let mut m = ir(FX);
    let u = unit_named(&m, "f");
    for n in 0..=25i32 {
        let nf = f64::from(n);
        let (f, st) = run(&mut m, EntryKind::F, &[1.5], &[nf], &[]);
        let mut out = [f64::NAN];
        let ust = machine::run_unit(
            &m.dag,
            u,
            &[nf],
            &[1.5],
            &[nf, 0.0],
            &[],
            &mut out,
            MAX_ACTIVE_CALLS,
        )
        .expect("runs");
        assert_eq!(out[0].to_bits(), f[0].to_bits(), "n={n}");
        assert_eq!(ust.max_depth, st.max_depth, "n={n}");
    }
}

#[test]
fn run_unit_leaves_nothing_behind() {
    // Two runs of different units, a failed one in between, then the first again: the global
    // flags set by one run must not show up as values in the next.
    let mut m = ir("par k=3\nfact(n)=if(n<=1)then(1)else(n*fact(n-1))\n\
                    pw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=fact(k)+pw(x,k)\ndone\n");
    let fu = unit_named(&m, "fact");
    let pu = unit_named(&m, "pw");
    let _ = m.entry(&EntryKind::F).unwrap();
    let go = |u: u32, args: &[f64], x: f64, cap: usize| {
        let mut out = [f64::NAN];
        let r = machine::run_unit(&m.dag, u, args, &[x], &[3.0, 0.0], &[], &mut out, cap);
        (r, out[0])
    };
    for round in 0..3 {
        let (r, v) = go(fu, &[10.0], 2.0, MAX_ACTIVE_CALLS);
        r.expect("fact");
        assert_eq!(v, 3_628_800.0, "round {round}");
        let (r, v) = go(pu, &[1.5, 6.0], 2.0, MAX_ACTIVE_CALLS);
        r.expect("pw");
        assert_eq!(v, 1.5f64.powi(6), "round {round}");
        // A cap error part way down: every slot is NaN, and the next call is good.
        let (r, v) = go(fu, &[10.0], 2.0, 4);
        assert!(matches!(r, Err(EvalError::Depth { .. })), "{r:?}");
        assert!(v.is_nan());
    }
}

#[test]
fn run_unit_survives_a_big_dag_then_a_small_one_then_a_big_one() {
    let mut big = ir(FX);
    let mut other = ir(PW);
    let _ = (
        big.entry(&EntryKind::F).unwrap(),
        other.entry(&EntryKind::F).unwrap(),
    );
    let (bu, ou) = (unit_named(&big, "f"), unit_named(&other, "pw"));
    let call = |m: &IrModel, u: u32, args: &[f64], x: f64, n: f64| {
        let mut out = [f64::NAN];
        machine::run_unit(
            &m.dag,
            u,
            args,
            &[x],
            &[n, 0.0],
            &[],
            &mut out,
            MAX_ACTIVE_CALLS,
        )
        .expect("runs");
        out[0]
    };
    for _ in 0..3 {
        assert_eq!(call(&big, bu, &[7.0], 1.5, 7.0), 1.5f64.powi(7));
        assert_eq!(call(&other, ou, &[1.5, 4.0], 1.5, 4.0), 1.5f64.powi(4));
    }
}

#[test]
fn run_unit_rejects_a_short_output_slice_and_a_bad_unit() {
    let mut m = ir(FX);
    let u = unit_named(&m, "f");
    let _ = m.entry(&EntryKind::F).unwrap();
    let mut none: [f64; 0] = [];
    let r = machine::run_unit(&m.dag, u, &[2.0], &[1.5], &[2.0, 0.0], &[], &mut none, 100);
    assert!(matches!(r, Err(EvalError::Malformed(_))), "{r:?}");
    let mut out = [f64::NAN];
    let r = machine::run_unit(&m.dag, 99, &[2.0], &[1.5], &[2.0, 0.0], &[], &mut out, 100);
    assert!(matches!(r, Err(EvalError::Malformed(_))), "{r:?}");
    assert!(out[0].is_nan());
    // The buffers are still good afterwards.
    machine::run_unit(&m.dag, u, &[2.0], &[1.5], &[2.0, 0.0], &[], &mut out, 100).unwrap();
    assert_eq!(out[0], 2.25);
}

#[test]
fn run_unit_does_not_keep_a_value_of_the_last_state() {
    // `f` reads the state `x` in its body, and `x` changes between runs of the same thread.
    let mut m = ir(FX);
    let u = unit_named(&m, "f");
    let _ = m.entry(&EntryKind::F).unwrap();
    for x in [1.5, 0.5, 2.0, 1.5, -3.0] {
        let mut out = [f64::NAN];
        machine::run_unit(&m.dag, u, &[5.0], &[x], &[5.0, 0.0], &[], &mut out, 100).unwrap();
        assert_eq!(out[0], f64::powi(x, 5), "x={x}");
    }
}
