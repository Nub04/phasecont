//! SPEC 8af-5, stage 2: the JIT runs recursive functions. Every entry is compared with the DAG
//! evaluator bit for bit, and the closed forms of the recursion tests give the values.

use pc_ir::eval::{try_eval_entry, try_eval_entry_err};
use pc_ir::machine::{EvalError, MAX_ACTIVE_CALLS};
use pc_ir::{Entry, EntryKind, IrModel, Slot, lower, lower_dde};
use pc_jit::{CompiledModel, JitError};
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

fn ulp_gap(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// Same bits, except that a NaN matches any NaN.
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

const PW: &str = "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n";
const FX: &str = "par k=3\nf(n)=if(n<1)then(1)else(x*f(n-1))\nx'=f(k)\ndone\n";

/// A model with its entries built on the IR side and the JIT on the same DAG.
struct Pair {
    cm: CompiledModel,
    entries: Vec<(EntryKind, Entry)>,
}

impl Pair {
    fn new(src: &str, kinds: &[EntryKind]) -> Pair {
        Pair::of(ir(src), kinds)
    }

    fn of(mut m: IrModel, kinds: &[EntryKind]) -> Pair {
        let entries = kinds
            .iter()
            .map(|k| {
                (
                    k.clone(),
                    m.entry(k).unwrap_or_else(|e| panic!("{k:?}: {e:?}")),
                )
            })
            .collect();
        Pair {
            cm: CompiledModel::new(m),
            entries,
        }
    }

    fn entry(&self, kind: &EntryKind) -> &Entry {
        &self
            .entries
            .iter()
            .find(|(k, _)| k == kind)
            .unwrap_or_else(|| panic!("{kind:?} was not built"))
            .1
    }

    /// The JIT result, with `t` appended to the parameters.
    fn jit(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        par: &[f64],
        v: &[f64],
    ) -> Result<Vec<f64>, JitError> {
        let mut p = par.to_vec();
        p.push(0.0);
        let mut out = vec![f64::NAN; self.entry(kind).outputs.len()];
        self.cm.call(kind, x, &p, v, &mut out)?;
        Ok(out)
    }

    /// The DAG evaluator result at the same point.
    fn dag(
        &self,
        kind: &EntryKind,
        x: &[f64],
        par: &[f64],
        v: &[f64],
    ) -> Result<Vec<f64>, EvalError> {
        let mut p = par.to_vec();
        p.push(0.0);
        let e = self.entry(kind);
        let mut out = vec![f64::NAN; e.outputs.len()];
        try_eval_entry(&self.cm.ir().dag, e, x, &p, v, &mut out)?;
        Ok(out)
    }

    /// The JIT result, checked to have the bits of the DAG evaluator.
    fn both(&mut self, kind: &EntryKind, x: &[f64], par: &[f64], v: &[f64]) -> Vec<f64> {
        let want = self.dag(kind, x, par, v).expect("dag evaluates");
        let got = self.jit(kind, x, par, v).expect("jit evaluates");
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                same_bits(*g, *w),
                "{kind:?}[{i}] at x={x:?} p={par:?}: jit {g:?} dag {w:?}"
            );
        }
        got
    }
}

fn dir2() -> EntryKind {
    EntryKind::Dir(vec![Slot::U, Slot::U])
}

#[test]
fn fact_is_exact_up_to_22() {
    let mut pr = Pair::new(
        "par k=3\nfact(n)=if(n<=1)then(1)else(n*fact(n-1))\nx'=fact(k)\ndone\n",
        &[EntryKind::F],
    );
    let mut want: u128 = 1;
    for n in 1..=22u32 {
        want *= u128::from(n);
        let out = pr.both(&EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(out[0], want as f64, "{n}!");
    }
}

#[test]
fn pw_gives_the_power_and_its_derivatives_to_1_ulp() {
    let mut pr = Pair::new(PW, &[EntryKind::F, EntryKind::JacU, dir2()]);
    for x in [1.5f64, 0.5] {
        for n in 0..=30i32 {
            let nf = f64::from(n);
            let f = pr.both(&EntryKind::F, &[x], &[nf], &[]);
            let j = pr.both(&EntryKind::JacU, &[x], &[nf], &[]);
            let d = pr.both(&dir2(), &[x], &[nf], &[1.0, 1.0]);
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
    let mut pr = Pair::new(
        "par k=3\nev(n)=if(n==0)then(1)else(od(n-1))\n\
         od(n)=if(n==0)then(0)else(ev(n-1))\nx'=ev(k)\naux q=od(k)\ndone\n",
        &[EntryKind::F, EntryKind::Aux],
    );
    for n in 0..40u32 {
        let f = pr.both(&EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        let a = pr.both(&EntryKind::Aux, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(f[0], f64::from(u8::from(n % 2 == 0)), "ev({n})");
        assert_eq!(a[0], f64::from(u8::from(n % 2 == 1)), "od({n})");
    }
}

#[test]
fn body_that_reads_a_state() {
    let mut pr = Pair::new(FX, &[EntryKind::F, EntryKind::JacU]);
    for n in 0..=20i32 {
        let nf = f64::from(n);
        let f = pr.both(&EntryKind::F, &[1.5], &[nf], &[]);
        let j = pr.both(&EntryKind::JacU, &[1.5], &[nf], &[]);
        assert_eq!(f[0], 1.5f64.powi(n), "n={n}");
        let w1 = if n >= 1 { nf * 1.5f64.powi(n - 1) } else { 0.0 };
        assert!(
            j[0] == w1 || ulp_gap(j[0], w1) <= 1,
            "n={n}: {:?} vs {w1:?}",
            j[0]
        );
    }
    // The state moves between calls of the same entry: no value of the first call may leak
    // into the second.
    for (x, n) in [(2.0, 5.0), (3.0, 5.0), (2.0, 5.0), (0.5, 7.0)] {
        let f = pr.both(&EntryKind::F, &[x], &[n], &[]);
        assert_eq!(f[0], x.powi(n as i32));
    }
}

#[test]
fn nested_if_gives_min_of_n_and_5() {
    let mut pr = Pair::new(
        "par k=3\nh(n)=if(n<1)then(0)else(if(n>5)then(h(n-1))else(1+h(n-1)))\n\
         x'=h(k)\ndone\n",
        &[EntryKind::F],
    );
    for n in 0..30u32 {
        let f = pr.both(&EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        assert_eq!(f[0], f64::from(n.min(5)), "h({n})");
    }
}

#[test]
fn shared_call_in_two_arms() {
    let mut pr = Pair::new(
        "par k=3\ng(n)=(if(n>0)then(g(n-1))else(0))+(if(n>5)then(g(n-1))else(1))\n\
         x'=g(k)\ndone\n",
        &[EntryKind::F],
    );
    for n in 0..30u32 {
        let f = pr.both(&EntryKind::F, &[0.0], &[f64::from(n)], &[]);
        let want = if n <= 5 {
            f64::from(n + 1)
        } else {
            6.0 * 2f64.powi(n as i32 - 5)
        };
        assert_eq!(f[0], want, "g({n})");
    }
}

#[test]
fn untaken_arm_makes_no_call() {
    // `f(0)` never ends: it is the arm that must not run when k < 0.
    let src = "par k=-1\nf(n)=if(n<0)then(1)else(f(n+1))\n\
               x'=if(k<0)then(0)else(f(0))\ndone\n";
    let mut pr = Pair::new(src, &[EntryKind::F, EntryKind::JacU]);
    let f = pr.both(&EntryKind::F, &[2.0], &[-1.0], &[]);
    assert_eq!(f[0], 0.0);
    let j = pr.both(&EntryKind::JacU, &[2.0], &[-1.0], &[]);
    assert_eq!(j[0], 0.0);
    // The taken arm runs and reaches the cap.
    match pr.jit(&EntryKind::F, &[2.0], &[1.0], &[]) {
        Err(JitError::Eval(EvalError::Depth { func, cap })) => {
            assert_eq!((func.as_str(), cap), ("f", MAX_ACTIVE_CALLS));
        }
        other => panic!("{other:?}"),
    }
    // An error leaves the entry usable.
    let f = pr.both(&EntryKind::F, &[2.0], &[-1.0], &[]);
    assert_eq!(f[0], 0.0);
}

#[test]
fn untaken_arm_with_a_real_body_still_matches_the_dag() {
    let src = "par k=-1\nf(n)=if(n<1)then(1)else(x*f(n-1))\n\
               x'=if(k<0)then(0)else(f(k))\ndone\n";
    let mut pr = Pair::new(src, &[EntryKind::F, EntryKind::JacU, dir2()]);
    for k in [-1.0, 0.0, 1.0, 3.0, 6.0] {
        for x in [2.0, 0.5, -1.5] {
            pr.both(&EntryKind::F, &[x], &[k], &[]);
            pr.both(&EntryKind::JacU, &[x], &[k], &[]);
            pr.both(&dir2(), &[x], &[k], &[1.0, 1.0]);
        }
    }
    let f = pr.both(&EntryKind::F, &[2.0], &[3.0], &[]);
    assert_eq!(f[0], 8.0);
}

/// Shapes of `If` in the entry itself: nested, a call used in an arm and again after the
/// merge, and an `If` as the argument of a call.
#[test]
fn if_in_the_entry_matches_the_dag() {
    let defs = "par k=3\nf(n)=if(n<1)then(1)else(x*f(n-1))\ng(n)=if(n<1)then(0)else(n+g(n-1))\n";
    for (name, rhs) in [
        (
            "nested",
            "x'=if(k<0)then(0)else(if(k>5)then(f(k)*2)else(f(k)+1))",
        ),
        ("reused", "x'=(if(k<0)then(0)else(f(k)))+f(k)"),
        ("argument", "x'=f(if(k<0)then(0)else(g(k)))"),
        (
            "two ifs",
            "x'=(if(k<2)then(g(k))else(f(k)))*(if(k>4)then(f(k))else(g(k)))",
        ),
    ] {
        let src = format!("{defs}{rhs}\ndone\n");
        let mut pr = Pair::new(&src, &[EntryKind::F, EntryKind::JacU, dir2()]);
        for k in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0] {
            for x in [1.1, 0.9, -0.7] {
                pr.both(&EntryKind::F, &[x], &[k], &[]);
                pr.both(&EntryKind::JacU, &[x], &[k], &[]);
                pr.both(&dir2(), &[x], &[k], &[1.0, 1.0]);
            }
        }
        let _ = name;
    }
}

#[test]
fn delay_inside_a_recursive_body() {
    let src = "par k=3\nf(n)=if(n<1)then(1)else(delay(x,1)*f(n-1))\nx'=f(k)\ndone\n";
    let m = load_src(src, true).expect("lower_dde");
    assert_eq!(m.delays.len(), 1);
    let mut pr = Pair::of(m, &[EntryKind::F]);
    // p = [k, delay slot], then t.
    let f = pr.both(&EntryKind::F, &[0.0], &[3.0, 1.5], &[]);
    assert_eq!(f[0], 1.5f64.powi(3));
}

#[test]
fn sum_reaches_the_cap_and_one_more_fails() {
    let mut pr = Pair::new(
        "par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n",
        &[EntryKind::F],
    );
    let out = pr.both(&EntryKind::F, &[0.0], &[999_999.0], &[]);
    assert_eq!(out[0], 999_999.0 * 1_000_000.0 / 2.0);
    match pr.jit(&EntryKind::F, &[0.0], &[1_000_000.0], &[]) {
        Err(JitError::Eval(EvalError::Depth { func, cap })) => {
            assert_eq!((func.as_str(), cap), ("sum", MAX_ACTIVE_CALLS));
        }
        other => panic!("{other:?}"),
    }
    // The next call is clean: no error flag, no frames left from the failed one.
    let out = pr.both(&EntryKind::F, &[0.0], &[10.0], &[]);
    assert_eq!(out[0], 55.0);
}

#[test]
fn error_agrees_with_the_dag_error() {
    let mut pr = Pair::new(
        "par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n",
        &[EntryKind::F],
    );
    let jit = pr.jit(&EntryKind::F, &[0.0], &[2_000_000.0], &[]);
    let dag = pr.dag(&EntryKind::F, &[0.0], &[2_000_000.0], &[]);
    match (jit, dag) {
        (Err(JitError::Eval(a)), Err(b)) => assert_eq!(a, b),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_model_without_recursion_still_has_no_call_helper() {
    // A call-free model compiles through the same entry and runs as before.
    let mut pr = Pair::new("par a=2\nf(u)=u*u+a\nx'=f(x)-f(a)\ndone\n", &[EntryKind::F]);
    let out = pr.both(&EntryKind::F, &[3.0], &[2.0], &[]);
    assert_eq!(out[0], 9.0 + 2.0 - 6.0);
}

#[test]
fn the_public_compile_still_refuses_calls() {
    let mut m = ir(FX);
    let e = m.entry(&EntryKind::F).unwrap();
    match pc_jit::compile(&[(EntryKind::F, &e)], &m.dag) {
        Err(JitError::Unsupported(t)) => assert!(t.contains("recursive"), "{t}"),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("a plain compile must refuse a call"),
    }
}

#[test]
fn check_rows_of_a_recursive_model_pass() {
    use pc_ir::Fallback;
    for src in [PW, FX] {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        let m = ir(src);
        let fb = Fallback::new(&l.expanded, &l.info, &m);
        let rows = pc_jit::check::jit_rows(m, &fb, 6, 7, 0.5);
        assert!(!rows.is_empty());
        for r in &rows {
            assert!(r.pass, "{}", r.text);
        }
    }
}

/// Cost of one call into a recursive entry, against the DAG evaluator. Run with
/// `cargo test --release -p pc-jit --test recursion -- --ignored --nocapture`.
#[test]
#[ignore = "timing"]
fn timing_of_a_recursive_entry() {
    use std::time::Instant;
    const SUM: &str = "par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n";
    let cases: [(&str, &str, &[f64], &[EntryKind]); 2] = [
        (
            "pw",
            PW,
            &[3.0, 30.0, 300.0, 1000.0],
            &[EntryKind::F, EntryKind::JacU],
        ),
        ("sum", SUM, &[100.0, 10_000.0, 100_000.0], &[EntryKind::F]),
    ];
    for (name, src, depths, kinds) in cases {
        let mut pr = Pair::new(src, kinds);
        for &n in depths {
            for kind in kinds {
                let (x, par) = ([1.000_1], [n]);
                // About 2e7 calls of the body per measurement, whatever the depth.
                let reps = (20_000_000.0 / n).clamp(20.0, 20_000.0) as u32;
                let t = Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(pr.jit(kind, &x, &par, &[]).unwrap());
                }
                let jit = t.elapsed().as_secs_f64() / f64::from(reps) * 1e9;
                let t = Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(pr.dag(kind, &x, &par, &[]).unwrap());
                }
                let dag = t.elapsed().as_secs_f64() / f64::from(reps) * 1e9;
                println!("{name} n={n:<7} {kind:?}: jit {jit:.0} ns, dag {dag:.0} ns");
            }
        }
    }
}
#[test]
fn error_bounds_are_not_needed_but_the_dag_error_entry_agrees() {
    // `try_eval_entry_err` is what the check rows use; it must see the same values.
    let mut pr = Pair::new(PW, &[EntryKind::F]);
    let e = pr.entry(&EntryKind::F).clone();
    let p = [3.0, 0.0];
    let (mut a, mut b) = ([f64::NAN], [f64::NAN]);
    try_eval_entry_err(&pr.cm.ir().dag, &e, &[1.5], &p, &[], &mut a, &mut b).unwrap();
    let got = pr.jit(&EntryKind::F, &[1.5], &[3.0], &[]).unwrap();
    assert_eq!(got[0].to_bits(), a[0].to_bits());
}
