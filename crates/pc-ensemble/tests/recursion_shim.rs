//! Recursive function calls in the ensemble kernel text (SPEC section 8af-5, stage 4b). The
//! kernel runs a call tape on a per-member stack of 4,096 frames, so each case runs the CPU
//! backend under the same cap and the kernel, compiled as plain C, must give the same bits,
//! statuses and counts. Every model is our own.

mod common;

use common::shim::check_in;
use common::*;
use pc_ensemble::kernel::Layout;
use pc_ensemble::tape::with_max_calls;
use pc_ensemble::{EnsembleResult, Job, Method, Output, Params, Precision, Status};

/// The frames of the GPU stack, which the CPU tape takes as its cap in these tests.
const FRAMES: usize = 4096;

const BOTH: [Precision; 2] = [Precision::Single, Precision::Double];

fn methods(p: Precision) -> [Method; 2] {
    let dp5 = match p {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.05 }, dp5]
}

fn tag(name: &str, p: Precision, m: Method, l: Layout) -> String {
    let kind = match m {
        Method::Rk4 { .. } => "rk4",
        Method::Dp5 { .. } => "dp5",
        Method::Iter => "iter",
    };
    let l = match l {
        Layout::Private => "priv",
        Layout::Global => "glob",
    };
    format!("{name}-{}-{kind}-{l}", p.name())
}

/// The kernel against the CPU tape at the GPU cap. `None` after a loud skip.
fn same(what: &str, j: &Job, l: Layout) -> Option<EnsembleResult> {
    with_max_calls(FRAMES, || check_in(what, j, l))
}

/// Runs one model with per-member `k` under RK4 and DP5 in both precisions and both
/// layouts, and returns the statuses of each run.
fn each(name: &str, src: &str, x0: &[f64], k: &[f64]) -> Vec<Vec<Status>> {
    let m = build_model(src);
    let mut seen = Vec::new();
    for p in BOTH {
        for method in methods(p) {
            for l in [Layout::Private, Layout::Global] {
                let mut j = job(&m, k.len(), x0, &[], 1.0, method, p);
                j.params = Params::PerMember(k);
                let Some(r) = same(&tag(name, p, method, l), &j, l) else {
                    return seen;
                };
                seen.push(r.status);
            }
        }
    }
    seen
}

#[test]
fn fib_matches_the_cpu() {
    let src = "par k=5\nfib(n)=if(n<2)then(n)else(fib(n-1)+fib(n-2))\nx'=-x*fib(k)/8\ndone\n";
    let k = [2.0, 3.0, 5.0, 6.0, 7.0];
    for s in each("fib", src, &[1.0, 0.9, 0.8, 0.7, 0.6], &k) {
        assert_eq!(s, [Status::Ok; 5]);
    }
}

#[test]
fn mutual_recursion_with_different_arities_matches_the_cpu() {
    // `ev` takes two arguments and `od` one; `od` passes the state to `ev`.
    let src = "par k=3\nev(n,a)=if(n<1)then(a)else(od(n-1))\n\
               od(n)=if(n<1)then(0)else(ev(n-1,x))\nx'=-x*(1+ev(k,0.5))\ndone\n";
    let k = [0.0, 1.0, 2.0, 3.0, 6.0];
    for s in each("evod", src, &[1.0, 0.9, 0.8, 0.7, 0.6], &k) {
        assert_eq!(s, [Status::Ok; 5]);
    }
}

#[test]
fn a_body_that_reads_a_state_matches_the_cpu() {
    let src = "par k=2\nf(n)=if(n<1)then(1)else(x*f(n-1))\nx'=-f(k)\ndone\n";
    let k = [0.0, 1.0, 2.0, 3.0];
    for s in each("state", src, &[0.5, 0.6, 0.7, 0.8], &k) {
        assert_eq!(s, [Status::Ok; 4]);
    }
}

#[test]
fn an_argument_free_call_inside_a_recursive_body_matches_the_cpu() {
    // `s(3)` in the body of `f` does not depend on `n`, so the tape evaluates it once per
    // right-hand side through a global subroutine called from inside a frame of `f`.
    let src = "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\n\
               f(n)=if(n<1)then(1)else(f(n-1)+s(3)*x)\nx'=-x*f(k)/10\ndone\n";
    let k = [0.0, 1.0, 4.0, 9.0];
    for s in each("argfree", src, &[1.0, 0.9, 0.8, 0.7], &k) {
        assert_eq!(s, [Status::Ok; 4]);
    }
}

#[test]
fn a_call_shared_by_both_arms_matches_the_cpu() {
    // `h(n-1)` sits in both arms of the inner `if`, so the tape evaluates it in a lazy local
    // subroutine of the frame.
    let src = "par k=3\nh(n)=if(n<1)then(0)else(if(n>2)then(h(n-1)*2)else(h(n-1)+1))\n\
               x'=-x*h(k)/20\ndone\n";
    let k = [0.0, 1.0, 3.0, 6.0];
    for s in each("shared", src, &[1.0, 0.9, 0.8, 0.7], &k) {
        assert_eq!(s, [Status::Ok; 4]);
    }
}

#[test]
fn a_global_call_that_reenters_itself_ends_alone() {
    // At k >= 1 the body of `f` needs `f(3)`, which needs itself: the CPU tape recurses to
    // its cap and the kernel finds the subroutine already running. Both end the member with
    // `recursion_depth`.
    let src = "par k=0\nf(n)=if(n<1)then(1)else(f(3))\nx'=-x*f(k)\ndone\n";
    let k = [0.0, 3.0, 0.0];
    for s in each("reenter", src, &[1.0, 0.9, 0.8], &k) {
        assert_eq!(s, [Status::Ok, Status::RecursionDepth, Status::Ok]);
    }
}

#[test]
fn the_kernel_admits_exactly_4096_frames() {
    // `s(4095)` takes 4,096 frames and `s(4096)` one more.
    let src = "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n";
    let k = [4095.0, 4096.0, 5.0];
    for s in each("cap", src, &[1.0, 2.0, 3.0], &k) {
        assert_eq!(s, [Status::Ok, Status::RecursionDepth, Status::Ok]);
    }
}

#[test]
fn a_dp5_member_that_reaches_the_cap_mid_run_matches_the_cpu() {
    // `x' = 1` from 0.4 and the depth of `y'` passes 4,096 frames once `x` passes 0.68, so a
    // stage fails mid-run. The member ends with `recursion_depth` and no rejected step.
    let m = build_model("s(n)=if(n<1)then(0)else(1+s(n-1))\nx'=1\ny'=s(x*6000)*0\ndone\n");
    for (p, rtol, atol) in [
        (Precision::Double, 1e-8, 1e-10),
        (Precision::Single, 1e-4, 1e-6),
    ] {
        let method = Method::Dp5 {
            rtol,
            atol,
            hmax: 0.05,
        };
        for l in [Layout::Private, Layout::Global] {
            let mut j = job(&m, 2, &[0.4, 0.0, 0.1, 0.0], &[], 0.5, method, p);
            j.output = Output::Samples(vec![0.1, 0.2, 0.5]);
            let what = tag("midrun", p, method, l);
            let Some(r) = same(&what, &j, l) else {
                return;
            };
            assert_eq!(r.status, [Status::RecursionDepth, Status::Ok], "{what}");
            assert_eq!(r.rejected[0], 0, "{what}");
            assert!(r.accepted[0] > 0, "{what}");
        }
        // RK4 reaches the cap inside a step too.
        let mut j = job(
            &m,
            2,
            &[0.4, 0.0, 0.1, 0.0],
            &[],
            0.5,
            Method::Rk4 { dt: 0.1 },
            p,
        );
        j.output = Output::Samples(vec![0.1, 0.2, 0.5]);
        let what = tag("midrun", p, j.method, Layout::Private);
        let Some(r) = same(&what, &j, Layout::Private) else {
            return;
        };
        assert_eq!(r.status, [Status::RecursionDepth, Status::Ok], "{what}");
        assert_eq!(r.accepted[0], 2, "{what}");
    }
}

#[test]
fn a_map_member_past_the_cap_matches_the_cpu() {
    let m = build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx(t+1)=s(k)*x/k/2\ndone\n");
    let k = [3.0, 5000.0, 4095.0];
    for p in BOTH {
        for l in [Layout::Private, Layout::Global] {
            let mut j = job(&m, 3, &[1.0, 2.0, 3.0], &[], 10.0, Method::Iter, p);
            j.params = Params::PerMember(&k);
            let what = tag("map", p, Method::Iter, l);
            let Some(r) = same(&what, &j, l) else {
                return;
            };
            assert_eq!(
                r.status,
                [Status::Ok, Status::RecursionDepth, Status::Ok],
                "{what}"
            );
        }
    }
}

#[test]
fn the_stack_is_sized_by_the_largest_frame() {
    use pc_ensemble::Tape;
    use pc_ensemble::kernel::stack_len;
    let plain = Tape::build(&build_model("par k=3\nx'=-k*x\ndone\n")).unwrap();
    assert_eq!(stack_len(&plain), 0);
    let calls = Tape::build(&build_model(
        "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n",
    ))
    .unwrap();
    let len = stack_len(&calls);
    assert!(len > 0 && len.is_multiple_of(FRAMES), "{len}");
}

#[test]
fn every_dialect_passes_the_stack_to_every_right_hand_side() {
    use pc_codegen::kernel::{Dialect, Scalar};
    use pc_ensemble::Tape;
    use pc_ensemble::kernel::{Integrator, kernel_source_in};
    let tape = Tape::build(&build_model(
        "par k=3\nfib(n)=if(n<2)then(n)else(fib(n-1)+fib(n-2))\nx'=-x*fib(k)/8\ndone\n",
    ))
    .unwrap();
    for d in [Dialect::C, Dialect::OpenCl, Dialect::Cuda, Dialect::Msl] {
        for i in [Integrator::Rk4, Integrator::Dp5, Integrator::Iter] {
            for s in [Scalar::F32, Scalar::F64] {
                // Metal has no double.
                if d == Dialect::Msl && s == Scalar::F64 {
                    continue;
                }
                for l in [Layout::Private, Layout::Global] {
                    let what = format!("{d:?} {i:?} {s:?} {l:?}");
                    let k = kernel_source_in(&tape, i, s, d, l).expect(&what);
                    assert!(k.contains("#define PC_RECURSION_DEPTH 6u"), "{what}");
                    let calls = k
                        .lines()
                        .filter(|t| !t.starts_with("PC_FN"))
                        .filter(|t| {
                            ["pc_rhs(", "pc_hinit(", "pc_dp5_step("]
                                .iter()
                                .any(|f| t.contains(f))
                        })
                        .filter(|t| t.ends_with(");"))
                        .collect::<Vec<_>>();
                    assert!(!calls.is_empty(), "{what}");
                    for c in calls {
                        assert!(c.ends_with(", stk, cerr);"), "{what}: {c}");
                    }
                    assert!(k.contains("stk + m * PC_STK"), "{what}");
                    if d == Dialect::Msl {
                        assert!(k.contains("[[buffer(12)]]"), "{what}");
                    }
                }
            }
        }
    }
}
