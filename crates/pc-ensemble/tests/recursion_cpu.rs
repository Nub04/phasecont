//! Recursive function calls on the CPU ensemble (SPEC section 8af-5, stage 4a). A member
//! whose right-hand side goes past the frame cap ends with `recursion_depth`, and the other
//! members go on with the same bits as a run without it. Every model is our own.

mod common;

use common::*;
use pc_ensemble::{Method, Output, Params, Precision, Status};

/// `s(n)` counts down to zero, so `s(k) = k` for a whole `k` and the call takes `k + 1`
/// frames. `x' = -s(k) x / k` is `x' = -x` written through the call.
const COUNTED: &str = "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n";
const PLAIN: &str = "par k=3\nx'=-k*x/k\ndone\n";

/// Past the default cap of 1,000,000 frames.
const DEEP: f64 = 2e6;

#[test]
fn a_member_past_the_cap_ends_alone_and_the_others_keep_their_bits() {
    let calls = build_model(COUNTED);
    let plain = build_model(PLAIN);
    let x0 = [1.0, 2.0, 3.0];
    let k = [3.0, DEEP, 5.0];
    for precision in [Precision::Double, Precision::Single] {
        let single = precision == Precision::Single;
        for method in [
            Method::Rk4 { dt: 0.01 },
            if single { DP5_SINGLE } else { DP5_STD },
        ] {
            let what = format!("{method:?} {precision:?}");
            let mut j = job(&calls, 3, &x0, &[], 1.0, method, precision);
            j.params = Params::PerMember(&k);
            j.output = Output::Samples(vec![0.5, 1.0]);
            let r = run(&j);
            assert_eq!(
                r.status,
                [Status::Ok, Status::RecursionDepth, Status::Ok],
                "{what}"
            );
            assert_eq!(r.t_reached[1], 0.0, "{what}");
            assert_eq!(r.accepted[1], 0, "{what}");
            assert!(
                r.state(1, 0)[0].is_nan() && r.state(1, 1)[0].is_nan(),
                "{what}"
            );
            // The members that finish match the call-free model bit for bit.
            let mut p = job(&plain, 3, &x0, &[], 1.0, method, precision);
            p.params = Params::PerMember(&k);
            p.output = Output::Samples(vec![0.5, 1.0]);
            let want = run(&p);
            for m in [0, 2] {
                for s in 0..2 {
                    assert_eq!(
                        bits(r.state(m, s)),
                        bits(want.state(m, s)),
                        "{what} {m} {s}"
                    );
                }
                assert_eq!(r.accepted[m], want.accepted[m], "{what} {m}");
            }
        }
    }
}

#[test]
fn a_member_that_reaches_the_cap_mid_run_keeps_its_last_accepted_state() {
    // `x' = 1` from 0.4, and the depth of `y'` grows with `x`: past 1,000,000 frames once
    // `x` passes 2/3, so the stage at t = 0.3 fails and the member stops at t = 0.2.
    let m = build_model("s(n)=if(n<1)then(0)else(1+s(n-1))\nx'=1\ny'=s(x*1.5e6)*0\ndone\n");
    let r = run(&job(
        &m,
        1,
        &[0.4, 0.0],
        &[],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    ));
    assert_eq!(r.status, [Status::RecursionDepth]);
    assert_eq!(r.accepted[0], 2);
    assert!((r.t_reached[0] - 0.2).abs() < 1e-12, "{}", r.t_reached[0]);
}

#[test]
fn a_dp5_member_that_reaches_the_cap_mid_run_is_not_a_rejected_step() {
    // The same model as above under DP5 with steps of at most 0.05. A stage past x = 2/3
    // fails; the step must end the member with `recursion_depth`, not count as a rejected
    // step that shrinks `h` until `step_too_small`, and no NaN may reach a stored sample.
    let m = build_model("s(n)=if(n<1)then(0)else(1+s(n-1))\nx'=1\ny'=s(x*1.5e6)*0\ndone\n");
    for (precision, rtol, atol, eps) in [
        (Precision::Double, 1e-8, 1e-10, 1e-12),
        (Precision::Single, 1e-4, 1e-6, 1e-6),
    ] {
        let what = format!("{precision:?}");
        let method = Method::Dp5 {
            rtol,
            atol,
            hmax: 0.05,
        };
        let mut j = job(&m, 1, &[0.4, 0.0], &[], 1.0, method, precision);
        j.output = Output::Samples(vec![0.1, 0.2, 0.5, 1.0]);
        let r = run(&j);
        assert_eq!(r.status, [Status::RecursionDepth], "{what}");
        let t = r.t_reached[0];
        // The last accepted step ends before x = 2/3, and at most one step of 0.05 before.
        assert!(
            t > 2.0 / 3.0 - 0.4 - 0.05 - eps && t < 2.0 / 3.0 - 0.4,
            "{what} {t}"
        );
        assert!(r.accepted[0] > 0, "{what}");
        assert_eq!(r.rejected[0], 0, "{what}");
        for (s, ts) in [(0, 0.1), (1, 0.2)] {
            let y = r.state(0, s);
            assert!(
                (y[0] - (0.4 + ts)).abs() < eps && y[1] == 0.0,
                "{what} {s} {y:?}"
            );
        }
        for s in [2, 3] {
            assert!(r.state(0, s).iter().all(|v| v.is_nan()), "{what} {s}");
        }
    }
}

#[test]
fn a_map_member_past_the_cap_ends_alone() {
    let calls =
        build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx(t+1)=s(k)*x/k/2\ndone\n");
    let plain = build_model("par k=3\nx(t+1)=k*x/k/2\ndone\n");
    let x0 = [1.0, 2.0];
    let k = [3.0, DEEP];
    let mut j = job(&calls, 2, &x0, &[], 10.0, Method::Iter, Precision::Double);
    j.params = Params::PerMember(&k);
    let r = run(&j);
    assert_eq!(r.status, [Status::Ok, Status::RecursionDepth]);
    assert_eq!(r.t_reached[1], 0.0);
    let mut p = job(&plain, 2, &x0, &[], 10.0, Method::Iter, Precision::Double);
    p.params = Params::PerMember(&k);
    assert_eq!(bits(r.state(0, 0)), bits(run(&p).state(0, 0)));
    assert_eq!(r.state(0, 0)[0], 1.0 / 1024.0);
}

/// A cancel inside one right-hand side that recurses for minutes returns `Cancelled`, not a
/// result whose members read `recursion_depth`.
#[test]
fn a_cancel_inside_a_long_recursive_right_hand_side_returns_cancelled() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};
    // fib(60) makes about 3e12 calls, at most 60 frames deep.
    let m =
        build_model("par k=60\nfib(n)=if(n<2)then(n)else(fib(n-1)+fib(n-2))\nx'=-fib(k)*x\ndone\n");
    for members in [1, 4] {
        let x0 = vec![1.0; members];
        let j = job(
            &m,
            members,
            &x0,
            &[60.0],
            1.0,
            Method::Rk4 { dt: 0.1 },
            Precision::Double,
        );
        let flag = AtomicBool::new(false);
        let t0 = Instant::now();
        let res = std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(Duration::from_millis(100));
                flag.store(true, Ordering::Relaxed);
            });
            pc_ensemble::run_cpu(&j, u64::MAX, &flag)
        });
        assert_eq!(
            res.map(|r| r.status).unwrap_err(),
            pc_ensemble::EnsembleError::Cancelled,
            "{members}"
        );
        assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
    }
}

#[test]
fn recursion_depth_has_code_six() {
    assert_eq!(Status::RecursionDepth.code(), 6);
    assert_eq!(Status::RecursionDepth.name(), "recursion_depth");
    assert_eq!(Status::from_code(6), Some(Status::RecursionDepth));
    assert_eq!(Status::from_code(7), None);
}

/// Nanoseconds per right-hand side of the call tape in both precisions against the DAG
/// evaluator, on the same models as the stage 2 timing in `pc-jit/tests/recursion.rs`. Run in
/// release with `--ignored --nocapture`.
#[test]
#[ignore = "timing"]
fn timing_of_the_call_tape() {
    use pc_ensemble::tape::CpuTape;
    use pc_ir::EntryKind;
    use pc_ir::eval::try_eval_entry;
    use std::hint::black_box;
    use std::time::Instant;
    const PW: &str = "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n";
    const SUM: &str = "par k=3\nsum(n)=if(n<1)then(0)else(n+sum(n-1))\nx'=sum(k)\ndone\n";
    let cases: [(&str, &str, &[f64]); 2] = [
        ("pw", PW, &[3.0, 30.0, 300.0, 1000.0]),
        ("sum", SUM, &[100.0, 10_000.0, 100_000.0]),
    ];
    for (name, src, depths) in cases {
        let mut m = build_model(src);
        let tape = pc_ensemble::Tape::build(&m).unwrap();
        let (t64, t32) = (CpuTape::<f64>::new(&tape), CpuTape::<f32>::new(&tape));
        let (mut s64, mut s32) = (t64.slab(), t32.slab());
        let e = m.entry(&EntryKind::F).unwrap();
        for &n in depths {
            let reps = (20_000_000.0 / n).clamp(20.0, 20_000.0) as u32;
            let (x, p) = ([1.000_1], [n, 0.0]);
            let (xf, pf) = ([1.000_1f32], [n as f32, 0.0]);
            let mut out = [0.0];
            let mut outf = [0.0f32];
            let time = |f: &mut dyn FnMut()| {
                let t = Instant::now();
                for _ in 0..reps {
                    f();
                }
                t.elapsed().as_secs_f64() / f64::from(reps) * 1e9
            };
            let a = time(&mut || {
                t64.eval(black_box(&x), &p, &mut s64, &mut out);
                black_box(out);
            });
            let b = time(&mut || {
                t32.eval(black_box(&xf), &pf, &mut s32, &mut outf);
                black_box(outf);
            });
            let c = time(&mut || {
                try_eval_entry(&m.dag, &e, black_box(&x), &p, &[], &mut out).unwrap();
                black_box(out);
            });
            assert!(!s64.failed() && !s32.failed());
            println!("{name} n={n:<7} F: tape f64 {a:.0} ns, tape f32 {b:.0} ns, dag {c:.0} ns");
        }
    }
}
