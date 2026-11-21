//! The depth keys of the `auto` stack layout (SPEC section 8af-5): one right-hand side per
//! member on the CPU at `(x0, t0)`, with the kernel cap of 4,096 frames. Every model is our
//! own.

mod common;

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::*;
use pc_ensemble::order::{Depth, depth_keys};
use pc_ensemble::{Method, Params, Precision, Tape, run_cpu};

const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";
const FIB: &str =
    "par n=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\nx'=-x*fib(n)/(1+fib(n))\ndone\n";

/// The depth `base + spread * u - spread / 2` of member i, with u from a multiplicative hash
/// in [0, 1), clamped to [0, 4095]: the depths of the GPU sweeps.
fn depths(members: usize, base: f64, spread: f64) -> Vec<f64> {
    (0..members)
        .map(|i| {
            let u = ((i as u64 * 2_654_435_761) % 4_294_967_296) as f64 / 4_294_967_296.0;
            (base - spread / 2.0 + u * spread)
                .floor()
                .clamp(0.0, 4095.0)
        })
        .collect()
}

fn fib(k: u64) -> u64 {
    if k < 2 { k } else { fib(k - 1) + fib(k - 2) }
}

#[test]
fn the_keys_count_frames_as_the_recursion_enters_them() {
    let ks = [0.0, 1.0, 5.0, 4095.0, 4096.0, 5000.0];
    for (src, name) in [(SUM, "sum"), (FIB, "fib")] {
        let model = build_model(src);
        let tape = Tape::build(&model).expect("tape");
        for prec in [Precision::Double, Precision::Single] {
            let k: Vec<f64> = if name == "sum" {
                ks.to_vec()
            } else {
                vec![0.0, 1.0, 2.0, 10.0]
            };
            let x0 = vec![1.0; k.len()];
            let mut j = job(
                &model,
                k.len(),
                &x0,
                &[],
                0.01,
                Method::Rk4 { dt: 0.01 },
                prec,
            );
            j.params = Params::PerMember(&k);
            let got = depth_keys(&tape, &j, &AtomicBool::new(false))
                .expect("keys")
                .expect("a tape with calls");
            let want: Vec<Depth> = k
                .iter()
                .map(|&v| {
                    let v = v as u64;
                    if name == "sum" {
                        // s(v) enters v + 1 frames, one at a time, so its peak is v + 1, up to
                        // the cap, where the frame past it is refused.
                        let d = (v + 1).min(4096);
                        Depth {
                            peak: d as u32,
                            calls: d,
                        }
                    } else {
                        // fib(v) enters 2 fib(v + 1) - 1 frames and is v deep (1 for v < 2).
                        Depth {
                            peak: v.max(1) as u32,
                            calls: 2 * fib(v + 1) - 1,
                        }
                    }
                })
                .collect();
            assert_eq!(got, want, "{name} {prec:?}");
        }
    }
    // A tape without calls has no keys.
    let model = build_model("x'=-x\ndone\n");
    let tape = Tape::build(&model).expect("tape");
    let x0 = [1.0];
    let j = job(
        &model,
        1,
        &x0,
        &[],
        0.01,
        Method::Rk4 { dt: 0.01 },
        Precision::Double,
    );
    assert_eq!(
        depth_keys(&tape, &j, &AtomicBool::new(false)).expect("keys"),
        None
    );
}

#[test]
fn a_set_cancel_flag_stops_the_key_pass() {
    let model = build_model(SUM);
    let tape = Tape::build(&model).expect("tape");
    let k = depths(1000, 2048.0, 0.0);
    let x0 = vec![1.0; k.len()];
    let mut j = job(
        &model,
        k.len(),
        &x0,
        &[],
        0.01,
        Method::Rk4 { dt: 0.01 },
        Precision::Double,
    );
    j.params = Params::PerMember(&k);
    let e = depth_keys(&tape, &j, &AtomicBool::new(true)).expect_err("cancelled");
    assert!(matches!(e, pc_ensemble::EnsembleError::Cancelled), "{e:?}");
}

/// The cost of the key pass against one CPU RK4 step (four right-hand sides) per member, on
/// the sweep jobs. Run in release with `--ignored --nocapture`.
#[test]
#[ignore]
fn time_the_key_pass() {
    let members = 16_384;
    for (src, name, base, spread) in [
        (SUM, "sum", 2048.0, 2048.0),
        (SUM, "sum", 256.0, 0.0),
        (FIB, "fib", 10.0, 16.0),
    ] {
        let model = build_model(src);
        let tape = Tape::build(&model).expect("tape");
        let k = depths(members, base, spread);
        let x0 = vec![1.0; members];
        for prec in [Precision::Double, Precision::Single] {
            let mut j = job(
                &model,
                members,
                &x0,
                &[],
                0.01,
                Method::Rk4 { dt: 0.01 },
                prec,
            );
            j.params = Params::PerMember(&k);
            let never = AtomicBool::new(false);
            let _ = depth_keys(&tape, &j, &never).expect("warm");
            let t = Instant::now();
            let reps = 5;
            for _ in 0..reps {
                let _ = depth_keys(&tape, &j, &never).expect("keys");
            }
            let keys_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t = Instant::now();
            let _ = run_cpu(&j, u64::MAX, &never).expect("run");
            let step_ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{name} {base}+-{} {prec:?}: keys {keys_ms:.1} ms, one RK4 step {step_ms:.1} ms, ratio {:.3}",
                spread / 2.0,
                keys_ms / step_ms
            );
        }
    }
}
