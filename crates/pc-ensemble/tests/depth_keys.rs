//! The depth keys of the `auto` stack layout (SPEC section 8af-5): one right-hand side per
//! member on the CPU at `(x0, t0)`, with the kernel cap of 4,096 frames. Every model is our
//! own.

mod common;

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::*;
use pc_ensemble::order::{Depth, depth_keys, sort_order};
use pc_ensemble::{Method, Params, Precision, Tape, run_cpu};

const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";
const FIB: &str =
    "par n=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\nx'=-x*fib(n)/(1+fib(n))\ndone\n";
const PW: &str = "par n=0\npw(a,m)=if(m<1)then(1)else(a*pw(a,m-1))\nx'=-0.01*pw(x,n)\ndone\n";
/// A short tree beside a chain.
const TREE: &str = "par n=2,k=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\n\
s(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(fib(n)+s(k))/(1+fib(n)+s(k))\ndone\n";
/// The depth follows the state.
const DRIFT: &str =
    "par w=1\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=w+1e-12*s(flr(2048*(1+sin(x))))\ndone\n";

/// A fixed number in [0, 1) for member `i`.
fn u(i: usize) -> f64 {
    ((i as u64 * 2_654_435_761) % 4_294_967_296) as f64 / 4_294_967_296.0
}

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

/// The stable order of `keys` by (peak, calls), the sort key before 2026-11-26.
fn peak_first(keys: &[Depth]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_by_key(|&m| (keys[m as usize].peak, keys[m as usize].calls));
    order
}

/// Sorting by frames entered first gives the order that sorting by peak first gave on every
/// chain and on a lone tree, the jobs of the GPU sweeps, so only a tree beside a chain moves.
#[test]
fn the_frames_first_order_is_the_peak_first_order_except_on_a_tree_beside_a_chain() {
    let n = 4096;
    let ninety: Vec<f64> = (0..n)
        .map(|i| {
            if u(i * 7 + 3) < 0.1 {
                (u(i) * 4095.0).floor()
            } else {
                2048.0
            }
        })
        .collect();
    let tree: Vec<f64> = (0..n)
        .flat_map(|i| {
            [
                (2.0 + 13.0 * u(i * 5 + 1)).floor(),
                (201.0 * u(i * 11 + 2)).floor(),
            ]
        })
        .collect();
    let drift: Vec<f64> = (0..n).map(|i| 0.5 + 2.5 * u(i)).collect();
    let cases: [(&str, &str, f64, Vec<f64>, bool); 7] = [
        ("sum-scattered", SUM, 1.0, depths(n, 2048.0, 2048.0), true),
        (
            "sum-scattered-narrow",
            SUM,
            1.0,
            depths(n, 256.0, 8.0),
            true,
        ),
        ("ninety-ten", SUM, 1.0, ninety, true),
        ("pw-scattered", PW, 0.999, depths(n, 2048.0, 2048.0), true),
        ("fib-scattered", FIB, 1.0, depths(n, 10.0, 16.0), true),
        ("drift-scattered", DRIFT, 0.0, drift, true),
        ("tree-wide", TREE, 1.0, tree, false),
    ];
    for (name, src, x0, params, same) in cases {
        let model = build_model(src);
        let tape = Tape::build(&model).expect("tape");
        let x0 = vec![x0; n];
        for prec in [Precision::Double, Precision::Single] {
            let mut j = job(&model, n, &x0, &[], 0.2, Method::Rk4 { dt: 0.01 }, prec);
            j.params = Params::PerMember(&params);
            let keys = depth_keys(&tape, &j, &AtomicBool::new(false))
                .expect("keys")
                .expect("a tape with calls");
            let new = sort_order(&keys).unwrap_or_else(|| (0..n as u32).collect());
            let old = peak_first(&keys);
            assert_eq!(new == old, same, "{name} {prec:?}");
        }
    }
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
