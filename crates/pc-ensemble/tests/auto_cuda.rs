//! The `auto` stack layout on a real CUDA device (SPEC section 8af-5): the layout it picks per
//! batch, the member order, the fallback to `member` and the label, with the bits of a
//! `member` run for every member. Every model is our own.
//!
//! A machine with no CUDA device skips with a `SKIPPED` line, and `PC_REQUIRE_GPU=cuda` turns
//! the skip into a failure. All cases run in one function because `force_members` and the
//! repay factor are process-wide switches and tests in one binary run in parallel. The factor
//! is 0, so the compile gate passes as soon as the member kernel is kept; `auto_cache` tests
//! the gate itself.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::batch::force_members;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::order::{REPAY, set_repay_factor};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleResult, Job, Method, Params, Precision, StackLayout, StackRan, Status};

/// A chain `s(n)` that enters `n + 1` frames, with the depth per member in `n`.
const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";

/// The same chain at a depth that follows the state: every member starts at one depth, and
/// the depths spread as the members move at their own speed `w`.
const DRIFT: &str =
    "par w=1\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=w+1e-12*s(flr(256*(1+sin(x))))\ndone\n";

/// Held for the whole of each test: the batch switch and the repay factor are process-wide.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Puts the batch switch and the repay factor back, also when a test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_members(None);
        set_repay_factor(REPAY);
    }
}

fn on(device: &DeviceInfo, j: &Job, stack: StackLayout) -> EnsembleResult {
    let mut o = j.clone();
    o.stack = stack;
    run_on(device, &o, &AtomicBool::new(false))
        .unwrap_or_else(|e| panic!("{}: the device run failed: {e}", stack.name()))
}

/// Runs `j` under `auto` and under `member` and checks that every member has the same bits,
/// statuses and counts, then returns the label fields of the `auto` run.
fn auto(device: &DeviceInfo, what: &str, j: &Job) -> (StackRan, bool) {
    let want = on(device, j, StackLayout::Member);
    let got = on(device, j, StackLayout::Auto);
    assert_eq!(want.label.stack, StackLayout::Member, "{what}");
    assert_eq!(want.label.stack_ran, StackRan::Member, "{what}");
    assert!(!want.label.sorted, "{what}");
    assert_eq!(got.label.stack, StackLayout::Auto, "{what}");
    assert_eq!(want.status, got.status, "{what}: status");
    assert_eq!(bits(&want.states), bits(&got.states), "{what}: states");
    assert_eq!(bits(&want.t_reached), bits(&got.t_reached), "{what}: t");
    assert_eq!(want.accepted, got.accepted, "{what}: accepted");
    assert_eq!(want.rejected, got.rejected, "{what}: rejected");
    (got.label.stack_ran, got.label.sorted)
}

/// `0..len` in a fixed shuffled order.
fn shuffled(len: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..len).collect();
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..len).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.swap(i, (s % (i as u64 + 1)) as usize);
    }
    v
}

#[test]
fn auto_picks_the_layout_per_batch_on_a_cuda_device() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    eprintln!("cuda device 0: {} ({})", device.name, device.driver);
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _reset = Reset;
    set_repay_factor(0.0);
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    } else {
        eprintln!("SKIPPED: double precision, which this device does not have");
    }
    let sum = build_model(SUM);
    let rk4 = Method::Rk4 { dt: 0.05 };

    for &p in &precisions {
        let dp5 = match p {
            Precision::Single => DP5_SINGLE,
            Precision::Double => DP5_STD,
        };
        for method in [rk4, dp5] {
            let tag = |name: &str| {
                let kind = if method == rk4 { "rk4" } else { "dp5" };
                format!("{name} {} {kind}", p.name())
            };
            let x0: Vec<f64> = (0..64).map(|i| 1.0 + i as f64 / 64.0).collect();

            // Depths 100 to 107 in order: narrow groups, no reordering, warp.
            let narrow: Vec<f64> = (0..64).map(|i| 100.0 + (i / 8) as f64).collect();
            let mut j = job(&sum, 64, &x0, &[], 1.0, method, p);
            j.params = Params::PerMember(&narrow);
            assert_eq!(
                auto(&device, &tag("narrow"), &j),
                (StackRan::Warp, false),
                "{}",
                tag("narrow")
            );

            // The same depths and states in a shuffled order: sorted, still warp.
            let order = shuffled(64);
            let k: Vec<f64> = order.iter().map(|&i| narrow[i]).collect();
            let xs: Vec<f64> = order.iter().map(|&i| x0[i]).collect();
            let mut j = job(&sum, 64, &xs, &[], 1.0, method, p);
            j.params = Params::PerMember(&k);
            assert_eq!(
                auto(&device, &tag("shuffled"), &j),
                (StackRan::Warp, true),
                "{}",
                tag("shuffled")
            );

            // Depths 0 to 1,890 in steps of 30: every sorted group is wide, member.
            let wide: Vec<f64> = order.iter().map(|&i| (i * 30) as f64).collect();
            let mut j = job(&sum, 64, &xs, &[], 1.0, method, p);
            j.params = Params::PerMember(&wide);
            assert_eq!(
                auto(&device, &tag("wide"), &j),
                (StackRan::Member, true),
                "{}",
                tag("wide")
            );

            // Fewer than 32 members: member, in the caller's order.
            let mut j = job(&sum, 20, &xs[..20], &[], 1.0, method, p);
            j.params = Params::PerMember(&k[..20]);
            assert_eq!(
                auto(&device, &tag("small"), &j),
                (StackRan::Member, false),
                "{}",
                tag("small")
            );
        }

        // Below the floor: RK4 with 4 * 4 = 16 right-hand sides per member runs member,
        // with no key pass and so no reordering.
        let order = shuffled(64);
        let k: Vec<f64> = order.iter().map(|&i| 100.0 + (i / 8) as f64).collect();
        let x0 = vec![1.0; 64];
        let mut j = job(&sum, 64, &x0, &[], 0.2, rk4, p);
        j.params = Params::PerMember(&k);
        let what = format!("floor {}", p.name());
        assert_eq!(
            auto(&device, &what, &j),
            (StackRan::Member, false),
            "{what}"
        );
        // Five steps make 20 right-hand sides: the keys run, and the narrow groups pick warp.
        let mut j = job(&sum, 64, &x0, &[], 0.25, rk4, p);
        j.params = Params::PerMember(&k);
        let what = format!("at the floor {}", p.name());
        assert_eq!(auto(&device, &what, &j), (StackRan::Warp, true), "{what}");

        // Every member starts at depth 257, so the keys pick warp; the depths spread as the
        // members move apart, and the run falls back to member.
        let drift = build_model(DRIFT);
        let w: Vec<f64> = (0..64).map(|i| 0.5 + 2.5 * i as f64 / 64.0).collect();
        let x0 = vec![0.0; 64];
        let mut j = job(&drift, 64, &x0, &[], 5.0, Method::Rk4 { dt: 0.01 }, p);
        j.params = Params::PerMember(&w);
        let what = format!("drift {}", p.name());
        assert_eq!(auto(&device, &what, &j), (StackRan::Mixed, false), "{what}");

        // Members past the 4,096-frame stack among narrow ones: the keys run, and each such
        // member ends with `recursionDepth` in `auto` as in `member`.
        let order = shuffled(64);
        let k: Vec<f64> = order
            .iter()
            .map(|&i| {
                if i % 16 == 3 {
                    5000.0
                } else {
                    100.0 + (i / 8) as f64
                }
            })
            .collect();
        let x0 = vec![1.0; 64];
        let mut j = job(&sum, 64, &x0, &[], 1.0, rk4, p);
        j.params = Params::PerMember(&k);
        let what = format!("cap {}", p.name());
        auto(&device, &what, &j);
        let r = on(&device, &j, StackLayout::Auto);
        for (m, s) in r.status.iter().enumerate() {
            let want = if k[m] > 4095.0 {
                Status::RecursionDepth
            } else {
                Status::Ok
            };
            assert_eq!(*s, want, "{what}: member {m}");
        }

        // 140 members in two batches of 70: after sorting, the shallow half is narrow and
        // runs warp, the deep half is wide and runs member.
        let deep: Vec<f64> = (0..140)
            .map(|i| {
                if i < 70 {
                    10.0 + (i / 16) as f64
                } else {
                    200.0 + 25.0 * (i - 70) as f64
                }
            })
            .collect();
        let order = shuffled(140);
        let k: Vec<f64> = order.iter().map(|&i| deep[i]).collect();
        let x0: Vec<f64> = order.iter().map(|&i| 1.0 + i as f64 / 140.0).collect();
        let mut j = job(&sum, 140, &x0, &[], 1.0, rk4, p);
        j.params = Params::PerMember(&k);
        let whole = on(&device, &j, StackLayout::Member);
        force_members(Some(70));
        let what = format!("batches {}", p.name());
        assert_eq!(auto(&device, &what, &j), (StackRan::Mixed, true), "{what}");
        force_members(None);
        assert_eq!(
            bits(&whole.states),
            bits(&on(&device, &j, StackLayout::Auto).states)
        );
    }
}

/// Under `fma` the device compiler fuses on its own, and the member, warp and probe kernels
/// are separate sources. This pins that the three still compute the same bits, so an `auto`
/// run in `fma` does not depend on where the clock put the switch.
#[test]
fn fma_gives_the_same_bits_in_every_layout_on_a_cuda_device() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    if !device.fp64 {
        return skip_gpu("cuda", "fma runs only in double precision");
    }
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _reset = Reset;
    set_repay_factor(0.0);
    let rk4 = Method::Rk4 { dt: 0.05 };
    let members = 2048;
    let x0: Vec<f64> = (0..members).map(|i| 1.0 + i as f64 / 4096.0).collect();

    // A narrow chain stays on warp; a drifting one ends mixed.
    let sum = build_model(SUM);
    let narrow: Vec<f64> = (0..members).map(|i| 100.0 + (i / 64) as f64).collect();
    let mut j = job(&sum, members, &x0, &[], 1.0, rk4, Precision::Double);
    j.params = Params::PerMember(&narrow);
    j.math = pc_ensemble::Math::Fma;
    let warp = on(&device, &j, StackLayout::Warp);
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(bits(&warp.states), bits(&member.states), "narrow: warp");
    auto(&device, "narrow fma", &j);

    let drift = build_model(DRIFT);
    let w: Vec<f64> = (0..members).map(|i| 1.0 + (i % 97) as f64 / 97.0).collect();
    let mut j = job(&drift, members, &x0, &[], 4.0, rk4, Precision::Double);
    j.params = Params::PerMember(&w);
    j.math = pc_ensemble::Math::Fma;
    let warp = on(&device, &j, StackLayout::Warp);
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(bits(&warp.states), bits(&member.states), "drift: warp");
    let (ran, _) = auto(&device, "drift fma", &j);
    eprintln!("drift fma ran {}", ran.name());
}
