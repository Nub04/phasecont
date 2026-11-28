//! The CUDA call stack is not cleared before a launch (SPEC section 8af-5), so it holds what an
//! earlier launch, frame or call left at each depth. Each case runs on stacks filled with
//! several bytes and on unset stacks, under the `member`, `warp` and `auto` stack layouts, and
//! the bits must not change with the fill. Fixed-step RK4 must also give the CPU's bits, and
//! the key kernel the CPU's keys. The models put frame values where a slot is easy to read
//! before it is written: a call shared by both arms of an `if` and read after it, a call in
//! one arm only, a call whose argument is a call, and mutual recursion between units with
//! frames of different sizes. Every model is our own.
//!
//! A machine with no CUDA device skips with a `SKIPPED` line, and `PC_REQUIRE_GPU=cuda` turns
//! the skip into a failure. All cases run in one function because the stack fill and the
//! repay factor are process-wide switches and tests in one binary run in parallel.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::cuda::{device_keys, set_stack_fill};
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::order::{REPAY, depth_keys, set_repay_factor};
use pc_ensemble::run::run_on;
use pc_ensemble::tape::with_max_calls;
use pc_ensemble::{EnsembleResult, Job, Method, Params, Precision, StackLayout, Tape};

/// The frames of the GPU stack, which the CPU tape takes as its cap here.
const FRAMES: usize = 4096;

/// `q(m)` is shared by both arms of the `if` in `b` and read again after it.
const SHARED: &str = "par n=0\nq(m)=if(m<1)then(0)else(1+q(m-1))\n\
b(m)=if(m<1)then(1)else((if(m>3)then(q(m)+b(m-1))else(2*q(m)))+q(m))\n\
x'=-x*b(n+flr(4*(1+sin(x))))/(1+b(n+flr(4*(1+sin(x)))))\ndone\n";

/// `q(m)` runs in one arm only.
const ONE_ARM: &str = "par n=0\nq(m)=if(m<1)then(0)else(1+q(m-1))\n\
v(m)=if(m<1)then(1)else(if(m>2)then(v(m-1)+q(m))else(m))\n\
x'=-x*v(n+flr(3*(1+sin(x))))/(1+v(n+flr(3*(1+sin(x)))))\ndone\n";

/// The argument of `c` is a call.
const NESTED: &str = "par n=0\na(m)=m-0.5\nc(m)=if(m<1)then(0)else(1+c(a(m-1)))\n\
x'=-x*c(n+flr(5*(1+cos(x))))/(1+c(n+flr(5*(1+cos(x)))))\ndone\n";

/// `ev` and `od` call each other, and their frames differ in size.
const MUTUAL: &str = "par n=0\nev(m)=if(m<1)then(1)else(od(m-1,0.01*m))\n\
od(m,w)=if(m<1)then(w)else(w*w+0.5*ev(m-1)+w)\n\
x'=-x*ev(n+flr(4*(1+sin(x))))/(1+ev(n+flr(4*(1+sin(x)))))\ndone\n";

/// The fills of the stack: two bytes, zero, and an unset stack, which holds what the runs
/// before it left.
const FILLS: [Option<u8>; 4] = [Some(0xFF), Some(0x41), Some(0x00), None];

/// Held for the whole test: the stack fill and the repay factor are process-wide.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Puts the fill and the repay factor back, also when the test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        set_stack_fill(if cfg!(debug_assertions) {
            Some(0xFF)
        } else {
            None
        });
        set_repay_factor(REPAY);
    }
}

/// A fixed number in [0, 1) for member `i`.
fn u(i: usize) -> f64 {
    ((i as u64 * 2_654_435_761) % 4_294_967_296) as f64 / 4_294_967_296.0
}

fn on(device: &DeviceInfo, what: &str, j: &Job, stack: StackLayout) -> EnsembleResult {
    let mut o = j.clone();
    o.stack = stack;
    run_on(device, &o, &AtomicBool::new(false))
        .unwrap_or_else(|e| panic!("{what} {}: the device run failed: {e}", stack.name()))
}

/// Every layout under every fill gives the bits of `member` on a stack of 0xFF bytes, and
/// RK4 gives the CPU's bits.
fn same_bits(device: &DeviceInfo, what: &str, j: &Job) {
    set_stack_fill(FILLS[0]);
    let want = on(device, what, j, StackLayout::Member);
    if let Method::Rk4 { .. } = j.method {
        let cpu = with_max_calls(FRAMES, || run(j));
        assert_eq!(cpu.status, want.status, "{what}: status against the CPU");
        assert_eq!(
            bits(&cpu.states),
            bits(&want.states),
            "{what}: states against the CPU"
        );
        assert_eq!(
            bits(&cpu.t_reached),
            bits(&want.t_reached),
            "{what}: t against the CPU"
        );
    }
    for fill in FILLS {
        set_stack_fill(fill);
        for stack in [StackLayout::Member, StackLayout::Warp, StackLayout::Auto] {
            let got = on(device, what, j, stack);
            let w = format!("{what} {} fill {fill:?}", stack.name());
            assert_eq!(want.status, got.status, "{w}: status");
            assert_eq!(bits(&want.states), bits(&got.states), "{w}: states");
            assert_eq!(bits(&want.t_reached), bits(&got.t_reached), "{w}: t");
            assert_eq!(want.accepted, got.accepted, "{w}: accepted");
            assert_eq!(want.rejected, got.rejected, "{w}: rejected");
        }
    }
}

/// Under every fill the key kernel gives the CPU keys.
fn same_keys(device: &DeviceInfo, what: &str, j: &Job) {
    let tape = Tape::build(j.model).expect("tape");
    let want = depth_keys(&tape, j, &AtomicBool::new(false))
        .expect("cpu keys")
        .expect("a tape with calls");
    for fill in FILLS {
        set_stack_fill(fill);
        for chunk in [32, j.members] {
            let got = device_keys(device, j, chunk)
                .unwrap_or_else(|e| panic!("{what} fill {fill:?}: the key kernel failed: {e}"))
                .expect("a tape with calls");
            assert_eq!(got, want, "{what} fill {fill:?} chunk {chunk}: keys");
        }
    }
}

#[test]
fn unset_call_stacks_give_the_bits_of_cleared_ones_on_a_cuda_device() {
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
    // The compile gate passes at once, so `auto` runs its sorted and `warp` batches.
    set_repay_factor(0.0);
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    } else {
        eprintln!("SKIPPED: double precision, which this device does not have");
    }
    let members = 200;
    // Depths from 0 to 60, scattered, so the lanes of a warp take different branches and
    // reuse frames at different depths.
    let n: Vec<f64> = (0..members).map(|i| (u(i) * 60.0).floor()).collect();
    let x0: Vec<f64> = (0..members).map(|i| 0.25 + 2.5 * u(i * 7 + 3)).collect();
    let models = [
        ("shared", build_model(SHARED)),
        ("one arm", build_model(ONE_ARM)),
        ("nested", build_model(NESTED)),
        ("mutual", build_model(MUTUAL)),
    ];
    for (name, model) in &models {
        for &p in &precisions {
            let dp5 = match p {
                Precision::Single => DP5_SINGLE,
                Precision::Double => DP5_STD,
            };
            for method in [Method::Rk4 { dt: 0.05 }, dp5] {
                let kind = if let Method::Rk4 { .. } = method {
                    "rk4"
                } else {
                    "dp5"
                };
                let what = format!("{name} {} {kind}", p.name());
                let mut j = job(model, members, &x0, &[], 0.5, method, p);
                j.params = Params::PerMember(&n);
                same_bits(&device, &what, &j);
                if kind == "rk4" {
                    same_keys(&device, &what, &j);
                }
            }
        }
    }
}
