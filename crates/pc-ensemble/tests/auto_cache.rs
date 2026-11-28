//! The kernels of an `auto` run on CUDA (SPEC section 8af-5): a batch that stays on warp
//! runs the kernel of an explicit `warp` run and compiles only the probe beside it, and a
//! batch that the keys put on member runs the kernel of an explicit `member` run. Before that,
//! the compile gate runs a new model as `member` until the member kernel has repaid the warp
//! and probe kernels. Every model is our own.
//!
//! A machine with no CUDA device skips with a `SKIPPED` line, and `PC_REQUIRE_GPU=cuda` turns
//! the skip into a failure. The compile counter and the repay factor are process-wide, so the
//! steps run in one test.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::cuda::{auto_gate_passed, kernel_compiles};
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::order::{REPAY, set_repay_factor};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleResult, Job, Method, Params, Precision, StackLayout, StackRan};

/// A chain `s(n)` that enters `n + 1` frames, with the depth per member in `n`.
const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";

/// The same chain in a model of its own, so its kernels are not those of `SUM`.
const GATED: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(2+s(n))/(2+n)\ndone\n";

/// Puts the repay factor back, also when the test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        set_repay_factor(REPAY);
    }
}

fn on(device: &DeviceInfo, j: &Job, stack: StackLayout) -> EnsembleResult {
    let mut o = j.clone();
    o.stack = stack;
    run_on(device, &o, &AtomicBool::new(false))
        .unwrap_or_else(|e| panic!("{}: the device run failed: {e}", stack.name()))
}

#[test]
fn auto_reuses_the_kernels_of_warp_and_member() {
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
    let _reset = Reset;
    let x0: Vec<f64> = (0..64).map(|i| 1.0 + i as f64 / 64.0).collect();
    let p = Precision::Single;
    let method = Method::Rk4 { dt: 0.05 };

    // The compile gate at its factor of 10. Depths 100 to 107 in reverse, which `auto` sorts
    // once past the gate: until then each call runs the member kernel alone, unsorted.
    let gated = build_model(GATED);
    let reversed: Vec<f64> = (0..64).map(|i| 107.0 - (i / 8) as f64).collect();
    let mut j = job(&gated, 64, &x0, &[], 1.0, method, p);
    j.params = Params::PerMember(&reversed);
    let base = kernel_compiles();
    let first = on(&device, &j, StackLayout::Auto);
    assert!(!auto_gate_passed(), "the member kernel was not kept");
    assert_eq!(
        kernel_compiles(),
        base + 1,
        "the first call compiles member only"
    );
    assert_eq!(first.label.stack_ran, StackRan::Member);
    assert!(!first.label.sorted);
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(
        kernel_compiles(),
        base + 1,
        "auto compiled the member kernel"
    );
    assert_eq!(bits(&member.states), bits(&first.states));
    let second = on(&device, &j, StackLayout::Auto);
    assert!(
        !auto_gate_passed(),
        "two short runs do not repay two compiles"
    );
    assert_eq!(kernel_compiles(), base + 1);
    assert_eq!(second.label.stack_ran, StackRan::Member);
    assert_eq!(bits(&member.states), bits(&second.states));
    // With the factor at 0 the member kernel has repaid them.
    set_repay_factor(0.0);
    let third = on(&device, &j, StackLayout::Auto);
    assert!(auto_gate_passed());
    assert_eq!(
        kernel_compiles(),
        base + 3,
        "past the gate auto compiles the warp and probe kernels"
    );
    assert_eq!(third.label.stack_ran, StackRan::Warp);
    assert!(third.label.sorted);
    assert_eq!(bits(&member.states), bits(&third.states));

    // The factor stays 0 for the rest, so the gate passes once the member kernel is kept.
    let sum = build_model(SUM);

    // Depths 100 to 107 in order: the batch stays on warp.
    let narrow: Vec<f64> = (0..64).map(|i| 100.0 + (i / 8) as f64).collect();
    let mut j = job(&sum, 64, &x0, &[], 1.0, method, p);
    j.params = Params::PerMember(&narrow);
    let base = kernel_compiles();
    let warp = on(&device, &j, StackLayout::Warp);
    assert_eq!(kernel_compiles(), base + 1, "warp compiles its kernel");
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(kernel_compiles(), base + 2, "member compiles its kernel");
    assert_eq!(bits(&warp.states), bits(&member.states));
    let auto = on(&device, &j, StackLayout::Auto);
    assert_eq!(auto.label.stack_ran, StackRan::Warp);
    assert_eq!(
        kernel_compiles(),
        base + 3,
        "auto on warp compiles only the probe"
    );
    assert_eq!(bits(&warp.states), bits(&auto.states));
    on(&device, &j, StackLayout::Auto);
    assert_eq!(
        kernel_compiles(),
        base + 3,
        "a second auto run compiles nothing"
    );

    // Depths 0 to 1,890 in steps of 30: the keys put the batch on member.
    let wide: Vec<f64> = (0..64).map(|i| (i * 30) as f64).collect();
    let mut j = job(&sum, 64, &x0, &[], 1.0, method, p);
    j.params = Params::PerMember(&wide);
    let base = kernel_compiles();
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(
        kernel_compiles(),
        base,
        "member runs the kernel kept from the narrow depths"
    );
    let auto = on(&device, &j, StackLayout::Auto);
    assert_eq!(auto.label.stack_ran, StackRan::Member);
    assert_eq!(kernel_compiles(), base, "auto on member compiles nothing");
    assert_eq!(bits(&member.states), bits(&auto.states));

    // A model without calls has no stack: `auto` runs the member kernel with no key pass and
    // no probe, so it compiles nothing after a `member` run and gives the same bits.
    let plain = build_model("par k=1\nx'=-k*x\ndone\n");
    let k: Vec<f64> = (0..64).map(|i| 0.5 + i as f64 / 64.0).collect();
    let mut j = job(&plain, 64, &x0, &[], 1.0, method, p);
    j.params = Params::PerMember(&k);
    let base = kernel_compiles();
    let member = on(&device, &j, StackLayout::Member);
    assert_eq!(
        kernel_compiles(),
        base + 1,
        "member compiles the plain kernel"
    );
    let auto = on(&device, &j, StackLayout::Auto);
    assert_eq!(auto.label.stack, StackLayout::Auto);
    assert_eq!(auto.label.stack_ran, StackRan::Member);
    assert!(!auto.label.sorted);
    assert_eq!(
        kernel_compiles(),
        base + 1,
        "auto without calls compiles nothing"
    );
    assert_eq!(bits(&member.states), bits(&auto.states));
}
