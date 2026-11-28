//! The depth keys of the `auto` stack layout from the key kernel on a CUDA device (SPEC
//! section 8af-5): the same peak and call count per member as the CPU keys of
//! `order::depth_keys`, in both precisions, for every chunk size. Every model is our own.
//!
//! A machine with no CUDA device skips with a `SKIPPED` line, and `PC_REQUIRE_GPU=cuda` turns
//! the skip into a failure.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::cuda::device_keys;
use pc_ensemble::device::{Backend, BackendState, find_device, list_devices};
use pc_ensemble::order::depth_keys;
use pc_ensemble::{Job, Method, Params, Precision, Tape};

const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";
const PW: &str = "par n=0\npw(a,m)=if(m<1)then(1)else(a*pw(a,m-1))\nx'=-0.01*pw(x,n)\ndone\n";
const FIB: &str =
    "par n=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\nx'=-x*fib(n)/(1+fib(n))\ndone\n";
const TREE: &str = "par n=2,k=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\n\
s(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(fib(n)+s(k))/(1+fib(n)+s(k))\ndone\n";
/// The depth follows the state.
const DRIFT: &str =
    "par w=1\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=w+1e-12*s(flr(2048*(1+sin(x))))\ndone\n";
const PLAIN: &str = "par a=1\nx'=-a*x\ndone\n";

/// A fixed number in [0, 1) for member `i`.
fn u(i: usize) -> f64 {
    ((i as u64 * 2_654_435_761) % 4_294_967_296) as f64 / 4_294_967_296.0
}

/// The CPU and device keys of `j`, for every chunk size in `chunks`, are the same.
fn same_keys(device: &pc_ensemble::device::DeviceInfo, what: &str, j: &Job, chunks: &[usize]) {
    let tape = Tape::build(j.model).expect("tape");
    let want = depth_keys(&tape, j, &AtomicBool::new(false))
        .expect("cpu keys")
        .expect("a tape with calls");
    for &chunk in chunks {
        let got = device_keys(device, j, chunk)
            .unwrap_or_else(|e| panic!("{what} chunk {chunk}: the key kernel failed: {e}"))
            .expect("a tape with calls");
        assert_eq!(got.len(), want.len(), "{what} chunk {chunk}: length");
        if let Some(m) = (0..want.len()).find(|&m| got[m] != want[m]) {
            panic!(
                "{what} chunk {chunk}: member {m} has device key {:?} and CPU key {:?}",
                got[m], want[m]
            );
        }
    }
}

#[test]
fn the_key_kernel_gives_the_cpu_keys_on_a_cuda_device() {
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
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    } else {
        eprintln!("SKIPPED: double precision, which this device does not have");
    }
    let rk4 = Method::Rk4 { dt: 0.05 };
    let members = 1000;
    let chunks = [1, 7, 32, 33, 300, members];
    let ones = vec![1.0; members];

    // Depths from 0 past the cap of 4,096 frames, scattered.
    let sum = build_model(SUM);
    let deep: Vec<f64> = (0..members)
        .map(|i| match i % 10 {
            0 => 0.0,
            1 => 4095.0,
            2 => 4096.0,
            3 => 5000.0,
            _ => (u(i) * 4200.0).floor(),
        })
        .collect();
    // A per-member state, so a member that reads another's row shows.
    let x0: Vec<f64> = (0..members).map(|i| 0.5 + u(i * 3 + 1)).collect();
    let pw = build_model(PW);
    let pw_x0: Vec<f64> = (0..members).map(|i| 0.99 + 0.009 * u(i)).collect();
    let fib = build_model(FIB);
    let fib_n: Vec<f64> = (0..members).map(|i| (u(i) * 22.0).floor()).collect();
    let tree = build_model(TREE);
    let tree_nk: Vec<f64> = (0..members)
        .flat_map(|i| {
            [
                (2.0 + 13.0 * u(i * 5 + 1)).floor(),
                (201.0 * u(i * 11 + 2)).floor(),
            ]
        })
        .collect();
    let drift = build_model(DRIFT);
    let drift_x0 = vec![0.0; members];
    let drift_w: Vec<f64> = (0..members).map(|i| 0.5 + 2.5 * u(i)).collect();

    for &p in &precisions {
        let tag = |name: &str| format!("{name} {}", p.name());

        let mut j = job(&sum, members, &x0, &[], 0.2, rk4, p);
        j.params = Params::PerMember(&deep);
        same_keys(&device, &tag("sum deep"), &j, &chunks);

        // One shared row: the kernel reads it with a stride of 0.
        let shared = [300.0];
        let j = job(&sum, members, &ones, &shared, 0.2, rk4, p);
        same_keys(&device, &tag("sum shared"), &j, &chunks);

        let mut j = job(&pw, members, &pw_x0, &[], 0.2, rk4, p);
        j.params = Params::PerMember(&deep);
        same_keys(&device, &tag("pw deep"), &j, &chunks);

        let mut j = job(&fib, members, &ones, &[], 0.2, rk4, p);
        j.params = Params::PerMember(&fib_n);
        same_keys(&device, &tag("fib"), &j, &chunks);

        let mut j = job(&tree, members, &ones, &[], 0.2, rk4, p);
        j.params = Params::PerMember(&tree_nk);
        same_keys(&device, &tag("tree"), &j, &chunks);

        let mut j = job(&drift, members, &drift_x0, &[], 0.5, rk4, p);
        j.params = Params::PerMember(&drift_w);
        same_keys(&device, &tag("drift"), &j, &chunks);
    }

    // A model without calls has no keys.
    let plain = build_model(PLAIN);
    let j = job(&plain, 4, &[1.0; 4], &[1.0], 0.1, rk4, Precision::Single);
    assert!(device_keys(&device, &j, 4).expect("plain").is_none());
}
