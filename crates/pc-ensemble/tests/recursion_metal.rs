//! Recursive function calls in the Metal kernel on every Metal device (SPEC section 8af-5,
//! stage 4d). The kernel runs a call tape on a per-member stack of 4,096 frames in device
//! memory (buffer 12), so each case runs the CPU backend under the same cap and each device
//! must give the same statuses, counts and bits. Metal has no double type, so only the single
//! precision cases run, against the CPU tape in `f32`. Every model is our own.
//!
//! A machine with no Metal device skips with a `SKIPPED` line, and `PC_REQUIRE_GPU=metal`
//! turns the skip into a failure. All cases run in one function because `force_layout` and
//! `force_members` are process-wide switches and tests in one binary run in parallel.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::batch::force_members;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, list_devices};
use pc_ensemble::kernel::Layout;
use pc_ensemble::metal::force_layout;
use pc_ensemble::run::run_on;
use pc_ensemble::tape::with_max_calls;
use pc_ensemble::{EnsembleResult, Job, Method, Output, Params, Precision, Status};

/// A model run per member: name, source, start states, `k` per member and the statuses.
type Case<'a> = (&'a str, &'a str, &'a [f64], &'a [f64], &'a [Status]);

/// The frames of the GPU stack, which the CPU tape takes as its cap here.
const FRAMES: usize = 4096;

/// Puts the layout and batch switches back, also when a test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_layout(None);
        force_members(None);
    }
}

/// The device run of `j` under layout `l` against the CPU at the GPU cap. Fixed-step RK4 and
/// maps use only `+ - *` and correctly rounded `/`, so they match bit for bit. Dormand-Prince
/// takes a power for its step size, and the Metal `pow` is not promised to match the CPU's, so
/// its states only have to agree to `rel`; the statuses must still be the same.
fn same(device: &DeviceInfo, what: &str, j: &Job, l: Layout) -> EnsembleResult {
    let want = with_max_calls(FRAMES, || run(j));
    force_layout(Some(l));
    let got = run_on(device, j, &AtomicBool::new(false))
        .unwrap_or_else(|e| panic!("{what}: the device run failed: {e}"));
    force_layout(None);
    assert_eq!(got.label.backend, "metal", "{what}");
    assert_eq!(want.status, got.status, "{what}: status");
    match j.method {
        Method::Dp5 { .. } => {
            let rel = match j.precision {
                Precision::Single => 1e-3,
                Precision::Double => 1e-6,
            };
            close(&want.states, &got.states, rel, what);
        }
        _ => {
            assert_eq!(bits(&want.states), bits(&got.states), "{what}: states");
            assert_eq!(bits(&want.t_reached), bits(&got.t_reached), "{what}: t");
            assert_eq!(want.accepted, got.accepted, "{what}: accepted");
        }
    }
    got
}

/// Every finite value of `want` is within `rel` of `got`, relative to `1 + |want|`, and a
/// NaN (a sample a member did not reach) sits at the same place in both.
fn close(want: &[f64], got: &[f64], rel: f64, what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    for (k, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(
            w.is_nan(),
            g.is_nan(),
            "{what}: value {k} is {g:e} against {w:e}"
        );
        if !w.is_nan() {
            assert!(
                (w - g).abs() <= rel * (1.0 + w.abs()),
                "{what}: value {k} is {g:e} on the device and {w:e} on the CPU"
            );
        }
    }
}

fn methods(p: Precision) -> [Method; 2] {
    let dp5 = match p {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.05 }, dp5]
}

#[test]
fn recursive_models_run_on_every_metal_device() {
    let list = list_devices();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Metal)
        .map(|(_, s)| s.clone())
        .expect("metal is listed");
    match state {
        BackendState::Available => {}
        BackendState::Unavailable(why) => return skip_gpu("metal", &why),
        BackendState::NotBuilt => return skip_gpu("metal", "not built into this binary"),
    }
    let devices: Vec<DeviceInfo> = list
        .devices
        .into_iter()
        .filter(|d| d.backend == Backend::Metal)
        .collect();
    assert!(
        !devices.is_empty(),
        "metal is available and lists no device"
    );
    let _reset = Reset;
    for device in &devices {
        eprintln!(
            "metal device {}: {} ({})",
            device.index, device.name, device.driver
        );
        on_device(device);
    }
}

fn on_device(device: &DeviceInfo) {
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    } else {
        eprintln!(
            "SKIPPED: double precision on {}, which it does not have",
            device.name
        );
    }
    let layouts = [Layout::Private, Layout::Global];

    let cases: [Case; 7] = [
        (
            "fib",
            "par k=5\nfib(n)=if(n<2)then(n)else(fib(n-1)+fib(n-2))\nx'=-x*fib(k)/8\ndone\n",
            &[1.0, 0.9, 0.8, 0.7, 0.6],
            &[2.0, 3.0, 5.0, 6.0, 7.0],
            &[Status::Ok; 5],
        ),
        (
            "evod",
            "par k=3\nev(n,a)=if(n<1)then(a)else(od(n-1))\n\
             od(n)=if(n<1)then(0)else(ev(n-1,x))\nx'=-x*(1+ev(k,0.5))\ndone\n",
            &[1.0, 0.9, 0.8, 0.7, 0.6],
            &[0.0, 1.0, 2.0, 3.0, 6.0],
            &[Status::Ok; 5],
        ),
        (
            "state",
            "par k=2\nf(n)=if(n<1)then(1)else(x*f(n-1))\nx'=-f(k)\ndone\n",
            &[0.5, 0.6, 0.7, 0.8],
            &[0.0, 1.0, 2.0, 3.0],
            &[Status::Ok; 4],
        ),
        (
            "argfree",
            "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\n\
             f(n)=if(n<1)then(1)else(f(n-1)+s(3)*x)\nx'=-x*f(k)/10\ndone\n",
            &[1.0, 0.9, 0.8, 0.7],
            &[0.0, 1.0, 4.0, 9.0],
            &[Status::Ok; 4],
        ),
        (
            "shared",
            "par k=3\nh(n)=if(n<1)then(0)else(if(n>2)then(h(n-1)*2)else(h(n-1)+1))\n\
             x'=-x*h(k)/20\ndone\n",
            &[1.0, 0.9, 0.8, 0.7],
            &[0.0, 1.0, 3.0, 6.0],
            &[Status::Ok; 4],
        ),
        (
            "reenter",
            "par k=0\nf(n)=if(n<1)then(1)else(f(3))\nx'=-x*f(k)\ndone\n",
            &[1.0, 0.9, 0.8],
            &[0.0, 3.0, 0.0],
            &[Status::Ok, Status::RecursionDepth, Status::Ok],
        ),
        (
            // `s(4095)` takes 4,096 frames and `s(4096)` one more.
            "cap",
            "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n",
            &[1.0, 2.0, 3.0],
            &[4095.0, 4096.0, 5.0],
            &[Status::Ok, Status::RecursionDepth, Status::Ok],
        ),
    ];
    for (name, src, x0, k, want) in cases {
        let m = build_model(src);
        for &p in &precisions {
            for method in methods(p) {
                for l in layouts {
                    let what = format!("{name} {} {method:?} {l:?}", p.name());
                    let mut j = job(&m, k.len(), x0, &[], 1.0, method, p);
                    j.params = Params::PerMember(k);
                    let r = same(device, &what, &j, l);
                    assert_eq!(r.status, want, "{what}");
                }
            }
        }
    }

    // A member that reaches the cap mid-run: `x' = 1` from 0.4 and the depth of `y'` passes
    // 4,096 frames once `x` passes 0.68.
    let m = build_model("s(n)=if(n<1)then(0)else(1+s(n-1))\nx'=1\ny'=s(x*6000)*0\ndone\n");
    for &p in &precisions {
        let (rtol, atol) = match p {
            Precision::Double => (1e-8, 1e-10),
            Precision::Single => (1e-4, 1e-6),
        };
        let dp5 = Method::Dp5 {
            rtol,
            atol,
            hmax: 0.05,
        };
        for method in [dp5, Method::Rk4 { dt: 0.1 }] {
            for l in layouts {
                let what = format!("midrun {} {method:?} {l:?}", p.name());
                let mut j = job(&m, 2, &[0.4, 0.0, 0.1, 0.0], &[], 0.5, method, p);
                j.output = Output::Samples(vec![0.1, 0.2, 0.5]);
                let r = same(device, &what, &j, l);
                assert_eq!(r.status, [Status::RecursionDepth, Status::Ok], "{what}");
                assert_eq!(r.rejected[0], 0, "{what}");
                assert!(r.accepted[0] > 0, "{what}");
            }
        }
    }

    // A map member past the cap.
    let m = build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx(t+1)=s(k)*x/k/2\ndone\n");
    for &p in &precisions {
        for l in layouts {
            let what = format!("map {} {l:?}", p.name());
            let mut j = job(&m, 3, &[1.0, 2.0, 3.0], &[], 10.0, Method::Iter, p);
            let k = [3.0, 5000.0, 4095.0];
            j.params = Params::PerMember(&k);
            let r = same(device, &what, &j, l);
            assert_eq!(
                r.status,
                [Status::Ok, Status::RecursionDepth, Status::Ok],
                "{what}"
            );
        }
    }

    // Batches of two members give the same result as one launch.
    let m = build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n");
    let k = [3.0, 4096.0, 5.0, 40.0, 4095.0];
    let x0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    for &p in &precisions {
        let mut j = job(&m, 5, &x0, &[], 1.0, Method::Rk4 { dt: 0.1 }, p);
        j.params = Params::PerMember(&k);
        let whole = run_on(device, &j, &AtomicBool::new(false)).expect("one launch");
        force_members(Some(2));
        let what = format!("batches {}", p.name());
        let parts = same(device, &what, &j, Layout::Private);
        force_members(None);
        assert_eq!(bits(&whole.states), bits(&parts.states), "{what}");
        assert_eq!(whole.status, parts.status, "{what}");
        assert_eq!(parts.status[1], Status::RecursionDepth, "{what}");
    }
}
