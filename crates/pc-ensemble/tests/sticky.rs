//! A sticky CUDA error and the recovery code of the CUDA backend, on a real device.
//!
//! NVIDIA documents that after an illegal address or a launch failure the process must be
//! terminated and started again. The backend answers a device error by dropping its context
//! and making a new one on the next run, which is right only if a new primary context starts
//! clean in the same process. This test makes the error with a PTX kernel that writes through a
//! null pointer, then looks at what a new context does. It changes the state of the GPU for
//! the whole process, so it runs only with `PC_RUN_STICKY=1`, and it prints what it found.
//!
//! `PC_RUN_STICKY=1 cargo test -p pc-ensemble --test sticky -- --ignored --nocapture`

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Writes 1 through the null pointer.
const BAD: &str = ".version 7.5\n.target sm_75\n.address_size 64\n\
.visible .entry bad()\n{\n.reg .b64 %rd<2>;\n.reg .b32 %r<2>;\n\
mov.u64 %rd1, 0;\nmov.u32 %r1, 1;\nst.global.u32 [%rd1], %r1;\nret;\n}\n";

/// Writes 42 through its argument.
const GOOD: &str = ".version 7.5\n.target sm_75\n.address_size 64\n\
.visible .entry good(.param .u64 p)\n{\n.reg .b64 %rd<2>;\n.reg .b32 %r<2>;\n\
ld.param.u64 %rd1, [p];\nmov.u32 %r1, 42;\nst.global.u32 [%rd1], %r1;\nret;\n}\n";

fn cfg() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Runs the good kernel on a fresh context and returns the value it wrote.
fn good_run() -> Result<u32, String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("create context: {e:?}"))?;
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(GOOD))
        .map_err(|e| format!("load module: {e:?}"))?;
    let f = module
        .load_function("good")
        .map_err(|e| format!("load function: {e:?}"))?;
    let mut buf = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| format!("alloc: {e:?}"))?;
    let mut b = stream.launch_builder(&f);
    b.arg(&mut buf);
    // SAFETY: the kernel takes one device pointer and writes one u32 through it.
    unsafe { b.launch(cfg()) }.map_err(|e| format!("launch: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("synchronize: {e:?}"))?;
    let host = stream
        .clone_dtoh(&buf)
        .map_err(|e| format!("copy: {e:?}"))?;
    Ok(host[0])
}

#[test]
#[ignore = "poisons the CUDA context of the process: set PC_RUN_STICKY=1 and pass --ignored"]
fn a_new_context_after_a_sticky_error() {
    if std::env::var("PC_RUN_STICKY").as_deref() != Ok("1") {
        eprintln!("SKIPPED: PC_RUN_STICKY is not 1");
        return;
    }
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_eq!(good_run(), Ok(42), "the good kernel works before the fault");

    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_src(BAD)).expect("load bad");
    let f = module.load_function("bad").expect("function bad");
    let mut b = stream.launch_builder(&f);
    // SAFETY: this launch is meant to fault. It takes no arguments.
    let launched = unsafe { b.launch(cfg()) };
    let synced = stream.synchronize();
    eprintln!("bad launch: {launched:?}");
    eprintln!("bad synchronize: {synced:?}");
    assert!(
        launched.is_err() || synced.is_err(),
        "the null write must fail"
    );
    eprintln!(
        "alloc on the same context: {:?}",
        stream.alloc_zeros::<u32>(1).map(|_| ())
    );

    drop(b);
    drop(f);
    drop(module);
    drop(stream);
    drop(ctx);
    let again = good_run();
    eprintln!("good run on a new context: {again:?}");
    // Found on the RTX 5060 test machine (driver 13.x, WSL2): the new context fails to be
    // made, with the same error. If this changes, the recovery of the CUDA backend can go
    // back to working in the same process, and SPEC section 9 and the note in `cuda.rs` need
    // an update.
    assert!(
        again.is_err_and(|m| m.contains("ILLEGAL_ADDRESS")),
        "a new context after error 700 no longer fails: update SPEC and the STICKY note"
    );
}
